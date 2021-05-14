from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from Mobility import RouteAgent, Map_to_Graph, Human

from mesa_geo.geoagent import GeoAgent, AgentCreator

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import random
import numpy as np

import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon, LineString, Point
import contextily as ctx

from io import StringIO
import json

from iper import GeoSpacePandas
import osmnx as ox

import logging
logging.basicConfig(level = logging.DEBUG)
_log = logging.getLogger(__name__)

class VirusInformation(object):
  def __init__(self):
    self.r0 = 2.5

    

class BCNCovid2020(Model):

  def __init__(self, N, basemap):
    super().__init__()
    _log.info("Initalizing model")   
    self.DateTime = datetime(year=2021, month=1, day=1, hour= 0, minute=0, second=0) 
    
    self._basemap = basemap
    self.grid = GeoSpacePandas()
    #self.transport_grid = NetworkGrid()
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusInformation()

    _log.info("Loading shapefiles")
    self.loadShapefiles()

    _log.info("Loading Trajectories")
    #self.load_public_transport_dataset()

    _log.info("Initalizing agents")
    self.createAgents(N) 

    _log.info("Initializing Routes")
    #self.createRouteAgents() 

  def place_at(self, agent, loc):
    #if self.boundaries["bbox"].contains(loc):
    self.grid.update_shape(agent, loc)

  def createAgents(self, N):
    #base = self._xs["centroid"]

    AC = AgentCreator(Human, {"model": self})
    agents = []
 
    for i in range(N):
      random_node = random.choice(list(self.walkMap.G_proj.nodes))
      x, y = self.walkMap.G_proj.nodes[random_node]['x'],self.walkMap.G_proj.nodes[random_node]['y']
      _a = AC.create_agent(Point(x,y), i)
      agents.append(_a)
    
    _log.info("Adding %d agents..."%len(agents))
    self.grid.add_agents(agents)
    for agent in agents:
      self.schedule.add(agent)

  def createRouteAgents(self):
    for route in self.routes.loc[:,'route_id']:
      a = RouteAgent(route, self)
      self.schedule.add(a)
      _log.info("Route Agent created %s"%str(route))

  def plotAll(self):

    fig, ax1 = plt.subplots(figsize=(8,8))
    self.walkMap.plot_graph(ax = ax1, figsize = fig, node_size = 0, show=False, close=False)
    plt.tight_layout()
    # Plot agents
    self.grid._agdf.plot(ax=ax1)
    
  def loadShapefiles(self):
    self.walkMap = Map_to_Graph(self._basemap, 'walk')  #Load the shapefiles 
    self.boundaries = self.walkMap.get_boundaries()
    print(self.boundaries) 
    self.boundaries['centroid'] = LineString(
        (
          (self.boundaries["w"], self.boundaries["s"]),
          (self.boundaries["e"], self.boundaries["n"])
        )
      ).centroid
    self.boundaries["bbox"] = Polygon.from_bounds(
      self.boundaries["w"], self.boundaries["s"],
      self.boundaries["e"], self.boundaries["n"])
    self.boundaries["dx"] = 111.32; #One degree in longitude is this in KM 
    self.boundaries["dy"] = 40075 * math.cos( self.boundaries["centroid"].y ) / 360 
    _log.info("Arc amplitude at this latitude %f, %f"%(self.boundaries["dx"], self.boundaries["dy"]))

  def load_public_transport_dataset(self):
    _log.info("Loading GTFS data")
    path = os.getcwd()
    self.trips = pd.read_csv(path + '/GTFS/bus_metro/trips.txt', dtype = str)
    self.routes = pd.read_csv(path + '/GTFS/bus_metro/routes.txt', dtype = str)
    self.st = pd.read_csv(path + "/GTFS/bus_metro/stop_times.txt", dtype = str)
    self.stops = pd.read_csv(path + "/GTFS/bus_metro/stops.txt", dtype = str)
    self.calendar = pd.read_csv(path + "/GTFS/bus_metro/calendar_dates.txt", dtype = str)
    _log.info("GTFS data Loaded")
    self.routes = self.routes.loc[self.routes['route_id'] == '2.129.2968']

    # Add Geometry
    self.stops = gpd.GeoDataFrame(self.stops, geometry=gpd.points_from_xy(self.stops.stop_lon, self.stops.stop_lat))
    _log.info("Geometry added")

    # Pass to datetime.date
    t_dates = []
    for item in self.calendar['date']:
        date = datetime(year=int(item[0:4]), month=int(item[4:6]), day=int(item[6:8])).date()
        t_dates.append(date)        
    self.calendar['t_date'] = t_dates
    _log.info("date format changed")

    # Join together arrival and departure times
    a = self.st[['trip_id','arrival_time', 'stop_id', 'stop_sequence']]
    b = self.st[['trip_id','departure_time', 'stop_id', 'stop_sequence']]
    a.loc[:,'time'] = a.loc[:,'arrival_time']
    b.loc[:,'time'] = b.loc[:,'departure_time']
    self.st = pd.concat([a,b])
    self.st = self.st.drop(columns=['arrival_time', 'departure_time'])
    self.st.sort_values(['trip_id', 'stop_id'])

    # Pass to datetime.time
    t = []
    for each_time in self.st['time']:
        try:
            h, m, s = each_time.split(':')
            each_t_time = timedelta(hours=int(h),minutes=int(m),seconds=int(s))
            t.append(each_t_time)
        except:
            if each_time is np.nan: 
                t.append(np.nan)
            else:
                print('error on a_time')
                break

    self.st['t_time'] = t
    _log.info("time format changed")
    
  def step(self):
    self.schedule.step()
    
  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()
      self.DateTime += timedelta(seconds=60)

    

#https://opendata-ajuntament.barcelona.cat/data/en/dataset/qualitat-aire-detall-bcn


#Modelo regression linial cuando contaminacion hay si hay tanto trafico. 

#TIempo Dinero

#Gerarchicla planing   https://github.com/oubiwann/pyhop
