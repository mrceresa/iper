from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid

from mesa_geo.geoagent import GeoAgent, AgentCreator

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import random
import numpy as np

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon, LineString, Point
import contextily as ctx

from io import StringIO
import json

from iper import GeoSpacePandas

import logging
logging.basicConfig(level = logging.DEBUG)
_log = logging.getLogger(__name__)

class VirusInformation(object):
  def __init__(self):
    self.r0 = 2.5

class Human(GeoAgent):
  def __init__(self, unique_id, model, shape, probs=None):
    super().__init__(unique_id, model, shape)
    # Markov transition matrix
    self._trans = probs
    self._vel1step = 0.4 #Km per hora

  def place_at(self, newPos):
    self.model.place_at(self, newPos)
  
  def get_pos(self):
    return (self.shape.x, self.shape.y)

  def step(self):
    _log.debug("*** Agent %d stepping"%self.unique_id) 
    nx = random.uniform(-1.0, 1.0)*self._vel1step / self.model._xs["dx"]
    ny = random.uniform(-1.0, 1.0)*self._vel1step / self.model._xs["dy"]
    ox, oy = self.get_pos()
    newPos = Point(ox + nx, oy + ny)

    self.place_at(newPos)
    #neighbors = self.model.grid.get_neighbors(self)

  def __repr__(self):
    return "Agent " + str(self.unique_id)
    

class BCNCovid2020(Model):

  def __init__(self, N, basemap):
    super().__init__()
    _log.info("Initalizing model")    
    
    self._basemap = basemap
    self.grid = GeoSpacePandas()
    self.transport_grid = NetworkGrid()
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusInformation()

    _log.info("Loading shapefiles")
    self.loadShapefiles()

    _log.info("Loading Trajectories")
    self.load_trajectories()
    self.start_times = []
    self.get_start_times_sorted()
    self.traj_count = 0
    
    _log.info("Initalizing agents")
    self.createAgents(N)   

  def place_at(self, agent, loc):
    if self._xs["bbox"].contains(loc):
      self.grid.update_shape(agent, loc)

  def createAgents(self, N):
    #base = self._xs["centroid"]
    AC = AgentCreator(Human, {"model": self})
    agents = []
    for i in range(N):
      _a = AC.create_agent( 
        Point( 
          random.uniform(self._xs["w"],self._xs["e"]),
          random.uniform(self._xs["n"],self._xs["s"])
        ), i)
      agents.append(_a)
    
    _log.info("Adding %d agents..."%len(agents))
    self.grid.add_agents(agents)
    for agent in agents:
      self.schedule.add(agent)

  #Maybe will not be used....
  def createTransportAgents(self, traj_list, transport_type): 
    #traj_list: list of integers being the id of the trajectories in collection
    AC = AgentCreator(transport_type , {"model", self})
    agents = []
    for i in traj_list:
      _a = AC.create_agent( 
        Point( 
          collection.get_trajectory(i).get_start_location().x,
          collection.get_trajectory(i).get_start_location().y
        ), i)
      agents.append(_a)

    _log.info("Adding %d agent... of type ......."%len(agents))
    self.transport_grid.add_agents(agents)
    for agent in agents: 
      self.schedule.add(agent)

  def createTransportAgent(self, trajectory, traj_id, transport_type): 
    AC = AgentCreator(transport_type , {"model", self})
    _a = AC.create_agent( 
        Point( 
          trajectory.get_start_location().x,
          trajectory.get_start_location().y
        ), traj_id)

    self.transport_grid.add_agents(_a)
    self.schedule.add(_a)

  def removeTransportAgents(self):
    # Two options:
    # 1: Create a get_end_times and perform like the creation of instances but to eraise them, 
    # 2: Check when ever they arrived at its end location and eraise them then. 
    pass

  def plotAll(self):

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()

    ctx.plot_map(self._loc, ax=ax1)
    _c = ["red", "blue"]
    for i, _r in enumerate(self._roads):
      _r.plot(ax=ax1, facecolor='none', edgecolor=_c[i])    
    plt.tight_layout()

    # Plot agents
    self.grid._agdf.plot(ax=ax1)
    
  def loadShapefiles(self):

    self._loc = ctx.Place(self._basemap, zoom_adjust=0)  # zoom_adjust modifies the auto-zoom
    # Print some metadata
    self._xs = {}
    
    # Longitude w,e Latitude n,s
    for attr in ["w", "s", "e", "n", "place", "zoom", "n_tiles"]:
      self._xs[attr] = getattr(self._loc, attr)
      print("{}: {}".format(attr, self._xs[attr]))

    self._xs["centroid"] = LineString(
        (
          (self._xs["w"], self._xs["s"]),
          (self._xs["e"], self._xs["n"])
        )
      ).centroid

    self._xs["bbox"] = Polygon.from_bounds(
          self._xs["w"], self._xs["s"],
          self._xs["e"], self._xs["n"]
        )

    self._xs["dx"] = 111.32; #One degree in longitude is this in KM 
    self._xs["dy"] = 40075 * math.cos( self._xs["centroid"].y ) / 360 
    _log.info("Arc amplitude at this latitude %f, %f"%(self._xs["dx"], self._xs["dy"]))

    path = os.getcwd()
    _log.info("Loading geo data from path:"+path)
    roads_1 = gpd.read_file(os.path.join(path, "shapefiles","1","roads-line.shp"))
    roads_2 = gpd.read_file(os.path.join(path, "shapefiles","2","roads-line.shp"))
    self._roads = [roads_1, roads_2]

  def load_trajectories(self):
    _log.info("Loading GTFS data")
    trips = pd.read_csv(ROOT_DIR + '/GTFS/bus_metro/trips.txt', dtype = str).head(1)
    routes = pd.read_csv(ROOT_DIR + '/GTFS/bus_metro/routes.txt', dtype = str)
    st = pd.read_csv(ROOT_DIR + "/GTFS/bus_metro/stop_times.txt", dtype = str)
    stops = pd.read_csv(ROOT_DIR + "/GTFS/bus_metro/stops.txt", dtype = str)
    calendar = pd.read_csv(ROOT_DIR + "/GTFS/bus_metro/calendar_dates.txt", dtype = str)
    _log.info("GTFS data Loaded")

    # Add Geometry
    stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat))
    _log.info("Geometry added")

    # Pass to datetime.date
    t_dates = []
    for item in calendar['date']:
        date = datetime(year=int(item[0:4]), month=int(item[4:6]), day=int(item[6:8]))
        t_dates.append(date)        

    calendar['t_date'] = t_dates
    _log.info("date format changed")

    # Join together arrival and departure times
    st.loc[st['trip_id'] == '1.1.2373316']
    a = st[['trip_id','arrival_time', 'stop_id', 'stop_sequence']]
    b = st[['trip_id','departure_time', 'stop_id', 'stop_sequence']]
    a.loc[:,'time'] = a.loc[:,'arrival_time']
    b.loc[:,'time'] = b.loc[:,'departure_time']
    st = pd.concat([a,b])
    st = st.drop(columns=['arrival_time', 'departure_time'])
    st.sort_values(['trip_id', 'stop_id'])

    # Pass to datetime.time
    t = []
    for each_time in st['time']:
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

    st['t_time'] = t
    _log.info("time format changed")

    # Merge the tables to get the desired attributes. 
    stops_by_route = pd.merge(pd.merge(pd.merge(trips[['trip_id','route_id','service_id']],
                                        st[['trip_id','stop_id','stop_sequence','t_time']]),
                                        stops[['stop_id','geometry']]),
                                        routes[['route_id', 'route_type']])
    stops_by_route_date = pd.merge(stops_by_route,calendar[['service_id','t_date']])
    _log.info("Tables merged")

    # Group the trips by trip_id which are the trayectories on global dates. --- Expected lenght 47967 == # trips
    grouped = stops_by_route_date.groupby(stops_by_route_date.trip_id)

    # For each trip get all the days its running
    traj_id = 0
    trip_num = 0
    count = 0
    traj_list = []
    for t_id in trips.trip_id:
        trajectories_in_trip = grouped.get_group(t_id)
        dates = trajectories_in_trip.loc[:,'t_date'].unique()
        grouped_by_date = trajectories_in_trip.groupby(trajectories_in_trip.t_date) 

        for day in dates:
            trajectory = grouped_by_date.get_group(day)
            trajectory.loc[:,'t'] = trajectory.loc[:,'t_time'] + trajectory.loc[:,'t_date']
            trajectory.set_index('t', inplace=True)
            gdf_trajectory = gpd.GeoDataFrame(trajectory, crs=CRS(31256))
            traj = mpd.Trajectory(gdf_trajectory, traj_id)
            traj_list.append(traj)
            #traj.plot()
        
            traj_id += 1
            if traj_id%100 == 0:
                print(traj_id)
                
        trip_num += 1
        if trip_num%100 == 0:
            print('trip num is', trip_num)
            
        count += 1
        if count == 5:
            break

    self.collection = mpd.TrajectoryCollection(traj_list)

  def get_start_times_sorted(self):
    for traj_id, trajectory in enumerate(self.collection):
      self.start_times.append([trajectory.get_start_time(),traj_id])
    sorted(self.start_times)
    
  def getStartingTrajectories(self, datetype):
    while self.traj_count < len(self.start_times): 
      if self.datetime == self.start_times[self.traj_count][0]:
        trajectory = collection.get_trajectory(self.start_times[self.traj_count][1])
        route_type = trajectory.df['route_type'].unique()
        if route_type == 1:
          self.createTransportAgent(trajectory, self.traj_count, Subway)
        elif route_type == 3:
          self.createTransportAgent(trajectory, self.traj_count, Bus)
        elif route_type == 7:
          self.createTransportAgent(trajectory, self.traj_count, Funicular)
        else:
          _log.warning("Transport type error, id: %d"%(self.traj_count))
            
        self.traj_count += 1
      else: 
        break

  def step(self):
    #Somehow I need the time for the step. 
    self.getStartingTrajectories() 
    #self.createTransportAgents(traj_list = ,transport_type = ) Inside getStartingTrajectories
    self.schedule.step()
    self.removeTransportAgents()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()

    

