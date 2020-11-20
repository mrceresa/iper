#latitude how much north, lon how much east

from mesa import Agent, Model
from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent, AgentCreator

import math
import networkx as nx
import osmnx as ox
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
_log = logging.getLogger(__name__)

from Mobility import Map_to_Graph

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
    nx = random.uniform(-1.0, 1.0)*self._vel1step / self.model.boundaries["dx"]
    ny = random.uniform(-1.0, 1.0)*self._vel1step / self.model.boundaries["dy"]
    ox, oy = self.get_pos()
    newPos = oy + ny, ox + nx
    lat, lng = self.model.driveMap.get_lat_lng_from_point(newPos)
    newPosNode = Point(lng,lat)
    self.place_at(newPosNode)
    #neighbors = self.model.grid.get_neighbors(self)

  def __repr__(self):
    return "Agent " + str(self.unique_id)
       
class BCNCovid2020(Model):

  def __init__(self, N, basemap):
    super().__init__()
    _log.info("Initalizing model")    
    
    self._basemap = basemap
    self.grid = GeoSpacePandas()
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusInformation()

    _log.info("Loading shapefiles")

    self.loadShapefiles()
    
    _log.info("Initalizing agents")
    self.createAgents(N)   

  def place_at(self, agent, loc):
    if self.boundaries["bbox"].contains(loc):          #mirar si el punt està dins del mapa
      self.grid.update_shape(agent, loc)
    else:
      _log.info("Agent out of range")

  def createAgents(self, N):
      
    base = self.boundaries["centroid"]
    AC = AgentCreator(Human, {"model": self})
    agents = []
    for i in range(N):
      point = random.uniform(self.boundaries["n"],self.boundaries["s"]), random.uniform(self.boundaries["w"],self.boundaries["e"])
      lat, lng = self.driveMap.get_lat_lng_from_point(point)
      
      _a = AC.create_agent(Point(lng,lat), i)
      agents.append(_a)
    
    _log.info("Adding %d agents..."%len(agents))
    self.grid.add_agents(agents)
    for agent in agents:
      self.schedule.add(agent)
     
  def plotAll(self):
    fig, ax1 = plt.subplots(figsize=(8,8))

    self.driveMap.plot_graph(ax = ax1, figsize = fig, node_size = 0, show=False, close=False)
    plt.tight_layout()

    # Plot agents
    self.grid._agdf.plot(ax=ax1)

  # def plotAll(self):

  #   fig = plt.figure(figsize=(15, 15))
  #   ax1 = plt.gca()

  #   ctx.plot_map(self._loc, ax=ax1)
  #   _c = ["red", "blue"]
  #   for i, _r in enumerate(self._roads):
  #     _r.plot(ax=ax1, facecolor='none', edgecolor=_c[i])    
  #   plt.tight_layout()

  #   # Plot agents
  #   self.grid._agdf.plot(ax=ax1) 
    
  def loadShapefiles(self):
    self.driveMap = Map_to_Graph('Barcelona, Spain', 'drive')  #Load the shapefiles 

    self.boundaries = self.driveMap.get_boundaries()
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

    

  # def loadShapefiles(self):

  #   self._loc = ctx.Place(self._basemap, zoom_adjust=0)  # zoom_adjust modifies the auto-zoom
  #   # Print some metadata
  #   self._xs = {}
    
  #   # Longitude w,e Latitude n,s
  #   for attr in ["w", "s", "e", "n", "place", "zoom", "n_tiles"]:
  #     self._xs[attr] = getattr(self._loc, attr)
  #     print("{}: {}".format(attr, self._xs[attr]))

  #   self._xs["centroid"] = LineString(
  #       (
  #         (self._xs["w"], self._xs["s"]),
  #         (self._xs["e"], self._xs["n"])
  #       )
  #     ).centroid

  #   self._xs["bbox"] = Polygon.from_bounds(
  #         self._xs["w"], self._xs["s"],
  #         self._xs["e"], self._xs["n"]
  #       )

  #   self._xs["dx"] = 111.32; #One degree in longitude is this in KM 
  #   self._xs["dy"] = 40075 * math.cos( self._xs["centroid"].y ) / 360 
  #   _log.info("Arc amplitude at this latitude %f, %f"%(self._xs["dx"], self._xs["dy"]))

  #   path = os.getcwd()
  #   _log.info("Loading geo data from path:"+path)
  #   roads_1 = gpd.read_file(os.path.join(path, "shapefiles","1","roads-line.shp"))
  #   roads_2 = gpd.read_file(os.path.join(path, "shapefiles","2","roads-line.shp"))
  #   self._roads = [roads_1, roads_2]

  def step(self):
    self.schedule.step()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()



    
    

