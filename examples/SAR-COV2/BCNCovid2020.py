from mesa import Agent, Model
from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace

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

import logging
_log = logging.getLogger(__name__)

class BCNAgent(GeoAgent):

  def __init__(self, unique_id, model, destination, initial_pos):
    super(BCNAgent, self).__init__(unique_id, model)

  def step(self):
    _log.info("Not implemented")


class VirusInformation(object):
  def __init__(self):
    self.r0 = 2.5

class Human(GeoAgent):
  def __init__(self, unique_id, model, shape, probs=None):
    super().__init__(unique_id, model, shape)
    # Markov transition matrix
    self._trans = probs

  def step(self):
    _log.debug("*** Agent %d stepping"%self.unique_id)    
    #x = self.random.randrange(self.grid.width)
    #y = self.random.randrange(self.grid.height)
    #self.grid.place_agent(a, (x, y))  
    
    #neighbors = self.model.grid.get_neighbors(self)

  def __repr__(self):
    return "Agent " + str(self.unique_id)
    
    
class BCNCovid2020(Model):

  def __init__(self, N, basemap):
    _log.info("Initalizing model")    
    
    self._basemap = basemap
    self.grid = GeoSpace()
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusInformation()

    _log.info("Loading shapefiles")

    self.loadShapefiles()
    
    _log.info("Initalizing agents")
    self.createAgents(N)
    self.plotAll()    

  def createAgents(self, N):
  
    _h = """
    { 
      "type": "FeatureCollection",
      "crs": { 
        "type": "name", 
        "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } 
      },
      "features": []
    }
    """

    _ioH = StringIO(_h)
    _jsH = json.load(_ioH)

    _ag0 = """
          { "type": "Feature", "id": 0, "properties": {},
            "geometry": { "type": "Point", 
              "coordinates": [] 
            } 
          }"""
    
    
    base = self._xs["centroid"]
    
    for i in range(N):
      _ioAg = StringIO(_ag0)
      _jsAg = json.load(_ioAg)
      _jsAg["id"] = i 
      _jsAg["geometry"]["coordinates"] = [
          random.uniform(self._xs["w"],self._xs["e"]), 
          random.uniform(self._xs["n"],self._xs["s"])
          ]

      _jsH["features"].append( _jsAg )

    AC = AgentCreator(Human, {"model": self})
    agents = AC.from_GeoJSON(_jsH)
    _log.info("Adding %d agents..."%len(agents))
    self.grid.add_agents(agents)
    for agent in agents:
      self.schedule.add(agent)
     
  def plotAll(self):

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()
    
    ctx.plot_map(self._loc, ax=ax1)

    _c = ["red", "blue"]
    for i, _r in enumerate(self._roads):
      _r.plot(ax=ax1, facecolor='none', edgecolor=_c[i])    
      
    # Plot agents

    agentFeatures = self.grid.__geo_interface__
    gdf = gpd.GeoDataFrame.from_features(agentFeatures)
    gdf.plot(ax=ax1)
    
    
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
      
    self._xs["dx"] = 111.32; 
    self._xs["dy"] = 40075 * math.cos( self._xs["centroid"].y ) / 360
    _log.info("Arc amplitude at this latitude %f, %f"%(self._xs["dx"], self._xs["dy"]))

    path = os.getcwd()
    _log.info("Loading geo data from path:"+path)
    roads_1 = gpd.read_file(os.path.join(path, "shapefiles","1","roads-line.shp"))
    roads_2 = gpd.read_file(os.path.join(path, "shapefiles","2","roads-line.shp"))
    self._roads = [roads_1, roads_2]

  def step(self):
    self.schedule.step()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()

    
    

