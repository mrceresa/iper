from mesa import Agent, Model
from mesa.time import RandomActivation

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
_log = logging.getLogger(__name__)
import mapclassify
import geoplot
import attributes_Agent
class VirusInformation(object):
  def __init__(self):
    self.r0 = 2.5

class Human(GeoAgent):
  def __init__(self, unique_id, model, shape,family, job, age_group, probs=None):
    super().__init__(unique_id, model, shape)
    # Markov transition matrix
    self.family = family
    self.job = job
    #self.state = state
    self.age_group=age_group
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

  def __init__(self, N, basemap, distr_family, distr_job, distr_age):
    super().__init__()
    _log.info("Initalizing model")
    self.distr_family=distr_family        
    self.families = attributes_Agent.create_families(N,distr_family)
    self.distr_job=distr_job   
    self.distr_age=distr_age
    self._basemap = basemap
    self.grid = GeoSpacePandas()
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusInformation()

    _log.info("Loading shapefiles")

    self.loadShapefiles()
    
    _log.info("Initalizing agents")
    self.createAgents(N,self.families,self.distr_job,self.distr_age)   

  def place_at(self, agent, loc):
    if self._xs["bbox"].contains(loc):
      self.grid.update_shape(agent, loc)

  def createAgents(self, N, families, distr_job, distr_age):
    indice_family=0
    families_progressiv=[]
    families_progressiv.append(families[0]) 

    base = self._xs["centroid"]
    #AC = AgentCreator(Human, {"model": self,"family":"1-2","job":"1-2","age_group":"1-2"})
    agents = []
    for i in range(N):
      contatore_fam=1+i           
      if sum(families_progressiv)<contatore_fam:
        indice_family+=1
        families_progressiv.append(families[indice_family])
      age=attributes_Agent.age_(distr_age)
      work=attributes_Agent.job(distr_job)
      AC = AgentCreator(Human, {"model": self,"family":indice_family,"job":work,"age_group":age})
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
     
  def plotAll(self,figname):
    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()
    
    tot_people =self._blocks["density"]
    scheme = mapclassify.Quantiles(tot_people, k=5) 
 
    geoplot.choropleth(
    self._blocks, hue=tot_people, scheme=scheme,
    cmap='Oranges', figsize=(12, 8), ax=ax1
    )
    plt.savefig('density-'+figname)

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()

    ctx.plot_map(self._loc, ax=ax1)
    _c = ["red", "blue"]
    for i, _r in enumerate(self._roads):
      _r.plot(ax=ax1, facecolor='none', edgecolor=_c[i])
    self._blocks.plot(ax=ax1,facecolor='none', edgecolor="black")  
    

    plt.tight_layout()

    # Plot agents
    self.grid._agdf.plot(ax=ax1)
    plt.savefig('agents-'+figname)
    
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
    blocks = gpd.read_file(os.path.join(path, "shapefiles","quartieriBarca1.shp"))
    self._roads = [roads_1, roads_2]
    self._blocks= blocks

  def step(self):
    self.schedule.step()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()

    
    

