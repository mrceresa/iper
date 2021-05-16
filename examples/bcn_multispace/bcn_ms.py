from examples.EudaldMobility.Mobility import Map_to_Graph
from mesa import Agent, Model
from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import random
import numpy as np

import geopandas as gpd
import geoplot
import pandas as pd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon, LineString, Point
import contextily as ctx

from io import StringIO
import json

from iper import GeoSpacePandas
from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
from iper.space.Space import MeshSpace

import networkx as nx

import mapclassify

from shapely.geometry import Polygon
import numpy as np

import logging
from random import uniform
import time

from ...EudaldMobility.Mobility import Map_to_Graph

class CityModel(MultiEnvironmentWorld):

  def __init__(self, config):
    super().__init__(config)
    self.l.info("Initalizing model")
    self._basemap = config["basemap"]
    self.space = GeoSpacePandas()
    self.network = NetworkGrid(nx.Graph())
    self.l.info("Scheduler is " + str(self.schedule))
    self.schedule = RandomActivation(self)

    self.l.info("Loading geodata")
    self._initGeo()
    self._loadGeoData()
    
    self.loadShapefiles() # Eudald Mobility
  
  def loadShapefiles(self):
    self.walkMap = Map_to_Graph(self._basemap, 'walk')  #Load the shapefiles 
    self.boundaries = self.walkMap.get_boundaries()
    
    self.boundaries['centroid'] = LineString(
        (
          (self.boundaries["w"], self.boundaries["s"]),
          (self.boundaries["e"], self.boundaries["n"])
        )).centroid
    
    self.boundaries["bbox"] = Polygon.from_bounds(
      self.boundaries["w"], self.boundaries["s"],
      self.boundaries["e"], self.boundaries["n"])
    
    self.boundaries["dx"] = 111.32; #One degree in longitude is this in KM 
    self.boundaries["dy"] = 40075 * math.cos( self.boundaries["centroid"].y ) / 360 
    self.l.info("Arc amplitude at this latitude %f, %f"%(self.boundaries["dx"], self.boundaries["dy"]))

  def _initGeo(self):
    #Initialize geo data
    self._loc = ctx.Place(self._basemap, zoom_adjust=0)  # zoom_adjust modifies the auto-zoom
    # Print some metadata
    self._xs = {}
    
    # Longitude w,e Latitude n,s
    for attr in ["w", "s", "e", "n", "place", "zoom", "n_tiles"]:
      self._xs[attr] = getattr(self._loc, attr)
      self.l.debug("{}: {}".format(attr, self._xs[attr]))

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
    self.l.info("Arc amplitude at this latitude %f, %f"%(self._xs["dx"], self._xs["dy"]))

  def out_of_bounds(self, pos):
    xmin, ymin, xmax, ymax = self._xs["w"], self._xs["s"], self._xs["e"], self._xs["n"]  
  
    if pos[0] < xmin or pos[0] > xmax: return True
    if pos[1] < ymin or pos[1] > ymax: return True    
    return False

     
  def _loadGeoData(self):
    path = os.getcwd()
    self.l.info("Loading geo data from path:"+path)
    blocks = gpd.read_file(os.path.join(path, "shapefiles","quartieriBarca1.shp"))
    self._blocks= blocks    

  def plotAll(self,outdir, figname):
    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()
    
    tot_people =self._blocks["density"]
    scheme = mapclassify.Quantiles(tot_people, k=5) 
 
    geoplot.choropleth(
      self._blocks, hue=tot_people, scheme=scheme,
      cmap='Oranges', figsize=(12, 8), ax=ax1
    )
    #plt.colorbar()
    plt.savefig(os.path.join(outdir, "density-"+figname))

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()

    ctx.plot_map(self._loc, ax=ax1)
    self._blocks.plot(ax=ax1,facecolor='none', edgecolor="black")  

    plt.tight_layout()

    # Plot agents
    if self.space._gdf_is_dirty: self.space._create_gdf()
    self.space._agdf.plot(ax=ax1)

    #xmin, ymin, xmax, ymax = self._xs["w"], self._xs["s"], self._xs["e"], self._xs["n"]

    #dy = (xmax-xmin)/10
    #dx = (ymax-ymin)/10

    #cols = list(np.arange(xmin, xmax + dx, dx))
    #rows = list(np.arange(ymin, ymax + dy, dy))

    #polygons = []
    #for x in cols[:-1]:
    #    for y in rows[:-1]:
    #        polygons.append(Polygon([(x,y), (x+dx, y), (x+dx, y+dy), (x, y+dy)]))

    #grid = gpd.GeoDataFrame({'geometry':polygons})
    #grid.plot(ax=ax1, facecolor='none', edgecolor="red")

    plt.savefig(os.path.join(outdir, 'agents-'+figname))

  def step(self):
    self.schedule.step()
    if self.space._gdf_is_dirty: self.space._create_gdf

  def createAgents(self):
    for _agent in self._agentsToAdd:
      _agent.pos = (uniform(self._xs["w"], self._xs["e"]), 
                 uniform(self._xs["s"], self._xs["n"]))   
    super().createAgents()
    

