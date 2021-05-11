from mesa import Agent, Model
from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent, AgentCreator

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid, NetworkGrid
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

import logging
_log = logging.getLogger(__name__)

from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
from iper.space.Space import MeshSpace

import networkx as nx

import mapclassify

from shapely.geometry import Polygon
import numpy as np

class CityModel(MultiEnvironmentWorld):

  def __init__(self, config):
    super().__init__(config)
    _log.error("Initalizing model")
    self._basemap = config["basemap"]
    self.space = GeoSpacePandas()
    self.network = NetworkGrid(nx.Graph())
    _log.info("Scheduler is " + str(self.schedule))
    self.schedule = RandomActivation(self)

    _log.info("Loading geodata")
    self._initGeo()
    self._loadGeoData()
  
  def _initGeo(self):
    #Initialize geo data
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
     
  def _loadGeoData(self):
    path = os.getcwd()
    _log.info("Loading geo data from path:"+path)
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
    #self.grid._agdf.plot(ax=ax1)

    xmin, ymin, xmax, ymax = self._xs["w"], self._xs["s"], self._xs["e"], self._xs["n"]

    dy = (xmax-xmin)/10
    dx = (ymax-ymin)/10

    cols = list(np.arange(xmin, xmax + dx, dx))
    rows = list(np.arange(ymin, ymax + dy, dy))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+dx, y), (x+dx, y+dy), (x, y+dy)]))

    grid = gpd.GeoDataFrame({'geometry':polygons})
    grid.plot(ax=ax1, facecolor='none', edgecolor="red")

    plt.savefig(os.path.join(outdir, 'agents-'+figname))

  def step(self):
    self.schedule.step()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()

    
    

