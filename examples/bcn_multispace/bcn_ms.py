import hvplot
from mesa import Agent, Model, agent
from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import random
from networkx.classes.function import density
import numpy as np
import osmnx as ox
import cartopy.crs as ccrs

import geopandas as gpd
import geoplot
import pandas as pd
from datetime import datetime, timedelta
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

from EudaldMobility.Mobility import Map_to_Graph
from EudaldMobility.Pollution.pollution_model import Interpolation_Diffusion_Model

class CityModel(MultiEnvironmentWorld):

  def __init__(self, config):
    super().__init__(config)
    self.l.info("Initalizing model")
    self._basemap = config["basemap"]
    self.num_agents = config["num_agents"]
    self.network = NetworkGrid(nx.Graph())
    self.l.info("Scheduler is " + str(self.schedule))
    self.schedule = RandomActivation(self)

    self.l.info("Loading geodata")
    self._initGeo()
    self._loadGeoData()
    self.space = GeoSpacePandas(
      extent=(self._xs["w"], self._xs["s"],
              self._xs["e"], self._xs["n"]
            )
          )
    
    # Eudald Mobility
    self.PedCarBike_Map = Map_to_Graph('PedCarBike')  #Load the shapefiles 
    print('Pedestrian + Car + Bike Loaded')
    self.Ped_Map = Map_to_Graph('Pedestrian')  #Load the shapefiles 
    print('Pedestrian Loaded')
    self.PedCar_Map = Map_to_Graph('PedCar')  #Load the shapefiles 
    print('Pedestrian + Car Loaded')
    self.PedBike_Map = Map_to_Graph('PedBike')  #Load the shapefiles 
    print('Pedestrian + Bike Loaded')
    self.define_boundaries_from_graphs(self.Ped_Map) 
    self.DateTime = datetime(year=2021, month=1, day=1, hour= 0, minute=0, second=0) 
    self.current_hour = self.DateTime.hour
    self.time_step = timedelta(seconds=60)

    # Pollution
    self.pollution_model = Interpolation_Diffusion_Model(self.boundaries, self.DateTime)
  
  def define_boundaries_from_graphs(self, map):
    self.boundaries = map.get_boundaries()
    
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
    shpfilename = os.path.join(path,"shapefiles","quartieriBarca1.shp")
    if not os.path.exists(shpfilename):
      shpfilename = os.path.join(path, "examples/bcn_multispace/shapefiles","quartieriBarca1.shp")
    print("Loading shapefile from", shpfilename)
    blocks = gpd.read_file(shpfilename)
    self._blocks= blocks   

  def manage_pollution_model(self):
    if self.DateTime.hour != self.current_hour:
      # Read the new line from the dataset and update the particles
      self.pollution_model.update_pollution_next_hour(self.current_time)
      self.pollution_model.interpolate()
      self.pollution_model.init_particles() #location of each particle
      self.pollution_model.update_particles()
      self.current_hour = self.DateTime.hour
    else:
      #apply only the diffusion model
      self.pollution_model.diffusion()
      self.pollution_model.update_particles()
      #self.pollution_model.plot_3D()

  def plotAll(self,outdir, figname):
    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()
    
    tot_people = self._blocks["density"]
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
    if self.space._gdf_is_dirty: 
      self.space._create_gdf()
    #self.space._agdf.plot(ax=ax1)
    #plot = self.space._agdf.hvplot(crs=ccrs.UTM(31), tiles = 'OSM', width=700, height=400, title = "Agents's Initialization")
    #hvplot.show(plot)

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

    #plt.savefig(os.path.join(outdir, 'agents-'+figname))

  def step(self):
    self.schedule.step()
    self.DateTime += self.time_step
    self.manage_pollution_model()
    if self.space._gdf_is_dirty: self.space._create_gdf

  def createAgents(self):
    locations =  self.population_density()
    i = 0
    for _agent in self._agentsToAdd:
      # Random position for agents:
      #node = random.choice(list(self.Ped_Map.G.nodes))
      
      # Fix position for agents:
      #agent_loc, crs = ox.projection.project_geometry(Point(2.162902, 41.395852))
      
      # Density Distribution for agents:
      agent_loc, crs = ox.projection.project_geometry(locations[i])
      node = ox.nearest_nodes(self.Ped_Map.G, agent_loc.x, agent_loc.y, return_dist=False)
      #node = ox.nearest_nodes(self.Ped_Map.G, locations[i].x, locations[i].y, return_dist=False)

      _agent.pos = (self.Ped_Map.G.nodes[node]['x'],
                    self.Ped_Map.G.nodes[node]['y'])
      i +=1
    super().createAgents()
    
  def population_density(self):
    
    self._blocks['Population'] = self._blocks['density'] * self._blocks['area']
    self._blocks['Population_%'] = self._blocks['Population'] * 100 / self._blocks['Population'].sum()

    #create a list containg the polygon where each agent should be initialize
    init_polygon = []
    init_agents = 0
    for index, row in self._blocks.iterrows():
      agents_in_zone = round(self.num_agents * (row['Population_%']/100))
      init_agents += agents_in_zone
      for i in np.arange(agents_in_zone):
        init_polygon.append(row['geometry'])
    while init_agents < self.num_agents:
      init_polygon.append(self._blocks['geometry'].sample().item())
      init_agents += 1
    
    # Get the point where each agent will be initialized
    points = []
    for polygon in init_polygon[:self.num_agents]:
      minx, miny, maxx, maxy = polygon.bounds
      pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
      while polygon.contains(pnt) == False:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
      points.append(pnt)
    return points
