from enum import IntEnum

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

from iper import GeoSpacePandas

import logging
_log = logging.getLogger(__name__)

class VirusCovid(object):
  def __init__(self):
    self.r0 = 2.5
    self.recovery_days = 21
    self.recovery_days_sd = 7
    self.ptrans = 0.5
    self.death_rate = 0.02


class State(IntEnum):
  SUSC = 0
  INF = 1
  REC = 2
  DEAD = 3

class BasicHuman(Agent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.state = State.SUSC
    self.infection_time = 0

  def move(self):
    """Move the agent"""
    possible_steps = self.model.grid.get_neighborhood(
      self.pos,
      moore=True,
      include_center=False)
    new_position = self.random.choice(possible_steps)
    self.model.grid.move_agent(self, new_position)

  def status(self):
    """Check infection status"""

    if self.state == State.INF:
      drate = self.model.virus.death_rate
      alive = np.random.choice([0, 1], p=[drate, 1 - drate])
      if alive == 0:
        self.model.collector_counts["INF"] -= 1  # Adjust initial counts
        self.model.collector_counts["DEAD"] += 1
        self.state = State.DEAD
        #self.model.schedule.remove(self)
      t = self.model.schedule.time - self.infection_time
      if t >= self.recovery_time:
        self.model.collector_counts["INF"] -= 1  # Adjust initial counts
        self.model.collector_counts["REC"] += 1
        self.state = State.REC

  def contact(self):
    """Find close contacts and infect"""
    cellmates = self.model.grid.get_cell_list_contents([self.pos])
    if len(cellmates) > 1:
      for other in cellmates:
        if self.random.random() > self.model.virus.ptrans:
          continue
        if self.state is State.INF and other.state is State.SUSC:
          other.state = State.INF
          other.infection_time = self.model.schedule.time
          other.recovery_time = self.model.get_recovery_time()

  def update_stats(self):
    if self.state == State.INF:
      self.model.collector_counts['INF'] += 1
    if self.state == State.SUSC:
      self.model.collector_counts['SUSC'] += 1
    if self.state == State.REC:
      self.model.collector_counts['REC'] += 1
    if self.state == State.DEAD:
      self.model.collector_counts['DEAD'] += 1


  def step(self):
    self.status()
    self.move()
    self.contact()
    self.update_stats()


class BCNCovid2020(Model):

  def __init__(self, N, basemap, width=10, height=10):
    super().__init__()
    _log.info("Initalizing model")    
    
    self._basemap = basemap
    #self.grid = GeoSpacePandas()
    self.grid = MultiGrid(width, height, True)
    self.schedule = RandomActivation(self)
    self.initial_outbreak_size = 100
    self.virus = VirusCovid()

    #variables for data collector
    self.collector_counts = None
    self.reset_counts()
    self.collector_counts["SUSC"] = N
    #self.datacollector = DataCollector(agent_reporters={"State": "state"})
    self.datacollector = DataCollector(
      {"INF": get_infected_count, "SUSC": get_susceptible_count, "REC": get_recovered_count, "DEAD": get_dead_count, },
      tables = {"DC_Table": {"Steps":[], "Infected":[], "Susceptible":[], "Recovered":[], "Dead":[]}}
    )


    _log.info("Loading shapefiles")

    #self.loadShapefiles() for GEOSPACEPANDAS
    
    _log.info("Initalizing agents")
    #self.createAgents(N)
    self.createBasicAgents(N)

    self.datacollector.collect(self)


  def createBasicAgents(self, N):
    for i in range(N):
      a = BasicHuman(i, self)
      self.schedule.add(a)
      # Add the agent to a random grid cell
      x = self.random.randrange(self.grid.width)
      y = self.random.randrange(self.grid.height)
      self.grid.place_agent(a, (x, y))
      # make some agents infected at start
      infected = np.random.choice([0, 1], p=[0.98, 0.02])
      if infected == 1:
        a.state = State.INF
        self.collector_counts["SUSC"] -= 1
        self.collector_counts["INF"] += 1  # Adjust initial counts
        a.recovery_time = self.get_recovery_time()

  def get_recovery_time(self):
    return int(self.random.normalvariate(self.virus.recovery_days, self.virus.recovery_days_sd))

  def place_at(self, agent, loc):
    if self._xs["bbox"].contains(loc):
      self.grid.update_shape(agent, loc)

  def reset_counts(self):
    self.collector_counts = {"SUSC": 0, "INF": 0, "REC": 0, "DEAD": 0, }

  def plot_results(self, title=''):
    """Plot cases per country"""
    X = self.datacollector.get_table_dataframe("DC_Table")
    X.to_csv('sir_stats.csv', index=False)
    X.plot.line().get_figure().savefig('sir_stats.png')


  def update_DC_table(self):
    # collect "Strategies" table data
    step_num = self.schedule.steps
    next_row = {'Steps': step_num, 'Infected': get_infected_count(self),
                'Susceptible': get_susceptible_count(self), 'Recovered': get_recovered_count(self), 'Dead': get_dead_count(self)}
    self.datacollector.add_table_row("DC_Table", next_row, ignore_missing=True)


  def plotAll(self):

    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.gca()

    ctx.plot_map(self._loc, ax=ax1)
    _c = ["red", "blue"]
    for i, _r in enumerate(self._roads):
      _r.plot(ax=ax1, facecolor='none', edgecolor=_c[i])    
    plt.tight_layout()

    # Plot agents
    #self.grid._agdf.plot(ax=ax1) for GEOSPACEPANDAAS
    
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
    roads_1 = gpd.read_file(os.path.join(path, "../shapefiles", "1", "roads-line.shp"))
    roads_2 = gpd.read_file(os.path.join(path, "../shapefiles", "2", "roads-line.shp"))
    self._roads = [roads_1, roads_2]

  def step(self):
    self.reset_counts()
    self.schedule.step()
    self.datacollector.collect(self)
    self.update_DC_table()

  def run_model(self, n):
    for i in range(n):
      _log.info("Step %d of %d"%(i, n))
      self.step()


# Functions needed for datacollector
def get_infected_count(model):
  return model.collector_counts["INF"]


def get_susceptible_count(model):
  return model.collector_counts["SUSC"]


def get_recovered_count(model):
  return model.collector_counts["REC"]


def get_dead_count(model):
  return model.collector_counts["DEAD"]
    
    

