from enum import IntEnum

from scipy.spatial.distance import euclidean
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

from iper.brains import ScriptedBrain
from iper import GeoSpacePandas

import logging

_log = logging.getLogger(__name__)


class VirusCovid(object):
    def __init__(self):
        self.r0 = 2.5

        # from EXP to INF
        self.incubation_days = 2
        self.incubation_days_sd = 1

        # from INF to REC - HOSP
        self.infection_days = 21
        self.infection_days_sd = 7

        # from REC to SUSC
        self.immune_days = 30
        self.immune_days_sd = 10

        # from HOSP to REC - DEATH
        self.severe_days = 20
        self.severe_days_sd = 10

        self.ptrans = 0.5
        self.death_rate = 0.02
        self.severe_rate = 0.05


class State(IntEnum):
    SUSC = 0
    EXP = 1
    INF = 2
    REC = 3
    HOSP = 4
    DEAD = 5


class BasicHuman(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = State.SUSC
        self.days_in_current_state = 0 # variable to calculate time passed since last state transition
        self.friends = set()
        self.meet_w_friend = None

    def move(self):
        """Move the agent"""
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        print(f'Agent {self.unique_id} is following {self.meet_w_friend}')
        if self.meet_w_friend is None:
            if np.random.choice([0, 1], p=[0.75, 0.25]): self.look_for_friend()  # probability to meet with a friend
            new_position = self.random.choice(possible_steps)  # choose random step

        else:
            new_position = min(possible_steps, key=lambda c: euclidean(c,
                                                                       self.meet_w_friend.pos))  # check shortest step towards friend new position
            if self.pos == self.meet_w_friend.pos:
                self.meet_w_friend.meet_w_friend = None
                self.meet_w_friend = None

        self.model.grid.move_agent(self, new_position)

    def look_for_friend(self):
        """Move the agent"""
        id_friend = random.sample(self.friends, 1)[0]  # gets one random each time
        friend_agent = self.model.schedule.agents[id_friend]

        my_iter = iter(self.friends)
        while friend_agent.meet_w_friend is not None:
            try:
                # first_friend = next(iter(self.friends))          #get first one
                id_friend = next(my_iter)  # gets one random each time
                friend_agent = self.model.schedule.agents[id_friend]
            except StopIteration:
                friend_agent = None
                break

        if friend_agent is not None:
            self.meet_w_friend = friend_agent
            friend_agent.meet_w_friend = self

    def status(self):
        """Check infection status"""

        t = self.model.schedule.time - self.days_in_current_state

        if self.state == State.EXP:
            if t >= self.exposing_time:
                self.model.collector_counts["EXP"] -= 1  # Adjust initial counts
                self.model.collector_counts["INF"] += 1
                self.infecting_time = self.model.get_infection_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.INF

        elif self.state == State.INF:
            severe_rate = self.model.virus.severe_rate
            not_severe = np.random.choice([0, 1], p=[severe_rate, 1 - severe_rate])

            if not_severe == 0:
                self.model.collector_counts["INF"] -= 1  # Adjust initial counts
                self.model.collector_counts["HOSP"] += 1
                self.hospitalized_time = self.model.get_severe_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.HOSP
                # self.model.schedule.remove(self)

            if not_severe != 0 and t >= self.infecting_time:
                self.model.collector_counts["INF"] -= 1  # Adjust initial counts
                self.model.collector_counts["REC"] += 1
                self.immune_time = self.model.get_immune_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.REC

        elif self.state == State.REC:
            if t >= self.immune_time:
                self.model.collector_counts["REC"] -= 1  # Adjust initial counts
                self.model.collector_counts["SUSC"] += 1
                self.state = State.SUSC

        elif self.state == State.HOSP:
            death_rate = self.model.virus.death_rate
            alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
            if alive == 0:
                self.model.collector_counts["HOSP"] -= 1  # Adjust initial counts
                self.model.collector_counts["DEAD"] += 1
                self.state = State.DEAD
                # self.model.schedule.remove(self)

            if alive != 0 and t >= self.hospitalized_time:
                self.model.collector_counts["HOSP"] -= 1  # Adjust initial counts
                self.model.collector_counts["REC"] += 1
                self.immune_time = self.model.get_immune_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.REC



    def contact(self):
        """Find close contacts and infect"""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if self.random.random() > self.model.virus.ptrans:
                    continue
                if self.state is State.INF or self.state is State.EXP and other.state is State.SUSC:
                    other.state = State.EXP
                    other.days_in_current_state = self.model.schedule.time
                    other.exposing_time = self.model.get_incubation_time()

    def update_stats(self):
        if self.state == State.SUSC:
            self.model.collector_counts['SUSC'] += 1
        elif self.state == State.EXP:
            self.model.collector_counts['EXP'] += 1
        elif self.state == State.INF:
            self.model.collector_counts['INF'] += 1
        elif self.state == State.REC:
            self.model.collector_counts['REC'] += 1
        elif self.state == State.HOSP:
            self.model.collector_counts['HOSP'] += 1
        elif self.state == State.DEAD:
            self.model.collector_counts['DEAD'] += 1

    def step(self):
        self.status()
        self.move()
        self.contact()
        self.update_stats()


class BCNCovid2020(Model):

    def __init__(self, N, basemap, width=50, height=50):
        super().__init__()
        _log.info("Initalizing model")

        self._basemap = basemap
        # self.grid = GeoSpacePandas()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = 100
        self.virus = VirusCovid()

        # variables for data collector
        self.collector_counts = None
        self.reset_counts()
        self.collector_counts["SUSC"] = N
        # self.datacollector = DataCollector(agent_reporters={"State": "state"})
        self.datacollector = DataCollector(
            {"SUSC": get_susceptible_count, "EXP": get_exposed_count, "INF": get_infected_count,
             "REC": get_recovered_count,
             "HOSP": get_hosp_count, "DEAD": get_dead_count, },
            tables={"DC_Table": {"Steps": [], "Susceptible": [], "Exposed": [], "Infected": [], "Recovered": [],
                                 "Hospitalized": [], "Dead": []}}
        )

        _log.info("Loading shapefiles")

        # self.loadShapefiles() for GEOSPACEPANDAS

        _log.info("Initalizing agents")
        # self.createAgents(N)
        self.createBasicAgents(N)

        self.createFriendNetwork(N)
        self.printFriends()
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
            infected = np.random.choice([0, 1], p=[0.9, 0.1])
            if infected == 1:
                a.state = State.INF
                self.collector_counts["SUSC"] -= 1
                self.collector_counts["INF"] += 1  # Adjust initial counts
                a.infecting_time = self.get_infection_time()

    def createFriendNetwork(self, N):
        if N < 2:
            pass
        else:
            for a in self.schedule.agents:
                friend = random.choice([i for i in range(0, N) if i not in [a.unique_id]])
                a.friends.add(friend)
                self.schedule.agents[friend].friends.add(a.unique_id)

    def printFriends(self):
        for a in self.schedule.agents:
            print(f'{a.unique_id} has {a.friends} these friends')

    def get_incubation_time(self):
        return int(self.random.normalvariate(self.virus.incubation_days, self.virus.incubation_days_sd))

    def get_infection_time(self):
        return int(self.random.normalvariate(self.virus.infection_days, self.virus.infection_days_sd))

    def get_immune_time(self):
        return int(self.random.normalvariate(self.virus.immune_days, self.virus.immune_days_sd))

    def get_severe_time(self):
        return int(self.random.normalvariate(self.virus.severe_days, self.virus.severe_days_sd))

    def place_at(self, agent, loc):
        if self._xs["bbox"].contains(loc):
            self.grid.update_shape(agent, loc)

    def reset_counts(self):
        self.collector_counts = {"SUSC": 0, "EXP": 0, "INF": 0, "REC": 0, "HOSP": 0, "DEAD": 0, }

    def plot_results(self, title=''):
        """Plot cases per country"""
        X = self.datacollector.get_table_dataframe("DC_Table")
        X.to_csv('sir_stats.csv', index=False)
        X.plot.line().get_figure().savefig('sir_stats.png')

    def update_DC_table(self):
        # collect "Strategies" table data
        step_num = self.schedule.steps
        next_row = {'Steps': step_num, 'Susceptible': get_susceptible_count(self), 'Exposed': get_exposed_count(self),
                    'Infected': get_infected_count(self), 'Recovered': get_recovered_count(self),
                    'Hospitalized': get_hosp_count(self), 'Dead': get_dead_count(self)}
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
        # self.grid._agdf.plot(ax=ax1) for GEOSPACEPANDAAS

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

        self._xs["dx"] = 111.32;  # One degree in longitude is this in KM
        self._xs["dy"] = 40075 * math.cos(self._xs["centroid"].y) / 360
        _log.info("Arc amplitude at this latitude %f, %f" % (self._xs["dx"], self._xs["dy"]))

        path = os.getcwd()
        _log.info("Loading geo data from path:" + path)
        roads_1 = gpd.read_file(os.path.join(path, "shapefiles", "1", "roads-line.shp"))
        roads_2 = gpd.read_file(os.path.join(path, "shapefiles", "2", "roads-line.shp"))
        self._roads = [roads_1, roads_2]

    def step(self):
        self.reset_counts()
        self.schedule.step()
        self.datacollector.collect(self)
        self.update_DC_table()

    def run_model(self, n):
        for i in range(n):
            _log.info("Step %d of %d" % (i, n))
            self.step()


# Functions needed for datacollector
def get_susceptible_count(model):
    return model.collector_counts["SUSC"]


def get_exposed_count(model):
    return model.collector_counts["EXP"]


def get_infected_count(model):
    return model.collector_counts["INF"]


def get_recovered_count(model):
    return model.collector_counts["REC"]


def get_hosp_count(model):
    return model.collector_counts["HOSP"]


def get_dead_count(model):
    return model.collector_counts["DEAD"]
