from Covid_class import VirusCovid, State
from Human_Class import BasicHuman
from Hospital_class import Hospital, Workplace


from mesa import Model
from mesa.time import RandomActivation
from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
import random
import numpy as np
from scipy.spatial.distance import euclidean

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

class BCNCovid2020(Model):

    def __init__(self, N, basemap, width=50, height=50):
        super().__init__()
        _log.info("Initalizing model")

        self._basemap = basemap
        # self.grid = GeoSpacePandas()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = 100
        self.timetable = np.arange(0, 24, 0.25) # divide the day
        self.step_no = 0
        self.virus = VirusCovid(len(self.timetable))

        # variables for data collector
        self.collector_counts = None
        self.reset_counts()
        self.collector_counts["SUSC"] = N

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

        #Create human agents
        self.createBasicAgents(N)

        #create building agents
        N_hosp = 5
        N_work = int(np.ceil(N/4)) #round up
        self.createHospitals(N, N_hosp)
        self.createWorkplaces(N, N_hosp, N_work)

        # Set the friends and workplaces of human agents
        self.createSocialNetwork(N)

        self.datacollector.collect(self)

    def createBasicAgents(self, N):
        """ Create and place the Human agents into the map"""
        for i in range(N):
            a = BasicHuman(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            a.house = (x,y) # set a random place to be the house of agent
            self.grid.place_agent(a, (x, y)) #the agent starts at their house

            # make some agents infected at start
            infected = np.random.choice([0, 1], p=[0.9, 0.1])
            if infected == 1:
                a.state = State.INF
                self.collector_counts["SUSC"] -= 1
                self.collector_counts["INF"] += 1  # Adjust initial counts
                a.infecting_time = self.get_infection_time()
            print("Human agent "+ str(a.unique_id)+" created")

    def createHospitals(self, N, N_hosp):
        """ Create and place Hospital agents """
        for i in range(N, N+N_hosp):
            h = Hospital(i, self)
            self.schedule.add(h)
            self.grid.place_agent(h, (h.place[0], h.place[1]))
            print("Hospital agent "+str(h.unique_id)+" placed at " + str((h.place[0], h.place[1])) )

    def createWorkplaces(self, N, N_hosp, N_work, employment_rate = 0.95):
        """ Create and place Workplaces. Assign some of the Human agents to work on them """
        for i in range(N+N_hosp, N+N_hosp+N_work):
            w = Workplace(i, self)
            w.set_capacity(int(np.ceil(N_work / N)), 1)
            self.schedule.add(w)
            self.grid.place_agent(w, (w.place[0], w.place[1]))
            print("Workplace agent "+str(w.unique_id)+" placed at " + str((w.place[0], w.place[1])))

        # employ the human agents if the total capacity of the workplace is not full.
        for ag_id in range(0, N):
            if np.random.choice([True, False], p=[employment_rate, 1-employment_rate]):
                workplaces = random.sample([i for i in self.schedule.agents if isinstance(i, Workplace)], N_work)
                for workplace in workplaces:
                    if workplace.total_capacity != len(workplace.get_workers()):
                        ag = self.schedule.agents[ag_id]
                        ag.workplace = workplace.place
                        workplace.add_worker(ag)
                    break



    def createSocialNetwork(self, N):
        """ Add a set of friends for each of the human agents in the model"""
        if N < 2:
            pass
        else:
            for a in [a for a in self.schedule.agents]:
                if isinstance(a, BasicHuman):
                    friend = random.choice([i for i in range(0, N) if i != a])
                    a.friends.add(friend)
                    self.schedule.agents[friend].friends.add(a.unique_id)
            self.printSocialNetwork()



    def printSocialNetwork(self):
        """ Prints on the terminal screen the friends and workplace (if exists) of the human agents of the model. """
        for a in self.schedule.agents:
            if isinstance(a, BasicHuman):
                print(f'{a.unique_id} has {a.friends} these friends and works at {a.workplace}')

    def get_hour(self):
        """ Returns the 'hour' of the day the model is in. """
        return self.timetable[(self.step_no - 1) % len(self.timetable)]

    def get_day(self):
        """ Returns the day the model is in """
        return int( (self.step_no - 1) / len(self.timetable))

    def get_incubation_time(self):
        """ Returns the incubation time (EXP state) following a normal distribution """
        return int(self.random.normalvariate(self.virus.incubation_days, self.virus.incubation_days_sd))

    def get_infection_time(self):
        """ Returns the infeciton time (INF state) following a normal distribution """
        return int(self.random.normalvariate(self.virus.infection_days, self.virus.infection_days_sd))

    def get_immune_time(self):
        """ Returns the immune time (REC state) following a normal distribution """
        return int(self.random.normalvariate(self.virus.immune_days, self.virus.immune_days_sd))

    def get_severe_time(self):
        """ Returns the severe time (HOSP state) following a normal distribution """
        return int(self.random.normalvariate(self.virus.severe_days, self.virus.severe_days_sd))

    def place_at(self, agent, loc):
        """ Places an agent into a concrete location """
        if self._xs["bbox"].contains(loc):
            self.grid.update_shape(agent, loc)

    def reset_counts(self):
        """ Sets to 0 the counts for the datacollector """
        self.collector_counts = {"SUSC": 0, "EXP": 0, "INF": 0, "REC": 0, "HOSP": 0, "DEAD": 0, }

    def plot_results(self, title=''):
        """Plot cases per country"""
        X = self.datacollector.get_table_dataframe("DC_Table")
        X.to_csv('sir_stats.csv', index=False)
        X.plot.line().get_figure().savefig('sir_stats.png')

    def update_DC_table(self):
        """ Collects all statistics for the DC_Table """
        step_num = self.schedule.steps
        next_row = {'Steps': step_num, 'Susceptible': get_susceptible_count(self), 'Exposed': get_exposed_count(self),
                    'Infected': get_infected_count(self), 'Recovered': get_recovered_count(self),
                    'Hospitalized': get_hosp_count(self), 'Dead': get_dead_count(self)}
        self.datacollector.add_table_row("DC_Table", next_row, ignore_missing=True)


    def step(self):
        """ Runs one step on the model. Resets the count, collects the data, and updated the DC_Table. """
        self.step_no += 1
        self.reset_counts()
        self.schedule.step()
        self.datacollector.collect(self)
        self.update_DC_table()

    def run_model(self, n):
        """ Runs the model for the 'n' number of steps """
        for i in range(n):
            _log.info("Step %d of %d" % (i, n))
            self.step()


# Functions needed for datacollector
def get_susceptible_count(model):
    """ Returns the Susceptible human agents in the model """
    return model.collector_counts["SUSC"]


def get_exposed_count(model):
    """ Returns the Exposed human agents in the model """
    return model.collector_counts["EXP"]


def get_infected_count(model):
    """ Returns the Infected human agents in the model """
    return model.collector_counts["INF"]


def get_recovered_count(model):
    """ Returns the Recovered human agents in the model """
    return model.collector_counts["REC"]


def get_hosp_count(model):
    """ Returns the Hospitalized human agents in the model """
    return model.collector_counts["HOSP"]


def get_dead_count(model):
    """ Returns the Dead human agents in the model """
    return model.collector_counts["DEAD"]