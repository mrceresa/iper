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
from datetime import datetime, timedelta
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

    def __init__(self, N, basemap, width=50, height=50,
                 N_hosp = 5, Hosp_capacity = 10,
                 incubation_days=3, infection_days=5, immune_days=3,
                 severe_days=3, ptrans=0.7, pSympt=0.8, pTest=0.9, death_rate=0.002, severe_rate= 0.005):
        super().__init__()
        _log.info("Initalizing model")

        self._basemap = basemap
        # self.grid = GeoSpacePandas()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = 100
        self.DateTime = datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0)

        self.virus = VirusCovid(incubation_days=incubation_days, incubation_days_sd=int(incubation_days*0.4), infection_days=infection_days,
                                infection_days_sd=int(infection_days*0.4), immune_days=immune_days, immune_days_sd=int(immune_days*0.4), severe_days=severe_days,
                                severe_days_sd=int(severe_days*0.4), ptrans=ptrans, pSympt=pSympt, pTest=pTest, death_rate= (death_rate/(24*4)), severe_rate=(severe_rate/(24*4)))
        self.peopleTested = {}
        self.peopleToTest = {}
        self.peopleInMeeting = 5  # max people to meet with
        self.peopleInMeetingSd = self.peopleInMeeting *0.4
        friendsByAgent = 3  # friends each agent chooses (does not mean total number of friends by agent, it may coincide with min)

        # variables for model data collector
        self.collector_counts = None
        self.reset_counts()
        self.collector_counts["SUSC"] = N

        self.datacollector = DataCollector(
            {"SUSC": get_susceptible_count, "EXP": get_exposed_count, "INF": get_infected_count,
             "REC": get_recovered_count, "HOSP": get_hosp_count, "DEAD": get_dead_count, },
            tables={"Model_DC_Table": {"Day": [], "Susceptible": [], "Exposed": [], "Infected": [], "Recovered": [],
                                 "Hospitalized": [], "Dead": []}}
        )

        # variables for hospital data collector
        self.hosp_collector_counts = None
        self.reset_hosp_counts()
        self.hosp_collector_counts["H-SUSC"] = N
        self.hosp_collector = DataCollector(
            {"H-SUSC": get_h_susceptible_count, "H-INF": get_h_infected_count,
             "H-REC": get_h_recovered_count, "H-HOSP": get_h_hospitalized_count, "H-DEAD": get_h_dead_count, },
            tables={"Hosp_DC_Table": {"Day": [], "Hosp-Susceptible": [], "Hosp-Infected": [], "Hosp-Recovered": [],
                                 "Hosp-Hospitalized": [], "Hosp-Dead": []}}
        )

        _log.info("Loading shapefiles")

        # self.loadShapefiles() for GEOSPACEPANDAS

        _log.info("Initalizing agents")

        # Create human agents
        self.createBasicAgents(N)

        # create building agents
        N_hosp = N_hosp
        self.Hosp_capacity = Hosp_capacity
        N_work = int(np.ceil(N / 4))  # round up
        self.createHospitals(N, N_hosp)
        self.createWorkplaces(N, N_hosp, N_work)

        # Set the friends and workplaces of human agents
        self.createSocialNetwork(N, friendsByAgent)

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)

    def createBasicAgents(self, N):
        """ Create and place the Human agents into the map"""
        for i in range(N):
            a = BasicHuman(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            a.house = (x, y)  # set a random place to be the house of agent
            self.grid.place_agent(a, (x, y))  # the agent starts at their house

            # make some agents infected at start
            infected = np.random.choice([0, 1], p=[0.9, 0.1])
            if infected == 1:
                a.state = State.INF
                self.collector_counts["SUSC"] -= 1
                self.collector_counts["INF"] += 1  # Adjust initial counts
                a.infecting_time = self.get_infection_time()
            print("Human agent " + str(a.unique_id) + " created")

        self.update_DC_table()  # record the first day stats on table

    def createHospitals(self, N, N_hosp):
        """ Create and place Hospital agents """
        for i in range(N, N + N_hosp):
            h = Hospital(i, self)
            self.schedule.add(h)
            self.grid.place_agent(h, (h.place[0], h.place[1]))
            print(f"Hospital {h.unique_id} placed at {h.place[0], h.place[1]} with {h.PCR_availables} tests and {h.total_capacity} capacity")

    def getHospitalPosition(self, place=None):
        """ Returns the position of the Hospitals or the Hospital agent if position is given """
        if place is None:
            return [i.place for i in self.schedule.agents if isinstance(i, Hospital)]
        else:
            return [i for i in self.schedule.agents if isinstance(i, Hospital) and i.place == place][0]

    def createWorkplaces(self, N, N_hosp, N_work, employment_rate=0.95):
        """ Create and place Workplaces. Assign some of the Human agents to work on them """
        for i in range(N + N_hosp, N + N_hosp + N_work):
            w = Workplace(i, self)
            w.set_capacity(int(np.ceil(N / N_work)), 1)
            self.schedule.add(w)
            self.grid.place_agent(w, (w.place[0], w.place[1]))
            print(f"Workplace agent {w.unique_id} placed at {w.place[0], w.place[1]}")

        # employ the human agents if the total capacity of the workplace is not full.
        for ag_id in range(0, N):
            if np.random.choice([True, False], p=[employment_rate, 1 - employment_rate]):
                workplaces = random.sample([i for i in self.schedule.agents if isinstance(i, Workplace)], N_work)
                for workplace in workplaces:
                    if workplace.total_capacity != len(workplace.get_workers()):
                        ag = self.schedule.agents[ag_id]
                        ag.workplace = workplace
                        workplace.add_worker(ag)
                        break

    def createSocialNetwork(self, N, friendsByAgent):
        """ Add a set of friends for each of the human agents in the model"""
        if N < 2:
            pass
        else:
            for a in [a for a in self.schedule.agents]:
                if isinstance(a, BasicHuman):
                    friends = [i for i in range(0, N) if i != a.unique_id]
                    for f in range(0, friendsByAgent):
                        friend = random.choice(friends)
                        a.friends.add(friend)
                        self.schedule.agents[friend].friends.add(a.unique_id)
            self.printSocialNetwork()

    def printSocialNetwork(self):
        """ Prints on the terminal screen the friends and workplace (if exists) of the human agents of the model. """
        for a in self.schedule.agents:
            if isinstance(a, BasicHuman):
                print(f'{a.unique_id} has {a.friends} these friends and works at {a.workplace}')

    def get_minutes(self):
        return self.DateTime.minute

    def get_hour(self):
        """ Returns the 'hour' of the day the model is in. """
        return self.DateTime.hour
        # return self.timetable[(self.step_no - 1) % len(self.timetable)]

    def get_day(self):
        """ Returns the day the model is in """
        return self.DateTime.day
        # return int( (self.step_no - 1) / len(self.timetable))

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
        """ Sets to 0 the counts for the model datacollector """
        self.collector_counts = {"SUSC": 0, "EXP": 0, "INF": 0, "REC": 0, "HOSP": 0, "DEAD": 0, }

    def reset_hosp_counts(self):
        """ Sets to 0 the counts for the hospital datacollector """
        self.hosp_collector_counts = {"H-SUSC": 0, "H-INF": 0, "H-REC": 0, "H-HOSP": 0, "H-DEAD": 0, }

    def plot_results(self, title='sir_stats', hosp_title='hosp_sir_stats'):
        """Plot cases per country"""
        X = self.datacollector.get_table_dataframe("Model_DC_Table")
        X.to_csv(title+'.csv', index=False)  # get the csv
        colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]
        X.set_index('Day').plot.line(color=colors).get_figure().savefig(title)  # plot the stats

        Y = self.hosp_collector.get_table_dataframe("Hosp_DC_Table")
        Y.to_csv(hosp_title+'.csv', index=False)  # get the csv
        colors = ["Green", "Red", "Blue", "Gray", "Black"]
        Y.set_index('Day').plot.line(color=colors).get_figure().savefig(hosp_title+'.png')  # plot the stats

    def update_DC_table(self):
        """ Collects all statistics for the DC_Table """
        next_row = {'Day': self.DateTime.day, 'Susceptible': get_susceptible_count(self),
                    'Exposed': get_exposed_count(self),
                    'Infected': get_infected_count(self), 'Recovered': get_recovered_count(self),
                    'Hospitalized': get_hosp_count(self), 'Dead': get_dead_count(self)}
        self.datacollector.add_table_row("Model_DC_Table", next_row, ignore_missing=True)

        next_row2 = {'Day': self.DateTime.day, 'Hosp-Susceptible': get_h_susceptible_count(self),
                    'Hosp-Infected': get_h_infected_count(self), 'Hosp-Recovered': get_h_recovered_count(self),
                    'Hosp-Hospitalized': get_hosp_count(self), 'Hosp-Dead': get_h_dead_count(self)}
        self.hosp_collector.add_table_row("Hosp_DC_Table", next_row2, ignore_missing=True)

    def step(self):
        """ Runs one step on the model. Resets the count, collects the data, and updated the DC_Table. """
        # self.step_no += 1

        current_step = self.DateTime
        self.DateTime += timedelta(minutes=15)  # next step

        # next day
        self.reset_counts()

        self.schedule.step()

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)
        if current_step.day != self.DateTime.day:
            self.update_DC_table()
            self.clean_contact_list(current_step, Adays=2,
                                    Hdays=5, Tdays=10)  # clean contact lists from agents for faster computations
            self.plot_results(title="server_stats", hosp_title="server_hosp_stats")

    def clean_contact_list(self, current_step, Adays, Hdays, Tdays):
        """ Function for deleting past day contacts sets and arrange today's tests"""
        date = current_step.strftime('%Y-%m-%d')
        Atime = (current_step - timedelta(days=Adays)).strftime('%Y-%m-%d')
        Htime = (current_step - timedelta(days=Hdays)).strftime('%Y-%m-%d')

        Ttime = (current_step - timedelta(days=Tdays)).strftime('%Y-%m-%d')
        if Ttime in self.peopleTested: del self.peopleTested[Ttime]

        # People tested in the last recorded days
        peopleTested = set()
        for key in self.peopleTested:
            for elem in self.peopleTested[key]:
                peopleTested.add(elem)

        #print(f"Lista total de agentes a testear a primera hora: {self.peopleToTest}")

        # shuffle agent list to distribute to test agents among the hospitals
        agents_list = self.schedule.agents.copy()
        random.shuffle(agents_list)
        for a in agents_list:
            # delete contacts of human agents of Adays time
            if isinstance(a, BasicHuman):
                if Atime in a.contacts: del a.contacts[Atime]

            elif isinstance(a, Hospital):
                peopleTested = a.decideTesting(peopleTested)
                """if date in a.PCR_results:
                    if not date in self.peopleTested: self.peopleTested[date] = a.PCR_results[date]
                    else:
                        for elem in a.PCR_results[date]: self.peopleTested[date].add(elem)"""
                # delete contact tracing of Hdays time
                if Htime in a.PCR_testing: del a.PCR_testing[Htime]
                if Htime in a.PCR_results: del a.PCR_results[Htime]

                # print(f"Lista de contactos de hospital {a.unique_id} es {a.PCR_testing}")

        #print(f"Lista total de testeados: {self.peopleTested}")
        #print(f"Lista total de agentes a testear: {self.peopleToTest}")

    def run_model(self, n):
        """ Runs the model for the 'n' number of steps """
        for i in range(n):
            _log.info("Step %d of %d" % (i, n))
            self.step()


""" Functions for the data collectors """


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


def get_h_susceptible_count(model):
    """ Returns the Susceptible human agents in the model recorded by Hospital """
    return model.hosp_collector_counts["H-SUSC"]


def get_h_infected_count(model):
    """ Returns the Infected human agents in the model recorded by Hospital"""
    return model.hosp_collector_counts["H-INF"]


def get_h_recovered_count(model):
    """ Returns the Recovered human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-REC"]


def get_h_hospitalized_count(model):
    """ Returns the Hospitalized human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-HOSP"]


def get_h_dead_count(model):
    """ Returns the Dead human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-DEAD"]
