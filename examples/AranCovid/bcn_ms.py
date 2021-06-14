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
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon, LineString, Point
import contextily as ctx

from io import StringIO
import json

import sys

sys.path.insert(0, '../../')
from iper.space.geospacepandas import GeoSpacePandas
from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
from iper.space.Space import MeshSpace

import networkx as nx

import mapclassify

from shapely.geometry import Polygon
import numpy as np

import logging
from random import uniform
import time
from datetime import datetime, timedelta
from attributes_Agent import job, create_families, age_
from Hospital_class import Workplace, Hospital
from agents import HumanAgent
import DataCollector_functions as dc
from SEAIHRD_class import SEAIHRD_covid, Mask


class CityModel(MultiEnvironmentWorld):

    def __init__(self, config):
        super().__init__(config)
        self.l.info("Initalizing model")
        self._basemap = config["basemap"]
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
        self.DateTime = datetime(year=2021, month=1, day=1, hour=9, minute=0, second=0)
        # self.virus = VirusCovid(config["virus"])
        self.pTest = 0.95
        self.R0 = 0
        self.R0_obs = 0
        self.R0_observed = {}
        self.contact_count = [0,0]

        # alarm state characteristics
        self.alarm_state = config["alarm_state"]
        self.lockdown_total = False
        self.quarantine_period = 0
        self.night_curfew = 24
        self.masks_probs = [0.01, 0.64, 0.35]

        self.N_hospitals = config["hospitals"]
        self.Hosp_capacity = math.ceil((0.0046 * config["agents"]) / self.N_hospitals)  # 4.6 beds per 1,000 inhabitants

        self.PCR_tests = config["tests"] / config["hospitals"]
        # print("UCI BEDS: ", self.Hosp_capacity)
        # print("HOSP CAPACITY IS: ", self.Hosp_capacity)
        # print("TESTS PER HOSPITAL: ", self.PCR_tests)
        self.employment_rate = 0.70
        self.peopleTested = {}
        self.peopleToTest = {}

        self.peopleInMeeting = config["peopleMeeting"]  # max people to meet with
        self.peopleInMeetingSd = config["peopleMeeting"] * 0.2

        # variables for model data collector
        self.collector_counts = None
        dc.reset_counts(self)
        self.collector_counts["SUSC"] = config["agents"]

        self.datacollector = DataCollector(
            {"SUSC": dc.get_susceptible_count, "EXP": dc.get_exposed_count, "INF": dc.get_infected_count,
             "REC": dc.get_recovered_count, "HOSP": dc.get_hosp_count, "DEAD": dc.get_dead_count, "R0": dc.get_R0,
             "R0_Obs": dc.get_R0_Obs
             # , "Mcontacts": dc.get_R0_Obs0, "Quarantined": dc.get_R0_Obs1,"Contacts": dc.get_R0_Obs2,
             },
            tables={"Model_DC_Table": {"Day": [], "Susceptible": [], "Exposed": [], "Infected": [], "Recovered": [],
                                       "Hospitalized": [], "Dead": [], "R0": [], "R0_Obs": []
                                       # , "Mcontacts": [],"Quarantined": [], "Contacts": []
                                       }}
        )

        # variables for hospital data collector
        self.hosp_collector_counts = None
        dc.reset_hosp_counts(self)
        self.hosp_collector_counts["H-SUSC"] = config["agents"]
        self.hosp_collector = DataCollector(
            {"H-SUSC": dc.get_h_susceptible_count, "H-INF": dc.get_h_infected_count, "H-REC": dc.get_h_recovered_count,
             "H-HOSP": dc.get_h_hospitalized_count, "H-DEAD": dc.get_h_dead_count, },
            tables={"Hosp_DC_Table": {"Day": [], "Hosp-Susceptible": [], "Hosp-Infected": [], "Hosp-Recovered": [],
                                      "Hosp-Hospitalized": [], "Hosp-Dead": []}}
        )

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)

    def _initGeo(self):
        # Initialize geo data
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

        self._xs["dx"] = 111.32;  # One degree in longitude is this in KM
        self._xs["dy"] = 40075 * math.cos(self._xs["centroid"].y) / 360
        self.l.info("Arc amplitude at this latitude %f, %f" % (self._xs["dx"], self._xs["dy"]))

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

    def plotAll(self, outdir, figname):
        fig = plt.figure(figsize=(15, 15))
        ax1 = plt.gca()

        tot_people = self._blocks["density"]
        scheme = mapclassify.Quantiles(tot_people, k=5)

        geoplot.choropleth(
            self._blocks, hue=tot_people, scheme=scheme,
            cmap='Oranges', figsize=(12, 8), ax=ax1
        )
        # plt.colorbar()
        plt.savefig(os.path.join(outdir, "density-" + figname))

        fig = plt.figure(figsize=(15, 15))
        ax1 = plt.gca()

        ctx.plot_map(self._loc, ax=ax1)
        self._blocks.plot(ax=ax1, facecolor='none', edgecolor="black")

        plt.tight_layout()

        # Plot agents
        if self.space._gdf_is_dirty: self.space._create_gdf()
        self.space._agdf.plot(ax=ax1)

        # xmin, ymin, xmax, ymax = self._xs["w"], self._xs["s"], self._xs["e"], self._xs["n"]

        # dy = (xmax-xmin)/10
        # dx = (ymax-ymin)/10

        # cols = list(np.arange(xmin, xmax + dx, dx))
        # rows = list(np.arange(ymin, ymax + dy, dy))

        # polygons = []
        # for x in cols[:-1]:
        #    for y in rows[:-1]:
        #        polygons.append(Polygon([(x,y), (x+dx, y), (x+dx, y+dy), (x, y+dy)]))

        # grid = gpd.GeoDataFrame({'geometry':polygons})
        # grid.plot(ax=ax1, facecolor='none', edgecolor="red")

        plt.savefig(os.path.join(outdir, 'agents-' + figname))

    def plot_results(self, outdir, title='stats', hosp_title='hosp_stats', R0_title='R0_stats'):
        """Plot cases per country"""
        if isinstance(self.alarm_state['inf_threshold'], int) or self.alarm_state['inf_threshold'] == "2021-01-02":
            alarm_state = False
        else:
            alarm_state = True

        X = self.datacollector.get_table_dataframe("Model_DC_Table")

        X['Day'] = X['Day'].apply(pd.Timestamp)

        R0_df = X[['Day','R0', 'R0_Obs']]
        R0_df.to_csv(outdir + "/" + R0_title + '.csv', index=False)  # get the csv



        ### R0 plot ###
        columns = ['R0', 'R0_Obs']  # , 'Mcontacts', 'Quarantined', 'Contacts']
        colors = ["Orange", "Green", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('R0 values')
        # plt.gca().get_xaxis().set_visible(False)      #ax.xaxis.tick_top()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, R0_title))

        # Model stats plot
        X.drop(['R0', 'R0_Obs'], axis=1, inplace=True)
        X.to_csv(outdir + "/" + title + '.csv', index=False)  # get the csv

        columns = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Hospitalized', 'Dead']
        colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Model stats')
        # plt.gca().get_xaxis().set_visible(False)  # ax.xaxis.tick_top()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, title))

        Y = self.hosp_collector.get_table_dataframe("Hosp_DC_Table")
        Y['Day'] = Y['Day'].apply(pd.Timestamp)
        Y.to_csv(outdir + "/" + hosp_title + '.csv', index=False)  # get the csv

        # Hospital stats plot
        columns = ['Hosp-Susceptible', 'Hosp-Infected', 'Hosp-Recovered', 'Hosp-Hospitalized', 'Hosp-Dead']
        colors = ["Green", "Red", "Blue", "Gray", "Black"]

        Y.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Observed stats')
        # plt.gca().get_xaxis().set_visible(False)      #ax.xaxis.tick_top()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, hosp_title))
        #print(self.R0_observed)
        print(self.contact_count, self.DateTime)

    def getHospitalPosition(self, place=None):
        """ Returns the position of the Hospitals or the Hospital agent if position is given """
        if place is None:
            return [i.place for i in self.schedule.agents if isinstance(i, Hospital)]
        else:
            return [i for i in self.schedule.agents if isinstance(i, Hospital) and i.place == place][0]

    def step(self):

        current_step = self.DateTime
        self.DateTime += timedelta(minutes=15)  # next step

        self.schedule.step()

        if self.space._gdf_is_dirty:
            self.space._create_gdf()
            # self.space._create_gdf

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)
        if current_step.day != self.DateTime.day:
            dc.reset_counts(self)
            dc.reset_hosp_counts(self)
            dc.update_stats(self)
            self.calculate_R0(current_step)
            dc.update_DC_table(self)

            # clean contact lists from agents for faster computations
            self.clean_contact_list(current_step, Adays=2, Hdays=5, Tdays=10)
            self.changeAgentStates()

            # decide on applying stricter measures
            if isinstance(self.alarm_state['inf_threshold'], int):
                if self.hosp_collector_counts["H-INF"] >= self.alarm_state['inf_threshold'] and self.alarm_state[
                    'inf_threshold'] != 1:  # dont apply lockdown if threshold is set to 1
                    # print("NIGHT CURFEW: ", self.night_curfew, '\n', "MASKS PROBS: ", self.masks_probs, '\n QUARANTINE: ', self.quarantine_period, "\n MEETIGN:", self.peopleInMeeting)
                    self.activate_alarm_state()
                    # print("NIGHT CURFEW: ", self.night_curfew, '\n', "MASKS PROBS: ", self.masks_probs, '\n QUARANTINE: ', self.quarantine_period, "\n MEETIGN:", self.peopleInMeeting)
            # self.plot_results()  # title="server_stats", hosp_title="server_hosp_stats"

    def activate_alarm_state(self):
        self.alarm_state['inf_threshold'] = self.DateTime.strftime("%Y-%m-%d")

        if 'night_curfew' in self.alarm_state.keys():
            self.night_curfew = self.alarm_state['night_curfew']

        if 'masks' in self.alarm_state.keys():
            self.masks_probs = self.alarm_state['masks']

        if 'quarantine' in self.alarm_state.keys():
            self.quarantine_period = self.alarm_state['quarantine']

        if 'meeting' in self.alarm_state.keys():
            self.peopleInMeeting = self.alarm_state['meeting']
            self.peopleInMeetingSd = self.alarm_state['meeting'] * 0.2

        if 'remote-working' in self.alarm_state.keys() and self.alarm_state['remote-working'] < self.employment_rate:
            fire_employees = round(self.alarm_state['remote-working'] / self.employment_rate, 2)
            for human in self.schedule.agents:
                if isinstance(human, HumanAgent):
                    if human.workplace is not None and np.random.choice([False, True],
                                                                        p=[fire_employees, 1 - fire_employees]):
                        human.workplace = None

        if 'total_lockdown' in self.alarm_state.keys():
            self.lockdown_total = self.alarm_state['total_lockdown']

    def createAgents(self, Humanagents, Workplaces, friendsXagent=3):

        family_dist = create_families(Humanagents)
        agentsToBecreated = len(self._agentsToAdd) - 1
        index = 0

        while agentsToBecreated >= 0:
            if isinstance(self._agentsToAdd[agentsToBecreated], HumanAgent):
                # FAMILY PART
                if family_dist[index] == 0: index += 1
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))

                for i in range(0, family_dist[index]):
                    self._agentsToAdd[agentsToBecreated - i].pos = position
                    self._agentsToAdd[agentsToBecreated - i].house = position

                    # FRIENDS
                    friends = random.sample([fr for fr in range(0, Humanagents) if fr != agentsToBecreated - i],
                                            friendsXagent)  # get index position of random people to be friends
                    for friend_index in friends:
                        self._agentsToAdd[agentsToBecreated - i].friends.add(self._agentsToAdd[friend_index].id)
                        self._agentsToAdd[friend_index].friends.add(self._agentsToAdd[agentsToBecreated - i].id)

                    # INFECTION
                    infected = np.random.choice(["S", "E", "I"], p=[0.985, 0.005, 0.01])
                    if infected == "I":
                        self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(agentsToBecreated - i, "I",
                                                                                         age_())
                        self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                            list(range(1, 11)))
                        self.collector_counts["SUSC"] -= 1
                        self.collector_counts["INF"] += 1  # Adjust initial counts
                        self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.DateTime.strftime('%Y-%m-%d')] = [0,
                                                                                                                    round(
                                                                                                                        1 /
                                                                                                                        self._agentsToAdd[
                                                                                                                            agentsToBecreated - i].machine.rate[
                                                                                                                            'rIR']) -
                                                                                                                    self._agentsToAdd[
                                                                                                                        agentsToBecreated - i].machine.time_in_state,
                                                                                                                    0]
                    elif infected == "E":
                        self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(agentsToBecreated - i, "E",
                                                                                         age_())
                        self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                            list(range(1, 5)))

                        self.collector_counts["SUSC"] -= 1
                        self.collector_counts["EXP"] += 1  # Adjust initial counts
                        self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.DateTime.strftime('%Y-%m-%d')] = \
                            [0, round(1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rEI']) + round(
                                1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rIR']) - self._agentsToAdd[
                                 agentsToBecreated - i].machine.time_in_state, 0]

                    else:
                        self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(
                            self._agentsToAdd[agentsToBecreated - i].id, "S", age_())

                    # EMPLOYMENT

                    if np.random.choice([True, False], p=[self.employment_rate, 1 - self.employment_rate]) and 5 < \
                            self._agentsToAdd[agentsToBecreated - i].machine.age < 65:
                        workplaces = random.sample(
                            list(range(len(self._agentsToAdd) - Workplaces, len(self._agentsToAdd))), Workplaces)
                        for workplace in workplaces:
                            if self._agentsToAdd[workplace].total_capacity > len(
                                    self._agentsToAdd[workplace].get_workers()):
                                self._agentsToAdd[agentsToBecreated - i].workplace = self._agentsToAdd[workplace].id
                                self._agentsToAdd[workplace].add_worker(self._agentsToAdd[agentsToBecreated - i].id)
                            break

                agentsToBecreated -= family_dist[index]
                family_dist[index] = 0
            elif isinstance(self._agentsToAdd[agentsToBecreated], Hospital):
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                self._agentsToAdd[agentsToBecreated].place = position  # redundant
                self._agentsToAdd[agentsToBecreated].pos = position
                # print(f"Hospital {self._agentsToAdd[agents_created].id} created at {self._agentsToAdd[agents_created].place} place")
                agentsToBecreated -= 1
            elif isinstance(self._agentsToAdd[agentsToBecreated], Workplace):
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                self._agentsToAdd[agentsToBecreated].place = position  # redundant
                self._agentsToAdd[agentsToBecreated].pos = position
                self._agentsToAdd[agentsToBecreated].total_capacity = self.peopleInMeeting

                agentsToBecreated -= 1

        super().createAgents()
        dc.update_DC_table(self)

        #         # print(f'{a.unique_id} has {a.machine.age} these friends and works at {a.workplace} with state {a.machine.state}')

    def calculate_R0(self, current_step):
        """ R0: prob of transmission x contacts x days with disease """
        today = current_step.strftime('%Y-%m-%d')
        yesterday = (current_step - timedelta(days=1)).strftime('%Y-%m-%d')
        # use yesterday for detected people since it is the data recorded and given to hosp

        R0_values = [0, 0, 0]
        R0_obs_values = [0, 0, 0]
        hosp_count = 0
        agents_quarantined = 0

        for human in [agent for agent in self.schedule.agents if isinstance(agent, HumanAgent)]:
            if human.machine.state in ["E", "I", "A"] and yesterday != '2020-12-31' and yesterday in human.R0_contacts:
                if human.HospDetected:  # calculate R0 observed
                    hosp_count += 1

                    if human.quarantined is not None:
                        agents_quarantined += 1

                    try:
                        contacts = human.R0_contacts[yesterday][2]
                    except KeyError:  # is a exp agent new from today
                        contacts = human.R0_contacts[today][2]
                        yesterday = today

                    if contacts == 0: contacts = 1
                    # sorted(h.keys())[-1]
                    R0_obs_values[0] += human.R0_contacts[yesterday][0] / contacts  # mean value of transmission
                    R0_obs_values[1] += human.R0_contacts[yesterday][1]
                    R0_obs_values[2] += human.R0_contacts[yesterday][2]


                contacts = human.R0_contacts[today][2]
                if contacts == 0: contacts = 1
                R0_values[0] += human.R0_contacts[today][0] / contacts  # mean value of transmission
                R0_values[1] += human.R0_contacts[today][1]
                R0_values[2] += human.R0_contacts[today][2]

        total_inf_exp = self.collector_counts["INF"] + self.collector_counts["EXP"]
        if total_inf_exp == 0: total_inf_exp = 1

        self.R0 = round(
            (R0_values[0] / total_inf_exp) * (R0_values[1] / total_inf_exp) * (R0_values[2] / total_inf_exp), 3)

        if hosp_count == 0:
            hosp_count = 1

        old_R0_obs = self.R0_obs
        # self.R0_obs = (self.R0_obs + round((R0_obs_values[0] / hosp_count) * (R0_obs_values[1] / hosp_count) * (R0_obs_values[2] / hosp_count), 2))/2
        self.R0_obs = (old_R0_obs + round(
            (R0_obs_values[0] / hosp_count) * (R0_obs_values[1] / hosp_count) * (R0_obs_values[2] / hosp_count), 3)) / 2

        # print("HOY DIA", self.DateTime, "hay: ", hosp_count, "EN R0")

        # self.R0_observed[0] = round((R0_values[2] / total_inf_exp), 2)
        # self.R0_observed[1] = agents_quarantined / 10
        # # round((R0_obs_values[0] / hosp_count) * (R0_obs_values[1] / hosp_count) * (R0_obs_values[2] / hosp_count), 2)
        # self.R0_observed[2] = round((R0_obs_values[2] / hosp_count), 2)

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

        # print(f"Lista total de agentes a testear a primera hora: {self.peopleToTest}")

        # shuffle agent list to distribute to test agents among the hospitals
        agents_list = self.schedule.agents.copy()
        random.shuffle(agents_list)
        for a in agents_list:
            # delete contacts of human agents of Adays time
            if isinstance(a, HumanAgent):
                if Atime in a.contacts: del a.contacts[Atime]
                if Atime in a.R0_contacts: del a.R0_contacts[Atime]
                # create dict R0 for infected people in case it is not updated during the day
                if a.machine.state in ["I", "A"]:
                    a.R0_contacts[self.DateTime.strftime('%Y-%m-%d')] = [0, round(
                        1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]
                elif a.machine.state == "E":
                    a.R0_contacts[self.DateTime.strftime('%Y-%m-%d')] = [0, round(1 / a.machine.rate['rEI']) + round(
                        1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]

            elif isinstance(a, Hospital):
                peopleTested = a.decideTesting(peopleTested)
                """if date in a.PCR_results:
                    if not date in self.peopleTested: self.peopleTested[date] = a.PCR_results[date]
                    else:
                        for elem in a.PCR_results[date]: self.peopleTested[date].add(elem)"""
                # delete contact tracing of Hdays time
                if Htime in a.PCR_testing: del a.PCR_testing[Htime]
                if Htime in a.PCR_results: del a.PCR_results[Htime]

        self.peopleTested[self.DateTime.strftime('%Y-%m-%d')] = set()
        self.peopleToTest[self.DateTime.strftime('%Y-%m-%d')] = set()

        # print(f"Lista de contactos de hospital {a.unique_id} es {a.PCR_testing}")

        # print(f"Lista total de testeados: {self.peopleTested}")
        # print(f"Lista total de agentes a testear: {self.peopleToTest}")

    def changeAgentStates(self):
        """ UPDATE AGENTS STATE """
        asymptomatic = 0
        symptomatic = 0
        for human in [agent for agent in self.schedule.agents if isinstance(agent, HumanAgent)]:
            s = human.machine.state
            human.machine.check_state()

            if s != human.machine.state:

                if human.machine.state == "S":  # if s == "R"
                    # self.hosp_collector_counts['H-REC'] -= 1
                    # self.hosp_collector_counts['H-SUSC'] += 1
                    human.HospDetected = False

                elif human.machine.state == "I":  # infected, symptomatic, presents symptoms
                    symptomatic += 1

                    if self.quarantine_period == 0:
                        human.quarantined = self.DateTime + timedelta(days=1)  # quarantine
                    else:
                        human.quarantined = self.DateTime + timedelta(days=self.quarantine_period)  # quarantine
                    human.obj_place = min(self.getHospitalPosition(), key=lambda c: euclidean(c, human.pos))

                elif human.machine.state == "A":  # if s == "E":
                    asymptomatic += 1

                elif human.machine.state == "H":  # if s == "A":
                    # self.hosp_collector_counts['H-INF'] -= 1  # doesnt need to be, maybe it was not in the record
                    # self.hosp_collector_counts['H-HOSP'] += 1
                    human.HospDetected = False  # we assume hospitalized people do not transmit the virus

                    if self.hosp_collector_counts["H-HOSP"] >= (
                            self.Hosp_capacity * self.N_hospitals):  # hospital collapse
                        human.machine.dead(H_collapse=True)
                    else:
                        # look for the nearest hospital to treat the agent
                        human.obj_place = min(self.getHospitalPosition(), key=lambda c: euclidean(c, human.pos))
                        human.getWorld().space.move_agent(human, human.obj_place)
                        human.mask = Mask.FFP2

                        # adds patient to nearest hospital patients list
                        h = self.getHospitalPosition(human.obj_place)
                        h.add_patient(human)

                        human.quarantined = None
                        human.friend_to_meet = set()

                elif human.machine.state == "R":
                    """if s in ["I", "A"]:
                        if human.HospDetected:
                            self.hosp_collector_counts['H-INF'] -= 1
                            self.hosp_collector_counts['H-REC'] += 1"""

                    if s == "H":
                        h = self.getHospitalPosition(human.obj_place)
                        h.discharge_patient(human)
                        human.HospDetected = True
                        # self.hosp_collector_counts['H-HOSP'] -= 1
                        # self.hosp_collector_counts['H-REC'] += 1

                elif human.machine.state == "D":  # if s == "H":
                    h = self.getHospitalPosition(human.obj_place)
                    h.discharge_patient(human)
                    # self.hosp_collector_counts['H-HOSP'] -= 1
                    # self.hosp_collector_counts['H-DEAD'] += 1

            # change quarantine status if necessary
            if human.quarantined is not None and self.DateTime.day == human.quarantined.day:
                human.quarantined = None
                if human.machine.state in ["E", "I", "A"] and human.HospDetected:
                    # test them again
                    self.peopleToTest[self.DateTime.strftime('%Y-%m-%d')].add(human.id)

            if human.machine.state in ["A", "I"]:
                tup = (human.machine.time_in_state, human.machine.state)
                if not human.id in self.R0_observed.keys():
                    self.R0_observed[human.id] = [tup]
                else:
                    self.R0_observed[human.id].append(tup)
            # print("ANOTHER DAY:", "ASYMPT:", asymptomatic, " SYMPTOM:", symptomatic)

            """            if t > human.machine.time_in_state and s == human.machine.state:
                print("ERROR", s)
                a"""
