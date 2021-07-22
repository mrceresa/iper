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
from random import uniform, randint
import time
from datetime import datetime, timedelta
from attributes_Agent import job, create_families, age_
from Hospital_class import Workplace, Hospital
from agents import HumanAgent
import DataCollector_functions as dc
from SEAIHRD_class import SEAIHRD_covid, Mask

from EudaldMobility.Mobility import Map_to_Graph

class CityModel(MultiEnvironmentWorld):

    def __init__(self, config):
        super().__init__(config)

        self.l.info("Initalizing model")
        self._basemap = config["basemap"]
        self.network = NetworkGrid(nx.Graph())
        self.l.info("Scheduler is " + str(self.schedule))
        self.schedule = RandomActivation(self)
        self.count=0
        #L'istituto RKI, Robert Koch Institut calcola Rt come 
        # rapporto tra la somma del numero di contagiati negli ultimi 4 giorni e la somma del numero dei contagiati nei 4 giorni precedenti
        self.days_for_R0_RKI=3 #these are the days for the computation of the coefficient Rt according to the RKI method
        self.Infected_detects_for_RKI_today=0
        self.Infected_detects_for_RKI=[]
        self.Infected_type_for_RKI=["I","A"]
        self.Infected_for_RKI_today=0
        self.Infected_for_RKI=[]
        self.totalInStatesForDay=[] 

        self.agents_in_states={"S":0, "E":0,"A":0,"I":0,"H":0,"R":0,"D":0}
        self.E_today=0
        self.today1=0
        self.Hospitalized_total=0
        self.perc_vacc_day= 0.01
        self.vaccinations_on_day=math.ceil(self.perc_vacc_day * config["agents"])
        self.tobevaccinatedtoday=self.vaccinations_on_day

        self.l.info("Loading geodata")
        self._initGeo()
        self._loadGeoData()
        self.space = GeoSpacePandas(
            extent=(self._xs["w"], self._xs["s"],
                    self._xs["e"], self._xs["n"]
                    )
        )

        self.Ped_Map = Map_to_Graph('Pedestrian')  # Load the shapefiles
        self.define_boundaries_from_graphs(self.Ped_Map)

        # self.virus = VirusCovid(config["virus"])
        self.pTest = 0.95
        self.R0 = 0
        self.R0_obs = 0
        self.R0_observed = {}
        self.contact_count = [0, 0]
        self._contact_agents = set()

        # alarm state characteristics
        self.alarm_state = config["alarm_state"]
        self.lockdown_total = False
        self.quarantine_period = config["quarantine"]
        self.night_curfew = 24
        self.masks_probs = [1, 0, 0]  # [0.01, 0.66,0.33]

        self._hospitals = {}
        self.N_hospitals = config["hospitals"]
        self.Hosp_capacity = math.ceil((0.1 * config["agents"]) / self.N_hospitals)#math.ceil((0.0046 * config["agents"]) / self.N_hospitals)     # 4.6 beds per 1,000 inhabitants

        self.PCR_tests = config["tests"] / config["hospitals"]
        # self.PCR_tests = math.ceil(config["tests"] / config["hospitals"])

        self.employment_rate = config["employment_rate"]
        self.peopleTested = {}
        self.peopleToTest = {}

        self.peopleInMeeting = config["peopleMeeting"]  # max people to meet with
        self.peopleInMeetingSd = config["peopleMeeting"] * 0.2

        self.general_run = config["general_run"]

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

        self.boundaries["dx"] = 111.32;  # One degree in longitude is this in KM
        self.boundaries["dy"] = 40075 * math.cos(self.boundaries["centroid"].y) / 360
        self.l.info("Arc amplitude at this latitude %f, %f" % (self.boundaries["dx"], self.boundaries["dy"]))

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
        shpfilename = os.path.join(path, "shapefiles", "quartieriBarca1.shp")
        if not os.path.exists(shpfilename):
            shpfilename = os.path.join(path, "examples/bcn_multispace/shapefiles", "quartieriBarca1.shp")
        #print("Loading shapefile from", shpfilename)
        blocks = gpd.read_file(shpfilename)
        self._blocks = blocks

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
        self.l.info("PLOTTING RESULTS")
        self.l.info(self.Hospitalized_total)
        if isinstance(self.alarm_state['inf_threshold'], int) or self.alarm_state['inf_threshold'] == "2021-01-02":
            alarm_state = False
        else:
            alarm_state = True

        X = self.datacollector.get_table_dataframe("Model_DC_Table")

        X['Day'] = X['Day'].apply(pd.Timestamp)

        R0_df = X[['Day', 'R0', 'R0_Obs']]
        R0_df.to_csv(outdir + "/" + R0_title + str(self.general_run) + '.csv', index=False)  # get the csv

        ### R0 plot ###
        columns = ['R0', 'R0_Obs']  # , 'Mcontacts', 'Quarantined', 'Contacts']
        colors = ["Orange", "Green", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Rt values')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, R0_title))

        # Model stats plot
        X.drop(['R0', 'R0_Obs'], axis=1, inplace=True)
        X.to_csv(outdir + "/" + title + str(self.general_run) + '.csv', index=False)  # get the csv

        columns = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Hospitalized', 'Dead']
        colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]

        X.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Model stats')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, title))

        Y = self.hosp_collector.get_table_dataframe("Hosp_DC_Table")
        Y['Day'] = Y['Day'].apply(pd.Timestamp)
        Y.to_csv(outdir + "/" + hosp_title + str(self.general_run) + '.csv', index=False)  # get the csv

        # Hospital stats plot
        columns = ['Hosp-Susceptible', 'Hosp-Infected', 'Hosp-Recovered', 'Hosp-Hospitalized', 'Hosp-Dead']
        colors = ["Green", "Red", "Blue", "Gray", "Black"]

        Y.plot(x="Day", y=columns, color=colors)  # table=True
        if alarm_state: plt.axvline(pd.Timestamp(self.alarm_state['inf_threshold']), color='r', linestyle="dashed",
                                    label='Lockdown')
        plt.ylabel('Values')
        plt.title('Observed stats')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, hosp_title))



    def getHospitals(self):
        return self._hospitals.values()

    def getHospitalPosition(self):
        """ Returns the position of the Hospitals or the Hospital agent if position is given """
        return self._hospitals.keys() #TODO: place or position?

    def _check_movement_is_restricted(self):
        restricted = False
        if restricted:
            if 0 < self.getTime().hour < 6:
                return True

    def stepDay(self):
        """Called if there is a new day. Reimplement if neede"""
        super().stepDay()
        self.count+=1
        self.tobevaccinatedtoday=self.vaccinations_on_day

        self.totalInStatesForDay.append(self.agents_in_states)
        _t = self.datacollector.tables['Model_DC_Table']
        S,E,I,R,H,D = _t["Susceptible"][-1], _t["Exposed"][-1], _t["Infected"][-1], _t["Recovered"][-1], _t["Hospitalized"][-1], _t["Dead"][-1]
        self.l.info("************ S %d,E %d,I %d,R %d,H %d,D %d"%(S,E,I,R,H,D))
        dc.reset_counts(self)
        dc.reset_hosp_counts(self)
        dc.update_stats(self)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",self.hosp_collector_counts["H-INF"])

        self.Infected_detects_for_RKI.append(self.Infected_detects_for_RKI_today)
        self.Infected_for_RKI.append(self.Infected_for_RKI_today)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", self.Infected_detects_for_RKI)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", self.Infected_for_RKI)
        self.calculate_R0()
        self.Infected_detects_for_RKI_today=0
        self.Infected_for_RKI_today=0

        dc.update_DC_table(self)
        #print("T"*30,len(self._contact_agents))
        self._contact_agents = set()
        
        # clean contact lists from agents for faster computations
        self.clean_contact_list(Adays=2, Hdays=5, Tdays=10)           

        # decide on applying stricter measures
        if isinstance(self.alarm_state['inf_threshold'], int):
            if self.hosp_collector_counts["H-INF"] >= self.alarm_state['inf_threshold'] and self.alarm_state[
                'inf_threshold'] != 1:  # dont apply lockdown if threshold is set to 1
                # print("NIGHT CURFEW: ", self.night_curfew, '\n', "MASKS PROBS: ", self.masks_probs, '\n QUARANTINE: ', self.quarantine_period, "\n MEETIGN:", self.peopleInMeeting)
                self.activate_alarm_state()
                # print("NIGHT CURFEW: ", self.night_curfew, '\n', "MASKS PROBS: ", self.masks_probs, '\n QUARANTINE: ', self.quarantine_period, "\n MEETIGN:", self.peopleInMeeting)
        # self.plot_results()  # title="server_stats", hosp_title="server_hosp_stats"
        self.l.info("Plotting graphs...")
        tic = time.perf_counter()
        self._plotAgents(self.config["output_dir"], "step-%d-day-%d.png"%(self.getStep(), self.getTime().day))
        toc = time.perf_counter()
        self.l.info("Done plotting graphs for the day %.1f seconds" % (toc - tic))


    def step(self):
        #current_step = self.DateTime
        #self.DateTime += timedelta(minutes=self.timestep)  # next step
        #self.l.info("Current simulation time is %s"%str(self.DateTime))
        #self.schedule.step()

        if self.space._gdf_is_dirty:
            self.space._create_gdf()
            # self.space._create_gdf

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)

        super().step()

    
       #self.plotAll(self.config["output_dir"], "res%d.png"%self.getStep())
    
    def activate_alarm_state(self):
        self.alarm_state['inf_threshold'] = self.getTime().strftime("%Y-%m-%d")

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

    def createAgents(self, Humanagents, Workplaces, friendsXagent=3):# friendsXagent=3

        friendsXagent = self.peopleInMeeting
        family_dist = create_families(Humanagents)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++famiglie",family_dist)
        agentsToBecreated = len(self._agentsToAdd) - 1
        index = 0
        r =40.0/111100
        all_nodes = list(self.Ped_Map.G.nodes)
        while agentsToBecreated >= 0:
            if isinstance(self._agentsToAdd[agentsToBecreated], HumanAgent):
                # FAMILY PART
                if family_dist[index] == 0: index += 1
                node = random.choice(all_nodes)
                position = (self.Ped_Map.G.nodes[node]['lon'],
                            self.Ped_Map.G.nodes[node]['lat'])

                #position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                #print("+++++++++++++++++++++++++++++++++********************",position)

                for i in range(0, family_dist[index]):
                    #position = (position[0]+randint(-1,1)*r, position[1]+randint(-1,1)*r)

                    self._agentsToAdd[agentsToBecreated - i].pos = position
                    self._agentsToAdd[agentsToBecreated - i].house = position
                    #print("++++++++++++++++++++","id agente",self._agentsToAdd[agentsToBecreated - i].id,"   posizione iniziale agente", self._agentsToAdd[agentsToBecreated - i].pos)
                    for y in range(0, family_dist[index]):
                        if i != y:
                            self._agentsToAdd[agentsToBecreated - i].family.add(self._agentsToAdd[agentsToBecreated - y].id)



                    # FRIENDS
                    friends = random.sample([fr for fr in range(0, Humanagents) if fr != agentsToBecreated - i],
                                            friendsXagent)  # get index position of random people to be friends
                    for friend_index in friends:
                        self._agentsToAdd[agentsToBecreated - i].friends.add(self._agentsToAdd[friend_index].id)
                        self._agentsToAdd[friend_index].friends.add(self._agentsToAdd[agentsToBecreated - i].id)

                    # INFECTION
                    infected = np.random.choice(["S", "E","A","I"],  p=[0.995, 0, 0 ,0.005])#p=[0.985, 0, 0.015]) p=[0.970, 0.015,0 ,0.015])#
                    self.agents_in_states[infected]+= 1
                    if infected == "I":
                        self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(agentsToBecreated - i, "I",
                                                                                         age_(), agent=self._agentsToAdd[agentsToBecreated - i])
                        self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                            list(range(1, 11)))
                        self.collector_counts["SUSC"] -= 1
                        self.collector_counts["INF"] += 1  # Adjust initial counts
                        self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0,
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
                                                                                         age_(), agent=self._agentsToAdd[agentsToBecreated - i])
                        self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                            list(range(1, 5)))

                        self.collector_counts["SUSC"] -= 1
                        self.collector_counts["EXP"] += 1  # Adjust initial counts
                        self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.getTime().strftime('%Y-%m-%d')] = \
                            [0, round(1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rEI']) + round(
                                1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rIR']) - self._agentsToAdd[
                                 agentsToBecreated - i].machine.time_in_state, 0]

                    else:
                        self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(
                            self._agentsToAdd[agentsToBecreated - i].id, "S", age_(), agent=self._agentsToAdd[agentsToBecreated - i])

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
                #position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                node = random.choice(all_nodes)
                position = (self.Ped_Map.G.nodes[node]['lon'],
                            self.Ped_Map.G.nodes[node]['lat'])

                self._agentsToAdd[agentsToBecreated].pos = position
                self._hospitals[position] = self._agentsToAdd[agentsToBecreated]
                # print(f"Hospital {self._agentsToAdd[agents_created].id} created at {self._agentsToAdd[agents_created].pos} place")
                agentsToBecreated -= 1
            elif isinstance(self._agentsToAdd[agentsToBecreated], Workplace):
                #position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                node = random.choice(all_nodes)
                position = (self.Ped_Map.G.nodes[node]['lon'],
                            self.Ped_Map.G.nodes[node]['lat'])

                position = (position[0] + randint(-1, 1) * r, position[1] + randint(-1, 1) * r)
                self._agentsToAdd[agentsToBecreated].place = position  # redundant
                self._agentsToAdd[agentsToBecreated].pos = position
                self._agentsToAdd[agentsToBecreated].total_capacity = self.peopleInMeeting

                agentsToBecreated -= 1
            self.totalInStatesForDay.append(self.agents_in_states)    

        super().createAgents()
        dc.update_DC_table(self)

        # for a in [agent for agent in self.schedule.agents if isinstance(agent, Workplace)]:
        #     print(f'{a.unique_id} has {a.workers} ')

    def calculate_R0(self):

        if self.count<self.days_for_R0_RKI:
            self.R0=0
        else:

            if self.agents_in_states["I"]+self.agents_in_states["A"]==0:
                self.R0=0
            else:
                self.R0=(self.E_today*10)/(self.agents_in_states["I"]+self.agents_in_states["A"])

        
        self.E_today=0

        if len(self.Infected_detects_for_RKI)< 2*self.days_for_R0_RKI or sum(self.Infected_detects_for_RKI[-2*self.days_for_R0_RKI:-self.days_for_R0_RKI])==0:
            self.R0_obs = 0
        else:
            self.R0_obs =sum(self.Infected_detects_for_RKI[-self.days_for_R0_RKI:len(self.Infected_detects_for_RKI)])/sum(self.Infected_detects_for_RKI[-2*self.days_for_R0_RKI:-self.days_for_R0_RKI])


    def clean_contact_list(self, Adays, Hdays, Tdays):
        """ Function for deleting past day contacts sets and arrange today's tests"""
        date = self.getTime().strftime('%Y-%m-%d')
        Atime = (self.getTime() - timedelta(days=Adays)).strftime('%Y-%m-%d')
        Htime = (self.getTime() - timedelta(days=Hdays)).strftime('%Y-%m-%d')

        Ttime = (self.getTime() - timedelta(days=Tdays)).strftime('%Y-%m-%d')
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
                    a.R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0, round(
                        1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]
                elif a.machine.state == "E":
                    a.R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0, round(1 / a.machine.rate['rEI']) + round(
                        1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]

            elif isinstance(a, Hospital):
                peopleTested = a.decideTesting(peopleTested)
                """if date in a.PCR_results:
                    if not date in self.peopleTested: self.peopleTested[date] = a.PCR_results[date]
                    else:
                        for elem in a.PCR_results[date]: self.peopleTested[date].add(elem)"""
                # delete contact tracing of Hdays time
                #if Htime in a.PCR_testing: del a.PCR_testing[Htime]
                #if Htime in a.PCR_results: del a.PCR_results[Htime]

                #print(f"Lista de contactos de hospital {a.unique_id} es {a.PCR_testing}. Con {peopleTested}")

        today = self.getTime().strftime('%Y-%m-%d')
        #if not today in self.peopleTested:
        self.peopleTested[today] = set()
        self.peopleToTest[today] = set()

        # print(f"Lista total de testeados: {self.peopleTested}")
        # print(f"Lista total de agentes a testear: {self.peopleToTest}")

    def _on_agent_changed(self, agent, source, dest):
        # today = self.DateTime.day
        # if today != self.today1:
        #     self.E_today=0
        #print("-"*30,"MODEL", agent, source, dest)
        self.agents_in_states[source]+= -1
        self.agents_in_states[dest]+= 1
        if dest=="H":
            self.Hospitalized_total+=1
        #print( "----------------------------------------------------------------------------------------",self.agents_in_states)
        if dest=="E":
            self.E_today+=1
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",self.E_today)
        self.today1= self.getTime().day
        if dest in self.Infected_type_for_RKI:
            self.Infected_for_RKI_today += 1


            
        

   