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
from datetime import datetime, timedelta
from attributes_Agent import job, create_families
from Hospital_class import Workplace, Hospital
from agents import HumanAgent
from Covid_class import VirusCovid, State
import DataCollector_functions as dc


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

        self.DateTime = datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0)
        self.virus = VirusCovid(config["virus"])

        # variables for model data collector
        self.collector_counts = None
        dc.reset_counts(self)
        self.collector_counts["SUSC"] = config["agents"]

        self.datacollector = DataCollector(
            {"SUSC": dc.get_susceptible_count, "EXP": dc.get_exposed_count, "INF": dc.get_infected_count,
             "REC": dc.get_recovered_count,
             "HOSP": dc.get_hosp_count, "DEAD": dc.get_dead_count, "R0": dc.get_R0, "R0_Obs": dc.get_R0_Obs},
            tables={"Model_DC_Table": {"Day": [], "Susceptible": [], "Exposed": [], "Infected": [], "Recovered": [],
                                       "Hospitalized": [], "Dead": [], "R0": [], "R0_Obs": []}}
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
        self.l.info("Loading geo data from path:" + path)
        blocks = gpd.read_file(os.path.join(path, "shapefiles", "quartieriBarca1.shp"))
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

        X = self.datacollector.get_table_dataframe("Model_DC_Table")
        X.to_csv(outdir + "/" + title + '.csv', index=False)  # get the csv
        R_db = X[['Day', 'R0', 'R0_Obs']]
        X = X.drop(['R0', 'R0_Obs'], axis=1)
        colors = ["Green", "Yellow", "Red", "Blue", "Gray", "Black"]
        X.set_index('Day').plot.line(color=colors).get_figure().savefig(os.path.join(outdir, title))  # plot the stats
        R_db.set_index('Day').plot.line(color=["Orange", "Green"]).get_figure().savefig \
            (os.path.join(outdir, R0_title))  # plot the stats

        Y = self.hosp_collector.get_table_dataframe("Hosp_DC_Table")
        Y.to_csv(outdir + "/" + hosp_title + '.csv', index=False)  # get the csv
        colors = ["Green", "Red", "Blue", "Gray", "Black"]
        Y.set_index('Day').plot.line(color=colors).get_figure().savefig(
            os.path.join(outdir, hosp_title))  # plot the stats

    def step(self):
        self.schedule.step()
        if self.space._gdf_is_dirty: self.space._create_gdf

    def createAgents(self, Humanagents, friendsXagent=3):

        N = len(self._agentsToAdd)
        family_dist = create_families(Humanagents)
        agents_created = 0
        index = 0

        while agents_created < N:
            if isinstance(self._agentsToAdd[agents_created], HumanAgent):
                # FAMILY PART
                if family_dist[index] == 0: index += 1
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))

                for i in range(0, family_dist[index]):
                    self._agentsToAdd[agents_created + i].pos = position
                    self._agentsToAdd[agents_created + i].house = position

                    # FRIENDS
                    friends = random.sample([fr for fr in range(0, Humanagents) if fr != agents_created + i],
                                            friendsXagent)  # get index position of random people to be friends
                    for friend_index in friends:
                        self._agentsToAdd[agents_created + i].friends.add(self._agentsToAdd[friend_index])
                        self._agentsToAdd[friend_index].friends.add(self._agentsToAdd[agents_created + i])

                    # INFECTION
                    infected = np.random.choice([0, 1], p=[0.8, 0.2])
                    if infected:
                        self._agentsToAdd[agents_created + i].state = State.INF
                        self.collector_counts["SUSC"] -= 1
                        self.collector_counts["INF"] += 1  # Adjust initial counts
                        infection_time = dc.get_infection_time(self)
                        self._agentsToAdd[agents_created + i].infecting_time = infection_time
                        self._agentsToAdd[agents_created + i].R0_contacts[self.DateTime.strftime('%Y-%m-%d')] = [0,
                                                                                                                 infection_time,
                                                                                                                 0]

                agents_created += family_dist[index]
                family_dist[index] = 0
            elif isinstance(self._agentsToAdd[agents_created], Hospital):
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                self._agentsToAdd[agents_created].place = position  # redundant
                self._agentsToAdd[agents_created].pos = position
                print(
                    f"Hospital {self._agentsToAdd[agents_created].id} created at {self._agentsToAdd[agents_created].place} place")
                agents_created += 1
            elif isinstance(self._agentsToAdd[agents_created], Workplace):
                position = (uniform(self._xs["w"], self._xs["e"]), uniform(self._xs["s"], self._xs["n"]))
                self._agentsToAdd[agents_created].place = position  # redundant
                self._agentsToAdd[agents_created].pos = position
                print(
                    f"Workplace {self._agentsToAdd[agents_created].id} created at {self._agentsToAdd[agents_created].place} place")
                agents_created += 1

        super().createAgents()

    def printSocialNetwork(self):
        """ Prints on the terminal screen the friends and workplace (if exists) of the human agents of the model. """
        print("PRINTING SOCIAL NEEEEEEEEEEETWORK")
        for a in self.schedule.agents:
            print(a)
            if isinstance(a, HumanAgent):
                print(f'{a.unique_id} these friends and {a.state} state')
