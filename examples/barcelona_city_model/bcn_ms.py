from mesa.time import RandomActivation

from mesa_geo.geoagent import GeoAgent

import math
import networkx as nx
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
import random
import numpy as np

import geoplot
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os

#from io import StringIO
#import json

import sys

from iper.space.cities import CitySpace
from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
#from iper.space.Space import MeshSpace

import networkx as nx

import mapclassify

#from shapely.geometry import Polygon

import logging
#from random import uniform, randint
import time
from datetime import datetime, timedelta
from attributes_Agent import job, age_
from health import Workplace, Hospital
from agents import HumanAgent
import DataCollector_functions as dc
from SEAIHRD_class import SEAIHRD_covid, Mask


#from shapely.geometry import MultiLineString
#from shapely.ops import polygonize, cascaded_union

#import matplotlib.patches as mpatches
#from iper.xmlobjects import XMLObject

from plotter import CityPlotter
from analysis import CityStatistics
from infections import COVID2019Infection

class CityModel(MultiEnvironmentWorld):

    def __init__(self, config):
        super().__init__(config)

        self.l.info("Initalizing model")
        self._basemap = config["basemap"]

        self._infection = COVID2019Infection(config["agents"]) 

        self.network = NetworkGrid(nx.Graph())
        self.l.info("Scheduler is " + str(self.schedule))
        self.schedule = RandomActivation(self)
        self.count=0

        self.space = CitySpace(
            basemap=config["basemap"],
            extent=(2.1, 41.3170353, 41.4679135, 2.2283555),
            path_name = '/EudaldMobility/pickle_objects/projected/'
        )

        self._hospitals = {}
        self.N_hospitals = config["hospitals"]
        self.Hosp_capacity = math.ceil((0.1 * config["agents"]) / self.N_hospitals)#math.ceil((0.0046 * config["agents"]) / self.N_hospitals)     # 4.6 beds per 1,000 inhabitants

        self.employment_rate = config["employment_rate"]

        self.plotter = CityPlotter()
        self.statistics = CityStatistics()


    def _onRun(self):
      self._infection.startInfection()      

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
        if self.space._gdf_is_dirty:
            self.space._create_gdf()

        self.datacollector.collect(self)
        self.hosp_collector.collect(self)

        super().step()

    def createHouseholds(self, N,distr=[19.9 ,23.8 ,20.4, 24.8, 8.9, 2.2]):
      """    
      It generates a number of families given a number of individuals from a distribution list
      each term in the distr list represents the probability of generating a family
      with a number of individuals equal to the index of that element of distr
      for Barcellona:
          distr=[19.9 ,23.8 ,20.4, 24.8, 8.9, 2.2]
      """
      distr2=[]
      distr3=[]
      for i in distr:
          distr3.append(i)
          distr2.append(sum(distr3))
      M=0
      lista_fam=[]
      while M!=N :
          p=random.random()*100
          if N-M<len(distr):
              lista_fam.append(N-M)
          else:   
              for i in range(len(distr2)) :
                  if p<distr2[i]:
                      lista_fam.append(i+1)
                      break
                  else:                                
                      continue  
          M=sum(lista_fam)

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
      return lista_fam

    def createFriends(self):
      # FRIENDS
      friends = random.sample([fr for fr in range(0, Humanagents) if fr != agentsToBecreated - i],
                              friendsXagent)  # get index position of random people to be friends
      for friend_index in friends:
          self._agentsToAdd[agentsToBecreated - i].friends.add(self._agentsToAdd[friend_index].id)
          self._agentsToAdd[friend_index].friends.add(self._agentsToAdd[agentsToBecreated - i].id)


    def createAgents(self, Humanagents, Workplaces, friendsXagent=3):# friendsXagent=3

        #friendsXagent = self.peopleInMeeting
        
        super().createAgents()

        #self.createHouseholds()
        #self.createFriends()

        dc.update_DC_table(self)

        # for a in [agent for agent in self.schedule.agents if isinstance(agent, Workplace)]:
        #     print(f'{a.unique_id} has {a.workers} ')


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


            
        

   