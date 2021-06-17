import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

# MESA CLASSES
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator

# OTHER UTILITIES
import ogr
import shapely
from shapely.geometry import *
import geopandas as gpd
import ipdb
import ast

class Citizen (GeoAgent):
    ## Initializing the agent.
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.inventory = {"food": 100, "health": 100, "happiness": 100, "basic_goods" : 100, "housing": -1,  "funds": 1000}
        self.epsilon = 1.0
        self.knowledge = {} # key = business_id , value = known_price
        self.init_pos = shape
        self.home = None

        self.employed = False
        self.searching = False
        self.work_place = None # business_id, business_pos
        self.work_start = 99
        self.work_end = 99
   
    def think (self):


        if self.inventory['housing'] <= 0:
            self.shop('hotel')
        
        if self.inventory['food'] < 20: # speciality -> catering
            self.shop('catering')
        elif self.inventory['health'] < 45: # speciality -> health
            self.shop('health')
        elif self.inventory['happiness'] < 20: # speciality -> recreational_activities
            self.shop('recreational activities')
        elif self.inventory['basic_goods'] < 10: # speciality -> commerce
            self.shop('commerce')
        
        if self.inventory['funds'] > 1300 and self.home == None: 
            self.shop('real state')
            self.home = self.init_pos

    def shop(self, good_type):

        if not self.model.check_buy_policies(good_type):
            return
        if not self.model.check_move_policies():
            return
        # choose a business to visit
        aux , auxType = self.model.get_all_business()
        auxType = [k for k in auxType if auxType[k] == good_type] # Only needed type
        auxKeys = set(aux.keys()) & set(auxType) # Get all keys
        auxKeys = list(auxKeys)
        aux = {k: aux[k] for k in auxKeys} # Final filter

        business = None # id of chosen business

        explore = False

        if (len(self.knowledge) < len(aux)) and (random.random() < self.epsilon): 
            explore = True
        
        if len(set(aux.keys()).intersection(set(self.knowledge.keys()))) == 0:
            explore = True

        if explore: 
            # agent will explore
            entry_list = list(aux.keys())
            b = random.choice(entry_list) 
            if len(self.knowledge) == 0:
                business = b
            else:
                while b in self.knowledge.keys():  
                    b = random.choice(entry_list)
                business = b
        else :
            # agent won't explore
            known_poss = set(aux.keys()).intersection(set(self.knowledge.keys()))
            known_poss = {k: self.knowledge[k] for k in known_poss} # Final filter
            business = min(known_poss, key=known_poss.get)

            #if len( set(aux.keys()).intersection(set(self.knowledge.keys()))) == 0:
            #    print('I dont know')
            # business = min(self.knowledge.items(), key=lambda x: x[1]) 

        if good_type != 'transport':
          self.move(aux[business])
        
        has_stock, sell_price = self.model.eco_manager.make_offer(business)
        if has_stock and (sell_price <= self.inventory["funds"]): 
            self.model.eco_manager.buy_stock(self.unique_id, business)
        elif sell_price > self.inventory["funds"]:
            self.model.eco_manager.reject_offer(business)
        self.knowledge[business] = sell_price # update knowledge

        # update epsilon (when around 70% explored)
        if len(self.knowledge) > (len(aux)/1.4) : 
            self.epsilon = 0.1

    def work(self):

        if self.model.time.hour == self.work_start:
            self.move(self.work_place[1])
        else:
            self.inventory["happiness"] = self.inventory["happiness"] - 1

    def update_status(self):
        if (random.random() < 0.1):
            self.inventory["health"] -= 10
        self.inventory["food"] -= random.randint(0, 3)
        self.inventory["happiness"] -= random.randint(0, 1)
        if not (self.work_end > self.model.time.hour >= self.work_start):
            self.inventory["basic_goods"] -= random.randint(0, 5)
        if self.home == None and self.model.time.hour == 0:
            self.inventory['housing'] -= 1

    def move(self, newPos):
        # print("\tMoving to",newPos)
        self.shop('transport')
        self.model.move_citizen(self, newPos)

    def step(self):
        # print("Hi, I am citizen " + str(self.unique_id) + "! Pos : ")
        # print(self.shape)
        if (not self.employed and not self.searching):
            self.model.job_manager.job_request(self)
            self.searching = True

        if (self.work_end > self.model.time.hour >= self.work_start):
            self.work()
        else: 
            self.think()

        self.update_status()
