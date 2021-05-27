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
from Agent import Citizen
from Business import Business
from Employment import EmploymentManager

'''Simple Class to use when necessary'''
class Container (object):
    pass 

class city_model(Model):
    ## Initializing the model.
    def __init__(self, N, city_DF, width, height, init_w, init_h, curf_restrict = False, s_restrict = False):
        self.schedule = RandomActivation(self)
        self.grid = GeoSpace()
        self.df = city_DF
        self.num_agents = N
        self.num_stores = len(city_DF)
        self.agents = {}
        self.businesses = {}
        self.time = datetime.datetime(2019,1,1) # time starts at 1-1-2021 at midnight
        self.time_delta = datetime.timedelta(minutes=15) # each step will add 15 mins
        self.max_P = Point(width, height)
        self.min_P = Point(init_w, init_h)
        self.job_manager = EmploymentManager(self)

        self.curfew = curf_restrict
        self.shop_restriction = s_restrict

        ## CREATE BUSINESSES
        businesses_kwargs = dict(model=self)
        bc = AgentCreator(agent_class=Business, agent_kwargs=businesses_kwargs)
        self.init_businesses = bc.from_GeoDataFrame(city_DF)
        self.grid.add_agents(self.init_businesses)

        for _b in self.init_businesses:
            self.schedule.add(_b)
            self.businesses[_b.unique_id] = _b

        ## CREATE AGENTS 
        agent_list = []
        for i in range(self.num_agents):
            agent_list.append(Point(random.uniform(self.min_P.x, self.max_P.x),random.uniform(self.min_P.y, self.max_P.y)))
        agent_DF = gpd.GeoDataFrame(agent_list,columns=['geometry'],crs={'init': 'epsg:3857'})
        agent_DF.index = agent_DF.index + city_DF.index[-1] + 1
        
        citizens_kwargs = dict(model=self)
        ac = AgentCreator(agent_class=Citizen, agent_kwargs=citizens_kwargs)
        self.init_agents = ac.from_GeoDataFrame(agent_DF)
        self.grid.add_agents(self.init_agents)

        for _a in self.init_agents:
            self.schedule.add(_a)
            self.agents[_a.unique_id] = _a
        
        ## DATA COLLECTOR
        self.datacollector = DataCollector()
    
    ## All Business GET Functions
    def get_name(self, b_id):
        return self.df.iloc[b_id].Name
    
    def get_address(self, b_id):
        return self.df.iloc[b_id].Street , self.city_DF.iloc[b_id].AddNum

    def get_tag(self, b_id):
        return self.df.iloc[b_id].Tag

    def get_speciallity(self, b_id):
        return self.df.iloc[b_id].Speciallity

    def get_speciallityAdapt(self, b_id):
        return self.df.iloc[b_id].SpeciallityAdapt

    ## All Other Functions
    def get_all_business(self): # dict -> {key = business_id , value = business_pos}
        ret_buss = {}
        ret_type = {} 
        for id, b in self.businesses.items():
            ret_buss[id] = b.shape
            ret_type[id] = b.speciallity
        return ret_buss, ret_type
    
    def make_offer(self, b_id): # tuple -> (bool = has_stock , int = sell_price)
        b = self.businesses[b_id]
        if b.stock > 1:
            return True, b.sell_price
        else:
            return False, math.inf
    
    def reject_offer(self,b_id):
        b = self.businesses[b_id]
        b.sell_balance["total"] += 1
        b.sell_balance["rejected"] += 1
        return

    def buy_stock(self, a_id, b_id):
        a = self.agents[a_id]
        b = self.businesses[b_id]

        a.inventory["funds"] -= b.sell_price
        b.stock -= 1
        if b.speciallity in a.inventory.keys():
            a.inventory[b.speciallity] += 1
        else:
            a.inventory[b.speciallity] = 1
        b.funds += b.sell_price 

        b.sell_balance["total"] += 1
        b.sell_balance["accepted"] += 1

    def move_citizen(self, agent, newPos):
        if self.check_move_policies():
          self.agents[agent.unique_id].shape = newPos
        #self.grid.move_agent(agent, newPos)
    
    def check_move_policies(self):
        if not self.curfew: 
            return True
        else: 
            start_curfew = 23
            end_curfew = 6
            if my_event.start_time < self.time.hour < my_event.end_time:
                return False
            else:
                return True
    
    def check_buy_policies(self,speciality):
        if not self.shop_restriction:
            return True
        else:
            if speciality == 'catering':
                return True
            else:
                return False

    def step(self):  
        self.datacollector.collect(self)
        self.schedule.step()
        self.job_manager.resolve()
        self.time = self.time + self.time_delta