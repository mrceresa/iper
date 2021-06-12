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
from Economy import EconomyManager
import DataCollerctorFunc as dcf

'''Simple Class to use when necessary'''
class Container (object):
    pass 

class city_model(Model):
    ## Initializing the model.
    def __init__(self, N, city_DF, width, height, init_w, init_h, res_map, curf_restrict = False, s_restrict = False):
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
        self.resource_map = res_map
        
        #Modules
        self.job_manager = EmploymentManager(self)
        self.eco_manager = EconomyManager(self)


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
        self.datacollector = DataCollector(
            model_reporters={"Import(#)": dcf.get_import_amm, 
            "Import($)": dcf.get_import_val, 
            "Export(#)": dcf.get_export_amm, 
            "Export($)": dcf.get_export_val,
            "Commerce($)": dcf.get_agent_commerce})

    
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
    
    def check_resource_type(self, b_spec):
        return self.resource_map[b_spec]
    
    def check_agent_inv(self, spec):
        if spec == 'catering':
            return 'food'
        elif spec == 'hotel':
            return 'housing'
        elif spec == 'recreational activities':
            return 'happiness'
        elif spec == 'commerce':
            return 'basic_goods'
        else:
            return 'health'

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
        if (self.time.hour == 0): 
            self.eco_manager.export_all()
        self.time = self.time + self.time_delta