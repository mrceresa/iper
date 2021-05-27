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

class Business (GeoAgent):
    ## Initializing the agent.
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.funds = 5000
        self.stock = 0
        self.employees = []
        self.searching = False
        self.work_start = random.randrange(7, 11)
        self.work_end = self.work_start + 8
        
        ## Production info
        self.production_price = 20
        self.sell_price = 100
        self.sell_balance = {"total": 0, "accepted": 0, "rejected": 0}
        
        self.speciallity = self.model.get_speciallityAdapt(unique_id)
        self.search_employee() #we start searching for someone to produce

    def check_balance(self):
       
        # If 60% or more rejection rate -> price down 
        if (self.sell_balance["rejected"] / self.sell_balance["total"]) > 0.6:
            if self.sell_price > self.production_price: # can't sell at a loss
                self.sell_price -=  1
        # If 90% or more acceptance rate -> price up
        if (self.sell_balance["accepted"] / self.sell_balance["total"]) > 0.9:
            self.sell_price +=  1
            if len(self.employees) < 5:
                self.search_employee()
        
        self.sell_balance = {"total": 0, "accepted": 0, "rejected": 0} #Reset balance
    
    def produce(self):

        num_emp = len(self.employees)
        if (num_emp == 0) and (self.searching == False):
            self.search_employee()
        elif num_emp != 0:
            self.funds -= self.production_price * num_emp #consume funds to produce stock
            self.stock += num_emp  #produces more goods
    
    def search_employee (self):
        self.searching = True
        self.model.job_manager.job_offer(self)

    def pay_employees (self):
        for _id in self.employees:
            self.model.job_manager.pay_employee(self,_id)
      
    def step(self):
        # print("Hi, I am business " + str(self.unique_id) +"! POS : ")
        # print(self.shape)
        if self.sell_balance["total"] > 50:
            self.check_balance()
        if self.model.time.day == 28:
            self.pay_employees()
        self.produce()