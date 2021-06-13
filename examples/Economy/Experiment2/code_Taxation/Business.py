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
import random

class Business (GeoAgent):
    ## Initializing the agent.
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.funds = 5000
        self.stock = 0
        self.exp_stock = 0
        self.employees = []
        self.searching = False
        self.work_start = random.randrange(7, 11)
        self.work_end = self.work_start + 8

        ## Production info
        self.pending_res = False
        self.resources = 20
        self.sell_price = random.randint(15, 100)
        self.production_price = self.sell_price - 10
        self.sell_balance = {"total": 0, "accepted": 0, "rejected": 0}
        
        self.speciallity = self.model.get_speciallityAdapt(unique_id)

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
        self.model.eco_manager.buy_resources(self.unique_id, 'Information and communications', len(self.employees),one_time=True)
        self.searching = True
        self.model.job_manager.job_offer(self)

    def pay_employees (self):
        self.model.eco_manager.buy_resources(self.unique_id, 'financial activities', len(self.employees),one_time=True)
        for _id in self.employees:
            self.model.job_manager.pay_employee(self,_id)
      
    def step(self):
        # print("Hi, I am business " + str(self.unique_id) +"! POS : ")
        # print(self.shape)
        if (self.work_end > self.model.time.hour >= self.work_start):
            
            #check if excess stock
            if self.stock > 15:
                self.model.eco_manager.buy_resources(self.unique_id, 'public administration', len(self.employees) ,one_time=True)
                self.model.eco_manager.export_stock(self.unique_id, 5)
            
            # check if can produce
            if self.resources <= len(self.employees):
                if not self.pending_res:
                    if self.funds < (100 * self.production_price):
                        self.model.eco_manager.buy_resources(self.unique_id, self.speciallity, int(self.funds / self.production_price) - 1)
                    else:
                        self.model.eco_manager.buy_resources(self.unique_id, self.speciallity, 100)
                    self.pending_res = True
            else:
                self.produce()
            
            # check if balance prices
            if self.sell_balance["total"] > 50:
                self.check_balance()
              
        if self.model.time.day % 5 == 0:
                self.pay_employees()