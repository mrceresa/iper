import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator

class EconomyManager (object):
    def __init__(self, city):
        self.model = city
        self.agent_purchases = {}
        self.btob_purchases = {}
        self.imports = {}
        self.exports = {}
        self.pending_exports = []

    
    def make_offer(self, b_id): # tuple -> (bool = has_stock , int = sell_price)
        b = self.model.businesses[b_id]
        if b.stock > 1:
            return True, b.sell_price
        else:
            return False, math.inf
    
    def reject_offer(self,b_id):
        b = self.model.businesses[b_id]
        b.sell_balance["total"] += 1
        b.sell_balance["rejected"] += 1
        return

    def buy_stock(self, a_id, b_id):
        ammount = 1
        a = self.model.agents[a_id]
        b = self.model.businesses[b_id]

        a.inventory["funds"] -= b.sell_price
        b.stock -= 1
        a.inventory[self.model.check_agent_inv(b.speciallity)] = 100

        b.funds += b.sell_price 
        b.sell_balance["total"] += 1
        b.sell_balance["accepted"] += 1
        
        if self.agent_purchases.get(b.speciallity) is None:
            self.agent_purchases[b.speciallity] = {'ammount' : ammount, 'value' : ammount * b.sell_price}
        else:
            self.agent_purchases[b.speciallity] = {'ammount' : self.agent_purchases[b.speciallity]['ammount'] + ammount, 'value' : self.agent_purchases[b.speciallity]['value'] + ammount * b.sell_price} 

    def buy_resources(self, b_id, b_spec, buy_ammount, one_time=False):
        
        acc_b = self.model.businesses[b_id]
        if len(self.pending_exports) > 0:
            for e in self.pending_exports:
                ammount = e['ammount']
                exp_b = self.model.businesses[e['bid']]
                if exp_b.speciallity == self.model.check_resource_type(b_spec):
                    if exp_b.sell_price * ammount > acc_b.funds:
                        exp_b.sell_balance["total"] += 1
                        exp_b.sell_balance["rejected"] += 1
                    else:
                        exp_b.sell_balance["total"] += 1
                        exp_b.sell_balance["accepted"] += 1
                          
                        exp_b.stock -= ammount
                        exp_b.funds += exp_b.sell_price * ammount
                        
                        if self.btob_purchases.get(exp_b.speciallity) is None:
                            self.btob_purchases[exp_b.speciallity] = {'ammount' : ammount, 'value' : ammount * exp_b.sell_price}
                        else:
                            self.btob_purchases[exp_b.speciallity] = {'ammount' : self.btob_purchases[exp_b.speciallity]['ammount'] + ammount, 'value' : self.btob_purchases[exp_b.speciallity]['value'] + ammount * exp_b.sell_price} 
                        
                        if not one_time:
                            acc_b.stock += ammount
                            acc_b.pending_res = False
                        acc_b.funds -= exp_b.sell_price * ammount
                        
                        self.pending_exports.remove(e)
                        return 
        
        self.import_stock(b_id, buy_ammount, acc_b.production_price)
        self.model.businesses[b_id].pending_res = False
        

    def export_stock(self, b_id, ammount):
        self.pending_exports.append({'bid': b_id, 'ammount':ammount})

    def import_stock(self, b_id, ammount, price):
        
        b = self.model.businesses[b_id]
        b.stock += ammount
        b.funds -= price * ammount
        import_type = self.model.check_resource_type(b.speciallity)
        if self.imports.get(import_type) is None:
            self.imports[import_type] = {'ammount' : ammount, 'value' : (ammount * price)}
        else:
            self.imports[import_type] = {'ammount' : self.imports[import_type]['ammount'] + ammount, 'value' : self.imports[import_type]['value'] + (ammount * price)}
    
    def export_all (self):
        for e in self.pending_exports:

            b_id = e['bid']
            ammount = e['ammount']
            b = self.model.businesses[b_id]
            b.stock -= ammount
            b.funds += b.sell_price * ammount
            if self.exports.get(b.speciallity) is None:
                self.exports[b.speciallity] = {'ammount' : ammount, 'value' : ammount * b.sell_price}
            else:
                self.exports[b.speciallity] = {'ammount' : self.exports[b.speciallity]['ammount'] + ammount, 'value' : self.exports[b.speciallity]['value'] + ammount * b.sell_price} 

            self.pending_exports = self.pending_exports[1:]