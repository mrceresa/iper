import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator

def get_agent_commerce (model):
    if model.eco_manager.agent_purchases.get('commerce') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['commerce']['value']

def get_agent_catering (model):
	return model.eco_manager.agent_purchases['catering'].value

def get_agent_health (model):
	return model.eco_manager.agent_purchases['health'].value

def get_agent_recreation (model):
	return model.eco_manager.agent_purchases['recreational activities'].value

def get_agent_transport (model):
	return model.eco_manager.agent_purchases['transport'].value

def get_agent_hotel (model):
	return model.eco_manager.agent_purchases['hotel'].value

def get_agent_state (model):
	return model.eco_manager.agent_purchases['real state'].value

#################################################################################

def get_bus_state (model):
	return model.eco_manager.agent_purchases['real state'].value

#################################################################################

def get_export_amm (model):
    ans = 0
    for i in model.eco_manager.exports.values():
        ans = ans + i['ammount']
    return ans

def get_export_val (model):
    ans = 0
    for i in model.eco_manager.exports.values():
        ans = ans + i['value']
    return ans

def get_import_amm (model):
    ans = 0
    for i in model.eco_manager.imports.values():
        ans = ans + i['ammount']
    return ans

def get_import_val (model):
    ans = 0
    for i in model.eco_manager.imports.values():
        ans = ans + i['value']
    return ans
