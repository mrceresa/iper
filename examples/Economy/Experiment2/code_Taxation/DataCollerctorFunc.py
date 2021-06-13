import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator

def get_agent_commerce_val (model):
    if model.eco_manager.agent_purchases.get('commerce') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['commerce']['value']

def get_agent_catering_val (model):
    if model.eco_manager.agent_purchases.get('catering') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['catering']['value']

def get_agent_health_val (model):
    if model.eco_manager.agent_purchases.get('health') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['health']['value']

def get_agent_recreation_val (model):
    if model.eco_manager.agent_purchases.get('recreational activities') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['recreational activities']['value']

def get_agent_transport_val (model):
    if model.eco_manager.agent_purchases.get('transport') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['transport']['value']

def get_agent_hotel_val (model):
    if model.eco_manager.agent_purchases.get('hotel') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['hotel']['value']

def get_agent_state_val (model):
    if model.eco_manager.agent_purchases.get('real state') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['real state']['value']

def get_agent_commerce_amm(model):
    if model.eco_manager.agent_purchases.get('commerce') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['commerce']['ammount']

def get_agent_catering_amm (model):
    if model.eco_manager.agent_purchases.get('catering') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['catering']['ammount']

def get_agent_health_amm (model):
    if model.eco_manager.agent_purchases.get('health') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['health']['ammount']

def get_agent_recreation_amm (model):
    if model.eco_manager.agent_purchases.get('recreational activities') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['recreational activities']['ammount']

def get_agent_transport_amm (model):
    if model.eco_manager.agent_purchases.get('transport') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['transport']['ammount']

def get_agent_hotel_amm (model):
    if model.eco_manager.agent_purchases.get('hotel') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['hotel']['ammount']

def get_agent_state_amm (model):
    if model.eco_manager.agent_purchases.get('real state') is None:
        return 0
    else:
        return model.eco_manager.agent_purchases['real state']['ammount']

#################################################################################
###### BUSINESS TO BUSINESS #####################################################

def get_bus_state (model):
    if model.eco_manager.btob_purchases.get('real state') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['real state']['value']

def get_bus_social (model):
    if model.eco_manager.btob_purchases.get('social services') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['social services']['value']

def get_bus_professional (model):
    if model.eco_manager.btob_purchases.get('professional activities') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['professional activities']['value']

def get_bus_scientific (model):
    if model.eco_manager.btob_purchases.get('scientific activities') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['scientific activities']['value']

def get_bus_education (model):
    if model.eco_manager.btob_purchases.get('education') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['education']['value']

def get_bus_artistic (model):
    if model.eco_manager.btob_purchases.get('artistic activities') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['artistic activities']['value']

def get_bus_other (model):
    if model.eco_manager.btob_purchases.get('other') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['other']['value']

def get_bus_financial (model):
    if model.eco_manager.btob_purchases.get('financial activities') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['financial activities']['value']

def get_bus_information (model):
    if model.eco_manager.btob_purchases.get('Information and communications') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['Information and communications']['value']

def get_bus_public (model):
    if model.eco_manager.btob_purchases.get('public administration') is None:
        return 0
    else:
        return model.eco_manager.btob_purchases['public administration']['value']

#################################################################################
###### EXPORTS ##################################################################

def ex_state (model):
    if model.eco_manager.exports.get('real state') is None:
        return 0
    else:
        return model.eco_manager.exports['real state']['value']

def ex_catering (model):
    if model.eco_manager.exports.get('catering') is None:
        return 0
    else:
        return model.eco_manager.exports['catering']['value']

def ex_commerce (model):
    if model.eco_manager.exports.get('commerce') is None:
        return 0
    else:
        return model.eco_manager.exports['commerce']['value']

def ex_transport (model):
    if model.eco_manager.exports.get('transport') is None:
        return 0
    else:
        return model.eco_manager.exports['transport']['value']

def ex_hotel (model):
    if model.eco_manager.exports.get('hotel') is None:
        return 0
    else:
        return model.eco_manager.exports['hotel']['value']

def ex_public (model):
    if model.eco_manager.exports.get('public administration') is None:
        return 0
    else:
        return model.eco_manager.exports['public administration']['value']

def ex_education (model):
    if model.eco_manager.exports.get('education') is None:
        return 0
    else:
        return model.eco_manager.exports['education']['value']

def ex_health (model):
    if model.eco_manager.exports.get('health') is None:
        return 0
    else:
        return model.eco_manager.exports['health']['value']

def ex_social (model):
    if model.eco_manager.exports.get('social services') is None:
        return 0
    else:
        return model.eco_manager.exports['social services']['value']

def ex_professional (model):
    if model.eco_manager.exports.get('professional activities') is None:
        return 0
    else:
        return model.eco_manager.exports['professional activities']['value']

def ex_scientific (model):
    if model.eco_manager.exports.get('scientific activities') is None:
        return 0
    else:
        return model.eco_manager.exports['scientific activities']['value']

def ex_recreational (model):
    if model.eco_manager.exports.get('recreational activities') is None:
        return 0
    else:
        return model.eco_manager.exports['recreational activities']['value']

def ex_artistic (model):
    if model.eco_manager.exports.get('artistic activities') is None:
        return 0
    else:
        return model.eco_manager.exports['artistic activities']['value']

def ex_financial (model):
    if model.eco_manager.exports.get('financial activities') is None:
        return 0
    else:
        return model.eco_manager.exports['financial activities']['value']

def ex_other (model):
    if model.eco_manager.exports.get('other') is None:
        return 0
    else:
        return model.eco_manager.exports['other']['value']

def ex_information (model):
    if model.eco_manager.exports.get('Information and communications') is None:
        return 0
    else:
        return model.eco_manager.exports['Information and communications']['value']
#################################################################################
###### IMPORTS ##################################################################

def im_construction (model):
    if model.eco_manager.imports.get('construction') is None:
        return 0
    else:
        return model.eco_manager.imports['construction']['value']

def im_industrial (model):
    if model.eco_manager.imports.get('industrial production') is None:
        return 0
    else:
        return model.eco_manager.imports['industrial production']['value']

def im_food (model):
    if model.eco_manager.imports.get('food') is None:
        return 0
    else:
        return model.eco_manager.imports['food']['value']

def im_state (model):
    if model.eco_manager.imports.get('real state') is None:
        return 0
    else:
        return model.eco_manager.imports['real state']['value']

def im_education (model):
    if model.eco_manager.imports.get('education') is None:
        return 0
    else:
        return model.eco_manager.imports['education']['value']

def im_social (model):
    if model.eco_manager.imports.get('social services') is None:
        return 0
    else:
        return model.eco_manager.imports['social services']['value']

def im_professional (model):
    if model.eco_manager.imports.get('professional activities') is None:
        return 0
    else:
        return model.eco_manager.imports['professional activities']['value']

def im_scientific (model):
    if model.eco_manager.imports.get('scientific activities') is None:
        return 0
    else:
        return model.eco_manager.imports['scientific activities']['value']

def im_artistic (model):
    if model.eco_manager.imports.get('artistic activities') is None:
        return 0
    else:
        return model.eco_manager.imports['artistic activities']['value']

def im_other (model):
    if model.eco_manager.imports.get('other') is None:
        return 0
    else:
        return model.eco_manager.imports['other']['value']

'''
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
'''