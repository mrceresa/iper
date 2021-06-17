import subprocess
import pkg_resources
from osgeo import ogr
import sys

REQUIRED = {'mesa', 'mesa-geo','osmnx', 'geopandas', 'ipdb'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = REQUIRED - installed
if missing:
	for X in missing:
		subprocess.call(['pip', 'install', X])

# PYTHON libraries
import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

import networkx as nx
import osmnx as ox

# MESA CLASSES
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.space import NetworkGrid, MultiGrid
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator


# OTHER UTILITIES
import shapely
from shapely.geometry import *
import geopandas as gpd
import ipdb
import ast
from Model import city_model
import os
import time

##########################################

root_captioning = os.getcwd() + '/BCN_Resources/'
res_path = os.getcwd() + '/ModelResults/'

#### LOAD OSM FILE TO GDF ####
file = open(root_captioning + "bcn_spec_map.txt", "r")
contents = file.read()
spec_map = ast.literal_eval(contents)
file.close()

file = open(root_captioning + "resouce_map.txt", "r")
contents = file.read()
resource_map = ast.literal_eval(contents)
file.close()


# LOAD OSM FILE TO GDF
driver = ogr.GetDriverByName('OSM')
data = driver.Open(root_captioning + 'bcn.osm')
layer = data.GetLayer('points')
features=[x for x in layer]

data_list=[]
for feature in features:

    data = feature.ExportToJson(as_object=True)

    coords = data['geometry']['coordinates']
    shapely_geo = Point(coords[0], coords[1])
    #print(shapely_geo)
    name = data['properties']['name']
    other_tags = data['properties']['other_tags']

    street = None
    num = None
    if other_tags and 'addr:street' in other_tags : 
        aux = [x for x in other_tags.split(',') if 'addr:street' in x][0]
        street = aux[aux.rfind('>')+2:aux.rfind('"')]
    if other_tags and 'addr:housenumber' in other_tags :
        aux = [x for x in other_tags.split(',') if 'addr:housenumber' in x][0]
        num = aux[aux.rfind('>')+2:aux.rfind('"')]

    # If we cannot recognize by name, it is not usefull
    if name != None:
        if other_tags and 'shop' in other_tags:
            tag = 'shop'
            aux = [x for x in other_tags.split(',') if 'shop' in x][0]
            speciality = aux[aux.rfind('>')+2:aux.rfind('"')]
            data_list.append([name,street,num,tag,speciality,spec_map[speciality],shapely_geo])
        elif other_tags and 'tourism' in other_tags:
            tag = 'tourism'
            aux = [x for x in other_tags.split(',') if 'tourism' in x][0]
            speciality = aux[aux.rfind('>')+2:aux.rfind('"')]
            data_list.append([name,street,num,tag,speciality,spec_map[speciality],shapely_geo])
        elif other_tags and 'healthcare' in other_tags:
            tag = 'healthcare'
            aux = [x for x in other_tags.split(',') if 'healthcare' in x][0]
            speciality = aux[aux.rfind('>')+2:aux.rfind('"')]
            data_list.append([name,street,num,tag,speciality,spec_map[speciality],shapely_geo])
        elif other_tags and 'amenity' in other_tags:
            tag = 'amenity'
            aux = [x for x in other_tags.split(',') if 'amenity' in x][0]
            speciality = aux[aux.rfind('>')+2:aux.rfind('"')]
            data_list.append([name,street,num,tag,speciality,spec_map[speciality],shapely_geo])

gdf = gpd.GeoDataFrame(data_list,columns=['Name','Street','AddNum','Tag','Speciallity','SpeciallityAdapt','geometry'],crs={'init': 'epsg:3857'})
scient = gdf[gdf.SpeciallityAdapt.isin(['scientific activities'])].head(9)
rstate = gdf[gdf.SpeciallityAdapt.isin(['real state'])].iloc[[1]]
gdf = gdf.head(1000)
gdf = gdf.append([scient,rstate],ignore_index=True)

# ipdb.set_trace()
# gdf = gdf.reindex(index=gdf.index[::-1])
# gdf = gdf.reset_index(drop=True)

#### GET MAP LIMITS FOR AGENTS ####
max_x = -np.inf
max_y = -np.inf
min_x = np.inf
min_y = np.inf

for p in gdf['geometry']:
    if p.x > max_x:
        max_x = p.x
    if p.x < min_x:
        min_x = p.x
    if p.y > max_y:
        max_y = p.y
    if p.y < min_y:
        min_y = p.y

# NOW WE CAN START OUR MODEL BASED ON NEW COORDINATES
'''
model = city_model(25000, gdf, max_x, max_y, min_x, min_y, resource_map) #bcn_pop = 5541000

for j in range(1): # num of test
    for i in range(2880): # iters per test
        print('///// STEP %s ///// \t' % str(i+1))  
        t = time.time()
        model.step()
        print(time.time() - t)
        print('\n')
'''
import multiprocessing

def mod_run(_m,id,return_dict):
    model = _m
    for i in range(672): # iters per test
        if (i % 10 == 0):
            print('Model %s at step %s\n' % (str(id),str(i)))
        model.step()
    '''
    agent_df = model.agent_eco_datacollector.get_model_vars_dataframe()
    btob_df = model.btob_eco_datacollector.get_model_vars_dataframe()
    import_df = model.import_eco_datacollector.get_model_vars_dataframe()
    export_df = model.export_eco_datacollector.get_model_vars_dataframe()
    return_dict[id] = {'agents':agent_df, 'businesses':btob_df, 'import':import_df, 'export':export_df}
    '''
    stats_df = model.stats_datacollector.get_model_vars_dataframe()
    ammount_df = model.amm_datacollector.get_model_vars_dataframe()
    price_df = model.price_datacollector.get_model_vars_dataframe()
    return_dict[id] = {'stats':stats_df, 'ammount':ammount_df, 'price':price_df}
    
#PREPARE MODELS
procs = 10   # Number of processes to create
models = []
for k in range(procs):
    _m = city_model(5000, gdf, max_x, max_y, min_x, min_y, resource_map)
    models.append(_m)

#LAUNCH ALL MODELS

manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for i in range(0, procs):
    process = multiprocessing.Process(target=mod_run, args=(models[i],i,return_dict))
    jobs.append(process)

for j in jobs:
    j.start()

for j in jobs:
    j.join()

print('Retrieving results...\n')
for i in range(0, procs):
    '''
    df = return_dict[i]['agents']
    df.to_csv(res_path + 'Agents_Model%s.csv' % str(i), encoding='utf-8')
    df = return_dict[i]['businesses']
    df.to_csv(res_path + 'Business_Model%s.csv' % str(i), encoding='utf-8')
    df = return_dict[i]['import']
    df.to_csv(res_path + 'Import_Model%s.csv' % str(i), encoding='utf-8')
    df = return_dict[i]['export']
    df.to_csv(res_path + 'Export_Model%s.csv' % str(i), encoding='utf-8')
    '''
    df = return_dict[i]['stats']
    df.to_csv(res_path + 'Stats_Model%s.csv' % str(i), encoding='utf-8')
    df = return_dict[i]['ammount']
    df.to_csv(res_path + 'Ammount_Model%s.csv' % str(i), encoding='utf-8')
    df = return_dict[i]['price']
    df.to_csv(res_path + 'Price_Model%s.csv' % str(i), encoding='utf-8')
    
#ipdb.set_trace()
print('Task finished!')