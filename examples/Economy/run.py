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
from mesa.space import NetworkGrid, MultiGrid
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator

# OTHER UTILITIES
import ogr
import shapely
from shapely.geometry import *
import geopandas as gpd
import ipdb
import ast
from Model import city_model

##########################################

root_captioning = os.getcwd() + '/BCN_Resources/'

#### LOAD OSM FILE TO GDF ####
file = open(root_captioning + "bcn_spec_map.txt", "r")
contents = file.read()
spec_map = ast.literal_eval(contents)
file.close()

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


#### NOW WE CAN START OUR MODEL BASED ON NEW COORDINATES ####
model = city_model(10000, gdf, max_x, max_y, min_x, min_y)
# model = city_model(5541000, gdf, max_x, max_y, min_x, min_y) 

for i in range(200):
    print('///// STEP %s /////' % str(i+1))   
    model.step()
    print('\n')