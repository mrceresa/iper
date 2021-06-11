from networkx.classes.function import edges
import osmnx as ox
import pickle
import os
import random
import matplotlib.pyplot as plt 
import hvplot
from collections import Counter
from itertools import islice
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import movingpandas as mpd
from statistics import mean

#Fix length edges joined graph (Suport function not needed anymore)
def fix_length():
    root_path = os.getcwd()
    path_name = '/EudaldMobility/pickle_objects/'
    with open(root_path + path_name + 'BCN_' + 'PedCar' + '.p', 'rb') as f:
        db = pickle.load(f)
    nodes_proj = db[0]
    edges_proj = db[1]
    #G = ox.graph_from_gdfs(nodes_proj,edges_proj)

    for item in edges_proj.iterrows():
        id = item[0]
        values = item[1]
        if values['Type'] == 'Link_Edge':
            leng = values.geometry.length
            #print(edges_proj.loc[edges_proj.index == id])
            edges_proj.loc[edges_proj.index == id, 'length'] = leng
            #print(edges_proj.loc[edges_proj.index == id])
    #Pickle
    with open(root_path + path_name + 'BCN_PedCar_lengths.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)
    print("Pickle Pedestrian Done")

# Not used 
def plot_bars(x_value, y_value, x_lagel_name, y_label_name, title):
    plt.figure()
    plt.bar(x_value, y_value, color = '#bb1f2f')
    plt.xlabel(x_lagel_name)
    plt.ylabel(y_label_name)
    plt.title(title)
    plt.show()

def experiment_trasnport_histogram(agents):
    trasnport_used = []
    for agent in agents: #For each agent 
        for key, traj in agent.record_trajectories.items(): #For each route 
            aux_list = []
            for node in traj.df['node'][1:-1]:  # For each node of each route
                aux_list.append(node.split('-')[0])
            trasnport_used.append(list(set(aux_list)))
    
    trasnport_used = [item for sublist in trasnport_used for item in sublist]
    count = Counter(trasnport_used)
    plot_bars(count.keys(),count.values(), 'Transport Type', 'Number of Users', 'Number Of Users For Different Transports')

# Experiment 1.ii 
def plot_lines(x_value, y_value, x_lagel_name, y_label_name, title):
    plt.figure()
    plt.plot(x_value[0], y_value[0], label = "Walk")
    plt.plot(x_value[1], y_value[1], label = "Car")
    plt.plot(x_value[2], y_value[2], label = "Bike")
    #plt.yticks(y_value[0])
    #plt.xticks(x_value[0])
    plt.xlabel(x_lagel_name)
    plt.ylabel(y_label_name)
    plt.title(title)
    plt.legend()
    plt.show()

def experiment_transport_line_plot(agents, steps):
    walk, car, bike = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    start_time = datetime(year=2021, month=1, day=1, hour= 0, minute=0, second=0) 
    delta = timedelta(seconds=60)
    end_time =  start_time + (delta * steps)
    index_time = np.arange((start_time - start_time) / timedelta(seconds=60), (end_time - start_time) / timedelta(seconds=60), delta/timedelta(seconds=60)).astype(int)
    for agent in agents: 
        i = 0
        current_time = start_time
        old_row_name = ''
        for key, traj in agent.record_trajectories.items(): 
            while i < steps:
                row = traj.get_row_at(current_time)
                if old_row_name == row.name:
                    break
                else:
                    if row['node'].split('-')[0] == 'P':
                        walk[i] += 1
                    elif row['node'].split('-')[0] == 'C':
                        car[i] += 1
                    elif row['node'].split('-')[0] == 'B':
                        bike[i] += 1    
                    old_row_name = row.name    
                    current_time += delta
                    i += 1

    x_value = [index_time, index_time, index_time]
    y_value = [walk, car, bike]
    plot_lines(x_value, y_value, 'Time (min)', 'Number of Users', 'Number Of Users For Different Transports')

#Experiment 1.iii
def plot_box(data,x_lagel_name, y_label_name, title):
    plt.figure()
    plt.boxplot(data)
    plt.xticks([1, 2, 3], ['Walking', 'Driving', 'Riding'])
    plt.xlabel(x_lagel_name)
    plt.ylabel(y_label_name)
    plt.title(title)
    plt.show()

def experiment_mean_time_transport(agents, min = 10):
    driving, walking, riding = [], [], []  
    for agent in agents: #For each agent 
        ped_time, car_time, bike_time = timedelta(),timedelta(),timedelta()
        for index, traj in agent.record_trajectories.items(): #For each route 
            # Get data initial step
            start_item = traj.df.index[0]
            start_item_type = traj.df['node'][0].split("-")[0]
            for key, item in traj.df[1:].iterrows():
                if key >= datetime(year=2021, month=1, day=1, hour= 0, minute=min, second=0):
                    break
                else:
                    item_type = item['node'].split("-")[0]
                    if start_item_type != item_type:
                        ped_time += key - start_item
                        start_item = key
                        start_item_type = item_type
                    else:
                        time_dif = key - start_item
                        if item_type == 'P':
                            ped_time += time_dif
                        elif item_type == 'C':
                            car_time += time_dif
                        elif item_type == 'B':
                            bike_time += time_dif
                        start_item = key
                        start_item_type = item_type
        walking.append(ped_time.seconds/60)
        driving.append(car_time.seconds/60)
        riding.append(bike_time.seconds/60)
        
    plot_box([walking,driving,riding], 'Mobility Type', 'Time (min)', 'Box Plot')

#Experiment 2.i i 2.ii
def plot_all_routes_agent(agents):
    for agent in agents:
        route_list = []
        for index, traj in agent.record_trajectories.items(): #For each route 
            route_list.append(traj)
        traj_collect = mpd.TrajectoryCollection(route_list)
        plot = traj_collect.hvplot(geo = True, tiles='OSM', line_width=5, width=700, height=400) + traj_collect.hvplot(c='type', line_width=7.0, width=700, height=400, colorbar=True)
        hvplot.show(plot)
        break

# Experiment 2.iii

def plot_box_route_agent(agents):
    for agent in agents:
        data = []
        for index, traj in agent.record_trajectories.items(): #For each route 
            duration = traj.get_duration().seconds
            length = traj.get_length()
            traj.add_speed(overwrite=True)
            print(traj.df)
            speed = mean(traj.df['speed'].values)
            data.append([duration, length, speed])
        columns=['Times', 'Length', 'Speed']
        df = pd.DataFrame(data, columns = columns)
        plot = df.hvplot.box(y='Times', ylabel='seconds', xlabel = 'Duration') + df.hvplot.box(y='Length', xlabel='Length', ylabel='meters') + df.hvplot.box(y='Speed', xlabel='Speed', ylabel='meters/seconds') 
        hvplot.show(plot)
        break
def main():
    #list = ['C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'P', 'C', 'P']
    #plot_bars(list)
    read = pd.read_csv(os.path.join(os.getcwd(), 'examples/bcn_multispace/EudaldMobility/trajectories.csv'))
    #r = read['geometry']
    #r = read.loc[read['id'] == '1-1']
    print(read)

if __name__ == '__main__':
    main()

