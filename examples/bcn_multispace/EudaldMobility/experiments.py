from networkx.classes.function import edges
import osmnx as ox
import pickle
import os
import random
import matplotlib.pyplot as plt 
from collections import Counter
from itertools import islice
from datetime import datetime, timedelta
import numpy as np

#Fix length edges joined graph
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

# Exepriment 1 - a
# Plot Histogram Number of Users using different transports 
def toy_example_hist(n):
    agents = []
    transport_types = ['Walk', 'Car', 'Bike']
    for agent in range(n):
        r = random.choice(transport_types)
        agents.append(r)
    plot_bars(agents)

# array --> array where each position is an agent and its value correspond to string of the transport used. 
def plot_bars(x_value, y_value, x_lagel_name, y_label_name, title):
    plt.figure()
    plt.bar(x_value, y_value, color = '#bb1f2f')
    plt.xlabel(x_lagel_name)
    plt.ylabel(y_label_name)
    plt.title(title)
    plt.show()

def plot_lines(x_value, y_value, x_lagel_name, y_label_name, title):
    plt.figure()
    plt.plot(x_value[0], y_value[0], label = "Walk")
    plt.plot(x_value[1], y_value[1], label = "Car")
    plt.plot(x_value[2], y_value[2], label = "Bike")
    plt.xlabel(x_lagel_name)
    plt.ylabel(y_label_name)
    plt.title(title)
    plt.legend()
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

def experiment_transport_line_plot(agents, steps):
    walk, car, bike = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    start_time = datetime(year=2021, month=1, day=1, hour= 0, minute=0, second=0) 
    delta = timedelta(seconds=60)
    end_time =  start_time + (delta * steps)
    index_time = np.range(start_time / np.timedelta64(1, 'm'), end_time / np.timedelta64(1, 'm'), delta / np.timedelta64(1, 'm'))
    for agent in agents: 
        current_time = start_time
        for key, traj in agent.record_trajectories.items(): 
            for i in np.range(steps):
                row = traj.get_row_at(start_time)
                if row['node'].split('-')[0] == 'P':
                    walk[i] += 1
                elif row['node'].split('-')[0] == 'C':
                    car[i] += 1
                elif row['node'].split('-')[0] == 'B':
                    bike[i] += 1
                
                index_time[i] = start_time / np.timedelta64(1, 'm')    
                current_time += delta

    x_value = [index_time, index_time, index_time]
    y_value = [walk, car, bike]
    plot_lines(x_value, y_value, 'Transport Type', 'Number of Users', 'Number Of Users For Different Transports')

#Experiment 1 - b
# Plot grpah mean time people spend in each tranpsort 
def experiment_mean_time_transport(agents):
    num_agent = 0
    ped_time, car_time, bike_time = timedelta(),timedelta(),timedelta()
    for agent in agents: #For each agent 
        num_agent += 1
        for index, traj in agent.record_trajectories.items(): #For each route 
            # Get data initial step
            start_item = traj.df.index[0]
            start_item_type = traj.df['node'][0].split("-")[0]
            for key, item in traj.df[1:].iterrows():
                if key >= datetime(year=2021, month=1, day=1, hour= 0, minute=10, second=0):
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

    ped_total_time = (ped_time / num_agent) / np.timedelta64(1, 's')
    car_total_time = (car_time / num_agent) / np.timedelta64(1, 's')
    bike_total_time = (bike_time / num_agent) / np.timedelta64(1, 's')
    
    plot_bars(['Walk', 'Car', 'Bike'], [ped_total_time, car_total_time, bike_total_time], 
                'Transport Type', 'Seconds', 'Mean time in trasports')

def main():
    list = ['C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'P', 'C', 'P']
    plot_bars(list)
if __name__ == '__main__':
    main()

