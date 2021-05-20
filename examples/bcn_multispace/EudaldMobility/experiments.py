from networkx.classes.function import edges
import osmnx as ox
import pickle
import os
import random
import matplotlib.pyplot as plt 
from collections import Counter

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

# Exepriment 1 Check 
# Plot Histogram Number of Users using different transports 
def toy_example_hist(n):
    agents = []
    transport_types = ['Walk', 'Car', 'Bike']
    for agent in range(n):
        r = random.choice(transport_types)
        agents.append(r)
    plot_bars(agents)

# array --> array where each position is an agent and its value correspond to string of the transport used. 
def plot_bars(array):
    count = Counter(array)
    plt.bar(count.keys(), count.values(), color = '#bb1f2f')
    plt.xlabel('Transport Type')
    plt.ylabel('Number of Users')
    plt.title('Number Of Users For Different Transports')
    plt.show()

# Plot grpah mean time people spend in each tranpsort 


def main():
    toy_example_hist(100)

if __name__ == '__main__':
    main()

