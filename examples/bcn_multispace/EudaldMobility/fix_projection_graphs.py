import osmnx as ox
import os
import pickle


root_path = os.getcwd()
path_name = '/examples/bcn_multispace/EudaldMobility/pickle_objects/'
net_type = 'Pedestrian'
save_name = '/examples/bcn_multispace/EudaldMobility/pickle_objects/projected/'

with open(root_path + path_name + 'BCN_' + net_type + '.p', 'rb') as f:
    db = pickle.load(f)
print('read done')
G = ox.graph_from_gdfs(db[0],db[1])
G = ox.project_graph(G)
print('projected done')
nodes_proj, edges_proj = ox.graph_to_gdfs(G)
with open(root_path + save_name + 'BCN_' + net_type + '.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)
print('save done')