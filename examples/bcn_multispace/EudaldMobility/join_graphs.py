import osmnx as ox
import pickle

root_path = "./pickle_objects/"


# Load Car 
with open(root_path + 'BCN_Car.p', 'rb') as f:
    db = pickle.load(f)
nodes_proj = db[0]
edges_proj = db[1]

G_car = ox.graph_from_gdfs(nodes_proj, edges_proj)

ox.plot_graph(G_car)
