import osmnx as ox
import networkx as nx
import pickle
import os
import pandas as pd

# Load Car
def load_graphs(root_path, graph_name):
    with open(root_path + graph_name, 'rb') as f:
        db = pickle.load(f)
    nodes_proj = db[0]
    edges_proj = db[1]
    return nodes_proj, edges_proj

def main():
    root_path =  os.getcwd() +'/examples/bcn_multispace/EudaldMobility/pickle_objects/'

    print('loading graphs...')
    p_nodes, p_edges = load_graphs(root_path,'BCN_Pedestrian.p')
    c_nodes, c_edges = load_graphs(root_path,'BCN_Car.p')
    G_p = ox.graph_from_gdfs(p_nodes, p_edges)
    G_c = ox.graph_from_gdfs(c_nodes, c_edges)

    print('merging graphs...')
    # Join Graphs
    G = nx.union(G_p,G_c,('P-','C-'))
    i = 0
    c_nodes_dij = c_nodes.loc[c_nodes['dij'] == True]

    print('linking graphs...')
    for node in c_nodes_dij.iterrows():
        # Get x,y from node
        u_id = 'C-' + str(node[0])
        x = node[1]['x']
        y = node[1]['y']

        # Find nearest node in Pedestrian and dij True
        nearest, dist = ox.nearest_nodes(G_p,x,y,True)
        v_id = 'P-' + str(nearest)

        # Add an edges to the joined graph with the ids of the separate graph nodes. 
        G.add_edge(u_id,v_id,travel_time = dist/1.1, length =  dist, Type ='Link_Edge')
        G.add_edge(v_id,u_id,travel_time = dist/1.1, length =  dist, Type ='Link_Edge')

        if i % 100 == 0:
            print(i)
        i +=1 
    print(len(G_p.edges))
    print(len(G_c.edges))
    print(len(G_c.nodes))
    print(len(G.edges))
    nodes_proj, edges_proj = ox.graph_to_gdfs(G)
    with open(root_path + 'BCN_PedCar.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)

if __name__ == "__main__":
    main()
