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
    root_path =  os.getcwd() +'/examples/bcn_multispace/EudaldMobility/pickle_objects/Small/'

    print('loading graphs...')
    p_nodes, p_edges = load_graphs(root_path,'Part_BCN_Pedestrian.p')
    #c_nodes, c_edges = load_graphs(root_path,'Part_BCN_Car.p')
    b_nodes, b_edges = load_graphs(root_path,'Part_BCN_Bike.p')
    pc_nodes, pc_edges = load_graphs(root_path,'Part_BCN_PedCar.p')
    G_p = ox.graph_from_gdfs(p_nodes, p_edges)
    #G_c = ox.graph_from_gdfs(c_nodes, c_edges)
    G_b = ox.graph_from_gdfs(b_nodes, b_edges)
    G_pc = ox.graph_from_gdfs(pc_nodes, pc_edges)

    print('merging graphs...')
    # Join Graphs
    G = nx.union(G_pc,G_b,('','B-'))
    i = 0
    #c_nodes_dij = c_nodes.loc[c_nodes['dij'] == True]
    b_nodes_dij = b_nodes.loc[b_nodes['dij'] == True]

    print('linking graphs...')
    for node in b_nodes_dij.iterrows():
        # Get x,y from node
        u_id = 'B-' + str(node[0])
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
    print(len(G_b.edges))
    print(len(G_pc.edges))
    print(len(G_b.nodes))
    print(len(G.edges))
    nodes, edges = ox.graph_to_gdfs(G)
    with open(root_path + 'Part_BCN_PedCarBike.p', 'wb') as f:
        pickle.dump([nodes, edges], f)

if __name__ == "__main__":
    main()
