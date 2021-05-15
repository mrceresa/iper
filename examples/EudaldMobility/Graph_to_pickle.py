import osmnx as ox
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pedestrian', type=str, default="False", help="Pedestrian Graph")
parser.add_argument('-c','--car', type=str, default="False", help="Car Graph")
parser.add_argument('-b','--bike', type=str, default="False", help="Bike Graph")
parser.add_argument('-r','--root_path', type=str, default="./pickle_objects/", help="Save dir" )
args = parser.parse_args()  

if args.pedestrian == "True":
    print("Downloading Pedestrian Data...")
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'walk')
    
    print("Labeling Pedestrian Data...")
    # Add Labels and dij flag
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Pedestrian'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Pedestrian' 
        G.nodes[node_id]['dij'] = True

    #Add speeds
    hwy_speeds_walk = {'residential': 4,
        'living_street': 4,
        'secondary': 4,
        'tertiary': 4,
        'service': 4,
        'pedestrian': 4}
    G = ox.add_edge_speeds(G, hwy_speeds_walk)
    G = ox.add_edge_travel_times(G)

    #Convert to 
    nodes_proj, edges_proj = ox.graph_to_gdfs(G, nodes=True, edges=True)

    #Picke
    with open(args.root_path + 'BCN_Pedestrian.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)
    print("Pickle Pedestrian Done")

if args.car == "True":
    print("Downloading Car Data...")
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'drive')
    print("Labeling Car Data...")
    # Add Labels 
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Car'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Car' 
        G.nodes[node_id]['dij'] = True

    # Add speeds
    hwy_speeds_car = {'residential': 35,
            'living_street': 20,
            'primary': 120,
            'secondary': 80,
            'tertiary': 60,
            'service': 20}
    G = ox.add_edge_speeds(G, hwy_speeds_car)
    G = ox.add_edge_travel_times(G)

    #Convert to 
    nodes_proj, edges_proj = ox.graph_to_gdfs(G, nodes=True, edges=True)

    #Picke
    with open(args.root_path + 'BCN_Car.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)

    print("Pickle Car Done")

if args.bike == "True": 
    print("Downloading Bike Data...")
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'bike')
    print("Labeling Bike Data...")
    #Add Labels
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Bike'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Bike' 
        G.nodes[node_id]['dij'] = True

    #Add speeds
    hwy_speeds_bike = {'residential': 20,
            'living_street': 15,
            'secondary': 25,
            'tertiary': 25,
            'service': 15}
    G = ox.add_edge_speeds(G, hwy_speeds_bike)
    G = ox.add_edge_travel_times(G)

    #Convert to 
    nodes_proj, edges_proj = ox.graph_to_gdfs(G, nodes=True, edges=True)

    #Picke
    with open(args.root_path + 'BCN_Bike.p', 'wb') as f:
        pickle.dump([nodes_proj, edges_proj], f)
    print("Pickle Bike Done")