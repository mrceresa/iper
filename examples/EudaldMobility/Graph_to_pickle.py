import osmnx as ox
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pedestrian', type=str, default="False", help="Pedestrian Graph")
parser.add_argument('-c','--car', type=str, default="False", help="Car Graph")
parser.add_argument('-b','--bike', type=str, default="False", help="Bike Graph")
parser.add_argument('-r','--root_path', type=str, default="./pickle_objects", help="Save dir" )
args = parser.parse_args()  


if args.pedestrian == "True":
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'walk')

    # Add Labels 
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Pedestrian'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Pedestrian' 

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

if args.car == "True":
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'drive')

    # Add Labels 
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Car'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Car' 

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

if args.bike == "True": 
    G = ox.graph_from_place('Barcelona, Spain', network_type = 'bike')

    #Add Labels
    for edge_id in G.edges:
        G.edges[edge_id]['Type'] = 'Bike'
    for node_id in G.nodes:
        G.nodes[node_id]['Type'] = 'Bike' 

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
