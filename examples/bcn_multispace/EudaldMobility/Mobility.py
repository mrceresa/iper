import os
import osmnx as ox 
import pandas as pd
import numpy as np
import pickle

import logging
_log = logging.getLogger(__name__)

class Map_to_Graph():
    def __init__(self, net_type):
        root_path = os.getcwd()
        #path_name = '/examples/bcn_multispace/EudaldMobility/pickle_objects/Small/'
        path_name = '/examples/bcn_multispace/EudaldMobility/pickle_objects/projected/'
        try:
            #with open(root_path + path_name + 'Part_BCN_' + net_type + '.p', 'rb') as f:
            with open(root_path + path_name + 'BCN_' + net_type + '.p', 'rb') as f:
                db = pickle.load(f)
            self.nodes_proj = db[0] 
            self.edges_proj = db[1]
            self.G = ox.graph_from_gdfs(self.nodes_proj, self.edges_proj)
        except:  
            print("Error loading the pickle file with root: " + root_path + path_name + 'BCN_' + net_type + '.p')

    def get_boundaries(self):
        # Retrieve the maximum x value (i.e. the most eastern)
        eastern_node = self.nodes_proj['x'].max()
        western_node = self.nodes_proj['x'].min()
        northern_node = self.nodes_proj['y'].max()
        southern_node = self.nodes_proj['y'].min()
        
        return {'n': northern_node, 'e': eastern_node, 's': southern_node, 'w': western_node}

    def get_lat_lng_from_point(self, point):
        node_from_point = ox.nearest_nodes(self.G, point[0], point[1], return_dist=False)
        lat = self.nodes_proj.loc[self.nodes_proj['osmid'] == node_from_point]['lat'].item()
        lon = self.nodes_proj.loc[self.nodes_proj['osmid'] == node_from_point]['lon'].item()
        return lat, lon

    def get_graph_area(self):
        return self.nodes_proj.unary_union.convex_hull.area
         
    def get_basic_stats(self, stats = None):
        area_graph = self.get_graph_area()
        basic_stats = ox.basic_stats(self.G, area= area_graph, clean_intersects=True, tolerance=15, circuity_dist='euclidean')
        if stats == None: 
            return pd.Series(basic_stats)
        else:
            desired_stats = {}
            for stat in stats:
                desired_stats[stat] = basic_stats[stat]
            return pd.Series(desired_stats)

    def get_advanced_stats(self): 
        pass

    def graph_consolidation(self):
        self.G = ox.consolidate_intersections(self.G, rebuild_graph=True, tolerance=8, dead_ends=True)

    def routing_by_distance(self, origin_coord, destination_coord):
        origin_node = ox.nearest_nodes(self.G, origin_coord[0], origin_coord[1], return_dist=False)
        destination_node = ox.nearest_nodes(self.G, destination_coord[0], destination_coord[1], return_dist=False)
        route = ox.shortest_path(self.G ,origin_node, destination_node, weight='length')
        return route 
    
    def routing_by_travel_time(self, origin_coord, destination_coord):
        origin_node = ox.nearest_nodes(self.G, origin_coord[0], origin_coord[1], return_dist=False)
        destination_node = ox.nearest_nodes(self.G, destination_coord[0], destination_coord[1], return_dist=False)
        route = ox.shortest_path(self.G ,origin_node, destination_node, weight='travel_time')
        #routes = ox.k_shortest_paths(self.G, origin_node, destination_node, k = 5, weight = 'travel_time')
        #route = self.find_least_transfer_route(routes)
        return route 
        #_log.info("Destination dist to node: %d"%dist)
    
    def find_least_transfer_route(self, routes):
        min_transfers = np.inf
        for route in routes:
            # Start at node 1 because node 0 is the starting position an may refer to any map. 
            start_node = route[1]
            num_transfers = 0
            for end_node in route[2:-1]:
                if start_node.split('-')[0] != end_node.split('-')[0]:
                    num_transfers += 1
                start_node = end_node
            if num_transfers < min_transfers:
                min_transfers = num_transfers
                least_transfer_route = route

        return least_transfer_route

    def compare_routes(self, route1, route2):
        route1_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route1, 'length')))
        route2_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route2, 'length')))
        route1_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route1, 'travel_time')))
        route2_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route2, 'travel_time')))
        print('Route 1 is', route1_length, 'meters and takes', route1_time, 'seconds.')
        print('Route 2 is', route2_length, 'meters and takes', route2_time, 'seconds.')

    def plot_graph(self, ax=None, figsize=(8, 8), bgcolor="#111111", node_color="w", node_size=15, node_alpha=None, node_edgecolor="none", node_zorder=1, edge_color="#999999", edge_linewidth=1, edge_alpha=None, show=True, close=False, save=False, filepath=None, dpi=300, bbox=None):
        fig, ax = ox.plot_graph(self.G, ax=ax, figsize=figsize, bgcolor=bgcolor, node_color=node_color, node_size=node_size, node_alpha=node_alpha, node_edgecolor=node_edgecolor, node_zorder=node_zorder, edge_color=edge_color, edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, show=show, close=close, save=save, filepath=filepath, dpi=dpi, bbox=bbox)
        return fig, ax
    
    def plot_graph_route(self, route, route_color, show = True, save=False, filepath=None):
        fig, ax = ox.plot_graph_route(self.G, route=route, route_color=route_color, route_linewidth=6, node_size=0, show=show, save=save, filepath=filepath)

    def plot_graph_routes(self, routes, route_colors ):
        fig, ax = ox.plot_graph_routes(self.G, routes=routes, route_colors=route_colors, route_linewidth=6, node_size=0)

    def plot_route_by_transport_type(self, route, save, filepath):
        node_pairs = zip(route[:-1], route[1:])
        # plot graph
        fig, ax = ox.plot_graph(self.G, show=False, close=False, node_size = 0)

        # then plot colored route segments on top of it
        for (u, v) in node_pairs:
            data = min(self.G.get_edge_data(u, v).values(), key=lambda d: d["travel_time"])
            #print(data['Type'])
            if "geometry" in data:
                x, y = data["geometry"].xy
            else:
                x = self.G.nodes[u]["x"], self.G.nodes[v]["x"]
                y = self.G.nodes[u]["y"], self.G.nodes[v]["y"]       
            if data['Type'] == 'Pedestrian':
                ax.plot(x, y, color='r', lw=5)
            elif data['Type'] == 'Car':
                ax.plot(x, y, color='y', lw=5)
            elif data['Type'] == 'Bike':
                ax.plot(x,y, color='b', lw=5)
        if save == True:
            fig.savefig(filepath)
    