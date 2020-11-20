# Print stats in a good form 

import osmnx as ox 
import pandas as pd

class Map_to_Graph():
    def __init__(self, place, net_type):
        
        path_name = 'BCNgraphs/'+net_type+'.graphml'
        
        try:
            self.G = ox.load_graphml(path_name)
        except:
            self.G = ox.graph_from_place(place, network_type = net_type, buffer_dist = 500)
            ox.save_graphml(self.G, path_name)  

        self.G_proj = ox.project_graph(self.G)
        self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.G_proj, nodes=True, edges=True)
        #self.nodes_proj = self.nodes_proj.reset_index() # Sets the name index on the columns key names
        #self.edges_proj = self.edges_proj.reset_index() # Sets the name index on the columns key names

    def get_boundaries(self):
        # Retrieve the maximum x value (i.e. the most eastern)
        eastern_node = self.nodes_proj['lon'].max()
        western_node = self.nodes_proj['lon'].min()
        northern_node = self.nodes_proj['lat'].max()
        southern_node = self.nodes_proj['lat'].min()
        
        return {'n': northern_node, 'e': eastern_node, 's': southern_node, 'w': western_node}

    def get_lat_lng_from_point(self, point):
        node_from_point = ox.get_nearest_node(self.G, point)
        lat = self.nodes_proj.loc[self.nodes_proj['osmid'] == node_from_point]['lat'].item()
        lon = self.nodes_proj.loc[self.nodes_proj['osmid'] == node_from_point]['lon'].item()
        return lat, lon

    def get_graph_area(self):
        return self.nodes_proj.unary_union.convex_hull.area
         
    def get_basic_stats(self, stats = None):
        area_graph = self.get_graph_area()
        basic_stats = ox.basic_stats(self.G_proj, area= area_graph, clean_intersects=True, tolerance=15, circuity_dist='euclidean')
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
        # Check if it returns a graph projected 
        self.G = ox.consolidate_intersections(self.G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)

    def routing_by_distance(self, origin_coord, destination_coord):
        origin_node = ox.get_nearest_node(self.G, origin_coord)
        destination_node = ox.get_nearest_node(self.G, destination_coord)
        route = ox.shortest_path(self.G ,origin_node, destination_node, weight='length')
        return route 
    
    def routing_by_travel_time(self, origin_coord, destination_coord):
        origin_node = ox.get_nearest_node(self.G, origin_coord)
        destination_node = ox.get_nearest_node(self.G, destination_coord)
        hwy_speeds = {'residential': 35,
                    'living_street': 20,
                    'secondary': 50,
                    'tertiary': 60}
        self.G = ox.add_edge_speeds(self.G, hwy_speeds)
        self.G = ox.add_edge_travel_times(self.G)
        route = ox.shortest_path(self.G ,origin_node, destination_node, weight='travel_time')
        return route 

    def compare_routes(self, route1, route2):
        route1_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route1, 'length')))
        route2_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route2, 'length')))
        route1_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route1, 'travel_time')))
        route2_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G, route2, 'travel_time')))
        print('Route 1 is', route1_length, 'meters and takes', route1_time, 'seconds.')
        print('Route 2 is', route2_length, 'meters and takes', route2_time, 'seconds.')

    def plot_graph(self, ax=None, figsize=(8, 8), bgcolor="#111111", node_color="w", node_size=15, node_alpha=None, node_edgecolor="none", node_zorder=1, edge_color="#999999", edge_linewidth=1, edge_alpha=None, show=True, close=False, save=False, filepath=None, dpi=300, bbox=None):
        fig, ax = ox.plot_graph(self.G, ax=ax, figsize=figsize, bgcolor=bgcolor, node_color=node_color, node_size=node_size, node_alpha=node_alpha, node_edgecolor=node_edgecolor, node_zorder=node_zorder, edge_color=edge_color, edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, show=show, close=close, save=False, filepath=filepath, dpi=dpi, bbox=bbox)
        return fig, ax
    
    def plot_graph_routes(self, routes, route_colors ):
        fig, ax = ox.plot_graph_routes(self.G, routes=routes, route_colors=route_colors, route_linewidth=6, node_size=0)

class Map_to_gdf():
    def __init__(self, place, tags):
        self.Poi = ox.geometries_from_place(place, tags = tags)
        self.Poi_project = ox.project_gdf(self.Poi)
        
        self.Poi_project.to_csv('data.csv')
        

if __name__ == "__main__":
    tags = {
    'highway':['bus_stop','platform'],
    'public_transport':['stop_position', 'platform', 'station']
    }

    #Graph = Map_to_Graph('Barcelona, Spain', 'drive') 
    Gdf = Map_to_gdf('Barcelona, Spain', tags)
    #Graph.plot_graph()
    

    