from mesa import Agent
from mesa_geo.geoagent import GeoAgent
import pygtfs
import os
import osmnx as ox 
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import movingpandas as mpd
from datetime import datetime, timedelta
import numpy as np
from pyproj import CRS
import copy
import random


import timeit

import logging
_log = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

# Functions that may be needed
def load_merged_GTFS(self):
    self.merged_GTFS = pd.read_csv((ROOT_DIR + '/GTFS/merged_GTFS.csv'), dtype = str)
def save_merged_GTFS(self):
    self.merged_GTFS.to_csv(ROOT_DIR + '/GTFS/merged_GTFS.csv')
class pygtfs_Schedule():
    def __init__(self):
        sched = pygtfs.Schedule(":memory:")
        pygtfs.append_feed(sched, "GTFS/bus_metro")
        print(sched.agencies)

    def now(date, time):
        self.date = date
        self.time = time
# -----------------------------------------   

# Don't really know if it needs to be GEOAGENT
class RouteAgent(Agent):                       
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.route_id = unique_id
        transport_type = self.model.routes.loc[self.model.routes['route_id'] == self.route_id].route_type.item()
        self.name_transport_type = self.get_name_transport(transport_type)
        route_trips = self.model.trips.loc[self.model.trips['route_id'] == self.route_id]
        self.route_services = route_trips.loc[:,'service_id'].unique()
        route_service_group = route_trips.groupby(route_trips.service_id)

        self.route_dict = {}
        self.createRouteDict(route_service_group)

        self.service_by_days = self.getRunningDays()

        self.trips_currently_running = []

    def get_name_transport(self, transport_type):
        dictionary = {'0':'Tram', '1':'Subway', '3':'Bus', '7':'Funicular'}
        return dictionary[transport_type]

    def createRouteDict(self, groups):
        for s in self.route_services:
            group_i = groups.get_group(s)
            traj_dict = {}
            traj_dict = self.createTrajectories(group_i, traj_dict)
            self.route_dict[s] = traj_dict

    def createTrajectories(self, group, traj_dict):
        for t in group.trip_id:
            trajectory = self.model.st.loc[self.model.st['trip_id'] == t]
            trajectory = pd.merge(trajectory, self.model.stops[['stop_id','geometry']], on = 'stop_id')
            gdf_trajectory = gpd.GeoDataFrame(trajectory, crs=CRS(32631))
            traj = mpd.Trajectory(gdf_trajectory, t)
            traj_dict[t] = traj
        return traj_dict    

    def getRunningDays(self):
        schedule = pd.DataFrame()
        for s in self.route_services:
            schedule_tmp = self.model.calendar.loc[self.model.calendar['service_id'] == s]
            schedule = pd.concat([schedule, schedule_tmp])
        schedule = schedule.sort_values('date')
        return schedule

    def check_init_traj(self):
        for key_traj, value_traj in self.traj_today_service.items():
            if value_traj.get_start_time() == timedate:
                createTransportAgent(trajectory = value_traj, traj_id = key_traj, transport_type = self.name_transport_type)
                self.trips_currently_running.append(key_traj)

    def check_finish_traj(self):
        pass
        #for key_traj in self.trips_currently_running:
        #    trajecotry = 

    def createTransportAgent(self, trajectory, traj_id, transport_type): 
        ######## Vull tmb passar-li la trajectoria.#######
        AC = AgentCreator(transport_type , {"model", self.model})
        _a = AC.create_agent( 
            Point( 
            trajectory.get_start_location().x,
            trajectory.get_start_location().y
            ), traj_id)

        self.model.transport_grid.add_agents(_a)
        self.model.schedule.add(_a)

    def removeTransportAgents(self):
        # Two options:
        # 1: Create a get_end_times and perform like the creation of instances but to eraise them, 
        # 2: Check when ever they arrived at its end location and eraise them then. 
        pass

    def step(self):
        # Check every midnight the service running this day and load all the trajectories from that service
        #if self.model.time = datetime.min.time():
        if True:
            today_service = self.service_by_days.loc[self.service_by_days['date'] == timedate.date()].service_id.item()

            self.traj_today_service = copy.deepcopy(self.route_dict[today_service])
            for key, traj in self.traj_today_service.items():
                traj.df.loc[:,'time'] = traj.df.loc[:,'time'] + datetime.combine(timedate.date(), datetime.min.time())
                traj.df.set_index('time', inplace=True)


        check_init_traj()

class TransportAgent(GeoAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

    def get_stops(self, trajectory):
        return trajectory.df.geometry.unique()

class Human(GeoAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.has_goal = False
        self.life_goals = 0  
        self.record_trajectories = {}

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

    def define_goal(self):
        random_node = random.choice(list(self.model.walkMap.G_proj.nodes))
        node = self.model.walkMap.G_proj.nodes[random_node]
        return (node['x'], node['y'])
    
    def init_goal_traj(self):
        route = self.model.walkMap.routing_by_travel_time(self.get_pos(), self.goal)
        self.model.walkMap.plot_graph_route(route, 'y', show = False, save = True, filepath = 'plots/route_agent' + str(self.unique_id) + '_num' + str(self.life_goals) + '.png')
        
        df = pd.DataFrame()
        nodes, lats, lngs, times = [], [], [], []
        total_time = 0
        first_node = True

        for u, v in zip(route[:-1], route[1:]):
            travel_time = round(self.model.walkMap.G_proj.edges[(u, v, 0)]['travel_time'])
            
            if first_node == True:
                nodes.append(u)
                lats.append(self.model.walkMap.G_proj.nodes[u]['y'])
                lngs.append(self.model.walkMap.G_proj.nodes[u]['x'])
                times.append(0)
                first_node = False
            
            nodes.append(v)
            lats.append(self.model.walkMap.G_proj.nodes[v]['y'])
            lngs.append(self.model.walkMap.G_proj.nodes[v]['x'])
            times.append(total_time + travel_time)
            total_time += travel_time
            

        df['node'] = nodes
        df['time'] = pd.to_timedelta(times, unit = 'S')
        dfg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lngs, lats))
        gdf_trajectory = gpd.GeoDataFrame(dfg, crs=CRS(32631))
        traj = mpd.Trajectory(gdf_trajectory, self.life_goals)
        traj.df.loc[:,'time'] = traj.df.loc[:,'time'] + self.model.DateTime
        traj.df.set_index('time', inplace=True)

        self.record_trajectories[traj.df.index[0]] = traj
        return traj
       
    def update_goal_traj(self):
        try: 
            new_traj = self.goal_traj.df[self.goal_traj.df.index > self.model.DateTime]
            new_traj = new_traj.reset_index()
            new_traj.loc[new_traj.shape[0]] = (self.model.DateTime, '', self.goal_traj.get_position_at(self.model.DateTime))
            new_traj.set_index('time', inplace=True)
            new_traj = mpd.Trajectory(new_traj, 1)
            return new_traj
        except:
            self.has_goal = False
            return None

    def step(self):
        if self.has_goal == False:
            self.goal = self.define_goal()
            self.goal_traj = self.init_goal_traj()
            self.life_goals += 1
            self.has_goal = True
        else: 
            currentPos = self.get_pos()
            goal_df = self.goal_traj.df
            print(len(goal_df))
            newPos = self.goal_traj.get_position_at(self.model.DateTime)
            self.place_at(newPos)

            self.goal_traj = self.update_goal_traj()

            #neighbors = self.model.grid.get_neighbors(self)

    def __repr__(self):
        return "Agent " + str(self.unique_id)

class Tram(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        max_capacity = 100

    def step(self):
        pass

class Subway(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.max_capacity = 500
        self.passangers = []
        self.trajectory = self.collection.get_trajectory(unique_id)
        self.stops = self.get_stops(self.trajectory)
        self.currentStation = 0

    def step(self):
        currentPos = self.get_pos()
        newPos = self.trajectory.get_pos(model.time)  #time from the main
        self.place_at(newPos)

        #if self.stops[currentStation].x == newPos.x and self.stops[currentStation].y == newPos.y:
            #People can enter

        #Check whether the station has changed
        #if self.stops[currentStation].x == currentPos.x and self.stops[currentStation].y == currentPos.y and self.stops[currentStation].x != newPos.x and self.stops[currentStation].y != newPos.y:
        #    self.currentStation += 1
        #else:
            # We are still in the station

class Bus(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)   
    def step(self):
        pass

class Car(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def step(self):
        pass

class Bike(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
    def step(self):
        pass

class Map_to_Graph():
    def __init__(self, place, net_type):
        self.net_type = net_type
        root_path = os.getcwd()
        path_name = '/BCNgraphs/'+net_type+'.graphml'
        cheat = True
        if cheat == True:
            try: 
                self.G = ox.load_graphml(root_path + '/BCNgraphs/'+'cheat'+'.graphml')
            except:
                self.G = ox.graph_from_address('Plaça Catalunya, Barcelona, Spain', dist = 1000, network_type = 'walk')
                self.G_proj = ox.project_graph(self.G)
                self.graph_consolidation()
                ox.save_graphml(self.G_proj, root_path + '/BCNgraphs/'+'cheat'+'.osm')  
                
            
            self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.G_proj, nodes=True, edges=True)
        else:
            try:  
                self.G = ox.load_graphml(root_path + path_name)
            except:
                self.G = ox.graph_from_place(place, network_type = net_type)
                ox.save_graphml(self.G, root_path + path_name)  
            start = timeit.default_timer()
            self.G_proj = ox.project_graph(self.G)
            self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.G_proj, nodes=True, edges=True)
            #self.nodes_proj = self.nodes_proj.reset_index() # Sets the name index on the columns key names
            #self.edges_proj = self.edges_proj.reset_index() # Sets the name index on the columns key names
            stop = timeit.default_timer()
            print('Time: ', stop - start)

    def get_boundaries(self):
        # Retrieve the maximum x value (i.e. the most eastern)
        eastern_node = self.nodes_proj['lon'].max()
        western_node = self.nodes_proj['lon'].min()
        northern_node = self.nodes_proj['lat'].max()
        southern_node = self.nodes_proj['lat'].min()
        
        return {'n': northern_node, 'e': eastern_node, 's': southern_node, 'w': western_node}

    def get_lat_lng_from_point(self, point):
        node_from_point = ox.get_nearest_node(self.G_proj, point)
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
        self.G_proj = ox.consolidate_intersections(self.G_proj, rebuild_graph=True, tolerance=8, dead_ends=True)

    def routing_by_distance(self, origin_coord, destination_coord):
        origin_node = ox.get_nearest_node(self.G_proj, origin_coord)
        destination_node = ox.get_nearest_node(self.G_proj, destination_coord)
        route = ox.shortest_path(self.G_proj ,origin_node, destination_node, weight='length')
        return route 
    
    def routing_by_travel_time(self, origin_coord, destination_coord):
        origin_node, dist = ox.get_nearest_node(self.G_proj, (origin_coord[1],origin_coord[0]), method='euclidean', return_dist=True)
        #_log.info("Origin dist to node: %d"%dist)
        destination_node, dist = ox.get_nearest_node(self.G_proj, (destination_coord[1],destination_coord[0]), method='euclidean', return_dist=True)
        #_log.info("Destination dist to node: %d"%dist)
        if self.net_type == 'drive':
            hwy_speeds = {'residential': 35,
                        'living_street': 20,
                        'secondary': 50,
                        'tertiary': 60}
        elif self.net_type == 'walk':
            hwy_speeds = {'residential': 3,
                    'living_street': 3,
                    'secondary': 3,
                    'tertiary': 3}
        elif self.net_type == 'bike':
            hwy_speeds = {'residential': 20,
                    'living_street': 15,
                    'secondary': 25,
                    'tertiary': 25}
        self.G_proj = ox.add_edge_speeds(self.G_proj, hwy_speeds)
        self.G_proj = ox.add_edge_travel_times(self.G_proj)
        route = ox.shortest_path(self.G_proj ,origin_node, destination_node, weight='travel_time')
        return route 

    def compare_routes(self, route1, route2):
        route1_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route1, 'length')))
        route2_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route2, 'length')))
        route1_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route1, 'travel_time')))
        route2_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route2, 'travel_time')))
        print('Route 1 is', route1_length, 'meters and takes', route1_time, 'seconds.')
        print('Route 2 is', route2_length, 'meters and takes', route2_time, 'seconds.')

    def plot_graph(self, ax=None, figsize=(8, 8), bgcolor="#111111", node_color="w", node_size=15, node_alpha=None, node_edgecolor="none", node_zorder=1, edge_color="#999999", edge_linewidth=1, edge_alpha=None, show=True, close=False, save=False, filepath=None, dpi=300, bbox=None):
        fig, ax = ox.plot_graph(self.G_proj, ax=ax, figsize=figsize, bgcolor=bgcolor, node_color=node_color, node_size=node_size, node_alpha=node_alpha, node_edgecolor=node_edgecolor, node_zorder=node_zorder, edge_color=edge_color, edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, show=show, close=close, save=save, filepath=filepath, dpi=dpi, bbox=bbox)
        return fig, ax
    
    def plot_graph_route(self, route, route_color, show = True, save=False, filepath=None):
        fig, ax = ox.plot_graph_route(self.G_proj, route=route, route_color=route_color, route_linewidth=6, node_size=0, show=show, save=save, filepath=filepath)

    def plot_graph_routes(self, routes, route_colors ):
        fig, ax = ox.plot_graph_routes(self.G_proj, routes=routes, route_colors=route_colors, route_linewidth=6, node_size=0)

# DISCUSS
# Work done: 
    # Download GTFS
    # Transform GTFS to Trajectories.
    # Collect them in a collection
    # Get the starting and ending times of each trajectory
    # If the starting time of a trajectory coincides with the current time, create a new
    # instance of the correpsonding transport type.
    # Made the transport subway class 
    # Try to add humans to the transport.


# doubts: 
# 1 Need to deal with 173413642 line dataframe.
# 2 Now working with steps, i need time to make trjectories work. 
# 3 I init each trajectory inside model.step. Is it okay there? 
# 4 Remove still think how to do it explenation in the function
# 5 schedule random activation? First transports then humans. 
    # Maybe if we take into account seconds as time unit does not matter because we are in one stop for more 
    # than one step. 
# 6 NetworkGrid needs a G what should I use. 


# 7 Is the structure good? 
# 8 Add persons to transport_grid and therefore transport?
    # Get from the human_grid, the humans that are neigbours to the station location and then if they will, move them to the passangers list. 
    # transport_grid.add_agents(human_agent) --> Make new postions of the humans same as the train? 
    # Humans in list have the same position than the train?  
    # How to add them back into the worl grid? --> Remove them from the transport grid and add them in the real world? 
    # Do I need to remove the human from the worl grid when adding it to the transport grid?


# Ansewrs
# 1
# 2
# 3
# 4
# 5 
# 6
# 7
# 8

# HOW TIME WORKS
# TRAM FEED IS NOT VALID. SUGESTION: WORK FIRST WITH SUBWAY AND BUS AND THEN FIGURE OUT HOW TO WORK WITH TRAM
# WORKPLAN, NECESSARY TO BE IN THE PROJECT REPORT?    




# El start hauria de coincidir amb el inici de ruta. 
