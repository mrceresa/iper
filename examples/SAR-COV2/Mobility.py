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
import pickle
from scipy.spatial.distance import euclidean

from Covid_class import State, Mask

import timeit

import logging

_log = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


# Functions that may be needed
def load_merged_GTFS(self):
    self.merged_GTFS = pd.read_csv((ROOT_DIR + '/GTFS/merged_GTFS.csv'), dtype=str)


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
# Dictionary Route_dict --> Dictionary with services as keys and gorup of traj as items
# Dictionary traj_dict 
class RouteAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.route_id = unique_id
        transport_type = self.model.routes.loc[self.model.routes['route_id'] == self.route_id].route_type.item()
        self.name_transport_type = self.get_name_transport(transport_type)
        route_trips = self.model.trips.loc[self.model.trips['route_id'] == self.route_id]
        self.route_services = route_trips.loc[:, 'service_id'].unique()
        route_service_group = route_trips.groupby(route_trips.service_id)

        self.route_dict = {}
        self.createRouteDict(route_service_group)

        self.service_by_days = self.getRunningDays()

        self.trips_currently_running = {}

    def get_name_transport(self, transport_type):
        dictionary = {'0': 'Tram', '1': 'Subway', '3': 'Bus', '7': 'Funicular'}
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
            trajectory = pd.merge(trajectory, self.model.stops[['stop_id', 'geometry']], on='stop_id')
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
            if value_traj.get_start_time() == self.model.DateTime:
                self.createTransportAgent(trajectory=value_traj, traj_id=key_traj,
                                          transport_type=self.name_transport_type)
                self.trips_currently_running[key_traj] = value_traj

    def check_finish_traj(self):
        for key_traj, value_traj in self.trips_currently_running.items():
            if value_traj.get_end_time() == self.model.DateTime:
                self.removeTransportAgent(key_traj)

    def createTransportAgent(self, trajectory, traj_id, transport_type):
        ######## Vull tmb passar-li la trajectoria.#######
        AC = AgentCreator(transport_type, {"model", self.model})
        _a = AC.create_agent(
            Point(
                trajectory.get_start_location().x,
                trajectory.get_start_location().y
            ), traj_id)

        self.model.transport_grid.add_agents(_a)
        self.model.schedule.add(_a)

    def removeTransportAgent(self, agent_id):
        self.model.transport_grid.remove_agent(agent_id)
        self.model.schedule.remove(agent_id)

    def step(self):
        # Check every midnight the service running this day and load all the trajectories from that service
        if self.model.DateTime.time() == datetime.min.time():
            today_service = self.service_by_days.loc[
                self.service_by_days['date'] == self.model.DateTime.date()].service_id.item()

            self.traj_today_service = copy.deepcopy(self.route_dict[today_service])
            for key, traj in self.traj_today_service.items():
                traj.df.loc[:, 'time'] = traj.df.loc[:, 'time'] + datetime.combine(self.model.DateTime.date(),
                                                                                   datetime.min.time())
                traj.df.set_index('time', inplace=True)

        self.check_init_traj()
        self.check_finsih_traj()


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
        # self.has_goal = False
        self.goal_traj = None
        self.life_goals = 0
        self.has_car = random.random() < 0.39
        self.has_bike = random.random() < 0.05
        self.record_trajectories = {}

        self.house = None
        self.state = State.SUSC
        self.mask = Mask.NONE
        # variable to calculate time passed since last state transition
        self.days_in_current_state = self.model.DateTime
        self.presents_virus = False  # for symptomatic and detected asymptomatic people
        self.quarantined = None
        self.friends = set()
        self.contacts = {}  # dict where keys are days and each day has a list of people
        self.workplace = None  # to fill with a workplace if employed
        self.obj_place = None  # agent places to go
        self.friend_to_meet = set()  # to fill with Agent to meet

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

    def define_goal(self):
        random_node = random.choice(list(self.model.walkMap.G_proj.nodes))
        node = self.model.walkMap.G_proj.nodes[random_node]
        if node['x'] == self.shape.x and node['y'] == self.shape.y:
            self.define_goal()
        else:
            return (node['x'], node['y'])

    def init_goal_traj(self, goal_position):
        route = self.model.walkMap.routing_by_travel_time(self.get_pos(), goal_position)
        self.model.walkMap.plot_graph_route(route, 'y', show=False, save=True,
                                            filepath='plots/route_agent' + str(self.unique_id) + '_num' + str(
                                                self.life_goals) + '.png')

        df = pd.DataFrame()
        nodes, lats, lngs, times = [], [], [], []
        total_time = 0
        first_node = True

        for u, v in zip(route[:-1], route[1:]):
            travel_time = round(self.model.walkMap.G_proj.edges[(u, v, 0)]['travel_time'])
            if travel_time == 0:
                travel_time = 1

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
        df['time'] = pd.to_timedelta(times, unit='S')
        dfg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lngs, lats))
        gdf_trajectory = gpd.GeoDataFrame(dfg, crs=CRS(32631))
        traj = mpd.Trajectory(gdf_trajectory, self.life_goals)
        traj.df.loc[:, 'time'] = traj.df.loc[:, 'time'] + self.model.DateTime
        traj.df.set_index('time', inplace=True)

        self.record_trajectories[traj.df.index[0]] = traj
        return traj

    def update_goal_traj(self):
        try:
            new_traj = self.goal_traj.df[self.goal_traj.df.index > self.model.DateTime]
            new_traj = new_traj.reset_index()
            new_traj.loc[new_traj.shape[0]] = (
            self.model.DateTime, '', self.goal_traj.get_position_at(self.model.DateTime))
            new_traj.set_index('time', inplace=True)
            new_traj = mpd.Trajectory(new_traj, 1)
            return new_traj
        except:
            self.has_goal = False
            return None

    def step(self):
        """ Run one step taking into account the status, move, contact, and update_stats function. """
        self.status()
        self.routine()
        # self.contact()
        self.update_stats()

    def status(self):
        """Check for the infection status"""
        t = self.model.DateTime - self.days_in_current_state

        # For exposed people
        if self.state == State.EXP:
            # if have passed more days than the self.exposing time, it changes to an infectious state
            if t.days >= self.exposing_time:
                self.adjust_init_stats("EXP", "INF", State.INF)

                p_sympt = self.model.virus.pSympt  # prob to being Symptomatic
                self.presents_virus = np.random.choice([True, False], p=[p_sympt, 1 - p_sympt])
                if self.presents_virus:
                    # print(f"Agent {self.unique_id} presents symptoms and is going to test in the hospital")
                    self.quarantined = self.model.DateTime + timedelta(days=3)  # 3 day quarantine
                    self.obj_place = min(self.model.getHospitalPosition(), key=lambda c: euclidean(c, self.pos))
                    # print(f"Agent {self.unique_id} presents symptoms {self.presents_virus} and is quarantined until {self.quarantined.day}")

                inf_time = self.model.get_infection_time()
                self.infecting_time = inf_time
                # if self.unique_id < 5: print(f"Agent {self.unique_id} is now infected for {inf_time} days ")

                self.days_in_current_state = self.model.DateTime


        # For infected people
        elif self.state == State.INF:
            if self.presents_virus:  # if agent is symptomatic can be hospitalized or die
                # Calculate the prob of going severe
                severe_rate = self.model.virus.severe_rate
                not_severe = np.random.choice([0, 1], p=[severe_rate, 1 - severe_rate])

                # Agent is hospitalized
                if not_severe == 0:
                    self.adjust_init_stats("INF", "HOSP", State.HOSP)
                    self.model.hosp_collector_counts['H-INF'] -= 1  # doesnt need to be, maybe it was not in the record
                    self.model.hosp_collector_counts['H-HOSP'] += 1

                    sev_time = self.model.get_severe_time()
                    self.hospitalized_time = sev_time
                    # if self.unique_id < 5: print(f"Agent {self.unique_id} is now severe for {sev_time} days ")

                    self.days_in_current_state = self.model.DateTime

                    # look for the nearest hospital
                    self.obj_place = min(self.model.getHospitalPosition(), key=lambda c: euclidean(c, self.pos))
                    self.goal_traj = self.init_goal_traj(self.obj_place)

                    # adds patient to hospital patients list
                    h = self.model.getHospitalPosition(self.obj_place)
                    h.add_patient(self)

                    self.quarantined = None
                    self.friend_to_meet = set()

                # Only hospitalized agents die
                # death_rate = self.model.virus.pDeathRate(self.model)
                # alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
                # if alive == 0: self.adjust_init_stats("INF", "DEAD", State.DEAD)

            # agent is INF (not HOSP nor DEAD), has been INF for the infection time
            if self.state == State.INF and t.days >= self.infecting_time:
                self.adjust_init_stats("INF", "REC", State.REC)
                self.presents_virus = False
                im_time = self.model.get_immune_time()
                self.immune_time = im_time
                # if self.unique_id < 5: print(f"Agent {self.unique_id} is now immune for {im_time} days ")

                self.days_in_current_state = self.model.DateTime



        # For recovered people
        elif self.state == State.REC:
            # if have passed more days than self.immune_time, agent is susceptible again
            if t.days >= self.immune_time:
                self.adjust_init_stats("REC", "SUSC", State.SUSC)
                # if self.unique_id < 5: print(f"Agent {self.unique_id} is SUSC again")

        # For hospitalized people
        elif self.state == State.HOSP:
            # Calculate the prob of dying
            death_rate = self.model.virus.pDeathRate(self.model)
            alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
            # Agent dies
            if alive == 0:
                # discharge patient
                h = self.model.getHospitalPosition(self.obj_place)
                h.discharge_patient(self)

                self.adjust_init_stats("HOSP", "DEAD", State.DEAD)
                self.model.hosp_collector_counts['H-HOSP'] -= 1
                self.model.hosp_collector_counts['H-DEAD'] += 1
                # self.model.schedule.remove(self)
            # Agent still alive, if have passed more days than hospitalized_time, change state to Recovered
            if alive != 0 and t.days >= self.hospitalized_time:
                # discharge patient
                h = self.model.getHospitalPosition(self.obj_place)
                h.discharge_patient(self)

                self.adjust_init_stats("HOSP", "REC", State.REC)
                self.model.hosp_collector_counts['H-HOSP'] -= 1
                self.model.hosp_collector_counts['H-REC'] += 1

                im_time = self.model.get_immune_time()
                self.immune_time = im_time

                self.days_in_current_state = self.model.DateTime

        # change quarantine status if necessary
        if self.quarantined is not None:
            if self.model.DateTime.day == self.quarantined.day:
                self.quarantined = None

    def routine(self):
        # new_position = None
        # possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        if self.state == State.SUSC or self.state == State.EXP or self.state == State.INF or self.state == State.REC:
            if self.quarantined is None:  # if agent not hospitalized or dead
                # sleeping time
                if self.model.DateTime.hour <= 6:
                    self.friend_to_meet = set()  # meetings are cancelled
                    self.obj_place = None

                # working time
                elif 6 < self.model.DateTime.hour <= 16:  # working time
                    # employed agent
                    if self.workplace is not None:
                        if self.model.DateTime.hour == 7 and self.model.DateTime.minute == 0:
                            self.mask = Mask.RandomMask()  # wear mask for walk
                            self.goal_traj = self.init_goal_traj(self.workplace)
                            # new_position = min(possible_steps,key=lambda c: euclidean(c,self.workplace.place))  # check shortest path to work


                        else:
                            if self.pos == self.workplace.place:  # employee at workplace
                                self.mask = self.workplace.mask
                                """cellmates = self.model.grid.get_cell_list_contents([self.pos])
                                if len(cellmates) > 1:
                                    for other in [i for i in cellmates if i != self]:
                                        if isinstance(other, Human):
                                            self.add_contact(other)"""


                # leisure time
                elif 16 < self.model.DateTime.hour <= 21:  # leisure time
                    if self.model.DateTime.hour == 17 and self.model.DateTime.minute == 0: self.mask = Mask.RandomMask()  # wear mask for walk
                    if not self.friend_to_meet:
                        if np.random.choice([0, 1],
                                            p=[0.75, 0.25]): self.look_for_friend()  # probability to meet with a friend
                        #new_position = self.random.choice(possible_steps)  # choose random step

                    else:  # going to a meeting
                        #cellmates = set(self.model.grid.get_cell_list_contents([self.pos]))
                        if self.pos == self.obj_place: #and self.friend_to_meet.issubset(cellmates):  # wait for everyone at the meeting
                            """for friend in self.friend_to_meet:
                                self.add_contact(friend)"""
                            self.friend_to_meet = set()
                            self.obj_place = None
                            self.goal_traj = None

                # go back home
                else:  # Time to go home
                    if self.pos != self.house and self.goal_traj is None:
                        self.goal_traj = self.init_goal_traj(self.house)
                    elif self.pos == self.house:  # agent at home
                        self.mask = Mask.NONE
            # Agent is self.quarantined
            elif self.quarantined is not None:
                if self.pos != self.house and self.obj_place is None and self.goal_traj is None:  # if has been tested, go home
                    #new_position = min(possible_steps,key=lambda c: euclidean(c, self.house))  # check shortest path to house
                    self.goal_traj = self.init_goal_traj(self.house)

                elif self.pos == self.house and self.obj_place is None and self.goal_traj is None: # agent has returned home
                    self.mask = Mask.NONE

                elif self.obj_place is not None:

                    if self.obj_place != self.pos and self.model.DateTime.hour == 7 and self.goal_traj is None:  #check this # if has to go testing, just go
                        # print(f"Agent {self.unique_id} on their way to testing")
                        #new_position = min(possible_steps, key=lambda c: euclidean(c, self.obj_place))
                        self.goal_traj = self.init_goal_traj(self.obj_place)

                    elif self.obj_place == self.pos:
                        # once at hospital, is tested and next step will go home to quarantine
                        self.mask = Mask.FFP2
                        h = self.model.getHospitalPosition(self.obj_place)
                        h.doTest(self)
                        self.obj_place = None

        # Ill agents move to nearest hospital to be treated
        elif self.state == State.HOSP:
            if self.pos == self.obj_place:
                self.mask = Mask.FFP2

        # if new_position: self.model.grid.move_agent(self, new_position)
        self.movement()

    def look_for_friend(self):
        """ Check the availability of friends to meet and arrange a meeting """
        available_friends = [self.model.schedule.agents[friend] for friend in self.friends if
                             not self.model.schedule.agents[friend].friend_to_meet and self.model.schedule.agents[
                                 friend].quarantined is None and self.model.schedule.agents[
                                 friend].state != State.HOSP and self.model.schedule.agents[friend].state != State.DEAD]
        peopleMeeting = int(self.random.normalvariate(self.model.peopleInMeeting,
                                                      self.model.peopleInMeetingSd))  # get total people meeting
        if len(available_friends) > 0:

            pos_x = [self.pos.x]
            pos_y = [self.pos.y]

            while peopleMeeting > len(
                    self.friend_to_meet) and available_friends:  # reaches max people in meeting or friends are unavailable
                friend_agent = random.sample(available_friends, 1)[0]  # gets one random each time
                available_friends.remove(friend_agent)

                self.friend_to_meet.add(friend_agent)
                friend_agent.friend_to_meet.add(self)

                pos_x.append(friend_agent.pos.x)
                pos_y.append(friend_agent.pos.y)

            # update the obj position to meet for all of them
            meeting_position = (int(sum(pos_x) / len(pos_x)), int(sum(pos_y) / len(pos_y)))
            self.obj_place = meeting_position
            self.goal_traj = self.init_goal_traj(meeting_position)

            for friend in self.friend_to_meet:
                friend.friend_to_meet.update(self.friend_to_meet)
                friend.friend_to_meet.remove(friend)
                friend.obj_place = meeting_position
                friend.goal_traj = friend.init_goal_traj(meeting_position)

    def movement(self):

        if self.goal_traj is not None:
            end_time = self.goal_traj.get_end_time()
            if end_time == self.model.DateTime:
                self.goal_traj = None
            else:
                try:
                    print(self.model.DateTime)
                    newPos = self.goal_traj.get_position_at(self.model.DateTime)
                except:
                    print('---------------------')
                    print(self.get_pos())
                    print(self.goal_traj.df)
                    print('---------------------')
                    # exit()
                self.place_at(newPos)

            # I would use it for print only the trajectory left...
            # self.goal_traj = self.update_goal_traj()
            # neighbors = self.model.grid.get_neighbors(self)

    def contact(self):
        """ Find close contacts and infect """
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if isinstance(other, Human) and other != self:
                    pTrans = self.model.virus.pTrans(self.mask, other.mask)
                    trans = np.random.choice([0, 1], p=[pTrans, 1 - pTrans])
                    if trans == 0 and (
                            self.state is State.INF or self.state is State.EXP) and other.state is State.SUSC:
                        other.state = State.EXP
                        other.days_in_current_state = self.model.DateTime
                        in_time = self.model.get_incubation_time()
                        other.exposing_time = in_time

    def update_stats(self):
        """ Update Status dictionaries for data collector. """
        if self.state == State.SUSC:
            self.model.collector_counts['SUSC'] += 1
        elif self.state == State.EXP:
            self.model.collector_counts['EXP'] += 1
        elif self.state == State.INF:
            self.model.collector_counts['INF'] += 1
        elif self.state == State.REC:
            self.model.collector_counts['REC'] += 1
        elif self.state == State.HOSP:
            self.model.collector_counts['HOSP'] += 1
        elif self.state == State.DEAD:
            self.model.collector_counts['DEAD'] += 1

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
        newPos = self.trajectory.get_pos(model.time)  # time from the main
        self.place_at(newPos)

        # if self.stops[currentStation].x == newPos.x and self.stops[currentStation].y == newPos.y:
        # People can enter

        # Check whether the station has changed
        # if self.stops[currentStation].x == currentPos.x and self.stops[currentStation].y == currentPos.y and self.stops[currentStation].x != newPos.x and self.stops[currentStation].y != newPos.y:
        #    self.currentStation += 1
        # else:
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
        path_name = '/Mobility Jupyter Files/OSMnx/pickle_objects/'
        cheat = True  # toy example
        if cheat == True:
            with open(root_path + path_name + 'PlaçaCat_walk_proj.p', 'rb') as f:
                self.G_proj = pickle.load(f)
            self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.G_proj, nodes=True, edges=True)

        else:
            start = timeit.default_timer()
            with open(root_path + path_name + 'road_network_projected.p', 'rb') as f:
                self.G_proj = pickle.load(f)
            try:
                with open(root_path + path_name + 'road_network_projected_db.p', 'rb') as f:
                    db = pickle.load(f)
                self.nodes_proj = db[0]
                self.edges_proj = db[1]
            except:
                self.nodes_proj, self.edges_proj = ox.graph_to_gdfs(self.G_proj, nodes=True, edges=True)
                with open(root_path + path_name + 'road_network_projected_db.p', 'wb') as f:
                    pickle.dump([self.nodes_proj, self.edges_proj], f)

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

    def get_basic_stats(self, stats=None):
        area_graph = self.get_graph_area()
        basic_stats = ox.basic_stats(self.G_proj, area=area_graph, clean_intersects=True, tolerance=15,
                                     circuity_dist='euclidean')
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
        route = ox.shortest_path(self.G_proj, origin_node, destination_node, weight='length')
        return route

    def routing_by_travel_time(self, origin_coord, destination_coord):
        origin_node, dist = ox.get_nearest_node(self.G_proj, (origin_coord[1], origin_coord[0]), method='euclidean',
                                                return_dist=True)
        # _log.info("Origin dist to node: %d"%dist)
        destination_node, dist = ox.get_nearest_node(self.G_proj, (destination_coord[1], destination_coord[0]),
                                                     method='euclidean', return_dist=True)
        # _log.info("Destination dist to node: %d"%dist)
        route = ox.shortest_path(self.G_proj, origin_node, destination_node, weight='travel_time')
        return route

    def compare_routes(self, route1, route2):
        route1_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route1, 'length')))
        route2_length = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route2, 'length')))
        route1_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route1, 'travel_time')))
        route2_time = int(sum(ox.utils_graph.get_route_edge_attributes(self.G_proj, route2, 'travel_time')))
        print('Route 1 is', route1_length, 'meters and takes', route1_time, 'seconds.')
        print('Route 2 is', route2_length, 'meters and takes', route2_time, 'seconds.')

    def plot_graph(self, ax=None, figsize=(8, 8), bgcolor="#111111", node_color="w", node_size=15, node_alpha=None,
                   node_edgecolor="none", node_zorder=1, edge_color="#999999", edge_linewidth=1, edge_alpha=None,
                   show=True, close=False, save=False, filepath=None, dpi=300, bbox=None):
        fig, ax = ox.plot_graph(self.G_proj, ax=ax, figsize=figsize, bgcolor=bgcolor, node_color=node_color,
                                node_size=node_size, node_alpha=node_alpha, node_edgecolor=node_edgecolor,
                                node_zorder=node_zorder, edge_color=edge_color, edge_linewidth=edge_linewidth,
                                edge_alpha=edge_alpha, show=show, close=close, save=save, filepath=filepath, dpi=dpi,
                                bbox=bbox)
        return fig, ax

    def plot_graph_route(self, route, route_color, show=True, save=False, filepath=None):
        fig, ax = ox.plot_graph_route(self.G_proj, route=route, route_color=route_color, route_linewidth=6, node_size=0,
                                      show=show, save=save, filepath=filepath)

    def plot_graph_routes(self, routes, route_colors):
        fig, ax = ox.plot_graph_routes(self.G_proj, routes=routes, route_colors=route_colors, route_linewidth=6,
                                       node_size=0)

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


# Juntar tots els trensports per que el shortest path els tingui en compte.
