from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random, numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import movingpandas as mpd
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from SEAIHRD_class import SEAIHRD_covid, Mask
import haversine as hs
import DataCollector_functions as dc
from geopandas.geodataframe import GeoDataFrame
from random import uniform, randint
import osmnx as ox
from shapely.geometry import Point


class RandomWalk(Action):
    def do(self, agent):
        _xs = agent.getWorld()._xs
        dx, dy = _xs["dx"], _xs["dy"]  # How much is 1deg in km?
        # Convert in meters
        dx, dy = (dx / 1000, dy / 1000)

        new_position = (agent.pos[0] + random.uniform(-dx, dx), agent.pos[1] + random.uniform(-dy, dy))

        if not agent.getWorld().out_of_bounds(new_position):
            agent.getWorld().space.move_agent(agent, new_position)

class MoveTo(Action):
    def init_goal_traj(self, agent):
        route = agent.map.routing_by_travel_time(agent.pos, agent.obj_place)
        # Plots (Better do it at the end)
        # agent.map.plot_graph_route(route, 'y', show = False, save = True, filepath = 'plots/route_agent' + str(agent.unique_id) + '_num' + str(agent.life_goals) + '.png')
        # agent.map.plot_route_by_transport_type(route, save = True, filepath = 'examples/bcn_multispace/plots/route_agent_' + str(agent.unique_id) + '_num' + str(agent.life_goals) + 'colors' + '.png')

        df = pd.DataFrame()
        nodes, lats, lngs, times, types = [], [], [], [], []
        total_time = 0
        first_node = True

        for u, v in zip(route[:-1], route[1:]):
            travel_time = round(agent.map.G.edges[(u, v, 0)]['travel_time'])
            if travel_time == 0:
                travel_time = 1

            if first_node == True:
                nodes.append(u)
                lats.append(agent.map.G.nodes[u]['y'])
                lngs.append(agent.map.G.nodes[u]['x'])
                times.append(0)
                types.append(agent.map.G.nodes[u]['Type'])
                first_node = False

            nodes.append(v)
            lats.append(agent.map.G.nodes[v]['y'])
            lngs.append(agent.map.G.nodes[v]['x'])
            times.append(total_time + travel_time)
            total_time += travel_time
            types.append(agent.map.G.nodes[v]['Type'])

        df['node'] = nodes
        df['type'] = types
        df['time'] = pd.to_timedelta(times, unit='S')
        # df['id'] = str(agent.unique_id) + '-' + str(agent.life_goals)
        dfg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lngs, lats))
        gdf_trajectory = gpd.GeoDataFrame(dfg, crs=CRS(32631))
        traj = mpd.Trajectory(gdf_trajectory, agent.life_goals)
        traj.df.loc[:, 'time'] = traj.df.loc[:, 'time'] + agent.model.DateTime
        traj.df.set_index('time', inplace=True)
        agent.record_trajectories[traj.df.index[0]] = traj
        # traj.df.to_csv(os.path.join(os.getcwd(), 'examples/bcn_multispace/EudaldMobility/trajectories.csv'), mode='a', header = False)
        return traj

    def accumulate_polution(self, agent):
        local_pollution = 1  # get_pollution(agent.pos)
        acc_traj_pollution = local_pollution * int(agent.model.time_step)
        agent.exposure_pollution += acc_traj_pollution

    def do(self, agent, pos):
        #print("***************************",pos)
        if agent.pos != pos:
            agent.goal_traj = self.init_goal_traj(agent)
            agent.life_goals += 1
            if agent.model.DateTime >= agent.goal_traj.get_end_time():
                newPos = agent.goal_traj.get_position_at(agent.goal_traj.get_end_time())
            else:
                newPos = agent.goal_traj.get_position_at(agent.model.DateTime)
            agent.getWorld().space.move_agent(agent, newPos)

        # Pollution
        pollution_value = agent.model.pollution_model.get_pollution_value((agent.pos[0], agent.pos[1]))
        agent.exposure_pollution.append(pollution_value)




class StandStill(Action):
    def do(self, agent):
        pass

class HumanAgent(XAgent):
    def __init__(self, unique_id, model):
        super().__init__(self.unique_id)
        self.unique_id = unique_id
        self.model = model
        self.has_goal = False
        self.life_goals = 0
        self.record_trajectories = {}
        self.exposure_pollution = []
        self.has_car = random.random() < 0.39
        self.has_bike = random.random() < 0.05
        # self.which_map()
        self.map = self.model.Ped_Map
        # If we want the predeterminated goals run:
        # self.goals_list = self.predeterminated_goals()

        # COVID ATTRIBUTES
        self.mask = Mask.NONE

        self.machine = None
        # variable to calculate time passed since last state transition
        self.days_in_current_state = model.DateTime
        # self.presents_virus = False  # for symptomatic and detected asymptomatic people
        self.quarantined = None

        self.family = set()
        self.friends = set()
        self.contacts = {}  # dict where keys are days and each day has a list of people

        self.workplace = None  # to fill with a workplace if employed
        self.hospital = None
        self.obj_place = None  # agent places to go
        # self.goal = ""
        self.friend_to_meet = set()  # to fill with Agent to meet

        self.HospDetected = False
        self.R0_contacts = {}

    def __repr__(self):
        return "Agent id " + str(self.id)

    def _postInit(self):
        self.obj_place = self.house

    def _se(self):
        """ This function is notified when a transition s->e happens"""
        pass

    def which_map(self):
        map_name = ""
        if self.has_car == True and self.has_bike == True:
            map_name = "Pedestrian + Car + Bike"
            self.map = self.model.PedCarBike_Map
        elif self.has_car == True and self.has_bike == False:
            map_name = "Pedestrian + Car"
            self.map = self.model.PedCar_Map
        elif self.has_car == False and self.has_bike == True:
            map_name = "Pedestrian + Bike"
            self.map = self.model.PedBike_Map
        else:
            map_name = "Pedestrian"
            self.map = self.model.Ped_Map

        print('Agent: ' + str(self.unique_id) + " is using the map: " + map_name)

    def working_time(self):

        if not self.workplace:
            return self.obj_place

        workplace = self.model.space.get_agent(self.workplace)

        # TODO: Ask Aran why we use a different mask
        # because it can be the case agents are not wearing masks on the outside but its mandatory in the inside
        # if self.pos == workplace.place:
        #     self.mask = workplace.mask

        self.mask = workplace.mask
        r = 20.0/111100
        work_position_rand = (workplace.place[0]+randint(-1,1)*r, workplace.place[1]+randint(-1,1)*r)
        return work_position_rand

    def leisure_time(self):
        if not self.friend_to_meet:
            if np.random.choice([0, 1], p=[0.75,0.25]):
                self.look_for_friend()  # probability to meet with a friend

        return self.obj_place


    def _thinkGoal(self):
        if self.model._check_movement_is_restricted():
            return self.house

        if self.quarantined: return self.obj_place
        # working time
        if 8 < self.model.DateTime.hour <= 16:  # working time
            self.goal = "WORK"
            return self.working_time()

        # leisure time
        if 16 < self.model.DateTime.hour <= self.model.night_curfew - 2:  # leisure time
            self.goal = "FUN"
            return self.leisure_time()

        if self.model.night_curfew - 2 < self.model.DateTime.hour <= self.model.night_curfew - 1:  # Time to go home
            if self.pos != self.house:
                self.obj_place = self.house
            return self.obj_place

        else:
            self.obj_place = self.house
            return self.obj_place

    def think(self):

        chosen_action, chosen_action_pars = StandStill(), [self]

        if not self.mask and self.pos != self.house:
            self.mask = Mask.RandomMask(self.model.masks_probs)  # wear mask for walk

        # If we have a previous goal, go to it
        if self.obj_place != self.pos:
            return MoveTo(), [self, self.obj_place]

        if self.quarantined and self.model.DateTime.hour == 9:  # test first thing in the morning
            if self.obj_place == self.pos and self.goal == "GO_TO_HOSPITAL":
                # once at hospital, is tested and next step will go home to quarantine
                self.mask = Mask.FFP2
                h = self.model._hospitals[self.obj_place]
                h.doTest(self)
        else:
            self.obj_place = self._thinkGoal()

        # If we have a new goal execute it
        if self.obj_place != self.pos:
            return MoveTo(), [self, self.obj_place]

        if self.pos == self.house and self.obj_place == self.pos:
            self.mask = Mask.NONE

        if self.machine.state in ["H", "D"]:
            pass

        return chosen_action, chosen_action_pars

    def getNeighs(self):
        cellmates = self.getWorld().space.agents_at(self.pos, radius=2.0/111100)  # pandas df [agentid, geometry, distance]
        cellmates = cellmates[(cellmates['agentid'].str.contains('Human'))]  # filter out buildings
        return cellmates

    def step(self):
        self.l.debug("*** Agent %s stepping" % str(self.id))
        #print( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",str(self.id),self.pos,"XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        cellmates = None

        if self.machine.state in ["I", "A"]:
            cellmates = self.getNeighs()
            self.contact(cellmates)
            for str_id in [x for x in cellmates["agentid"] if x != self.id]:
                self.add_contact(str_id)

        action, a_pars = self.think()

        #self.l.info("Agent %s is doing %s(%s)"%(self.id, action.__class__.__name__,str(a_pars)))
        action.do( *a_pars)

        #if not self.model.lockdown_total:
        #    self.move(cellmates)  # if not in total lockdown

        super().step()





    def contact(self, others):
        """ Find close contacts and infect """
        if others is None or len(others)==0: return
        others_agents = [self.model.space.get_agent(aid) for aid in others["agentid"] if aid != self.id]
        #others_agents = [self.model.space.get_agent(aid) for aid in others if aid != self.id]
        for other in others_agents:
            if other.machine.state == "S":
                other.machine.contact(self.mask, other.mask)
                self.model.contact_count[0] += 1
                if other.machine.state == "E":
                    self.model.contact_count[1] += 1
                    other.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')] = [0, round(1 / other.machine.rate['rEI']) + round(1 / other.machine.rate['rIR']), 0]
        #print(self.model.contact_count )

    def look_for_friend(self):
        """ Check the availability of friends to meet and arrange a meeting """


        available_friends = [friend for friend in self.friends if
                             not self.model.space.get_agent(friend).friend_to_meet and self.model.space.get_agent(
                                 friend).quarantined is None and self.model.space.get_agent(
                                 friend).machine.state not in ["H", "D"]]

        peopleMeeting = int(self.random.normalvariate(self.model.peopleInMeeting,
                                                      self.model.peopleInMeetingSd))  # get total people meeting
        if len(available_friends) > 0:

            pos_x = [self.pos[0]]
            pos_y = [self.pos[1]]

            while peopleMeeting > len(
                    self.friend_to_meet) and available_friends:  # reaches max people in meeting or friends are unavailable
                friend_agent = self.model.space.get_agent(random.sample(available_friends, 1)[0])  # gets one random each time
                available_friends.remove(friend_agent.id)

                self.friend_to_meet.add(friend_agent.id)
                friend_agent.friend_to_meet.add(self.id)

                pos_x.append(friend_agent.pos[0])
                pos_y.append(friend_agent.pos[1])

            # update the obj position to meet for all of them
            meeting_position = (round(sum(pos_x) / len(pos_x), 15), round(sum(pos_y) / len(pos_y), 15))
            self.obj_place = meeting_position

            for friend in self.friend_to_meet:
                self.model.space.get_agent(friend).friend_to_meet.update(self.friend_to_meet)
                self.model.space.get_agent(friend).friend_to_meet.remove(friend)
                self.model.space.get_agent(friend).obj_place = meeting_position


    def add_contact(self, contact):
        # check contacts for self agent

        already_registered_contact = False
        today  = self.model.DateTime.strftime('%Y-%m-%d')
        if not today in self.contacts:
            self.contacts[today] = {contact}  # initialize with contact

        elif contact not in self.contacts[today]:
            self.contacts[today].add(contact)  # add contact to today's date

        else:
            already_registered_contact = True

        # add contacts of infected people for R0 calculations
        if self.machine.state in ["E", "A", "I"]:
            contacts_mask = self.model.space.get_agent(contact).mask
            pTransMask1 = self.mask.maskPtrans(self.mask)
            pTransMask2 = contacts_mask.maskPtrans(contacts_mask)
            #print(self.R0_contacts)
            #print("************************",self.model.DateTime.strftime('%Y-%m-%d'))

            if not today in self.R0_contacts:
                self.R0_contacts[today] = [0,0,0]  # initialize

            if not already_registered_contact:
                self.R0_contacts[today][0] += self.machine.prob_inf * pTransMask1 * pTransMask2  #prob_infection(self.mask, contacts_mask)  # self.model.virus.pTrans
                self.R0_contacts[today][2] += 1
