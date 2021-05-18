from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random, numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from SEAIHRD_class import SEAIHRD_covid, Mask
import DataCollector_functions as dc


class RandomWalk(Action):
    def do(self, agent):
        _xs = agent.getWorld()._xs
        dx, dy = _xs["dx"], _xs["dy"]  # How much is 1deg in km?
        # Convert in meters
        dx, dy = (dx / 1000, dy / 1000)

        new_position = (agent.pos[0] + random.uniform(-dx, dx), agent.pos[1] + random.uniform(-dy, dy))

        if not agent.getWorld().out_of_bounds(new_position):
            agent.getWorld().space.move_agent(agent, new_position)


class HumanAgent(XAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id)
        self.mask = Mask.NONE

        self.machine = None
        # variable to calculate time passed since last state transition
        self.days_in_current_state = model.DateTime
        # self.presents_virus = False  # for symptomatic and detected asymptomatic people
        self.quarantined = None

        self.friends = set()
        self.contacts = {}  # dict where keys are days and each day has a list of people

        self.workplace = None  # to fill with a workplace if employed
        self.obj_place = None  # agent places to go
        self.friend_to_meet = set()  # to fill with Agent to meet

        self.HospDetected = False
        self.R0_contacts = {}

    def __repr__(self):
        return "Agent id " + str(self.id)

    def _postInit(self):
        pass

    def think(self):
        possible_actions = [RandomWalk()]
        chosen_action = random.choice(possible_actions)
        chosen_action.do(self)
        # return chosen_action

    def step(self):
        self.l.debug("*** Agent %s stepping" % str(self.id))
        # self.status()

        # self.think()
        self.move()

        if self.machine.state in ["E", "I"] and self.model.DateTime.hour > 7:
            self.contact()

        super().step()

    # def status(self):
    # t = self.model.DateTime - self.days_in_current_state
    # For exposed people
    # if self.machine.state == "E":
    # if have passed more days than the self.exposing time, it changes to an infectious state
    # if t.days >= self.exposing_time:
    #     self.adjust_init_stats("EXP", "INF", "I")
    #
    #     p_sympt = self.model.virus.pSympt  # prob to being Symptomatic
    #     self.presents_virus = np.random.choice([True, False], p=[p_sympt, 1 - p_sympt])
    #     if self.presents_virus:
    #         # print(f"Agent {self.unique_id} presents symptoms and is going to test in the hospital")
    #         self.quarantined = self.model.DateTime + timedelta(days=3)  # 3 day quarantine
    #         self.obj_place = min(self.model.getHospitalPosition(), key=lambda c: euclidean(c, self.pos))
    #         # print(f"Agent {self.unique_id} presents symptoms {self.presents_virus} and is quarantined until {self.quarantined.day}")
    #
    #     inf_time = dc.get_infection_time(self.model)
    #     self.infecting_time = inf_time
    #     # if self.unique_id < 5: print(f"Agent {self.unique_id} is now infected for {inf_time} days ")
    #     self.days_in_current_state = self.model.DateTime

    # For infected people
    # elif self.machine.state == "I":
    # if self.presents_virus:  # if agent is symptomatic can be hospitalized or die
    # Calculate the prob of going severe
    # severe_rate = self.model.virus.severe_rate
    # not_severe = np.random.choice([0, 1], p=[severe_rate, 1 - severe_rate])

    # Agent is hospitalized
    # if not_severe == 0:
    #     self.adjust_init_stats("INF", "HOSP", "H")
    #     self.model.hosp_collector_counts['H-INF'] -= 1  # doesnt need to be, maybe it was not in the record
    #     self.model.hosp_collector_counts['H-HOSP'] += 1
    #     self.HospDetected = False  # we assume hospitalized people do not transmit the virus
    #
    #     self.hospitalized_time = dc.get_severe_time(self.model)
    #     # if self.unique_id < 5: print(f"Agent {self.unique_id} is now severe for {sev_time} days ")
    #
    #     self.days_in_current_state = self.model.DateTime
    #
    #     # look for the nearest hospital
    #     self.obj_place = min(self.model.getHospitalPosition(), key=lambda c: euclidean(c, self.pos))
    #
    #     # adds patient to nearest hospital patients list
    #     h = self.model.getHospitalPosition(self.obj_place)
    #     h.add_patient(self)
    #
    #     self.quarantined = None
    #     self.friend_to_meet = set()

    # Only hospitalized agents die
    # death_rate = self.model.virus.pDeathRate(self.model)
    # alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
    # if alive == 0: self.adjust_init_stats("INF", "DEAD", State.DEAD)

    # agent is INF (not HOSP nor DEAD), has been INF for the infection time
    # if self.machine.state == "I" and t.days >= self.infecting_time:
    #     self.adjust_init_stats("INF", "REC", "R")
    #     self.presents_virus = False
    #     self.immune_time = dc.get_immune_time(self.model)
    #     # if self.unique_id < 5: print(f"Agent {self.unique_id} is now immune for {im_time} days ")
    #     self.days_in_current_state = self.model.DateTime
    #     if self.HospDetected == True:
    #         self.model.hosp_collector_counts['H-INF'] -= 1
    #         self.model.hosp_collector_counts['H-REC'] += 1

    # For recovered people
    # elif self.machine.state == "R":
    #    # if have passed more days than self.immune_time, agent is susceptible again
    #    if t.days >= self.immune_time:
    #        self.adjust_init_stats("REC", "SUSC", "S")
    #        if self.HospDetected == True:
    #            self.model.hosp_collector_counts['H-REC'] -= 1
    #            self.model.hosp_collector_counts['H-SUSC'] += 1
    #            self.HospDetected = False
    #        # if self.unique_id < 5: print(f"Agent {self.unique_id} is SUSC again")

    # For hospitalized people
    # elif self.machine.state == "H":
    # Calculate the prob of dying
    # death_rate = self.model.virus.pDeathRate(self.model)
    # alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
    # Agent dies
    # if alive == 0:
    #     # discharge patient
    #     h = self.model.getHospitalPosition(self.obj_place)
    #     h.discharge_patient(self)
    #
    #     self.adjust_init_stats("HOSP", "DEAD", "S")
    #     self.model.hosp_collector_counts['H-HOSP'] -= 1
    #     self.model.hosp_collector_counts['H-DEAD'] += 1
    # self.model.schedule.remove(self)
    # Agent still alive, if have passed more days than hospitalized_time, change state to Recovered
    # if alive != 0 and t.days >= self.hospitalized_time:
    #     # discharge patient
    #     h = self.model.getHospitalPosition(self.obj_place)
    #     h.discharge_patient(self)
    #
    #     self.adjust_init_stats("HOSP", "REC", "R")
    #     self.model.hosp_collector_counts['H-HOSP'] -= 1
    #     self.model.hosp_collector_counts['H-REC'] += 1
    #
    #     self.immune_time = dc.get_immune_time(self.model)
    #
    #     self.days_in_current_state = self.model.DateTime

    # change quarantine status if necessary
    # if self.quarantined is not None:
    #     if self.model.DateTime.day == self.quarantined.day:
    #         self.quarantined = None

    def move(self):
        # new_position = None
        # possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if self.machine.state in ["S", "E", "A", "I", "R"]:
            if self.quarantined is None:  # if agent not hospitalized or dead
                if self.model.DateTime.hour == 0:
                    self.friend_to_meet = set()  # meetings are cancelled
                    self.obj_place = None

                # sleeping time
                elif self.model.DateTime.hour == 6:
                    cellmates = self.getWorld().space.agents_at(self.pos,max_num=10)  # pandas df [agentid, geometry, distance]
                    cellmates = cellmates[(cellmates['agentid'].str.contains('Human')) & (cellmates['distance'] < 1)]  # filter out buildings and far away people .iloc[0:2]

                    if len(cellmates) > 1:
                        for str_id in [x for x in cellmates['agentid'] if x != self.id]:
                            # index = next((i for i, item in enumerate(self.model.schedule.agents) if item.id == str_id), -1)
                            if self.model.space.get_agent(str_id).house == self.house:
                                self.add_contact(str_id)

                # working time
                elif 6 < self.model.DateTime.hour <= 16:  # working time
                    workplace = self.model.space.get_agent(self.workplace)
                    if self.workplace is not None and self.pos != workplace.place:  # Employed and not at workplace
                        if self.model.DateTime.hour == 7 and self.model.DateTime.minute == 0: self.mask = Mask.RandomMask()  # wear mask for walk
                        # new_position = min(possible_steps,key=lambda c: euclidean(c,self.workplace.place))  # check shortest path to work
                        self.getWorld().space.move_agent(self, workplace.place)
                    # employee at workplace. Filter by time to avoid repeated loops
                    elif self.workplace is not None and self.pos == workplace.place and self.model.DateTime.hour == 15 and self.model.DateTime.minute == 0:

                        self.mask = workplace.mask
                        # self.getWorld().space._create_gdf()
                        cellmates = self.getWorld().space.agents_at(self.pos,max_num=10)  # pandas df [agentid, geometry, distance]
                        cellmates = cellmates[(cellmates['agentid'].str.contains('Human')) & (
                                cellmates['distance'] < 2)]  # filter out buildings and far away people .iloc[0:2]

                        if len(cellmates) > 1:
                            for str_id in [x for x in cellmates['agentid'] if x != self.id]:
                                # index = next((i for i, item in enumerate(self.model.schedule.agents) if item.id == str_id), -1)
                                if self.model.space.get_agent(str_id).workplace == workplace:
                                    self.add_contact(str_id)



                # leisure time
                elif 16 < self.model.DateTime.hour <= 21:  # leisure time
                    if self.model.DateTime.hour == 17 and self.model.DateTime.minute == 0: self.mask = Mask.RandomMask()  # wear mask for walk
                    if not self.friend_to_meet:
                        if np.random.choice([0, 1], p=[0.75,
                                                       0.25]) and self.model.DateTime.minute == 0: self.look_for_friend()  # probability to meet with a friend
                        # new_position = self.random.choice(possible_steps)  # choose random step
                        self.think()  # randomly

                    else:  # going to a meeting
                        if self.pos != self.obj_place:
                            # check shortest step towards friend new position
                            self.getWorld().space.move_agent(self, self.obj_place)
                            # new_position = min(possible_steps, key=lambda c: euclidean(c, self.obj_place))
                        else:
                            # self.getWorld().space._create_gdf()

                            cellmates = self.getWorld().space.agents_at(self.pos,
                                                                        max_num=10)  # pandas df [agentid, geometry, distance]
                            cellmates = cellmates[(cellmates['agentid'].str.contains('Human')) & (cellmates[
                                                                                                      'distance'] < 2)]  # filter out buildings and far away people .iloc[0:2]

                            human_cellmates = set([x for x in cellmates['agentid'] if x != self.id])

                            if self.friend_to_meet.issubset(human_cellmates):  # wait for everyone at the meeting
                                for friend in self.friend_to_meet:
                                    self.add_contact(friend)
                                self.friend_to_meet = set()
                                self.obj_place = None


                # go back home
                elif 21 < self.model.DateTime.hour <= 23:  # Time to go home
                    if self.pos != self.house:
                        self.getWorld().space.move_agent(self, self.house)
                        # new_position = min(possible_steps,key=lambda c: euclidean(c, self.house))  # check shortest path to house
                        # else:  # agent at home
                        self.mask = Mask.NONE
            # Agent is self.quarantined
            elif self.quarantined is not None:
                if self.pos != self.house and self.obj_place is None:  # if has been tested, go home
                    self.getWorld().space.move_agent(self, self.house)
                    # new_position = min(possible_steps,key=lambda c: euclidean(c, self.house))  # check shortest path to house

                elif self.pos == self.house and self.obj_place is None:
                    self.mask = Mask.NONE

                elif self.obj_place is not None:

                    if self.obj_place != self.pos and 7 < self.model.DateTime.hour <= 23:  # if has to go testing, just go
                        # print(f"Agent {self.unique_id} on their way to testing")
                        self.getWorld().space.move_agent(self, self.obj_place)
                        # new_position = min(possible_steps, key=lambda c: euclidean(c, self.obj_place))

                    elif self.obj_place == self.pos:
                        # once at hospital, is tested and next step will go home to quarantine
                        self.mask = Mask.FFP2
                        h = self.model.getHospitalPosition(self.obj_place)
                        h.doTest(self)  # H-SUSC H-INF
                        self.obj_place = None

        # Ill agents move to nearest hospital to be treated
        elif self.machine.state == "H":
            if self.pos != self.obj_place:
                self.getWorld().space.move_agent(self, self.obj_place)
                # new_position = min(possible_steps,key=lambda c: euclidean(c, self.obj_place))  # check shortest path to hospital
            else:  # agent is at hospital
                self.mask = Mask.FFP2

        # if new_position: self.model.grid.move_agent(self, new_position)

    def contact(self):
        """ Find close contacts and infect """
        # self.getWorld().space._create_gdf()
        others = self.getWorld().space.agents_at(self.pos, max_num=10)  # pandas df [agentid, geometry, distance]
        others = others[(others['agentid'].str.contains('Human')) & (
                others['distance'] < 2)]
        others_agents = [self.model.space.get_agent(aid) for aid in others['agentid'] if aid != self.id]

        for other in others_agents:
            # if len(others):  # contact during daytime
            # for str_id in [x for x in others['agentid'] if x != self.id]:
            # index = next((i for i, item in enumerate(self.model.schedule.agents) if item.id == str_id), -1)
            # other = self.model.schedule.agents[index]

            # pTrans = self.model.virus.pTrans(self.mask, other.mask)
            # trans = np.random.choice([0, 1], p=[pTrans, 1 - pTrans])
            if other.machine.state is "S":  # trans == 0 and
                other.machine.contact(self.mask, other.mask)
                if other.machine.state == "E":
                    self.model.collector_counts['SUSC'] -= 1
                    self.model.collector_counts['EXP'] += 1
                    other.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')] = [0, round(
                        1 / other.machine.rate['rEI']) + round(1 / other.machine.rate['rIR']), 0]
                # other.machine.state = "E"
                # other.days_in_current_state = self.model.DateTime
                # other.exposing_time = dc.get_incubation_time(self.model)

    def look_for_friend(self):
        """ Check the availability of friends to meet and arrange a meeting """

        """for id_friend in self.friends:
            print(type(id_friend))
            index = next((i for i, item in enumerate(self.model.schedule.agents) if item.id == id_friend.id), -1)
            print(index)
            friend = self.model.schedule.agents[index]
            print(friend)
            if not friend.friend_to_meet and friend.quarantined is None and friend.state != State.HOSP and friend.state != State.DEAD:
                available_friends.append(index)"""

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
                friend_agent = self.model.space.get_agent(
                    random.sample(available_friends, 1)[0])  # gets one random each time
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

    # def adjust_init_stats(self, fr, to, state):
    #     self.model.collector_counts[fr] -= 1
    #     self.model.collector_counts[to] += 1
    #     self.machine.state = state

    def add_contact(self, contact):
        # check contacts for self agent
        if not self.model.DateTime.strftime('%Y-%m-%d') in self.contacts:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')] = {contact}  # initialize with contact

        else:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')].add(contact)  # add contact to now's date

        # add contacts of infected people for R0 calculations
        if self.machine.state in ["E", "A", "I"]:
            self.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')][0] += self.machine.prob_infection(self.mask,Mask.NONE) #self.model.virus.pTrans
            self.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')][2] += 1
