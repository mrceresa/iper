from mesa import Agent
from Covid_class import State, Mask
from Hospital_class import Workplace

import random
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean


class BasicHuman(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
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

    def __repr__(self):
        return "Agent " + str(self.unique_id)

    def move(self):
        new_position = None
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if self.state == State.SUSC or self.state == State.EXP or self.state == State.INF or self.state == State.REC:
            if self.quarantined is None:  # if agent not hospitalized or dead
                # sleeping time
                if self.model.get_hour() <= 6:
                    self.friend_to_meet = set()  # meetings are cancelled
                    self.obj_place = None

                # working time
                elif 6 < self.model.get_hour() <= 16:  # working time

                    if self.workplace is not None and self.pos != self.workplace.place:  # Employed and not at workplace
                        if self.model.get_hour() == 7 and self.model.get_minutes() == 0: self.mask = Mask.RandomMask()  # wear mask for walk
                        new_position = min(possible_steps,
                                           key=lambda c: euclidean(c,
                                                                   self.workplace.place))  # check shortest path to work


                    elif self.workplace is not None and self.pos == self.workplace.place:  # employee at workplace

                        self.mask = self.workplace.mask
                        cellmates = self.model.grid.get_cell_list_contents([self.pos])
                        if len(cellmates) > 1:
                            for other in [i for i in cellmates if i != self]:
                                if isinstance(other, BasicHuman):
                                    self.add_contact(other)


                # leisure time
                elif 16 < self.model.get_hour() <= 21:  # leisure time
                    if self.model.get_hour() == 17 and self.model.get_minutes() == 0: self.mask = Mask.RandomMask()  # wear mask for walk
                    if not self.friend_to_meet:
                        if np.random.choice([0, 1],
                                            p=[0.75, 0.25]): self.look_for_friend()  # probability to meet with a friend
                        new_position = self.random.choice(possible_steps)  # choose random step

                    else:  # going to a meeting
                        # check shortest step towards friend new position
                        new_position = min(possible_steps, key=lambda c: euclidean(c, self.obj_place))
                        cellmates = set(self.model.grid.get_cell_list_contents([self.pos]))
                        if self.pos == self.obj_place and self.friend_to_meet.issubset(
                                cellmates):  # wait for everyone at the meeting
                            for friend in self.friend_to_meet:
                                self.add_contact(friend)
                            self.friend_to_meet = set()
                            self.obj_place = None

                # go back home
                else:  # Time to go home
                    if self.pos != self.house:
                        new_position = min(possible_steps,
                                           key=lambda c: euclidean(c, self.house))  # check shortest path to house
                    else:  # agent at home
                        self.mask = Mask.NONE
            # Agent is self.quarantined
            elif self.quarantined is not None:
                if self.pos != self.house and self.obj_place is None:  # if has been tested, go home
                    new_position = min(possible_steps,
                                       key=lambda c: euclidean(c, self.house))  # check shortest path to house

                elif self.pos == self.house and self.obj_place is None:
                    self.mask = Mask.NONE

                elif self.obj_place is not None:

                    if self.obj_place != self.pos and 7 < self.model.get_hour() <= 23:  # if has to go testing, just go
                        # print(f"Agent {self.unique_id} on their way to testing")
                        new_position = min(possible_steps, key=lambda c: euclidean(c, self.obj_place))

                    elif self.obj_place == self.pos:
                        # once at hospital, is tested and next step will go home to quarantine
                        self.mask = Mask.FFP2
                        h = self.model.getHospitalPosition(self.obj_place)
                        h.doTest(self)
                        self.obj_place = None

        # Ill agents move to nearest hospital to be treated
        elif self.state == State.HOSP:
            if self.pos != self.obj_place:
                new_position = min(possible_steps,
                                   key=lambda c: euclidean(c, self.obj_place))  # check shortest path to work
            else:  # agent is at hospital
                self.mask = Mask.FFP2

        if new_position: self.model.grid.move_agent(self, new_position)

    def look_for_friend(self):
        """ Check the availability of friends to meet and arrange a meeting """
        available_friends = [self.model.schedule.agents[friend] for friend in self.friends if
                             not self.model.schedule.agents[friend].friend_to_meet and self.model.schedule.agents[
                                 friend].quarantined is None and self.model.schedule.agents[
                                 friend].state != State.HOSP and self.model.schedule.agents[friend].state != State.DEAD]
        peopleMeeting = int(self.random.normalvariate(self.model.peopleInMeeting,
                                                      self.model.peopleInMeetingSd))  # get total people meeting
        if len(available_friends) > 0:

            pos_x = [self.pos[0]]
            pos_y = [self.pos[1]]

            while peopleMeeting > len(
                    self.friend_to_meet) and available_friends:  # reaches max people in meeting or friends are unavailable
                friend_agent = random.sample(available_friends, 1)[0]  # gets one random each time
                available_friends.remove(friend_agent)

                self.friend_to_meet.add(friend_agent)
                friend_agent.friend_to_meet.add(self)

                pos_x.append(friend_agent.pos[0])
                pos_y.append(friend_agent.pos[1])

            # update the obj position to meet for all of them
            meeting_position = (int(sum(pos_x) / len(pos_x)), int(sum(pos_y) / len(pos_y)))
            self.obj_place = meeting_position

            for friend in self.friend_to_meet:
                friend.friend_to_meet.update(self.friend_to_meet)
                friend.friend_to_meet.remove(friend)
                friend.obj_place = meeting_position

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

    def contact(self):
        """ Find close contacts and infect """
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if isinstance(other, BasicHuman) and other != self:
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

    def adjust_init_stats(self, fr, to, state):
        self.model.collector_counts[fr] -= 1
        self.model.collector_counts[to] += 1
        self.state = state

    def add_contact(self, contact):
        # print(f"Agent {self.unique_id} has now added {contact.unique_id} as contact ! ")

        # check contacts for self agent
        if not self.model.DateTime.strftime('%Y-%m-%d') in self.contacts:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')] = {contact}  # initialize with contact

        else:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')].add(contact)  # add contact to now's date

        # print(f"Agent {self.unique_id} has now these contacts {self.contacts}")

    def step(self):
        """ Run one step taking into account the status, move, contact, and update_stats function. """
        # print("DIA: "+ str(self.model.get_day())+" hora:"+str(self.model.get_hour())+":"+str(self.model.get_minutes()))
        # if self.unique_id < 5: print(f"I am agent {self.unique_id}, I have been {self.state} for {self.days_in_current_state.day}")
        self.status()
        self.move()
        self.contact()
        self.update_stats()
        # print(f"Agent {self.unique_id} with friends {self.friend_to_meet} at place {self.obj_place}")
