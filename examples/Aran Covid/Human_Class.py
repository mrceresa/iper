from mesa import Agent
from Covid_class import State
from Hospital_class import Workplace

import random
import numpy as np
from scipy.spatial.distance import euclidean


class BasicHuman(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.house = None
        self.state = State.SUSC
        self.days_in_current_state = 0  # variable to calculate time passed since last state transition
        self.friends = set()
        self.workplace = None  # to fill with the coordinates of the workplace
        self.obj_place = None  # to fill with the coordinates of the meetings

    def move(self):
        """Move the agent according to the day time and employment condition """
        if 22 > self.model.get_hour() > 6: # agents are awake and ready to go to work
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

            if 6 > self.model.get_hour() > 17 and self.workplace is not None: # go and stay at work if employed

                if self.pos != self.workplace:
                    new_position = min(possible_steps, key=lambda c: euclidean(c,self.workplace))  # check shortest path to work
                    self.model.grid.move_agent(self, new_position)

            else: #leisure time or unemployed

                # print(f'Agent {self.unique_id} is following {self.obj_place}')
                if self.obj_place is None:
                    if np.random.choice([0, 1], p=[0.75, 0.25]): self.look_for_friend()  # probability to meet with a friend
                    new_position = self.random.choice(possible_steps)  # choose random step

                else:
                    new_position = min(possible_steps, key=lambda c: euclidean(c,
                                                                               self.obj_place.pos))  # check shortest step towards friend new position
                    if self.pos == self.obj_place.pos:
                        self.obj_place.obj_place = None
                        self.obj_place = None

            self.model.grid.move_agent(self, new_position)

    def look_for_friend(self):
        """ If the agent is in their free time, looks for a bored friend to meet. """
        id_friend = random.sample(self.friends, 1)[0]  # gets one random each time
        friend_agent = self.model.schedule.agents[id_friend]

        my_iter = iter(self.friends)
        while friend_agent.obj_place is not None:
            try:
                # first_friend = next(iter(self.friends))          #get first one
                id_friend = next(my_iter)  # gets one random each time
                friend_agent = self.model.schedule.agents[id_friend]
            except StopIteration:
                friend_agent = None
                break

        if friend_agent is not None:
            self.obj_place = friend_agent
            friend_agent.obj_place = self

    def status(self):
        """Check for the infection status"""

        t = self.model.schedule.time - self.days_in_current_state

        if self.state == State.EXP:
            if t >= self.exposing_time:
                self.model.collector_counts["EXP"] -= 1  # Adjust initial counts
                self.model.collector_counts["INF"] += 1
                self.infecting_time = self.model.get_infection_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.INF

        elif self.state == State.INF:
            severe_rate = self.model.virus.severe_rate
            not_severe = np.random.choice([0, 1], p=[severe_rate, 1 - severe_rate])

            if not_severe == 0:
                self.model.collector_counts["INF"] -= 1  # Adjust initial counts
                self.model.collector_counts["HOSP"] += 1
                self.hospitalized_time = self.model.get_severe_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.HOSP
                # self.model.schedule.remove(self)

            if not_severe != 0 and t >= self.infecting_time:
                self.model.collector_counts["INF"] -= 1  # Adjust initial counts
                self.model.collector_counts["REC"] += 1
                self.immune_time = self.model.get_immune_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.REC

        elif self.state == State.REC:
            if t >= self.immune_time:
                self.model.collector_counts["REC"] -= 1  # Adjust initial counts
                self.model.collector_counts["SUSC"] += 1
                self.state = State.SUSC

        elif self.state == State.HOSP:
            death_rate = self.model.virus.death_rate
            alive = np.random.choice([0, 1], p=[death_rate, 1 - death_rate])
            if alive == 0:
                self.model.collector_counts["HOSP"] -= 1  # Adjust initial counts
                self.model.collector_counts["DEAD"] += 1
                self.state = State.DEAD
                # self.model.schedule.remove(self)

            if alive != 0 and t >= self.hospitalized_time:
                self.model.collector_counts["HOSP"] -= 1  # Adjust initial counts
                self.model.collector_counts["REC"] += 1
                self.immune_time = self.model.get_immune_time()
                self.days_in_current_state = self.model.schedule.time
                self.state = State.REC

    def contact(self):
        """ Find close contacts and infect """
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for other in cellmates:
                if self.random.random() > self.model.virus.ptrans:
                    continue
                if self.state is State.INF or self.state is State.EXP and other.state is State.SUSC:
                    other.state = State.EXP
                    other.days_in_current_state = self.model.schedule.time
                    other.exposing_time = self.model.get_incubation_time()

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

    def step(self):
        """ Run one step taking into account the status, move, contact, and update_stats function. """
        #print("DIA: " + str(self.model.get_day()) + " hora:" + str(self.model.get_hour()))
        self.status()
        self.move()
        self.contact()
        self.update_stats()
