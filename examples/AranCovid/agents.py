from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random, numpy as np
from attributes_Agent import age_
from Covid_class import State, Mask
from SEAIHRD_class import SEAIHRD_covid


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
        self.age = age_()
        self.state = State.SUSC
        self.mask = Mask.NONE

        self.machine = SEAIHRD_covid(unique_id, "S", 20, self.age)
        # variable to calculate time passed since last state transition
        self.days_in_current_state = model.DateTime
        self.presents_virus = False  # for symptomatic and detected asymptomatic people
        self.quarantined = None

        self.friends = set()
        self.contacts = {}  # dict where keys are days and each day has a list of people

        self.workplace = None  # to fill with a workplace if employed
        self.obj_place = None  # agent places to go
        self.friend_to_meet = set()  # to fill with Agent to meet

        self.HospDetected = False
        self.R0_contacts = {}

    def __repr__(self):
        return "Agent " + str(self.id)

    def _postInit(self):
        pass

    def think(self):
        possible_actions = [RandomWalk()]
        return possible_actions

    def step(self):
        self.l.debug("*** Agent %s stepping" % str(self.id))
        _actions = self.think()
        _a = random.choice(_actions)
        _a.do(self)

        # if self.state is State.EXP or self.state is State.INF:
        self.contact()

        # print(f"Agent {self.id} at pos {self.pos}")
        # print(f"Agent {self.unique_id} is {self.age} and lives in {self.house}")
        super().step()

    def contact(self):
        """ Find close contacts and infect """
        others = self.getWorld().space.agents_at(self.pos, max_num=5)  # pandas df [agentid, geometry, distance]
        others = others[(others['agentid'].str.contains('Human')) & (
                others['distance'] < 100)]  # filter out buildings and far away people .iloc[0:2]

        if len(others):  # and self.model.DateTime.hour > 7:
            for other in [x.split(',') for x in others['agentid'] if x != self.id]:
                pass
                # pTrans = self.model.virus.pTrans(self.mask, other.mask)
                # trans = np.random.choice([0, 1], p=[pTrans, 1 - pTrans])
                # if trans == 0 and other.state is State.SUSC:
                #     self.model.collector_counts['SUSC'] -= 1
                #     self.model.collector_counts['EXP'] += 1
                #     other.state = State.EXP
                #     other.days_in_current_state = self.model.DateTime
                #     other.exposing_time = self.model.get_incubation_time()
                #     other.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')] = [0,
                #                                                                    other.exposing_time + self.model.virus.infection_days,
                #                                                                    0]
