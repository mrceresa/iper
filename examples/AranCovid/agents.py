from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random, numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from SEAIHRD_class import SEAIHRD_covid, Mask
import haversine as hs
import DataCollector_functions as dc
from geopandas.geodataframe import GeoDataFrame
from random import uniform, randint
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
    def do(self, agent, pos):
        #print("***************************",pos)
        if agent.pos != pos:
            agent.getWorld().space.move_agent(agent, pos)
            
 
class StandStill(Action):
    def do(self, agent):
        pass

class HumanAgent(XAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id)
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
        self.goal=""
        self.friend_to_meet = set()  # to fill with Agent to meet

        self.HospDetected = False
        self.R0_contacts = {}

    def __repr__(self):
        return "Agent id " + str(self.id)

    def _postInit(self):
        self.obj_place = self.house


    def working_time(self):

        
        
        if not self.workplace: 
            return self.obj_place
        
        workplace = self.model.space.get_agent(self.workplace)
        
        # TODO: Ask Aran why we use a different mask
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

    def think(self, step, curr_time, cellmates):

        chosen_action, chosen_action_pars = StandStill(), [self]

        if not self.mask and self.pos != self.house:
            self.mask = Mask.RandomMask(self.model.masks_probs)  # wear mask for walk

        # If we have a previous goal, go to it
        if self.obj_place != self.pos:
            return MoveTo(), [self, self.obj_place]

        if self.quarantined: 
            if self.obj_place == self.pos and self.goal == "GO_TO_HOSPITAL":
                # once at hospital, is tested and next step will go home to quarantine
                self.mask = Mask.FFP2
                h = self.model.getHospitalPosition(self.obj_place)
                h[0].doTest(self)

        self.obj_place = self._thinkGoal()

        # If we have a new goal execute it
        if self.obj_place != self.pos:
            return MoveTo(), [self, self.obj_place]

        if self.pos == self.house and self.obj_place == self.pos:
            self.mask = Mask.NONE

        if self.machine.state in ["H", "D"]:
            pass

        return chosen_action, chosen_action_pars


    def step(self):
        self.l.debug("*** Agent %s stepping" % str(self.id))
        #print( "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",str(self.id),self.pos,"XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        cellmates = self.getWorld().space.agents_at(self.pos, radius=2.0/111100)  # pandas df [agentid, geometry, distance]
        #import ipdb
        #ipdb.set_trace()
        cellmates = cellmates[(cellmates['agentid'].str.contains('Human'))]  # filter out buildings
        if self.machine.state in ["I", "A"]:
            self.contact(cellmates)
            for str_id in [x for x in cellmates["agentid"] if x != self.id]:
                self.add_contact(str_id)

        #cellmates = self.get_cellmates()
        _n = set([_aid for _aid in cellmates["agentid"]])
        _d = _n.difference(self.family)
        #if len(_d) > 1: print(len(_d))

        action, a_pars = self.think(self.model.currentStep, self.model.DateTime, cellmates)

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

        t = self.machine.time_in_state
        s = self.machine.state

        if not self.model.DateTime.strftime('%Y-%m-%d') in self.contacts:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')] = {contact}  # initialize with contact

        else:
            self.contacts[self.model.DateTime.strftime('%Y-%m-%d')].add(contact)  # add contact to now's date

        # add contacts of infected people for R0 calculations
        if self.machine.state in ["E", "A", "I"]:
            contacts_mask = self.model.space.get_agent(contact).mask
            pTransMask1 = self.mask.maskPtrans(self.mask)
            pTransMask2 = contacts_mask.maskPtrans(contacts_mask)
            #print(self.R0_contacts)
            #print("************************",self.model.DateTime.strftime('%Y-%m-%d'))

            if not self.model.DateTime.strftime('%Y-%m-%d') in self.R0_contacts: 
                self.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')] = [0,0,0]  # initialize

            self.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')][0] += self.machine.prob_inf * pTransMask1 * pTransMask2  #prob_infection(self.mask, contacts_mask)  # self.model.virus.pTrans
            self.R0_contacts[self.model.DateTime.strftime('%Y-%m-%d')][2] += 1
