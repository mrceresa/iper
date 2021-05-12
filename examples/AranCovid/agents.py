from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random
from attributes_Agent import age_
from Covid_class import State, Mask

class RandomWalk(Action):
  def do(self, agent):
    _xs = agent.getWorld()._xs
    dx, dy =  _xs["dx"], _xs["dy"] #How much is 1deg in km?
    # Convert in meters
    dx, dy = (dx/1000, dy/1000)

    new_position = ( agent.pos[0] + random.uniform(-dx, dx), agent.pos[0] + random.uniform(-dy,dy) )

    if not agent.getWorld().out_of_bounds(new_position):
      agent.getWorld().space.move_agent(agent, new_position)

class HumanAgent(XAgent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id)
    self.age = age_()
    self.state = State.SUSC
    self.mask = Mask.NONE
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




  def _postInit(self):
    pass

  def think(self):
    possible_actions = [RandomWalk()]
    return possible_actions

  def step(self):
    self.l.debug("*** Agent %s stepping"%str(self.id)) 
    _actions = self.think()
    _a = random.choice(_actions)
    _a.do(self)
    others = self.getWorld().space.agents_at(self.pos, max_num=2)
    #print(f"Agent {self.id} at pos {self.pos}")
    #print(f"Agent {self.unique_id} is {self.age} and lives in {self.house}")
    super().step()      

  def __repr__(self):
    return "Agent " + str(self.id)
