from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random

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
    super().step()      

  def __repr__(self):
    return "Agent " + str(self.id)
