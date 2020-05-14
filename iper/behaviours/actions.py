
import random
import logging
from ..xmlobjects import XMLObject

class Action(object):
  def __init__(self, name=None):
    if not name: name = self.__class__.__name__
    #XMLObject.__init__( self, name)
    self.name = name
    self.l = logging.getLogger(name)        

class TestAction(Action):
  def do(self, agent):
    print("Executed Test Action")
    
class AgingBehaviour(Action):
    
  def do(self, agent):
    age = int(agent._envGetVal("Demography","age"))
    agent._envSetVal("Demography","age",age+1)

  
class DieOldBehaviour(Action):
    
  def do(self, agent):
    age = int(agent._envGetVal("Demography","age"))
    max_age = int(agent._envGet("Demography","age","max"))
    if age > max_age:
      agent.removeAndClean("old age")
      
class DieOfStarvation(Action):
    
  def do(self, agent):
    en = int(agent._envGetVal("BasalMetabolism","energy"))
    min_en = int(agent._envGet("BasalMetabolism","energy","min"))
    if en < min_en:
      agent.removeAndClean("starvation")
    
class ConsumeBasal(Action):
    
  def do(self, agent):
    en = int(agent._envGetVal("BasalMetabolism","energy"))
    agent._envSetVal("BasalMetabolism","energy",en-1)    
    
class Eat(Action):
    
  def do(self, agent):
    location = agent.position
    world = agent.getWorld()
    _food = int(agent._envGetVal("FoodProduction","food"))
    if _food > 0:
      en = int(agent._envGetVal("BasalMetabolism","energy"))
      agent._envSetVal("BasalMetabolism","energy",en + 1)
      agent._envSetVal("FoodProduction","food", _food - 1) > 0

class Harvest(Action):
    
  def do(self, agent):
    location = agent.position
    world = agent.getWorld()
    raster_name = 'FoodProduction_food'
    has_food = world.getValue(raster_name, location)
    if has_food:
      _food = int(agent._envGetVal("FoodProduction","food"))
      agent._envSetVal("FoodProduction","food",_food + 1)
      world.setValue(raster_name, location, has_food - 1)

class RandomWalk(Action):

  def do(self, agent):

    new_pos = (
      agent.position._x + random.randint(-1, 2), 
      agent.position._y + random.randint(-1, 2)
                 )
    if agent.getWorld().checkPosition(new_pos):
      agent.position = new_pos

class TouchAndInfect(Action):

  def do(self, agent):
    _w = agent.getWorld()
    # Check all agents in the same cell
    for aid in _w.getAgentIds(agent.position, "all"):
      a = _w.getAgent(aid)
      if a is None: continue
      if a._envGetVal("SIRContagion","status") == "susceptible":
        infect_prob = float(agent._envGetVal("SIRContagion","infect_prob"))
        if random.random() < infect_prob:
          a._envSetVal("SIRContagion","status","infected")
          print(agent, "infected", a)

class RecoverOrDie(Action):

  def do(self, agent):
      if agent._envGetVal("SIRContagion","status") == "infected":
        recover_prob = float(agent._envGetVal("SIRContagion","recover_prob"))
        if random.random() < recover_prob:
          agent._envSetVal("SIRContagion","status","recovered")
        else:
          h = int(agent._envGetVal("SIRContagion","health"))
          h -= 1
          if h < agent._envGetVal("SIRContagion","min"):
            agent.removeAndClean("infection")          
          else:
            agent._envSetVal("SIRContagion","health",str(h))            
          


class ProductionRule(object):
  def __init__(self):
    pass
  
  def _has(self, obj):
    return True
  
  def _produce(self, obj, quant):
    pass
    
  def _consume(self, obj, quant):
    pass    

  def _check_broken(self, obj):
    return False
      
class Farm(ProductionRule):
   
  def do(self, agent):
    if self._has("wood") and self._has("tools"):
      self._produce("food", 4)
      self._consume("wood", 1)
      self.__check_broken("tools")
    elif self._has("wood") and not self._has("tools"):
      self._produce("food", 2)
      self._consume("wood", 1)                 

    
 
