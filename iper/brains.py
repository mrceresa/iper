from loguru import logger
from .xmlobjects import XMLObject
from . import _brainModelFactory

class Reward(object):
  def __init__(self):
    pass

class WorldState(object):
  def __init__(self):
    pass
    
class BaseBrain(XMLObject):
  def __init__(self, agent, model="default"):
    XMLObject.__init__( self)
    self._xd = XMLObject('Brain')
    self._xd.set("model", model)
    self._xd.set("agent", agent.id)
    self._agent = agent
    self._avActions = []
    self._model = _brainModelFactory.get(model)
  
    # Virtual function, implement in children classes.
  def think(self, status, reward):
    #self._model.updateActions(self._avActions)
    self._model._rewards.append(
      (self._agent.id,
       self._agent.getWorld().getStep(),
       self._agent.pos,
       reward       
      )
    )
    _actions = self._model.policy(status)
    return _actions
    
  # Virtual function, implement in children classes.
  # Execute decision making actions, that need to be done before taking final actions.
  def executeDecisionMakingActions(self, actionsToDo):
      return
    
  # Virtual function, implement in children classes.
  # Update agent after actions have been executed.
  def update(self):
      return
        
        
class ScriptedBrain(BaseBrain):
  def __init__(self, agent):
    super().__init__( agent )
    self._xd.set("type","Scripted")        
    
class GOAPBrain(BaseBrain):
  def __init__(self, agent):
    super().__init__( agent, model="goap" )
    self._xd.set("type","GOAP")
