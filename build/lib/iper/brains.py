import logging
from .xmlobjects import XMLObject
from . import _brainModelFactory

class WorldState(object):
  def __init__(self):
    pass
    
class BaseBrain(XMLObject):
  def __init__(self, agent, model="default"):
    XMLObject.__init__( self)
    self.l = logging.getLogger(self.__class__.__name__)        
    self._xd = XMLObject('Brain')
    self._agent = agent
    self._avActions = []
    self._model = _brainModelFactory.get(model)
  
    # Virtual function, implement in children classes.
  def think(self, status, reward):
    #self._model.updateActions(self._avActions)
    self._model._rewards.append(
      (self._agent.id,
       self._agent.getWorld().currentStep,
       self._agent.position,
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
    BaseBrain.__init__( self)        
    
    
class GOAPBrain(BaseBrain):
  def __init__(self, agent):
    BaseBrain.__init__( self)      
