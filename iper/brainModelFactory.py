import logging
import numpy as np
import operator
from iper.behaviours.actions import TestAction

class BrainModel(object):
  def __init__(self, name):
    self.l = logging.getLogger(self.__class__.__name__)
    self._name = name
    self._rewards = []
    self._nS = 1 # Num of states
    self._nA = 1 # Num of Actions    

  def train(self, ):
    self.l.info("Training model %s with %d new data"%(self._name, len(self._rewards)))
    
  def policy(self, status):
    return [] 
    
class QLearnBrainModel(BrainModel):
  def __init__(self, name):
    super(QLearnBrainModel, self).__init__(name)
    self._pol = {}
    self.reset()

  def reset(self):
    self._Q = np.zeros((self._nS, self._nA))
    self._pol = {}
    
  def train(self):
    self.l.info("Training model %s with %d new data"%(self._name, len(self._rewards)))
    self.l.info("Q: %s"%(str(self._Q)))
    self.l.info("pol: %s"%(str(self._pol)))    

  def policy(self, status):
    self.l.debug("Inside policy with status %s"%str(status))
    status = tuple(status) #List is not hashable
    _actions = self._pol.setdefault(status, [0 for _ in range(self._nA)])
    index, value = max(enumerate(_actions), key=operator.itemgetter(1))
    return index

class BrainModelFactory(object):
  def __init__(self):
    self.l = logging.getLogger(self.__class__.__name__)    
    self._models = {"default":BrainModel("default"),
                    "goap":BrainModel("goap"),
                    "Q":QLearnBrainModel("Q")}    

  def train(self):
    self.l.info("Updating models using batches...")
    for _m in self.get_all():
      _model = self.get(_m)
      _model.train()

  def get_all(self):
    return self._models.keys()
    
  def load_all(self):
    pass
    #for _a in behaviours.Action.__subclasses__():
    #  self.register(_a.__name__, _a(None))

  def get(self, name):
    return self._models.get(name.lower(), None)

  def register(self, name, action):
    if action is None: return False
    name = str(name)
    if name in self._models:
      self.l.error("A model with name %s is already registered"%name) 
      return False   
    self._models[name] = action
          
  def list_all(self):
    self.l.info("* This factory contains %d models"%len(self._models))
    for i, k in self._models.items():
      self.l.info("**** %s: %s"%(i, k)) 
