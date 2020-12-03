
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
import random
import numpy as np

import os

from iper import GeoSpacePandas
from iper import XAgent
from iper.behaviours.actions import MoveTo, Eat
from iper.brains import ScriptedBrain

import logging
_log = logging.getLogger(__name__)

class CatBrain(ScriptedBrain):
  def __init__(self, agent):
    super().__init__(agent)
    #self._model._nA = len(self._agent.getPossibleActions())
    #self._model._nS = len(self._agent.getPossibleStates())    
    #self._model.reset()

  def think(self, status, reward):

    cellmates = self._agent.model.grid.get_cell_list_contents([self._agent.pos])
    if len(cellmates) > 1:
      food = [_c for _c in cellmates if type(_c) == FoodAgent ]
      if food:
        return [Eat(food[0])]

    self._actionsmap = {
      "MoveUp": [MoveTo((0,1))],
      "MoveDown":[MoveTo((0,-1))],
      "MoveLeft":[MoveTo((-1,0))],
      "MoveRight":[MoveTo((1,0))],
      "Still":[MoveTo((0,0))]                  
    }          
    _action = random.choice(list(self._actionsmap.keys()))
    return self._actionsmap.get(_action)

class FoodAgent(XAgent):
  def _postInit(self, *argv, **kargv): 
    self._envSetVal("FoodProduction","food", 1)
  
  def step(self):
    if (float(self._envGetVal("FoodProduction","food")) <= 0):
      self.removeAndClean("Eaten up!")

  def __repr__(self):
    return "Cheese"

class Cat(XAgent):
  def __init__(self, unique_id, model):
    XAgent.__init__(self, unique_id, model)

  def _postInit(self, *argv, **kargv):
    _log.debug("*** Agent %s postInit called"%self.id) 
    
  def step(self):
    self.updateState()  

class Maze(MultiEnvironmentWorld):

  def __init__(self, N, config={"size":{"width":2,"height":2,"torus":False}}):
    super().__init__(config)
    _log.info("Initalizing model")   
    self.schedule = RandomActivation(self)
    _size = self.config["size"]
    self.grid = MultiGrid(_size["width"], _size["height"], _size["torus"])
    self.createAgents()

  def createAgents(self):
    self.l.info("Requesting agents for the simulation.")
    
    pr = PopulationRequest()
    pr._data = {
        "Cat": {"num": 1,
                    "type": Cat,
                    "templates":["BasicAnimal"],
                    "brain": CatBrain
                    },
        "Cheese": {"num": 2,
                    "type": FoodAgent
                    }                    
        }
    
    # Prepare agents for creation        
    _created = self.addPopulationRequest(pr)
    # Request their addition to the world
    super().createAgents()
    # Finish initialization of world-dependent data
     
  def plotAll(self):
    pass
    
  def run_model(self, n):
    self.info()
    self.currentStep = 0
    self.l.info("***** STARTING SIMULATION!!")
    for i in range(n):
      self.stepEnvironment()
      self.currentStep+=1
      self.schedule.step()
      _log.info("Step %d of %d"%(i, n))

    self.l.info("***** FINISHED SIMULATION!!")

    self.info()
  
  def info(self):
    self.l.info("*"*3 + "This is world %s",self.__class__.__name__ + "*"*3)

    self.l.info("*"*3 + "With %d agents: types and num %s"%(len(self._agentsById), str(self._agents)))
    self.l.info("*"*3 + "With %d environments: types and num %s"%(len(self._envs), str(self._envs)))

    for _a in self.getAgents():
      _log.debug("*"*3 + _a.info())    

    self.l.info("*"*3 + "...BYE!" + "*"*3)
