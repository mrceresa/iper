#!/usr/bin/python
import argparse
import os, sys, time
from datetime import datetime
import logging
import polisandbox as polis

from pyPandora import Config, SizeInt, Point2DInt
from polisandbox import MultiEnvironmentWorld, EnvironmentFactory, PopulationRequest
from polisandbox.xworlds import XAgent
from polisandbox.brains import BaseBrain
from polisandbox import Action, Event, RewardRule
from polisandbox.behaviours.behaviours import Harvest

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', 
          datefmt='%Y/%m/%d %I:%M:%S %p', 
          level=logging.DEBUG)

import numpy as np

def getNeighs(pos):
  ns = np.asarray([(-1,1), (0,1), (1,1),
       (-1,0), (1,0),
       (-1,-1), (0,-1), (1,-1)])
      
  pns = []
  for n in ns:
    _p = pos + n                 
    pns.append(Point2DInt(_p[0], _p[1]) )
  return pns

class MoveTo(Action):
  def __init__(self, direction):
    super(MoveTo, self).__init__()
    self._dir = direction
    
  def do(self, agent):
    new_pos = Point2DInt(
      agent.position._x + self._dir[0], 
      agent.position._y + self._dir[1]
                 )
    if agent.getWorld().checkPosition(new_pos):
      agent.position = new_pos
      
  def __str__(self):
    return "MoveTo%d_%d"%(self._dir[0],self._dir[1])
    
class MonkeyBrain(BaseBrain):
  def __init__(self, agent):
    super(MonkeyBrain, self).__init__(agent, model="Q")
    self._model._nA = len(self._agent.getPossibleActions())
    self._model._nS = len(self._agent.getPossibleStates())    
    self._model.reset()

  def think(self, status, reward):
    print(status, reward)
    self._actionsmap = {
      "MoveUp": [MoveTo((0,1))],
      "MoveDown":[MoveTo((0,-1))],
      "MoveLeft":[MoveTo((-1,0))],
      "MoveRight":[MoveTo((1,0))],
      "Still":[MoveTo((0,0))],                  
      "Harvest":[Harvest()]      
    }          
    _action = super(MonkeyBrain, self).think(status, reward)    
    _a = self._agent.getPossibleActions()[_action]
    return self._actionsmap.get(_a, [])

class FoodAgent(XAgent):
  def _postInit(self, *argv, **kargv): pass
  def updateState(self): pass
  def __repr__(self):
    return "Food"

class MonkeyAgent(XAgent):
  def __init__(self, id):
    super(MonkeyAgent, self).__init__(id)
    
  def _postInit(self, *argv, **kargv):
    self._brain = MonkeyBrain(self)

  def updateState(self):
    super(MonkeyAgent, self).updateState()

  @property
  def lookAround(self):
    neighs = getNeighs((self.position._x, self.position._y))
    around = []
    for _n in neighs:
      _a = [self.getWorld().getAgent(aid) 
        for aid in self.getWorld().getAgentIds(_n, "all")]        
      around.append(tuple(_a))
    return tuple(around)

class ForestWorld(MultiEnvironmentWorld):
  def __init__(self, config, output_dir):
    MultiEnvironmentWorld.__init__(self, config, output_dir)

    self.addRewardRule(RewardRule(Event("Monkey", "death"), 
                            ["a"], 
                            [-10])
                            )
    self.addRewardRule(RewardRule(Event("Monkey", "step"), 
                            ["a"], 
                            [-0.1])
                            )
    self.addRewardRule(RewardRule(Event("Monkey", "perform_eat", "food"), 
                            ["a"], 
                            [1.0])
                            )                                
    
  def stepEnvironment(self):    
    super(ForestWorld, self).stepEnvironment() 
    
  def createRasters(self):
    self.l.info("Creating rasters")  
    super(ForestWorld, self).createRasters()
    self.registerDynamicRaster("landCover", True)    
    
  def createAgents(self):
    self.l.info("Requesting agents for the simulation.")
    
    pr = PopulationRequest()
    pr._data = {
        "Monkey": {"num": 1,
                    "type": MonkeyAgent,
                    "templates": ["MonkeyAgent"]
                    },
        "Banana": {"num": 100,
                    "type": FoodAgent
                    }                    
        }
    
    # Prepare agents for creation        
    _created = self.addPopulationRequest(pr)
    # Request their addition to the world
    super(ForestWorld, self).createAgents()
    # Finish initialization of world-dependent data

def startSimulation(size, numTimeSteps, odir):
  _l = logging.getLogger(__name__)
  datadir = os.path.join(odir, "data")
  logfile = os.path.join(datadir, "results.h5")

  if os.path.exists(datadir):
    shutil.rmtree(datadir)
  
  os.makedirs(datadir)     
  
  worldSize = SizeInt(size, size)
  myConfig = Config(worldSize, numTimeSteps, logfile)
  

  _l.info("Start simulation")
  _l.info("Output in %s"%odir)    
  
  fw = ForestWorld(myConfig, datadir)
  ef = EnvironmentFactory()
  ef.load()        

  _m = ef.get("BasalMetabolism")
  fw.add_env(_m)

  fw.initialize()
  fw.run()  

  _l.info("End of simulation")  
  _l.info("Total agents created %d"%fw._totCreated)
  _l.info("Total agents destroyed %d"%fw._totDestroyed)  
  


def main(args, odir):

  i = 0
  while i < args.n_sims:
    startSimulation(
          args.size_raster, args.n_steps, odir)
     
    i = i + 1
  

      

if __name__ == "__main__":

  currDir = os.getcwd()

  parser = argparse.ArgumentParser()
  parser.add_argument("--n_sims", type=int, default=1)
  parser.add_argument("--n_steps", type=int, default=50)
  parser.add_argument("-r", "--size_raster", type=int, default=64)  
  parser.add_argument("-o","--output_dir", type=str, default=currDir)              
  args = parser.parse_args()

  startTime = time.time()
  date = datetime.fromtimestamp(startTime)
  odir = os.path.join(args.output_dir, "exps/%s"%date.strftime("%Y%m%d%H%M%S"))

  main(args, odir)
