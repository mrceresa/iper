#!/usr/bin/env python

import os, sys
from subprocess import call
from glob import glob

import random, math, time

from .xmlobjects import XMLObject, toStr, fromXmlFile
#from pyPandora import Agent, World, Point2DInt, SizeInt, Config
from mesa import Agent, Model

import uuid
import logging

from copy import deepcopy
from iper import _sandbox_defs, _agentTemplateFactory, _environmentFactory
from iper import BrainModelFactory
from iper.brains import BaseBrain
import inspect
import io, traceback

from .brains import BaseBrain, WorldState
from .behaviours.actions import Action

import pandas as pd

class PopulationRequest(object):
  def __init__(self):
    self._data = {}

class Event(object):
  def __init__(self, a, descr, b=None):
    self._a = a
    self._type = descr
    self._b = b    

class RewardRule(object):
  def __init__(self, event, receivers, values):
    self._ev = event
    self._rec = receivers        
    self._values = values    

class Sensor(XMLObject):
  def getWorldState(self, agent, world):
    state = [
      world.currentStep,
      agent.money,
      
    ]
    return state

class XAgent(Agent):
  """
    Extends pandora's agent with xml annotations a brain and a body
    which is capable of ranged perceptions and actions
  """
  @staticmethod
  def fromXMLFile(fname):
    agent = XAgent("0")
    agent._xd = agent._xd.createObjectFromXmlFile(fname)
    print(agent._xd)
    return agent
    
  def __init__(self, id, model=None):
    Agent.__init__( self, id, model)
    #_configure_logging()
    self.l = logging.getLogger(self.__class__.__name__)        
    self._xd = XMLObject('Agent')
    self._meta = self._xd.add("metadata")
    self._meta.el.set("id", str(id))
    self._events = self._xd.add("events")
    self._envdata = self._xd.add("environments")
    self._body = self._xd.add("body")
    self._behaviours = []
    self._sensors = [Sensor()]
    self._next_reward = 0.0
    self._last_reward = 0.0
    self.avStates = []
    self.avActions = []
    self.exists = False
    self._brain = BaseBrain(self)
    self.pos = (0,0)

  def _postInit(self):
    self.l.warn("You are calling virtual XAgent._postInit()")
    self.l.warn("This is to allow late initialization of you agent code")            
    self.l.warn("Consider overriding it in you agent")                

  @property
  def id(self):
    return self._meta.el.get("id")

  def getWorld(self):
    return self.model

  def toXmlFile(self, odir="./", suffix=""):
    fname = os.path.join(odir, "%s-%s.xml"%(self.id, suffix))
    self._xd.toXmlFile(fname)

  def getPossibleActions(self):
    return self.avActions

  def getPossibleStates(self):
    return self.avStates 

  def getBehaviours(self):
    return self._behaviours

  def setMetadata(self, _d):
    for k,v in _d.items():
      self._meta.el.set(str(k),str(v))
      
  def addTemplates(self, templates):
    for _t in templates:
      self.l.info("Adding template %s"%_t)
      _xd2 = _agentTemplateFactory.instantiate(_t)
      if (_xd2 is not None) and (len(_xd2) > 0):
        _att = _xd2.find("attributes").getchildren()
        if len(_att):
          self.l.error(" addTemplate.attributes NOT IMPLEMENTED")

        _evs = _xd2.find("events").getchildren()
        if len(_evs):
          self.l.error("addTemplate.events NOT IMPLEMENTED")
        
        _envs = _xd2.find("environments").getchildren()
        for _e in _envs:
          self.model.add_env(_environmentFactory.get(_e.tag))

      else:
        raise RuntimeError("Template %s is not defined"%_t)        
    
  def __getattr__(self, name):
    #print("Accessing XML attribute", name)
    res = self._xd.rootNode.el.xpath("attributes/%s"%name)
    if res:
      _attr = res[0]
      _v = _attr.get("val")
      _t = _attr.get("type")
      if _t == "int": _v = int(_v)
      if _t == "list": _v = map(str.strip, _v.split(","))

      return _v
    else:
      self.l.error("Failed to get attribute from XML: %s"%toStr(self._xd.rootNode.el))
      raise RuntimeError("Attribute %s is not defined for object of class %s"%(name, self.__class__.__name__))
  
  # This method is always needed or we have errors  
  def serialize(self):
    return

  def registerAttributes(self):    
    return

  def _notifyDeath(self):
    self.getWorld().notify(Event(self.id, "death"))

  def removeAndClean(self, reason):
    death = self._events.add("death")
    death.el.attrib["cause"] = reason
    death.el.attrib["simStep"] = str(self.getWorld().currentStep)
    if reason not in ["old age", "starvation"]:
      self.l.debug("Agent %s dead because of %s"%(self.id, reason))
    if self.exists:
      self._notifyDeath()


  def getReward(self):
    return self._next_reward
    
  def getLastReward(self): 
    return self._last_reward
    
  def addReward(self, reward):
    self.l.debug("Agent %s received reward %f"%(self.id, reward))          
    self._next_reward += reward

  def step(self):
    self.updateState()

  def updateState(self):
    #self._brain.setActions(self._behaviours)
    # Generate perceptions
    #status = [s.getWorldState(self.getWorld()) for s in self._sensors]
    status = [getattr(self, _s) for _s in self.getPossibleStates() if _s]

    toDo = self._brain.think(status, self.getReward())
    
    self.getWorld().notify(
              Event(self.id, "step")
            )
            
    for action in toDo:
      # Some actions may destroy the agent in the middle of the loop
      # if this is the case just return from this ghost shell!
      if not self.exists: return 
      self.l.debug("Agent %s executing action: %s"%(self.id, str(action)))
      try:
        self.getWorld().notify(
              Event(self.id, "perform_"+str(action.__class__.__name__))
            )      
        action.do(self)
      except Exception as e:
        exc_buffer = io.BytesIO()
        traceback.print_exc(file=exc_buffer)
        self.l.error(
                  'Uncaught exception running action %s:\n %s'
                  %(str(action), exc_buffer.getvalue()))

      for _behav in self.getBehaviours():
        if type(_behav) is not str:
          if not self.exists: return       
          self.l.debug("Agent %s executing behaviour: %s"%(self.id, _behav.__class__.__name__))
          self.getWorld().notify(
                Event(self.id, "perform_"+_behav.__class__.__name__)
              )        
          _behav.do(self)


  def _envGet(self, env_name,attr_name, var):
    env = self._envdata.el.find(env_name)
    if env is not None:
      attr = env.find("%s[@%s]"%(attr_name, var))
      if attr is not None:
        return attr.get(var)   
    return None

  def _envGetVal(self, env_name,attr_name):
    env = self._envdata.el.find(env_name)
    if env is not None:
      attr = env.find("%s[@%s]"%(attr_name,"val"))
      if attr is not None:
        return attr.get("val")   
    return None

  def _envSetVal(self,env_name,attr_name, val):
    env = self._envdata.el.find(env_name)
    if env is not None:
      attr = env.find("%s[@%s]"%(attr_name,"val"))
      if attr is not None:
        attr.set("val", str(val))
        
  def __str__(self):
    s = self.id 
    env = self._envdata.el
    if env is not None:
      for _el in env:
        for _attr in _el:
          s += ", " + _el.tag + "." + _attr.tag + ":" + _attr.attrib["val"]
    return s

  def info(self):
    s = toStr(self._xd.rootNode.el)
    s += str(self._brain._xd)
    s += str([_b.name for _b in self.getBehaviours()])
    return s

class MultiEnvironmentWorld(Model):
  def __init__(self, config, output_dir="./" ):
    super().__init__()
    self.config = config
    self.odir = output_dir
    self._aodir = os.path.join(self.odir, "agents")
    if not os.path.exists(self._aodir):
      os.makedirs(self._aodir)    
    #_configure_logging()
    self._af = None # No factory defined. Override!
    self.l = logging.getLogger(self.__class__.__name__)    
    self._envs = []
    self._rasters = []
    self._agentsToAdd = []
    self._agents = {}
    self._agentsById = {}
    self._events = []
    self._rewardRules = []      
    self._deathsinturn = []
    self._totCreated = 0
    self._totDestroyed = 0    
    self.currentStep = 0    


  def info(self):
    self.l.info("TOTAL agent types %d"%len(self._agents))    
    for k,v in self._agents.items():
      self.l.info("%s:%d"%(k, len(v)))  

  def step(self):
    self.schedule.step()
    
  def run(self, n):
    self.l.info("***** STARTING SIMULATION *******")
    for i in range(n):
      self.l.info("Step %d of %d"%(i, n))    
      self.info()          
      self.stepEnvironment()
      self.step()
      self.currentStep+=1
      

    self.l.info("***** FINISHED SIMULATION *******")

  def getAgents(self):
    return self._agentsById.values()

  def _onEvent(self, event):
    #self.l.info("** Applying %d reward rules on %d events "%(len(self._rewardRules),len(self._events)))
    # Add reward on events
    for _r in self._rewardRules:
      self.l.debug("Rule %s check on event %s"%(_r._ev._type, event._type))                
      if _r._ev._type == event._type:
        self.l.debug("Check on _a %s == %s? %s"%(event._a, _r._ev._a, str(event._a.startswith(_r._ev._a))))                
        if event._a.startswith(_r._ev._a):
          _agent = self.getAgent(event._a)
          val = _r._values[0]
          self.l.debug("Rule %s fired on %s"%(_r._ev._type, event._type))                          
          _agent.addReward(_r._values[0])

  def notify(self, event):
    self._events.append(event)
    self._onEvent(event)
    _a, _t, _b = event._a, event._type, event._b
    self.l.debug("Agent %s notified %s on %s"%(_a, _t, _b))
    _agent = self.getAgent(_a)    
    if _t == "death":
      self._deathsinturn.append(_a)
      _agent.toXmlFile(odir=self._aodir)
      if _agent in self._agents[type(_agent)]:
        self.l.debug("Removing dead agent")
        self.removeAgent(_agent)
      else:      
        self.l.error("Cannot find agent %s of type %s in %s"%(str(_agent), str(type(_agent)), str(self._agents)))
    else:
      _ev = _agent._events.add(_t)
      _ev.el.attrib["simStep"] = str(self.currentStep)

  def getAllAgentsIds(self) -> list:
    _w = self.getBoundaries()._size._width
    _h = self.getBoundaries()._size._height
    bag = []
    for i in range(_w):
      for j in range(_h):
        for aid in self.getAgentIds((i,j), "all"):
          self.l.info(aid)
          bag.append(aid)
        
    return bag

  def addRewardRule(self, rule):
    self._rewardRules.append(rule)
    
  def addAgent(self, agent):
    self.space.place_agent(agent, agent.pos)
    self.schedule.add(agent)
    
    self._totCreated += 1
    self._agents.setdefault(type(agent),[]).append(agent)
    self._agentsById[agent.id]=agent
    agent.model = self
    agent.exists=True
    agent._postInit()

  def removeAgent(self, agent):
    self.schedule.remove(agent)
    self.space.remove_agent(agent)
    self._agents[type(agent)].remove(agent)
    if not self._agents[type(agent)]:
      del self._agents[type(agent)]
    del self._agentsById[agent.id]    
    self._totDestroyed += 1

  def getAgent(self, aid):
    if aid in self._agentsById: 
      return self._agentsById[aid]
    
    return None

  def _check_env_reqs(self, env):
    self.l.info("Analyzing requirements for environment %s"%env.name)
    if len(env._req) == 0:
      self.l.info("Nothing required")
    else:
      for _r in env._req:  
        self.l.info("Requires: %s"%toStr(_r))
        _renv = _environmentFactory._capabilities.get(_r.tag, None)
        if _renv:
          self.add_env(_renv)
        else:    
          self.l.info("Cannot satisfy dependency of %s in %s"%(_r.tag, _environmentFactory._capabilities))
          return False
    return True
  
  def add_env(self, env):
    if (env is not None) and (env not in self._envs):
      if self._check_env_reqs(env):
        self._envs.append(env)
      else:
        raise ValueError("Cannot fullfill requirements for model " + str(env))    

  def stepEnvironment(self):
    #self.l.info("%d - "%self.currentStep)
    #self.l.info("Deaths: %d - "%len(self._deathsinturn))    
    for env in self._envs:
      # TODO: Step added environments
      pass
    self._events = []
    self._deathsinturn = []
    
  def addPopulationRequest(self, pr):
    _caller = ":".join(map(str, inspect.stack()[1][1:4])) # Get nane of calling method for debug
    for _prefix in pr._data:
      _d = pr._data[_prefix]
      _num = _d["num"]
      _agtClass = _d["type"]
      _templates = _d.get("templates", [])
      _behavs = _d.get("defaultBehaviours",[])
      _varsToSet =  _d.get("varsToSet",[])
      _brain = _d.get("brain", BaseBrain)

      for i in range(_num):
        _meta = {"prInvokedBy":_caller,
              "templates":str(_templates),
              "pythonClass": _agtClass.__name__
          }
        _agent = _agtClass(_prefix + "_" + str(uuid.uuid4()), self)
        _agent.setMetadata(_meta)
        _agent._brain = _brain(_agent)

        # Load templates if needed
        _agent.addTemplates(_templates)
        self._prepareAgentForAdd(_agent)
        for _v in _varsToSet:
          item, value = _v.split(":")
          env, attr = item.split(".")
          _agent._envSetVal(env, attr, value)
        for _behavClass in _behavs:
          _agent._behaviours.append(_behavClass())                  
        self._agentsToAdd.append(_agent)
    return self._agentsToAdd
    
  def createAgents(self):
    #Register the agent with the world
    if len(self._agentsToAdd) <= 0: return
    self.l.info("Creating %d agents"%len(self._agentsToAdd))
    for _agent in self._agentsToAdd:
      self.addAgent(_agent)
      
    #Distribute parameters of the agents according to the model of each environment
    for env in self._envs:
      env.sampleAgentPar(self._agentsToAdd)

    self._agentsToAdd = []
        
  def _prepareAgentForAdd(self, agent):
    # Generate random position
    _x = random.randint(0, self.config["size"]["width"]-1)
    _y = random.randint(0, self.config["size"]["height"]-1)
    agent.position = (_x, _y)
    # Configure environments
    self._applyEnvRequir(agent)
    
  def _applyEnvRequir(self, agent):
    for env in self._envs:
      self.l.debug("Requesting environment %s"%str(env.getName()))
      _ed = agent._envdata.add(env.getName())
      for a in env._aa:
        _tag = _ed.add(a.tag)
        if a.get("default"):
          _tag.el.set("val", a.get("default"))
        if a.get("max"):
          _tag.el.set("max", a.get("max"))
        if a.get("min"):
          _tag.el.set("min", a.get("min"))        
      _bhvs = env.rootNode.el.find("behaviours")
      if _bhvs is not None:
        it = _bhvs.iter()
        next(it) # Skip parent tag
        for el in it:
          classname = el.tag
          _mod = __import__("iper.behaviours.actions", globals(), locals(), (classname), 0)
          _class = getattr(_mod, classname)
          agent._behaviours.append(_class())
       
  def createRasters(self):
    for env in self._envs:
      for ras in env._ras:
        raster_name = env.getName() + "_" + ras.tag
        self.registerDynamicRaster(raster_name, True)
        self.getDynamicRaster(raster_name).setInitValues(0, 5, 5)
        self._rasters.append(raster_name)
    self.l.info(self._rasters)
       

