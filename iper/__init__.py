# -*- coding: utf-8 -*-
"""
IPER Social simulations

Core Objects: 

"""
import datetime
import os, sys
import logging

__all__ = []

__title__ = "iper"
__version__ = "0.1"
__license__ = "Apache 2.0"
__copyright__ = "Copyright %s BCNMedTech Universitat Pompeu Fabra" % datetime.date.today().year

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', 
          datefmt='%Y/%m/%d %I:%M:%S %p', 
          level=logging.INFO)


_pckg_basedir = os.path.dirname(os.path.abspath(__file__))
_sandbox_defs = {
  'environments':os.path.join(_pckg_basedir,"env_def"),
  'templates':os.path.join(_pckg_basedir,"templates/xmlTemplates"),
  'models':os.path.join(_pckg_basedir,"models/")  
}

from .behaviours.behaviourFactory import BehaviourFactory
from .behaviours.actions import Action
_behaviourFactory = BehaviourFactory()
_behaviourFactory.load_all()

from .brainModelFactory import BrainModelFactory
_brainModelFactory = BrainModelFactory()
_brainModelFactory.load_all()
_brainModelFactory.list_all()

from .templates.templateFactory import XAgentFactory
_agentTemplateFactory = XAgentFactory()
_agentTemplateFactory.load()

from .environments import EnvironmentFactory
_environmentFactory = EnvironmentFactory()
_environmentFactory.load()        

from .xworlds import XAgent, MultiEnvironmentWorld, PopulationRequest, Event, RewardRule
