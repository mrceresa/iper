# -*- coding: utf-8 -*-
"""
IPER Social simulations

Core Objects: 

"""
import datetime
import os, sys
from loguru import logger
from .space.geospacepandas import GeoSpacePandas

__all__ = [GeoSpacePandas]

__title__ = "iper"
__version__ = "0.1"
__license__ = "Apache 2.0"
__copyright__ = "Copyright %s BCNMedTech Universitat Pompeu Fabra" % datetime.date.today().year

_pckg_basedir = os.path.dirname(os.path.abspath(__file__))
logger.info("IPER package basedir: %s"%_pckg_basedir)

_sandbox_defs = {
  'environments':os.path.join(_pckg_basedir,"env_def"),
  'templates':os.path.join(_pckg_basedir,"templates/xmlTemplates"),
  'models':os.path.join(_pckg_basedir,"models/")  
}
logger.info("IPER sandbox definitions: %s"%_sandbox_defs)

from .behaviours.behaviourFactory import BehaviourFactory
from .behaviours.actions import Action
_behaviourFactory = BehaviourFactory()

from .brainModelFactory import BrainModelFactory
_brainModelFactory = BrainModelFactory()

from .templates.templateFactory import XAgentFactory
_agentTemplateFactory = XAgentFactory()

from .environments import EnvironmentFactory
_environmentFactory = EnvironmentFactory()

from .xworlds import XAgent, MultiEnvironmentWorld, PopulationRequest, Event, RewardRule

def load_all():
  _behaviourFactory.load_all()
  _brainModelFactory.load_all()
  _agentTemplateFactory.load()
  _environmentFactory.load()        
