#!/usr/bin/env python

import os
from glob import glob

from ..xmlobjects import toStr, fromXmlFile

from loguru import logger

from copy import deepcopy
from iper import _sandbox_defs

class XAgentFactory(object):
  def __init__(self, basedir=_sandbox_defs["templates"]):
    self._basedir = basedir
    if not os.path.exists(self._basedir):
      os.makedirs(self._basedir)
    self._templates = {}    
    
  def add(self, name, e):
    if name not in self._templates:
      if (e is not None):
        self._templates[name] = e
      return True
    else:
      return False
          
  def load(self):
    for xmlf in glob(os.path.join(self._basedir,"*.xml")):
      print("Loading %s..."%xmlf)
      el = fromXmlFile(xmlf).getroot()
      self.add(el.tag, el)
            
  def list_all(self):
    print("* This factory contains %d models"%len(self._templates))
    for i, m in enumerate(self._templates):
      print("**** Template %d: %s "%(i, m))
      print(toStr(self._templates[m]))
      print("****")

  def _mergeRecursively(self, el1, el2):
    for _c1 in el1:  
      if el2.xpath(_c1.tag):
        self._mergeRecursively(_c1, el2.xpath(_c1.tag)[0])
      else:
        el2.append(deepcopy(_c1))

  def _merge(self, t):
    deriveFrom = t.get("deriveFrom")
    if deriveFrom:
      _d = self._instantiate(deriveFrom)
      if _d is not None:
        self._mergeRecursively(_d, t)

    return t

  def _instantiate(self, templateName):
    if templateName in self._templates:
      #print("Creating new object from template ", templateName)
      _t = self._templates[templateName]
      _t = self._merge(_t) # Check if we need to merge data from derived templates
      return _t
    else:
      return None
    
  def instantiate(self, templateName):
    _t = self._instantiate(templateName)
    return _t
 
       

