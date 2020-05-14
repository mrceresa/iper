import logging

from .actions import Action

print(__name__, __package__)

class BehaviourFactory(object):
  def __init__(self):
    self._behaviours = {}    

  def get_all(self):
    return self._behaviours.keys()
    
  def load_all(self):
    for _a in Action.__subclasses__():
      self.register(_a.__name__, _a(None))

  def get(self, name):
    return self._behaviours.get(name, None)

  def register(self, name, action):
    if action is None: return False
    name = str(name)
    if name in self._behaviours:
      self.l.error("A method with name %s is already registered"%name) 
      return False   
    self._behaviours[name] = action
          
  def list_all(self):
    print("* This factory contains %d behaviours"%len(self._behaviours))
    for i, k in self._behaviours.items():
      print("**** %s: %s"%(i, k))
      
