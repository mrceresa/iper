from .xmlobjects import XMLObject, fromXmlFile, toStr
import os
from glob import glob
import importlib
from iper import _sandbox_defs
import logging

class EnvironmentFactory(object):
  def __init__(self, basedir=_sandbox_defs["environments"]):
    self._basedir=basedir
    if not os.path.exists(self._basedir):
      os.makedirs(self._basedir)
    self._environments = {}
    self._capabilities = {}    
    self.l = logging.getLogger(__name__)    
    
  def serialize(self):
    for e in self._environments:
      fname = os.path.join(self._basedir, "%s.xml"%e)
      v = self._environments[e]
      self.l.debug("Saving %s:"%fname)
      #print(str(v))
      v.toXmlFile(fname)
      
  def add(self, name, e):
    if name not in self._environments:
      if (e):
        self._environments[name] = e
      return True
    else:
      return False

     
  def load(self):
    for xmlf in glob(os.path.join(self._basedir,"*.xml")):
      self.l.debug("Loading %s..."%xmlf)
      el = fromXmlFile(xmlf).getroot()
      _e = Environment(el.tag, el)
      self.add(el.tag, _e)
      for _c in _e._prov:
        self._capabilities[_c.tag] = _e

  def list_all(self):
    self.l.debug("* This factory contains %d models"%len(self._environments))
    for i, m in enumerate(self._environments):
      self.l.debug("**** Environment %d: %s "%(i, m))
      self.l.debug(str(self._environments[m]))
      self.l.debug("****")
      
  def get(self, name):
    if name in self._environments:
      return self._environments[name]
    return None
  
class Environment(XMLObject):
  def __init__(self, name, el=None):
    super(Environment, self).__init__(name)
    self.name = name
    if el is not None:
      self.rootNode.el = el
    self._check_structure(self.rootNode.el)

  def sampleAgentPar(self, agents):
    n = len(agents)
    for attr in self._aa:
      classname = attr.get("sampleStrategy")
      if classname is not None:
        self.l.debug("Sampling agent population's' %s using %s distribution"%(attr.tag, classname))
        _mod = __import__("numpy.random", fromlist=[classname])
        _SamplingStrategy = getattr(_mod, classname)
        low = float(attr.get("min")); high = float(attr.get("max"));
        samples = _SamplingStrategy(low, high, n)
        
        _type = attr.get("type")
        module = importlib.import_module('__builtin__')
        _type_cls = getattr(module, _type)
        for i, agent in enumerate(agents):
          agent._envSetVal(self.getName(), attr.tag, _type_cls(samples[i]))

  def getName(self):
    return self.rootNode.el.tag

  def _check_structure(self, root):
    root.attrib["class"] = 'Environment'

    _els = ["requires", "provides", "agentAttributes", "behaviours", "rasters"]
    bag = root.findall(_els[0])
    if not bag:
      self._req = self.add(_els[0])
    elif len(bag) == 1:
      self._req = bag[0]
    elif len(bag) > 1:
      self.l.error("Environment._check_structure - Merging multiple requires not implemented")

    bag = root.findall(_els[1])
    if not bag:
      self._prov = self.add(_els[1])
    elif len(bag) == 1:
      self._prov = bag[0]      
    elif len(bag) > 1:
      self.l.error("Environment._check_structure - Merging multiple requires not implemented")

    bag = root.findall(_els[2])      
    if not bag:
      self._aa = self.add(_els[2])
    elif len(bag) == 1:
      self._aa = bag[0]      
    elif len(bag) > 1:
      self.l.error("Environment._check_structure - Merging multiple requires not implemented")
      
    bag = root.findall(_els[3])      
    if not bag:
      self._bhv = self.add(_els[3])
    elif len(bag) == 1:
      self._bhv = bag[0]      
    elif len(bag) > 1:
      self.l.error("Environment._check_structure - Merging multiple requires not implemented")
      
    bag = root.findall(_els[4])      
    if not bag:
      self._ras = self.add(_els[4])
    elif len(bag) == 1:
      self._ras = bag[0]      
    elif len(bag) > 1:
      self.l.error("Environment._check_structure - Merging multiple requires not implemented")                  
      
   
def create_demography_env():
  dem = Environment("Demography")  
  age = dem._aa.add("age")
  age.el.set("min",str(0))
  age.el.set("max",str(100))
  age.el.set("sampleStrategy","uniform")
  age.el.set("default",str(0))
  age.el.set("type","int")
  dem._bhv.add("AgingBehaviour")
  dem._bhv.add("DieOldBehaviour")
  
  return dem
  
def create_metabolism_env():
  bm = Environment("BasalMetabolism")
  energy = bm._aa.add("energy") 
  energy.el.set("min",str(0))
  energy.el.set("max",str(100))  
  energy.el.set("default",str(24))
  energy.el.set("type","int")  
  bm._bhv.add("DieOfStarvation")
  bm._bhv.add("ConsumeBasal") 
  bm._bhv.add("Eat")   
     
  return bm
  
def create_food_env():
  fp = Environment("FoodProduction")
  fp._prov.add("EnergySource")
  res = fp._ras.add("food")
  
  return fp  

def create_contagion_env():
  sir = Environment("SIRContagion")
  h = sir._aa.add("health")
  h.el.set("min",str(0))
  h.el.set("max",str(10))  
  h.el.set("default",str(10))
  h.el.set("type","int")  
  s = sir._aa.add("status")
  s.el.set("values",str(["susceptible","infected","recovered"]))
  s.el.set("default","susceptible")  
  s.el.set("type","str")
  rp = sir._aa.add("recover_prob")
  rp.el.set("min",str(0.0))
  rp.el.set("max",str(1.0))  
  rp.el.set("default",str(0.05))
  rp.el.set("type","float")    
  ip = sir._aa.add("infect_prob")
  ip.el.set("min",str(0.0))
  ip.el.set("max",str(1.0))  
  ip.el.set("default",str(0.95))
  ip.el.set("type","float")  
      
  return sir 
  
def create_all():
  dem = create_demography_env()
  bm = create_metabolism_env()
  fp = create_food_env()
  sir = create_contagion_env()
  
  ef = EnvironmentFactory()
  ef.add(dem.rootNode.el.tag, dem)
  ef.add(bm.rootNode.el.tag, bm)
  ef.add(fp.rootNode.el.tag, fp)
  ef.add(sir.rootNode.el.tag, sir)  
  ef.serialize()
  return ef
  
def load_all():  
  ef = EnvironmentFactory()
  ef.load()
  return ef
  
if __name__ == "__main__":
  ef = create_all()
  
  ef = load_all()
  ef.list_all()
  
  
  
