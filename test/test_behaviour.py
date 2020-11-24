import unittest
import iper
from iper import BehaviourFactory
from iper.behaviours.actions import TestAction
from utils import captured_output

class TestBehaviourFactory(unittest.TestCase):
  def setUp(self):
    self._bf = BehaviourFactory()

  def test_default_actions(self):
    self._bf.load_all()
    _ks = self._bf.get_all()
    exp = ['TestAction', 'AgingBehaviour', 'DieOldBehaviour', 'DieOfStarvation', 
           'ConsumeBasal', 'Eat', 'Harvest', 'RandomWalk', 'TouchAndInfect', 'RecoverOrDie']
    for _e in exp: 
      self.assertTrue(_e in _ks)
      
  def test_register(self):
    self._bf.register("default.TestAction",TestAction(None))
        
    a = self._bf.get("default.TestAction")
    self.assertTrue(a is not None)
    
    with captured_output() as (out, err):
      a.do(None)
      output = out.getvalue().strip()
      self.assertEqual(output, 'Executed Test Action')
    
    
if __name__ == "__main__":
  unittest.main()

