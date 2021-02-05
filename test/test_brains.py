import unittest
from iper import BehaviourFactory
from iper.behaviours.actions import TestAction
from utils import captured_output
from iper.brains import BaseBrain, WorldState, Reward

class TestBrains(unittest.TestCase):
  def setUp(self):
    self._bf = BehaviourFactory()
  
  def test_action_definitions(self):
    _b = BaseBrain(None)
    _b.setActions(self._bf.get_all())
    self.assertTrue(_b._actions is not None)

  def test_think(self):
    _b = BaseBrain(None)
    _b.setActions(self._bf.get_all())
    s = WorldState()
    r = Reward()
    toDo = _b.think(s, r)
    exp = ['TouchAndInfect', 'RecoverOrDie', 'DieOldBehaviour', 'AgingBehaviour',
         'DieOfStarvation', 'ConsumeBasal', 'TestAction', 'RandomWalk', 'Eat']    
    self.assertEqual(toDo, exp)    

    
if __name__ == "__main__":
  unittest.main()
