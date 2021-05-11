from iper import XAgent
from iper.behaviours.actions import Action

class RandomWalk(Action):
  def do(self, agent):
    possible_steps = agent.getWorld().space.get_neighborhood(
        agent.pos,
        moore=True,
        include_center=False)
    new_position = random.choice(possible_steps)
    if not agent.getWorld().space.out_of_bounds(new_position):
      agent.getWorld().space.move_agent(agent, new_position)

class HumanAgent(XAgent):
  def __init__(self, unique_id, model):
    super().__init__(unique_id)

  def _postInit(self):
    print("Initialized")

  def step(self):
    _log.debug("*** Agent %d stepping"%self.unique_id) 

  def __repr__(self):
    return "Agent " + str(self.unique_id)