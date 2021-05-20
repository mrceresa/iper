from iper import XAgent
from iper.behaviours.actions import Action
import logging
import random
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import movingpandas as mpd

class RandomWalk(Action):
  def do(self, agent):
    _xs = agent.getWorld()._xs
    dx, dy =  _xs["dx"], _xs["dy"] #How much is 1deg in km?
    # Convert in meters
    dx, dy = (dx/1000, dy/1000)

    new_position = ( agent.pos[0] + random.uniform(-dx, dx), agent.pos[0] + random.uniform(-dy,dy) )
    if not agent.getWorld().out_of_bounds(new_position):
      agent.getWorld().space.move_agent(agent, new_position)

class Move(Action):
  def define_goal(self,agent):
        random_node = random.choice(list(agent.model.Map.G.nodes))
        node = agent.model.Map.G.nodes[random_node]
        if node['x'] == agent.pos[0] and node['y'] == agent.pos[1]: 
            self.define_goal()
        else:
            return (node['x'], node['y'])

  def init_goal_traj(self,agent):
    route = agent.model.Map.routing_by_travel_time(agent.pos, agent.goal)
    #agent.model.Map.plot_graph_route(route, 'y', show = False, save = True, filepath = 'plots/route_agent' + str(agent.unique_id) + '_num' + str(agent.life_goals) + '.png')
    #agent.model.Map.plot_route_by_transport_type(route, save = True, filepath = 'plots/route_agent' + str(agent.unique_id) + '_num' + str(agent.life_goals) + 'colors' + '.png')

    df = pd.DataFrame()
    nodes, lats, lngs, times = [], [], [], []
    total_time = 0
    first_node = True

    for u, v in zip(route[:-1], route[1:]):
        travel_time = round(agent.model.Map.G.edges[(u, v, 0)]['travel_time'])
        if travel_time == 0:
            travel_time = 1
        
        if first_node == True:
            nodes.append(u)
            lats.append(agent.model.Map.G.nodes[u]['y'])
            lngs.append(agent.model.Map.G.nodes[u]['x'])
            times.append(0)
            first_node = False
        
        nodes.append(v)
        lats.append(agent.model.Map.G.nodes[v]['y'])
        lngs.append(agent.model.Map.G.nodes[v]['x'])
        times.append(total_time + travel_time)
        total_time += travel_time
        
    df['node'] = nodes
    df['time'] = pd.to_timedelta(times, unit = 'S')
    dfg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lngs, lats))
    gdf_trajectory = gpd.GeoDataFrame(dfg, crs=CRS(32631))
    traj = mpd.Trajectory(gdf_trajectory, agent.life_goals)
    traj.df.loc[:,'time'] = traj.df.loc[:,'time'] + agent.model.DateTime
    traj.df.set_index('time', inplace=True)
    agent.record_trajectories[traj.df.index[0]] = traj
    return traj

  def accumulate_polution(self,agent):
    local_pollution = 1 #get_pollution(agent.pos)
    acc_traj_pollution = local_pollution * int(agent.model.time_step)
    agent.exposure_pollution += acc_traj_pollution

  def do(self,agent):
    if agent.has_goal == False:
      agent.goal = self.define_goal(agent)
      agent.goal_traj = self.init_goal_traj(agent)
      agent.life_goals += 1
      agent.has_goal = True
    else:
      if agent.model.DateTime >= agent.goal_traj.get_end_time():
        #WAIT IN THE FINAL POSITION 
        #print('Waiting time: ' + str(agent.model.DateTime - agent.goal_traj.get_end_time()))
        newPos = agent.goal_traj.get_position_at(agent.goal_traj.get_end_time())
        agent.has_goal = False
      else:
        newPos = agent.goal_traj.get_position_at(agent.model.DateTime)
      agent.getWorld().space.move_agent(agent, (newPos.x,newPos.y))
  
class HumanAgent(XAgent):
  def __init__(self, unique_id, model):
    self.model = model
    self.has_goal = False
    self.life_goals = 0 
    self.record_trajectories = {}
    self.exposure_pollution = 0
    super().__init__(unique_id)

  def _postInit(self):
    pass

  def think(self):
    possible_actions = [Move()]
    return possible_actions

  def step(self):
    self.l.debug("*** Agent %s stepping"%str(self.id)) 
    _actions = self.think()
    _a = random.choice(_actions)
    _a.do(self)
    #others = self.getWorld().space.agents_at(self.pos, max_num=2)
    super().step()      

  def __repr__(self):
    return "Agent " + str(self.id)
