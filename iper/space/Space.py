from mesa_geo import GeoSpace, GeoAgent
import logging
_log = logging.getLogger(__name__)

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point


class GeoSpaceQR(GeoSpace):

  def place_agent(self, agent, newShape):
    if hasattr(agent, "shape"):
      _log.info("Deleting agent %d (%s)"%(id(agent),str(agent.shape.bounds)))
      self.idx.delete(id(agent), agent.shape.bounds)
      agent.shape = newShape
      _log.info("Inserting agent %d (%s)"%(id(agent),str(agent.shape.bounds)))
      self.idx.insert(id(agent), agent.shape.bounds, agent)
    else:
      raise AttributeError("GeoAgents must have a shape attribute")
    
    self.update_bbox()


class GeoSpacePandas(GeoSpace):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Override Index
      del self.idx
      df = pd.DataFrame(
        {'agentid': [],
        'geometry': []
        })

      self._agdf = gpd.GeoDataFrame(df)
      self._agents = {}


    def add_agents(self, agents):

      """Add a list of GeoAgents to the Geospace.

      GeoAgents must have a shape attribute. This function may also be called
      with a single GeoAgent."""
      if isinstance(agents, GeoAgent):
        agents = [agents]
      columns = list(self._agdf)
      data = []
      for agent in agents:
        if hasattr(agent, "shape"):
          _aid = id(agent)
          values = [_aid, agent.shape]
          zipped = zip(columns, values)
          a_dictionary = dict(zipped)
          data.append(a_dictionary)
          self._agents[_aid] = agent
        else:
          raise AttributeError("GeoAgents must have a shape attribute")
      self._agdf = self._agdf.append(data, ignore_index=True)
      self._agdf.crs = self.crs

    def remove_agent(self, agent):
      """Remove an agent from the GeoSpace."""
      _aid = id(agent)
      if _aid in self._agents:
        del self._agents[_aid]
        self._agdf = self._agdf.drop(index=self._agdf.loc[self._agdf['agentid']==_aid].index)

    def get_relation(self, agent, relation):
      """Return a list of related agents.

      Args:
          agent: the agent for which to compute the relation
          relation: must be one of 'intersects', 'within', 'contains',
              'touches'
          other_agents: A list of agents to compare against.
              Omit to compare against all other agents of the GeoSpace
      """
      raise ValueError("Not implemented") 

    def _get_rtree_intersections(self, agent):
      """Calculate rtree intersections for candidate agents."""
      raise ValueError("Not implemented") 

    def get_intersecting_agents(self, agent, other_agents=None):
      raise ValueError("Not implemented") 

    def get_neighbors_within_distance(
        self, agent, distance, center=False, relation="intersects"
    ):
      """Return a list of agents within `distance` of `agent`.

      Distance is measured as a buffer around the agent's shape,
      set center=True to calculate distance from center.
      """
      raise ValueError("Not implemented") 

    def agents_at(self, pos):
      """Return a list of agents at given pos."""
      raise ValueError("Not implemented") 

    def distance(self, agent_a, agent_b):
      """Return distance of two agents."""
      return agent_a.shape.distance(agent_b.shape)

    def _create_neighborhood(self):
      """Create a neighborhood graph of all agents."""
      raise ValueError("Not implemented") 

    def get_neighbors(self, agent):
      """Get (touching) neighbors of an agent."""
      raise ValueError("Not implemented") 

    def _recreate_rtree(self, new_agents):
      """Create a new rtree index from agents shapes."""
      raise ValueError("Not implemented") 

    def update_bbox(self, bbox=None):
      """Update bounding box of the GeoSpace."""
      raise ValueError("Not implemented") 

    @property
    def agents(self):
      return self._agents.values()

    @property
    def __geo_interface__(self):
      """Return a GeoJSON FeatureCollection."""
      raise ValueError("Not implemented") 