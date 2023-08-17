from mesa_geo import GeoSpace, GeoAgent
from mesa.space import MultiGrid, NetworkGrid
from loguru import logger

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator
from mpl_toolkits.mplot3d import Axes3D
#from mesa.agent import Agent
import pickle
import os
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from sklearn.neighbors import BallTree
import numpy as np
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors, BallTree, KDTree
from joblib import Parallel, effective_n_jobs, delayed
import random
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, cascaded_union

def _tree_query_parallel_helper(tree, *args, **kwargs):
    """Helper for the Parallel calls in KNeighborsMixin.kneighbors
    The Cython method tree.query is not directly picklable by cloudpickle
    under PyPy.
    """
    return tree.query(*args, **kwargs)

class GeoSpaceComposer(GeoSpace):

    def __init__(self, extent, N, M, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Override Index

      self._extent = extent
      self._agents = {}

      x =  np.linspace(extent[0], extent[1], N+1, endpoint=True)
      y =  np.linspace(extent[2], extent[3], M+1, endpoint=True)

      hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
      vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]

      self._grids = list(polygonize(MultiLineString(hlines + vlines)))
      self._spaces = {str(g): GeoSpacePandas(extent=g.bounds) for g in self._grids}

      self._gdf_is_dirty = False

    def _create_gdf(self, **kwargs):
      for s in self._spaces.values():
        if s._gdf_is_dirty:
          s._create_gdf(**kwargs)

    def _w(self, pos):
      pos = tuple(map(float,pos))
      for i, pol in enumerate(self._grids):
        if pol.contains(Point(*pos)): return pol
      return None

    def _s(self, pol):
      return self._spaces[str(pol)]

    def getAgentSpace(self, agent):
      g = self._w(agent.pos)
      sp = self._s(g)
      return sp

    def getRandomPos(self):
      pos = ( random.uniform(self._extent[0], self._extent[1]),
              random.uniform(self._extent[2], self._extent[3])
            )
      return pos 

    def get_agent(self, aid):
      aid=str(aid)
      if aid in self._agents:
        return self._agents[aid]


    def place_agent(self, agent, pos):
      _from = self._w(agent.pos)
      _to = self._w(pos)
      if _to is None: raise ValueError("Destination position %s is not within the extent of this space"%str(pos))
      self._agents[agent.id] = agent
      if _from is None: 
        self._s(_to).place_agent(agent, pos)
        return
      if _from == _to: 
        self._s(_to).move_agent(agent, pos)
      else:      
        self._s(_from).remove_agent(agent)
        self._s(_to).place_agent(agent, pos)

    def move_agent(self, agent, pos):
      if agent.id in self._agents:
        sp = self.getAgentSpace(agent)
        sp.move_agent(agent, pos)
        return True
      
      return False

    def remove_agent(self, agent):
      if type(agent) is str:
        agent = self._agents[agent]
      else:
        if not agent.id in self._agents: return

      sp = self.getAgentSpace(agent)
      if sp: sp.remove_agent(agent)

      del self._agents[agent.id]
      self._gdf_is_dirty = True

    def check_if_dirty(self):
      self._gdf_is_dirty = False
      for i, pol in enumerate(self._grids):
        if pol._gdf_is_dirty:
          self._gdf_is_dirty = True 

      return self._gdf_is_dirty 

    def _clear_gdf(self):
      for i, pol in enumerate(self._grids):
        if pol._gdf_is_dirty: pol._clear_gdf()

    def agents_at(self, pos, **kwargs):
      pol = self._w(pos)
      if pol is None: return []
      sp = self._s(pol)
      ids = sp.agents_at(pos, **kwargs)
      return ids

class GeoSpacePandas(GeoSpace):
    def __init__(self, extent, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Override Index
      self._crs = "EPSG:4326"
      self._extent = extent
      self._agents = {}
      self._clear_gdf()
      self._gdf_is_dirty = False

    def _clear_gdf(self):
      df = pd.DataFrame(
        {'agentid': [],
        'geometry': []
        })

      self._agdf = gpd.GeoDataFrame(df)  

    def get_agent(self, aid):
      aid=str(aid)
      if aid in self._agents:
        return self._agents[aid]

    def place_agent(self, agent, pos):
      if not hasattr(agent, "shape"): agent.shape = Point(pos[0], pos[1])
      if not agent.id in self._agents: self._agents[agent.id] = agent
      res = self.move_agent(agent, pos)
      if res: self._gdf_is_dirty = True
      return res

    def remove_agent(self, agent, pos):
      if not agent.id in self._agents: return
      del self._agents[agent.id]
      self._gdf_is_dirty = True

    def move_agent(self, agent, pos):
      if agent.id in self._agents:
        agent.pos = pos
        agent.shape = Point(pos[0], pos[1])
        self._gdf_is_dirty = True
        return True
      
      return False

    def _create_gdf(self, use_ntrees=False):
      self._clear_gdf()
      columns = list(self._agdf)
      data = []
      for agent in self._agents.values():
        _aid = agent.id
        values = [_aid, agent.shape]
        zipped = zip(columns, values)
        a_dictionary = dict(zipped)
        data.append(a_dictionary)

      self._agdf = self._agdf.append(data, ignore_index=True)
      self._agdf.set_crs(self._crs)
      # Ensure that index in right gdf is formed of sequential numbers
      self._right = self._agdf.copy().reset_index(drop=True)      
      _right_r = np.array(self._right["geometry"].apply(
        lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list()
        )      
      # Create tree from the candidate points
      #self._tree = BallTree(_right_r, leaf_size=15, metric='haversine')
      self._tree = BallTree(_right_r, leaf_size=2)    

      if use_ntrees:
        n_jobs = effective_n_jobs(mp.cpu_count())
        from sklearn.utils import gen_even_slices

        self._trees = [BallTree(_right_r[s], leaf_size=2) 
          for s in gen_even_slices(_right_r.shape[0], n_jobs)
        ]
        self._nn =  NearestNeighbors(n_neighbors=5, radius=2.0, n_jobs=n_jobs, algorithm="ball_tree", metric="haversine", leaf_size=2)
        self._nn.fit(_right_r)
      self._gdf_is_dirty = False
      
    def add_geo(self, agents):

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
      #self._agdf.crs = self.crs

    def remove_agent(self, agent):
      """Remove an agent from the GeoSpace."""
      _aid = id(agent)
      if _aid in self._agents:
        del self._agents[_aid]
        _idxToDel = self._agdf.loc[self._agdf['agentid']==_aid].index
        self._agdf = self._agdf.drop(index=_idxToDel)

    def update_shape(self, agent, newShape):
      """Update an agent shape in the GeoSpace."""
      _aid = id(agent)
      if _aid in self._agents:
        self._agents[_aid].shape = newShape
        self._agdf.geometry[self._agdf['agentid']==_aid] =newShape
      else:
        raise ValueError("Agent %s is not in geospace"%(str(agent)))

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

    def get_nearest(self, src_points, radius=2.0):
      """Find nearest neighbors for all source points from a set of candidate points"""

      # Find closest points and distances
      indices = self._tree.query_radius(src_points, radius)

      # Transpose to get indices into arrays
      indices = indices.transpose().tolist()

      # Return indices and distances
      return indices

    def get_nearest_mp(self, src_points, k_neighbors=5):
      """Find nearest neighbors for all source points from a set of candidate points"""

      # Find closest points and distances
      radius = 2.0
      distances, indices = self._nn.radius_neighbors(src_points, radius, return_distance=True)

      # Transpose to get distances and indices into arrays
      distances = np.asarray(distances.transpose().tolist()).squeeze()
      indices = np.asarray(indices.transpose().tolist())

      # Return indices and distances
      return (indices, distances)

    def get_nearest_mtree(self, src_points, k_neighbors=5):
      """Find nearest neighbors for all source points from a set of candidate points"""

      # Find closest points and distances
      radius = 2.0
      chunked_results = Parallel(len(self._trees), prefer="threads")(
          delayed(_tree_query_parallel_helper)(
                self._trees[i], src_points, k_neighbors, return_distance=True)
            for i in range(len(self._trees))
        )

      if chunked_results is not None:
        neigh_dist, neigh_ind = zip(*chunked_results)
        distances, indices = np.vstack(neigh_dist), np.vstack(neigh_ind)

        # Transpose to get distances and indices into arrays
        distances = np.asarray(distances.transpose().ravel().tolist()).squeeze()
        indices = np.asarray(indices.transpose().ravel().tolist())

        return (indices, distances)


      # Return indices and distances

    def agents_at(self, pos, max_num=5, radius=2.0):
      """Return a list of agents at given pos."""
     
      # Parse coordinates from points and insert them into a numpy array as RADIANS
      left_r = np.array(
         [ (pos[0] * np.pi / 180, pos[1] * np.pi / 180) ]
        )
      
      # Find the nearest points
      # -----------------------
      # closest ==> index in right_gdf that corresponds to the closest point
      
      closest = self.get_nearest(
        src_points=left_r, radius=radius
        )
      
      # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
      closest_points = self._right.loc[closest[0]]

      # Ensure that the index corresponds the one in left_gdf
      #closest_points = closest_points.reset_index(drop=True)
      
      return closest_points



    def agents_at_mp(self, pos, max_num=5):
      """Return a list of agents at given pos."""
     
      # Parse coordinates from points and insert them into a numpy array as RADIANS
      left_r = np.array(
         [ (pos[0] * np.pi / 180, pos[1] * np.pi / 180) ]
        )
      
      # Find the nearest points
      # -----------------------
      # closest ==> index in right_gdf that corresponds to the closest point
      # dist ==> distance between the nearest neighbors (in meters)
      
      closest, dist = self.get_nearest_mp(
        src_points=left_r, k_neighbors=max_num
        )
      
      # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
      closest = closest.reshape(-1)
      closest_points = self._right.loc[closest]

      # Ensure that the index corresponds the one in left_gdf
      closest_points = closest_points.reset_index(drop=True)

      # Convert to meters from radians
      earth_radius = 6371000  # meters
      closest_points['distance'] = dist * earth_radius
      
      return closest_points

    def agents_at_mtree(self, pos, max_num=5):
      """Return a list of agents at given pos."""
     
      # Parse coordinates from points and insert them into a numpy array as RADIANS
      left_r = np.array(
         [ (pos[0] * np.pi / 180, pos[1] * np.pi / 180) ]
        )
      
      # Find the nearest points
      # -----------------------
      # closest ==> index in right_gdf that corresponds to the closest point
      # dist ==> distance between the nearest neighbors (in meters)
      
      closest, dist = self.get_nearest_mtree(
        src_points=left_r, k_neighbors=max_num
        )
      
      # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
      closest = closest.reshape(-1)
      closest_points = self._right.loc[closest]

      # Ensure that the index corresponds the one in left_gdf
      closest_points = closest_points.reset_index(drop=True)

      # Convert to meters from radians
      earth_radius = 6371000  # meters
      closest_points['distance'] = dist * earth_radius
      
      return closest_points

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
      return list(self._agents.values())

    @property
    def __geo_interface__(self):
      """Return a GeoJSON FeatureCollection."""
      features = [a.__geo_interface__() for a in self.agents]
      return {"type": "FeatureCollection", "features": features}

    def getRandomPos(self):
      pos = ( random.uniform(self._extent[0], self._extent[1]),
              random.uniform(self._extent[2], self._extent[3])
            )
      return pos 
