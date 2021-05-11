from mesa_geo import GeoSpace, GeoAgent
from mesa.space import MultiGrid, NetworkGrid
import logging

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from trimesh import Trimesh
import meshio
#from mesa.agent import Agent
import trimesh
import pickle
import os
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from shapely.ops import nearest_points

from sklearn.neighbors import BallTree
import numpy as np

class MeshSpace(NetworkGrid):

  def __init__(self, mesh, debug=False, name="New MeshSpace", \
              compute_conn=True, g2=None, g3=None): 
              
    self.l = logging.getLogger(self.__class__.__name__)  
    self.name = name
    self._mesh = mesh
    self._info = {}
    self.has_surface = False
    self.has_volume = False
    self.is2d = False
    self.is3d = False
    self._2g =  g2
    self._3g =  g3

    self._parseMesh(compute_conn)

  def _parseMesh(self, compute_conn):
    self._v = self._mesh.points;
    self._info["points"] = self._v.shape
    if self._info["points"][1] == 2: 
      self.is2d = True
    elif self._info["points"][1] == 3: 
      self.is3d = True    
    else:
      raise RuntimeError("Points are not 2d or 3d?")
    

    _cd = self._mesh.cells_dict
    for k, v in _cd.items():
      self._info[k] = v.shape

    if "triangle" in self._info and self._info["triangle"][0] > 0: 
      self._tri = _cd.get("triangle",np.asarray([]))
      self.has_surface = True
      
    if "quad" in self._info and self._info["quad"][0] > 0: 
      self._quad = _cd.get("quad",np.asarray([]))
      self.has_surface = True
      
    if "tetra" in self._info and self._info["tetra"][0] > 0: 
      self._tetra = _cd.get("tetra",np.asarray([]))
      self.has_volume = True                  

    if compute_conn:
      self._computeConnectivity()

    # In case of multiple connectivities, which one to use?
    g = None
    if self.has_volume: 
      g = self._3g # If we have a volume connectivity we always use it
    else:
      if self.has_surface: 
        g = self._2g  # Otherwise use the 2D one
        
    # If none we will have an error
    assert(g is not None)
    
    super().__init__(g)

  def _computeConnectivity(self, debug=False):
    # Check if we have the info saved
    basedir = os.getcwd()

    self.l.info("Generating connectivity graphs for mesh %s"%self.name)
    self.l.info(str(self._mesh))
    
    # Generate triangulation
    from iper.space.utils import \
      parse_connectivity_3d_triangles, \
      parse_connectivity_3d_quads

    self.l.info("Calculating connectivity...")
    # Transform to a graph

    if self.has_surface and "triangle" in self._info:
      g, adj = parse_connectivity_3d_triangles(self)    
      assert(g is not None)  
      self._2g = g
      self._adj = adj

    if self.has_surface and "quad" in self._info:
      g, adj = parse_connectivity_3d_quads(self)      
      assert(g is not None)        
      self._2g = g
      self._adj = adj      
              
  def __repr__(self):
    s = "<MeshSpace>\n"
    
    if "points" in self._info:
      s += "\t Points: %d\n"%self._info["points"][0]

    if "vertex" in self._info:
      s += "\t Vertex: %d\n"%self._info["vertex"][0]
      
    if "line" in self._info:
      s += "\t Lines: %d\n"%self._info["line"][0]
      
    if "triangle" in self._info:
      s += "\t Triangles: %d\n"%self._info["triangle"][0]                  
      
    if "quad" in self._info:
      s += "\t Quads: %d\n"%self._info["quad"][0]                 
      
    if "tetra" in self._info:
      s += "\t Tetrahedrons: %d\n"%self._info["tetra"][0]                             

    if self.is2d: s+= "\t Mesh is 2D\n"
    if self.is3d: s+= "\t Mesh is 3D\n"
    
    if self.has_surface: s+= "\t Mesh contains 2D surfaces\n"
    if self.has_volume: s+= "\t Mesh contains 3D volume\n"
    
    if self._2g is not None: s+= "\t Mesh has 2D connectivity graph\n"
    if self._3g is not None: s+= "\t Mesh has 3D connectivity graph\n" 
   
    s += "</MeshSpace>"
    return s

  def place_agent(self, agent, agent_pos):
    node_id = agent_pos[0]  
    self._place_agent(agent, node_id)
    agent.pos = agent_pos

  def move_agent(self, agent , agent_pos):
    node_id = agent_pos[0]  
    self._remove_agent(agent, agent.pos)
    self.place_agent(agent, agent_pos)

  def remove_agent(self, agent):
    """ Remove an agent from a node. """

    node_id = agent.pos
    self._remove_agent(agent, agent.pos)
    
  def _remove_agent(self, agent, pos):
    """ Remove an agent from a node. """

    node_id = agent.pos
    if type(node_id) is tuple: node_id = node_id[0] 
    self.G.nodes[node_id]["agent"].remove(agent)    

  def getSurfaceSize(self):
    return (len(self._2g.nodes), )
    
  def getVolumeSize(self):
    return (len(self._3g.nodes), )
    
  def getSize(self):
    return (len(self.G.nodes), )

  

  #def find_cell(self, pos):

  #  pos = np.asarray([pos])
  #  print(pos, len(pos))

  #  _cp, _dist, _cellid = trimesh.proximity.closest_point(
  #    self._surface,
  #    pos
  #  )
    
    return int(_cellid)
    


#  def getNodeField(self, field):
#    for node_id in self.G.nodes:
#      _n = self.G.nodes[node_id]
#      if field in _n:
#        yield self.G.nodes[node_id][field]
#      else:
#        yield np.nan

  def getField(self, name): 
    return nx.get_node_attributes(self.G, name)
  
  def getFieldArray(self, name):
    return np.asarray(
      [self.G.nodes[node_id][name] if name in self.G.nodes[node_id] 
        else np.nan
        for node_id in self.G.nodes 
          
      ])

    
  def setField(self, name, values): 
    nx.set_node_attributes(self.G, values, name) 
 
  def out_of_bounds(self, pos):
    if type(pos) is tuple: pos = pos[0]

    if pos in self.G.nodes:
      return False
      
    return True
 
  def get_neighbors(self, agent_pos, include_center):
    agent_pos = agent_pos[0]
    _neighs = super().get_neighbors(agent_pos, include_center)
    _neighs_t = [(_i,) for _i in _neighs]
    return _neighs_t
  
#  def getField(self, field):
#    """ Go through all the nodes and get the correspondig value
#    """
#    for node_id in self.G.nodes:
#      _n = self.G.nodes[node_id]
#      _c = _n["centroid"]
#      if field in _n:
#        yield _c, self.G.nodes[node_id][field]
#      else:
#        yield _c, np.nan
        
  def plotSurface(self, alpha=1.0, savefig=None, field=None, 
      show=True, 
      title=None,
      cmap="Blues",
      ax=None
      ):

    if not self.has_surface:
      print("*"*5 + 
        "Mesh %s has NO surface??"%self.name, 
        self._info)

      return None, None

    if ax is None:
      fig, ax = plt.subplots(figsize=(12,9),subplot_kw =dict(projection="3d"))
    
    collec = ax.plot_trisurf(self._plt_tri, 
      self._elev, 
      cmap=cmap,
      alpha=alpha,
      antialiased=True,
      linewidth=1.0,
      shade=False
      )

    if field is not None:
      collec.set_array(field)
      collec.autoscale()

    if title is None: title = self.name
    ax.set_title(title)
    if savefig:
      plt.savefig(savefig)
    
    if show:
      plt.show()

    return ax, collec

class GeoSpacePandas(GeoSpace):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Override Index

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
      self.move_agent(agent, pos)
      self._gdf_is_dirty = True

    def remove_agent(self, agent, pos):
      if not agent.id in self._agents: return
      del self._agents[agent.id]
      self._gdf_is_dirty = True

    def move_agent(self, agent, pos):
      if agent.id in self._agents:
        agent.pos = pos
        agent.shape = Point(pos[0], pos[1])
        self._gdf_is_dirty = True
      else:
        raise AttributeError("No agent %s"%agent.id)

    def _create_gdf(self):
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
      # Ensure that index in right gdf is formed of sequential numbers
      self._right = self._agdf.copy().reset_index(drop=True)      
      _right_r = np.array(self._right["geometry"].apply(
        lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list()
        )      
      # Create tree from the candidate points
      self._tree = BallTree(_right_r, leaf_size=15, metric='haversine')        
    
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

    def get_nearest(self, src_points, k_neighbors=5):
      """Find nearest neighbors for all source points from a set of candidate points"""

      # Find closest points and distances
      distances, indices = self._tree.query(src_points, k=k_neighbors)

      # Transpose to get distances and indices into arrays
      distances = distances.transpose()
      indices = indices.transpose()

      # Return indices and distances
      return (indices, distances)


    def agents_at(self, pos, max_num=5):
      """Return a list of agents at given pos."""
     
      # Parse coordinates from points and insert them into a numpy array as RADIANS
      left_r = np.array(
         [ (pos[0] * np.pi / 180, pos[1] * np.pi / 180) ]
        )
      
      # Find the nearest points
      # -----------------------
      # closest ==> index in right_gdf that corresponds to the closest point
      # dist ==> distance between the nearest neighbors (in meters)
      
      closest, dist = self.get_nearest(
        src_points=left_r, k_neighbors=max_num
        )
      
      # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
      closest_points = self._right.loc[closest.reshape(-1)]

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
