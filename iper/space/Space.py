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
from mesa.agent import Agent
import trimesh
import pickle
import os

class MeshSpace(NetworkGrid):

  def __init__(self, mesh, debug=False, name="New MeshSpace"): 
    self.l = logging.getLogger(self.__class__.__name__)  
    self.name = name
    self._mesh = mesh
    self._info = {}
    self.has_surface = False
    self.has_volume = False
    self.is2d = False
    self.is3d = False
    self._2g =  None
    self._3g =  None

    self._parseMesh()

  def _parseMesh(self):
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
    
    cache_fname = os.path.join(basedir, "%s.conn_cached"%self.name)

    self.l.info("Looking for cache file %s"%cache_fname)    
    if os.path.exists(cache_fname):
      self.l.info("Loading connectivity from file %s"%cache_fname)        
      with open(cache_fname, "rb") as fp:
        self._2g, self._3g, self._adj = pickle.load(fp)
      return

    # Generate triangulation
    from iper.space.utils import \
      parse_connectivity_3d_triangles, \
      parse_connectivity_3d_quads, \
      parse_connectivity_3d_tetra

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
      
    if self.has_volume and "tetra" in self._info:
      g, adj = parse_connectivity_3d_tetra(self)      
      assert(g is not None)        
      self._3g = g   
      self._adj = adj         
    
    self.l.info("Caching connectivity graphs to file %s"%cache_fname)        
    with open(cache_fname, "wb") as fp:
      pickle.dump( (self._2g, self._3g, self._adj), fp, protocol=pickle.HIGHEST_PROTOCOL)
    
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

  def remove_agent(self, agent: Agent) -> None:
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

class GeoSpaceQR(GeoSpace):

  def place_agent(self, agent, newShape):
    if hasattr(agent, "shape"):
      self.l.info("Deleting agent %d (%s)"%(id(agent),str(agent.shape.bounds)))
      self.idx.delete(id(agent), agent.shape.bounds)
      agent.shape = newShape
      self.l.info("Inserting agent %d (%s)"%(id(agent),str(agent.shape.bounds)))
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
      return list(self._agents.values())

    @property
    def __geo_interface__(self):
      """Return a GeoJSON FeatureCollection."""
      features = [a.__geo_interface__() for a in self.agents]
      return {"type": "FeatureCollection", "features": features}
