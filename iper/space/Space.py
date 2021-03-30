from mesa_geo import GeoSpace, GeoAgent
from mesa.space import MultiGrid, NetworkGrid
import logging
_log = logging.getLogger(__name__)

import pandas as pd
import geopandas as gpd
import trimesh
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from trimesh import Trimesh
import meshio

class MeshSpace(NetworkGrid):
  """
    _mesh
    _v
    _tri
    _elev
    G
    _adj
  """

  @staticmethod
  def read(filename):
    mesh = meshio.read(filename)
    return MeshSpace(mesh, name=filename), mesh

  def from_meshio(points, cells):
    mesh = meshio.Mesh(points, cells)
    return MeshSpace(mesh, name="Meshio object")

  @staticmethod
  def from_vertices(points, elevation=None):
    tess = Delaunay(points)
    #triang = mtri.Triangulation(x=points[:, 0], y=points[:, 1], triangles=tri)
    if elevation is not None:
      nx, ny = points.shape
      _tp = np.zeros((nx, ny+1))
      _tp[:,:-1] = points 
      _tp[:,-1] = elevation
      points = _tp    

    cells = [("triangle", tess.simplices)]
    mesh = meshio.Mesh(points, cells)
    ms = MeshSpace(mesh)
    return ms


  @staticmethod
  def from_meshgrid(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0, z=0.0, xp=10, yp=10):
    nx, ny = (xp, yp)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy= np.meshgrid(x, y)
    points = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    ms = MeshSpace.from_vertices(points, z)
    #plt.triplot(points[:,0], points[:,1], tri.simplices)
    #plt.plot(xx, yy, "o")
    #plt.show()
  

    return ms, [xx, yy]    



  def __init__(self, mesh, debug=False, name="New MeshSpace"): 
    self.name = name
    self._mesh = mesh
    self._info = {}
    self.has_surface = False
    self.has_volume = False

    self._v = self._mesh.points;
    self._info["points"] = self._v.shape

    self._tri = self._mesh.cells_dict.get("triangle",np.asarray([]))
    self._info["triangle"] = self._tri.shape
    if self._info["triangle"][0] > 0: self.has_surface = True

    self._tetra = self._mesh.cells_dict.get("tetra",np.asarray([]))
    self._info["tetra"] = self._tetra.shape
    if self._info["tetra"][0] > 0: self.has_volume = True

    g = self._processMesh(debug)

    super().__init__(g)

  def _processMesh(self, debug=False):
    # Generate triangulation

    print("Processing mesh", self.name)
    _v = self._v
    x, y, z = _v[:,0], _v[:,1], _v[:,2]
    if debug:
      print(self._info)

    # Transform to a graph
    g = nx.Graph()

    if self.has_surface:
      _surface = trimesh.Trimesh(vertices=self._v, faces=self._tri)
      triang = Triangulation(x, y, triangles=self._tri)
      self._plt_tri = triang; self._elev = z

      _ad = trimesh.graph.face_adjacency(mesh=_surface, return_edges=False)
      self._adj = _ad
      _nodes = []
      for i, f in enumerate(self._tri):
        loop = (_v[f[0]], _v[f[1]], _v[f[2]])
        _c = np.mean(np.asarray(loop), axis=0)
        _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

      g.add_nodes_from(_nodes)
      g.add_edges_from(_ad)

    return g


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
