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
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from trimesh import Trimesh

class MeshSpace(NetworkGrid):

  @staticmethod
  def from_meshgrid(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0, z=0.0, xp=10, yp=10):
    nx, ny = (xp, yp)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    points3d = np.vstack(list(map(np.ravel, [xx,yy,zz] ))).T    
    tri = Delaunay(points)
    mesh = Trimesh(vertices=points3d,
      faces=tri.simplices)
    #plt.triplot(points[:,0], points[:,1], tri.simplices)
    #plt.plot(xx, yy, "o")
    #plt.show()

    return mesh, [xx, yy, zz]



  def __init__(self, mesh):
    self._mesh = mesh
    g = self._processMesh()
    super().__init__(g)


  def _processMesh(self):
    # Generate triangulation
    _v = self._mesh.vertices
    x, y, z = self._mesh.vertices[:,0], self._mesh.vertices[:,1], self._mesh.vertices[:,2]
    triang = tri.Triangulation(x, y, triangles=self._mesh.faces)
    self._v = _v; self.triang = triang; self.elevation = z

    # Transform to a graph
    g = nx.Graph()

    _ad = trimesh.graph.face_adjacency(mesh=self._mesh, return_edges=False)
    self._ad = _ad
    _nodes = []
    for i, f in enumerate(self._mesh.faces):
      loop = np.asarray([_v[f[0]], _v[f[1]], _v[f[2]]])
      _c = np.mean(np.asarray(loop), axis=0)
      _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

    g.add_nodes_from(_nodes)
    g.add_edges_from(_ad)
    return g

  def plot(self, alpha=1.0, savefig=None):
    fig, ax = plt.subplots(figsize=(12,9),subplot_kw =dict(projection="3d"))
    ax.plot_trisurf(self.triang, 
      self.elevation, 
      cmap="jet",
      alpha=alpha,
      antialiased=True,
      linewidth=2.0
      )
    ax.set_title('Agent mesh space plot')
    if savefig:
      plt.savefig(savefig)
    else:
      plt.show()

    return fig, ax



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