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
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, effective_n_jobs, delayed
import random


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
      print("VOLUME NODES: %d"%len(g.nodes))      
    else:
      if self.has_surface: 
        g = self._2g  # Otherwise use the 2D one
        print("SURFACE NODES: %d"%len(g.nodes))              

    self._lnodes = list(g.nodes)
    self._nnodes = len(g.nodes)    
    
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

    if not self.out_of_bounds(node_id):
      self._place_agent(agent, node_id)
      agent.pos = agent_pos
      return True
    
    return False

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

  def getRandomPos(self):
    pos = random.choice(self._lnodes)
    return (pos, )

  #def find_cell(self, pos):

  #  pos = np.asarray([pos])
  #  print(pos, len(pos))

  #  _cp, _dist, _cellid = trimesh.proximity.closest_point(
  #    self._surface,
  #    pos
  #  )
    
  #  return int(_cellid)
    


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
    #print("Inside out of bounds", pos, type(pos))
    if type(pos) is tuple and len(pos) == 1: 
      pos = pos[0]    
    elif type(pos) is int:
      pass
    else:
      print(pos)
      assert False, "Please set pos either as an integer (node number) or as a 1-D tuple (n,)"
    
    if pos not in self.G.nodes:
    #if pos > len(self.G.nodes):
      #print(pos,"is out of bound",self._nnodes)      
      #import ipdb
      #ipdb.set_trace()
      return True

    #print(pos,"is inside",self._nnodes)      
    #if int(pos)>len(self.G.nodes) and pos in self.G.nodes:
    #  print("BUG IN GRAPH!")          
    #  import ipdb
    #  ipdb.set_trace()    

    return False
 
  def get_neighbors(self, agent_pos, include_center):
    agent_pos = agent_pos[0]
    _neighs = super().get_neighbors(agent_pos, include_center)
    _neighs_t = [(_i,) for _i in _neighs if not self.out_of_bounds(_i)]
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
