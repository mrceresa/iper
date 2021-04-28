import unittest
import iper
from utils import captured_output
import trimesh

from iper.space.Space import MeshSpace
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from descartes.patch import PolygonPatch
import numpy as np
import networkx as nx
from mesa import Agent
import random
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.interpolate as interp
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from os.path import expanduser
import os

from iper.space.utils import read as read_mesh, _graph_as_fig

class TestMeshAsGraph(unittest.TestCase):
    
  def test2DTriang(self):
   
    fname = os.path.join("test","meshes","a_2D_triang.msh")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    
    ms, mesh = read_mesh(fname)
    print(mesh)
    print(ms._info)
    print(ms)
    
    num_triangles = ms._info["triangle"][0]
    num_cells = len(ms.G.nodes)
    assert(num_triangles == num_cells)
    
    for _l in ms._adj:
      assert(ms._adj2[_l[0], _l[1]] == 1)
    
    centroids = ms.getFieldArray("centroid")
    _graph_as_fig("2D_triang.png", ms.G, pos=centroids)

        
if __name__ == "__main__":
  unittest.main()

