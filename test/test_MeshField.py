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


class TestMeshField(unittest.TestCase):

  def testFieldOnPlane(self):
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    _nx, _ny = (10, 10)
    x = np.linspace(xmin, xmax, _nx)
    y = np.linspace(ymin, ymax, _ny)
    xx, yy= np.meshgrid(x, y)
    points = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    self.assertEqual(points.shape, (_nx*_ny, 2))    
    ms = MeshSpace.from_vertices(points, 0.0)
    
    self.assertEqual(ms.getSize(), (162,3))
    
    field = {nid: nid%2 for nid in ms.G.nodes}
    ms.setField("mod", field)
    modf = ms.getFieldArray("mod")

    ms.plotSurface(savefig="test/field.png", 
      show=False, 
      title="Test planar field",
      cmap="jet",
      field=modf) 
    plt.close()    
  
    _pos = (0,0,0)
    cell_pos = ms.find_cell(_pos)
      
  def testFieldOnCube(self):
    ms, _grid = MeshSpace.read("test/meshes/cube_plane.msh")
    
    field = {nid: nid%2 for nid in ms.G.nodes}
    ms.setField("mod", field)
    modf = ms.getFieldArray("mod")

    ms.plotSurface(savefig="test/fieldCube.png", 
      show=False, 
      title="Test cube field",
      cmap="jet",
      field=modf) 
    plt.close()

  def testFieldOnSphere(self):
    ms, _grid = MeshSpace.read("test/meshes/sphere.msh")
    
    field = {nid: nid%2 for nid in ms.G.nodes}
    ms.setField("mod", field)
    modf = ms.getFieldArray("mod")

    ms.plotSurface(savefig="test/fieldSphere.png", 
      show=False, 
      title="Test sphere field",
      cmap="jet",
      field=modf) 
    plt.close()

if __name__ == "__main__":
  unittest.main()

