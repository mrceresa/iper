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

class TestMeshAsGraph(unittest.TestCase):
    
  def testHexa(self):
    home = expanduser("~")
    fname = os.path.join(home, "Downloads","alveolar_sac_Oriol_Cuxart.msh")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    #ms, _grid = MeshSpace.read(fname, debug=True)
    #ms.plotSurface()
    #plt.close()    

  def testPolydata(self):
   
    home = expanduser("~")
    fname = os.path.join(home, "Downloads","malla_sup.vtk")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    
    ms, _ = MeshSpace.read_vtk(fname) 
    ms.plotSurface(show=False, 
      title="Test vtk polydata alveolar mesh (decimated)",
      cmap="viridis",
      savefig="test/testVTKPolydata.png",
      alpha=0.6)
    plt.close() 
    
if __name__ == "__main__":
  unittest.main()

