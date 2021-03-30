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

  def testField(self):
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    _nx, _ny = (10, 10)
    x = np.linspace(xmin, xmax, _nx)
    y = np.linspace(ymin, ymax, _ny)
    xx, yy= np.meshgrid(x, y)
    points = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    print(points.shape)    
    ms = MeshSpace.from_vertices(points, 0.0)
    print("Mesh points:", ms._mesh.points.shape)
    _mb = ms._mesh.cells[0]    
    print("Mesh cells:", len(_mb))    
    print("Mesh nodes:", len(ms.G.nodes))
    
    field = {nid: nid%2 for nid in ms.G.nodes}
    ms.setField("mod", field)
    modf = ms.getFieldArray("mod")

    ms.plotSurface(savefig="test/field.png", 
      show=False, 
      title="Test meshgrid space",
      cmap="jet",
      field=modf) 
    plt.close()    

if __name__ == "__main__":
  unittest.main()

