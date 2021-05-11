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
import vtk

from iper.space.utils import read as read_mesh, _graph_as_fig

class TestMeshAsGraph(unittest.TestCase):

  def test2DTriang(self):
   
    fname = os.path.join("test","meshes","a_2D_triang.msh")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    
    ms, mesh = read_mesh(fname)
    centroids = ms.getFieldArray("centroid")    
    _graph_as_fig("test/2D_triang.png", ms.G, pos=centroids)    
    
    num_triangles = ms._info["triangle"][0]
    num_cells = len(ms.G.nodes)
    assert(num_triangles == num_cells)
   
    import pickle
    with open("2D_triang_MeshSpace.pkl", "wb") as fp:
      pickle.dump(ms, fp) 
    #for _l in ms._adj:
    #  assert(ms._adj2[_l[0], _l[1]] == 1)
    
  def test2DQuads(self):
   
    fname = os.path.join("test","meshes","a_2D_quads.msh")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    
    ms, mesh = read_mesh(fname)
    centroids = ms.getFieldArray("centroid")    
    _graph_as_fig("test/2D_quad.png", ms.G, pos=centroids)    
    
    num_quad = ms._info["quad"][0]
    num_cells = len(ms.G.nodes)
    assert(num_quad == num_cells)
    
    #for _l in ms._adj:
    #  assert(ms._adj2[_l[0], _l[1]] == 1)

  def test3DTetra(self):
   
    fname = os.path.join("test","meshes","a_3d_tetra.vtu")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    
    ms, mesh = read_mesh(fname)
    centroids = ms.getFieldArray("centroid")    
    _graph_as_fig("test/3D_tetra.png", ms.G, pos=centroids)    
    
    print(mesh)
    print(ms)    
    
    num_triangles = ms._info["tetra"][0]
    num_cells = len(ms.G.nodes)
    #assert(num_triangles == num_cells)
    
    #for _l in ms._adj:
    #  assert(ms._adj2[_l[0], _l[1]] == 1)ç
    
  def testMultiBlock(self):
    filein="/home/mario/Downloads/alveolo/test_0_0.vtu"
    #rd = vtk.vtkXMLMultiBlockDataReader()
    rd = vtk.vtkXMLUnstructuredGridReader()
    rd.SetFileName(filein)
    rd.Update()
    
    mesh = rd.GetOutput()
    pd = mesh.GetAttributes(0)
    codno = pd.GetArray("CODNO")

    import ipdb
    ipdb.set_trace()
        
if __name__ == "__main__":
  unittest.main()

