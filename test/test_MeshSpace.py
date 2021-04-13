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

  def testMeshGrid(self):
    space, grid = MeshSpace.from_meshgrid(z=1.0)
    space.plotSurface(savefig="test/testFromMeshgrid.png", 

      show=False, 
      title="Test meshgrid space",
      cmap="jet") 
    plt.close()
    #plt.triplot(mesh.vertices[:,0], points[:,1], tri.simplices)
    #plt.plot(xx, yy, "o")
    #plt.show()


  def testGraph3D(self):
    #print("Vertices",self._grid.vertices)
    #print("Faces (%d)"%len(self._grid.faces),self._grid.faces)
    ms, _grid = MeshSpace.read("test/meshes/cube_plane.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="test/testCubeMesh3D.png", show=False, title="Test cubic space")
    plt.close()

    ms, _grid = MeshSpace.read("test/meshes/sphere.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="test/testShpereMesh3D.png", show=False, title="Test shperical space")
    plt.close()

    ms, _grid = MeshSpace.read("test/meshes/indheat.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="test/testComplexMesh3D.png", alpha=0.3, show=False, title="Test complex space")
    plt.close()

  def testGraph2D(self):
    ms, _grid = MeshSpace.read("test/meshes/plane.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="test/testPlaneMesh2D.png",show=False, title="Test plane mesh")
    plt.close()

    ms, _grid = MeshSpace.read("test/meshes/cube_plane.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="testCubeMesh3D.png", show=False, title="Test cubic space")
    plt.close()

    ms, _grid = MeshSpace.read("test/meshes/sphere.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="testShpereMesh3D.png", show=False, title="Test shperical space")
    plt.close()

    ms, _grid = MeshSpace.read("test/meshes/indheat.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="testComplexMesh3D.png", alpha=0.3, show=False, title="Test complex space")
    plt.close()

  def testPlane2D(self):
    ms, _grid = MeshSpace.read("test/meshes/plane.msh")
    
    self.assertTrue(len(_grid.cells_dict["triangle"]) == len(ms.G.nodes))
    ms.plotSurface(savefig="testPlaneMesh2D.png", 
      show=False, 
      title="Test Plane space",
      cmap="jet")
    plt.close()


  def testGraphMovement2D(self):
    ms, _grid = MeshSpace.read("test/meshes/plane.msh")

    nodes = ms.G.nodes
    starting_node = 0
    ending_node = 10
    a = Agent(0, None)
    short_path = (nx.shortest_path(ms.G,starting_node,ending_node))
    # check short path 
    self.assertEqual(short_path, [0, 19, 10])
    ms.place_agent(a, starting_node)
    track_move = 0

    while a.pos != ending_node:
      possible_steps = ms.get_neighbors(a.pos,include_center=False)
      #check neighbors for pos 0 pos 1 and pos 2
      if a.pos == short_path[0]:
        self.assertEqual(possible_steps, [4, 19, 2])
      elif a.pos == short_path[1]:
        self.assertEqual(possible_steps, [14, 10, 0])
     
      
      for neighbor in possible_steps: 
        if neighbor == short_path[track_move+1]:
          new_position = neighbor
          break
      
      #check we have found the new path 
      self.assertNotEqual(a.pos, new_position)

      #new_position = random.choice(possible_steps)
      ms.move_agent(a, new_position)
      self.assertTrue(a.pos == new_position)
      track_move += 1

    #Check that we have arrived to our destination
    self.assertTrue(a.pos == ending_node)

  def testGraphMovement3D(self):

    ms, _grid = MeshSpace.read("test/meshes/sphere.msh")
    nodes = ms.G.nodes
    starting_node = 0
    a = Agent(0, None)
    ms.place_agent(a, starting_node)
    possible_steps = ms.get_neighbors(
        a.pos,
        include_center=False
    )
    self.assertEqual(possible_steps, [31, 43, 6])
    new_position = random.choice(possible_steps)
    ms.move_agent(a, new_position)
    self.assertTrue(a.pos == new_position)  

  
  def testTetra(self):
    home = expanduser("~")
    fname = os.path.join(home, "Downloads","malla-Mario_0_0.vtu")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)
    ms, _grid = MeshSpace.read(fname)
    ms.plotSurface()
    plt.close()

  def testPolydata(self):
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkIdList
    
    home = expanduser("~")
    fname = os.path.join(home, "Downloads","malla_sup.vtk")
    self.assertTrue(os.path.exists(fname), "Missing mesh file %s"%fname)

    reader = vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()

    points = vtk_to_numpy(polydata.GetPoints().GetData())

    #def getCellIds(data):
    #  cells = data.GetPolys()
    #  cell_ids = []
    #  points_per_cell = []
    #  idList = vtkIdList()
    #  cells.InitTraversal()
    #  while cells.GetNextCell(idList):
    #      points_per_cell.append(idList.GetNumberOfIds())
    #      for i in range(0, idList.GetNumberOfIds()):
    #          pId = idList.GetId(i)
    #          cell_ids.append(pId)
    #  return np.array(cell_ids), points_per_cell

    #cell_ids = np.array(cell_ids)
    cells = polydata.GetPolys()
    nCells = cells.GetNumberOfCells()
    array = cells.GetData()
    # This holds true if all polys are of the same kind, e.g. triangles.
    assert(array.GetNumberOfValues()%nCells==0)
    nCols = array.GetNumberOfValues()//nCells
    numpy_cells = vtk_to_numpy(array)
    numpy_cells = numpy_cells.reshape((-1,nCols))
    # Drop first cell that is only the number of points
    numpy_cells = [("triangle", numpy_cells[::10,1:])] 
    ms = MeshSpace.from_meshio(points, numpy_cells) 
    ms.plotSurface(show=False, 
      title="Test vtk polydata alveolar mesh (decimated)",
      cmap="viridis",
      savefig="test/testVTKPolydata.png",
      alpha=0.6)
    plt.close()
    
if __name__ == "__main__":
  unittest.main()

