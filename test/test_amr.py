import unittest
import iper
from utils import captured_output
import trimesh
trimesh.util.attach_to_log()

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

def plot(mesh, ax1):
  _v = mesh.vertices
  x, y = mesh.vertices[:,0], mesh.vertices[:,1]
  triang = tri.Triangulation(x, y, triangles=mesh.faces)
  ax1.set_aspect('equal')
  ax1.triplot(triang, 'bo-', lw=1)
  ax1.set_title('triplot')

  _cs = {}
  for i, e in enumerate(mesh.edges_sorted):
    loop = [_v[e[0]], _v[e[1]]]
    _c = np.mean(np.asarray(loop), axis=0)
    _cs.setdefault((_c[0],_c[1]), []).append("%d"%i)

  for _k, _t in _cs.items():
    plt.annotate(",".join(_t), (_k[0],_k[1]), color='r', ha='center', va='center', size=10)

  for i, f in enumerate(mesh.faces):
    #print(i, f)
    loop = [_v[f[0]], _v[f[1]], _v[f[2]]]
    #_p = Polygon( loop )
    #_c = _p.centroid
    #print(np.asarray(loop), "X:", _c.x, "Y:", _c.y, "np", np.mean(np.asarray(loop), axis=0))
    _c = np.mean(np.asarray(loop), axis=0)
    plt.annotate("%d"%i, (_c[0], _c[1]), color='b', ha='center', va='center', size=12)
    #patch = PolygonPatch(_p, facecolor=(1,0,0), edgecolor=(1,0,0),
    #                 alpha=0.5, zorder=1)
    #ax1.add_patch(patch)
  
  #plt.show()

class TestMeshAsGraph(unittest.TestCase):
  def setUp(self):
    #self._grid = AMRSpace()
    #self._grid.load_example_mesh("lattice_3x3")
    # mesh objects can be created from existing faces and vertex data
    pass


  def testGraph3D(self):
    #print("Vertices",self._grid.vertices)
    #print("Faces (%d)"%len(self._grid.faces),self._grid.faces)
    _grid = trimesh.load("test/meshes/cube_plane.msh")
    ms = MeshSpace(_grid)
    
    self.assertTrue(len(_grid.faces) == len(ms.G.nodes))
    ms.plot(savefig="testCubeMesh3D.png")
    plt.close()

    _grid = trimesh.load("test/meshes/sphere.msh")
    ms = MeshSpace(_grid)
    
    self.assertTrue(len(_grid.faces) == len(ms.G.nodes))
    ms.plot(savefig="testShpereMesh3D.png")
    plt.close()

    _grid = trimesh.load("test/meshes/indheat.msh")
    ms = MeshSpace(_grid)
    
    self.assertTrue(len(_grid.faces) == len(ms.G.nodes))
    ms.plot(savefig="testComplexMesh3D.png")
    plt.close()

  def testGraph2D(self):
    #print("Vertices",self._grid.vertices)
    #print("Faces (%d)"%len(self._grid.faces),self._grid.faces)
    _grid = trimesh.load("test/meshes/plane.msh")

    ms = MeshSpace(_grid)
    
    self.assertTrue(len(_grid.faces) == len(ms.G.nodes))
    ms.plot(savefig="testPlaneMesh2D.png")
    plt.close()
    #fig1, ax1 = plt.subplots(figsize=(12,9), projection='3d')

    #plot(grid, ax1)
    #plt.savefig("testGraph.png")

    #self.assertTrue(self._grid.size == (3,3))

    #self.grid.refine()

  def testGraphMovement2D(self):
    _grid = trimesh.load("test/meshes/plane.msh")

    ms = MeshSpace(_grid)
    nodes = ms.G.nodes
    starting_node = 0
    a = Agent(0, None)
    ms.place_agent(a, starting_node)
    possible_steps = ms.get_neighbors(
        a.pos,
        include_center=False
    )
    self.assertEqual(possible_steps, [4, 19, 2])
    new_position = random.choice(possible_steps)
    ms.move_agent(a, new_position)
    self.assertTrue(a.pos == new_position)

  def testGraphMovement3D(self):
    _grid = trimesh.load("test/meshes/sphere.msh")

    ms = MeshSpace(_grid)
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

            
    
if __name__ == "__main__":
  unittest.main()
