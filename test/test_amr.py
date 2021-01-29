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

def create_poisson(x, y, xx, yy):
    p_an = np.sinh(1.5*np.pi*yy / x[-1]) /\
      (np.sinh(1.5*np.pi*y[-1]/x[-1]))*np.sin(1.5*np.pi*xx/x[-1])

    pts = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    return pts, p_an

def create_paraboloid(n_angles = 20,   n_radii = 10, min_radius = 0.15):
  # First create the x and y coordinates of the points.
  radii = np.linspace(min_radius, 0.95, n_radii)
  angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
  angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
  angles[:, 1::2] += math.pi/n_angles
  x = (radii*np.cos(angles)).ravel()
  y = (radii*np.sin(angles)).ravel()

  pts = np.vstack([x, y]).T
  z = x*x + y*y
  return pts, z

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
  
  def testPoisson(self):
    nx, ny = (41, 16)
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    xx, yy = np.meshgrid(x,y)

    pts, p_an = create_poisson(x,y,xx,yy)
    p_an = p_an.ravel()
    space = MeshSpace.from_vertices(points=pts, elevation=p_an)
    space.plotSurface(cmap="viridis",show=False,savefig="test/testPoisson.png")
    plt.close()


  def testTetra(self):
    ms, _grid = MeshSpace.read("/Users/mario/Downloads/malla-Mario_0_0.vtu")
    import ipdb
    #ipdb.set_trace()  
    ms.plotSurface()

  def testAlveolo(self):
    ms, _grid = MeshSpace.read("/Users/mario/Downloads/TFG_ORIOL_CUXART/healthy.vtk")
    import ipdb
    #ipdb.set_trace()  
    ms.plotSurface()


  def testInterpolationPoissonOnPlane(self):
    nx, ny = (41, 16)
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    xx, yy = np.meshgrid(x,y)

    #pts = np.vstack(list(map(np.ravel, [xx,yy] ))).T
    pts, p_an = create_poisson(x,y,xx,yy)
    #print(pts.shape, p_an.shape)    
    p_an = p_an.ravel()

    pspace = MeshSpace.from_vertices(pts, elevation=p_an)
    space = MeshSpace.from_vertices(pts, elevation=0.0)
    
    #print(pts.shape, pspace._tri.triangles.shape)
    #print(pts.shape, pspace._mesh.faces.shape)
    #print(pts.shape, space._tri.triangles.shape)
    #print(pts.shape, space._mesh.faces.shape)

    f = interp.interp1d(np.arange(p_an.size),p_an)
    colors = f(
      np.linspace(
        0,
        p_an.size-1, 
        space._tri.shape[0]
        )
      )
    ax, collec = space.plotSurface(show=False, 
      title="Test plane space with Poisson field",
      cmap="viridis")
    collec.set_array(colors)
    collec.autoscale()
    plt.colorbar(collec)

    pspace.plotSurface(show=False, 
      title="Test plane space with Poisson field",
      cmap="viridis",
      alpha=0.6,
      ax=ax)
    
    plt.savefig("test/testInterpPoissonOnPlane.png")
    
  def testSurfaceCustomField(self):
    pts, z = create_paraboloid()

    space = MeshSpace.from_vertices(points=pts, elevation=z)

    # Plotting
    ax, collec = space.plotSurface(show=False,
      title="Custom field on surface",
      cmap="jet")
    vals = -z #np.sin(pts[:,0]) * np.cos(pts[:,1])
    colors = np.mean(vals[space._tri], axis=1)
    collec.set_array(colors)
    collec.autoscale()
    plt.colorbar(collec)
    plt.savefig("testCustomField.png")
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_trisurf(space._tri, space._elev)
    #plt.show()


  def testRefinement(self):
    """ We start from a simple mesh, calculate a field on it and then refine
        we should check that the field information is copied correctly and then interpolated
    """
    create_paraboloid(n_angles = 20,   n_radii = 10)
    return

    # First create a mesh and put some information on it
    space, grid = MeshSpace.from_meshgrid()
    _h = grid[0]
    x = np.squeeze(grid[0][0,:])
    y = np.squeeze(grid[1][:,0])
    #plt.contourf(x, y, np.squeeze(_h))
    #plt.show()
    #print(space._mesh.faces)
    #print(_h[space._mesh.faces])
    #colors = np.mean(_h[triangles], axis=1)



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cmap = plt.get_cmap('jet')
    collec = ax.plot_trisurf(space._tri, space._elev, cmap=cmap, shade=False, linewidth=0.)
    collec.set_array(_h.ravel())
    collec.autoscale()
    ax.colorbar()
    plt.show()
    #for nid in space.G.nodes:
    #space.G[nid]["h"] = np.sin(grid[0]**2 + grid[1]**2) / (grid[0] + grid[1])
    #  _h = np.sin(grid[0]**2 + grid[1]**2) / (grid[0] + grid[1])
    #space.getField("h")
    # Plot with a colormap for the face values
    # Refine the mesh and copy the info interpolating
    # Determine the mapping from old to new mesh

    # Make it coarser and average


    #space, grid = MeshSpace.from_meshgrid(z=1.0)
    #space.plot() 
    #plt.triplot(mesh.vertices[:,0], points[:,1], tri.simplices)
    #plt.plot(xx, yy, "o")
    #plt.show()            
    
if __name__ == "__main__":
  unittest.main()

