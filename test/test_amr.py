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
    #cubic interpolation
    fig, ax = plt.subplots(figsize=(12,9),subplot_kw =dict(projection="3d"))
    ax.scatter(pspace._v[:,0], pspace._v[:,1], p_an, marker='o')
    ax.scatter(space._v[:,0], space._v[:,1], 0.0, marker='^')
    #plt.show()

    #fzc = LinearTriInterpolator(pspace._plt_tri,p_an)
    #colors = fzc(pts[:,0],pts[:,1])

    f = interp.interp1d(np.arange(p_an.size),p_an)
    colors = f(
      np.linspace(
        0,
        p_an.size-1, 
        space._tri.shape[0]
        )
      )
    #import ipdb
    #ipdb.set_trace()
    ax, collec = space.plotSurface(show=False, 
      title="Test plane space with Poisson field",
      cmap="viridis")
    collec.set_array(colors.ravel())
    collec.autoscale()
    plt.colorbar(collec)

    pspace.plotSurface(show=False, 
      title="Test plane space with Poisson field",
      cmap="viridis",
      alpha=0.6,
      ax=ax)
    
    plt.savefig("test/testInterpPoissonOnPlane.png")
    plt.close()
    
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
    plt.close()
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

