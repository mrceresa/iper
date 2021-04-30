
import os, sys
import networkx as nx
import meshio
from iper.space.Space import MeshSpace
import logging
_log = logging.getLogger(__name__)
from trimesh.grouping import group_rows, hashable_rows, float_to_int
#from matplotlib.tri import Triangulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def parse_connectivity_3d_triangles(space):
  _log.info("Parsing 3d triangular surface...")
  _v = space._v
  x, y, z = _v[:,0], _v[:,1], _v[:,2]  
  
  #_surface = trimesh.Trimesh(vertices=space._v, faces=space._tri)
  #triang = Triangulation(x, y, triangles=space._tri)
  #space._plt_tri = triang; space._elev = z
  #space._surface = _surface

  _ad = face_adjacency(faces=space._tri, eltype="triang")
   #space._adj = _ad
  _nodes = []
  for i, f in enumerate(space._tri):
    loop = (_v[f[0]], _v[f[1]], _v[f[2]])
    _c = np.mean(np.asarray(loop), axis=0)
    _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

  g = nx.Graph()
  g.add_nodes_from(_nodes)
  g.add_edges_from(_ad)  
  
  _adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))  
  return g, _adj

def parse_connectivity_3d_quads(space):
  _log.info("Parsing 3d quadrangular surface...")
  _v = space._v
  x, y, z = _v[:,0], _v[:,1], _v[:,2]  
  
  #_surface = trimesh.Trimesh(vertices=space._v, faces=space._tri)
  #triang = Triangulation(x, y, triangles=space._tri)
  #space._plt_tri = triang; space._elev = z
  #space._surface = _surface

  _ad = face_adjacency(faces=space._quad, eltype="quad")

  #space._adj = _ad
  _nodes = []
  for i, f in enumerate(space._quad):
    loop = (_v[f[0]], _v[f[1]], _v[f[2]], _v[f[3]])
    _c = np.mean(np.asarray(loop), axis=0)
    _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

  g = nx.Graph()
  g.add_nodes_from(_nodes)
  g.add_edges_from(_ad)
  
  _adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))  
  return g, _adj
  
def parse_connectivity_3d_tetra(space):
  _log.info("Parsing 3d tetraedrical volume...")
  _v = space._v
  x, y, z = _v[:,0], _v[:,1], _v[:,2]  
  
  #_surface = trimesh.Trimesh(vertices=space._v, faces=space._tri)
  #triang = Triangulation(x, y, triangles=space._tri)
  #space._plt_tri = triang; space._elev = z
  #space._surface = _surface

  _ad = face_adjacency_tetra(cells=space._tetra, eltype="tetra")
  #space._adj = _ad
  _nodes = []
  for i, f in enumerate(space._tetra):
    loop = (_v[f[0]], _v[f[1]], _v[f[2]], _v[f[3]])
    _c = np.mean(np.asarray(loop), axis=0)
    _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

  g = nx.Graph()
  g.add_nodes_from(_nodes)
  g.add_edges_from( _ad )  
  
  _adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))  
  return g, _adj

def face_adjacency_tetra(cells, eltype):
  cells = np.asanyarray(cells)
  if eltype == "tetra": #We are working with triangles
    # each face has six edges
    faces = cells[:, [0, 1, 2, 0, 0, 1, 3, 0, 0, 2, 3, 0, 1, 2, 3, 1]]
    faces_dim = 4
   
  faces = faces.reshape((-1, faces_dim))

  # edges are in order of faces due to reshape
  face_index = np.tile(np.arange(len(cells)),
                       (faces_dim, 1)).T.reshape(-1)
 
  # make sure face rows are sorted
  faces.sort(axis=1)
  
  unq, cnt = np.unique(faces, axis=0, return_counts=True)
  _, inv_idx = np.unique(faces, axis=0, return_inverse=True)   
  
  _shared = np.where(cnt>1)[0] #Position of shared edges
  
  adjacency = []
  for _s in _shared:
    adjacency.append(face_index[np.in1d(inv_idx, _s).nonzero()[0]])
  
  return adjacency


def face_adjacency(faces, eltype, return_edges=False, debug=False):

  # first generate the list of edges for the current faces
  # also return the index for which face the edge is from
  edges, edges_face = faces_to_edges(faces, eltype, return_index=True)

  # make sure edge rows are sorted
  edges.sort(axis=1)

  ed_unq, ed_cnt = np.unique(edges, axis=0, return_counts=True)
  _, ed_inv_idx = np.unique(edges, axis=0, return_inverse=True)    

  _shared = np.where(ed_cnt>1)[0] #Position of shared edges
  adjacency = []
  for _s in _shared:
    adjacency.append(edges_face[np.in1d(ed_inv_idx, _s).nonzero()[0]])
  
  # the pairs of all adjacent faces
  # so for every row in face_idx, self.faces[face_idx[*][0]] and
  # self.faces[face_idx[*][1]] will share an edge
  adjacency = edges_face[edge_groups]

  # degenerate faces may appear in adjacency as the same value
  nondegenerate = adjacency[:, 0] != adjacency[:, 1]
  adjacency = adjacency[nondegenerate]

  # sort pairs in-place so we can search for indexes with ordered pairs
  adjacency.sort(axis=1)

  return adjacency


def faces_to_edges(faces, eltype, return_index=False):
  """
  Given a list of faces (n,3), return a list of edges (n*3,2)
  Parameters
  -----------
  faces : (n, 3) int
    Vertex indices representing faces
  Returns
  -----------
  edges : (n*3, 2) int
    Vertex indices representing edges
  """
  faces = np.asanyarray(faces)
  if eltype == "triang": #We are working with triangles
    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]]
    faces_dim = 3
  elif eltype == "quad": #We are working with quads
    # each face has four edges
    edges = faces[:, [0, 1, 1, 2, 2, 3, 3, 0]]
    faces_dim = 4
  elif eltype == "tetra": #We are working with tetra
    # each face has six edges
    edges = faces[:, [0, 1, 1, 2, 1, 3, 2, 3, 2, 0, 3, 0]]
    faces_dim = 6
    
  edges = edges.reshape((-1, 2))

  if return_index:
      # edges are in order of faces due to reshape
      face_index = np.tile(np.arange(len(faces)),
                           (faces_dim, 1)).T.reshape(-1)
      return edges, face_index
  return edges


def _graph_as_fig(figfile, g, colors=None, cm='gray', clim=None, pos=None):
  #spring_3D = nx.spring_layout(g, k = 0.5)
  fig = plt.figure()
  ax = Axes3D(fig)  
  if colors is None:
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], edgecolors='k', alpha=0.7)  
  else:
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, edgecolors='k', alpha=0.7)
    #nx.draw(g, node_color=colors, cmap=cm)
    vmin = min(colors)
    vmax = max(colors)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []  
    plt.colorbar(sm)      

  for _e in g.edges:
    ax.plot3D(pos[_e,0], pos[_e,1], pos[_e,2],"gray")
  plt.savefig(figfile)  
  plt.close(fig)




  
def read(filename, debug=False):
  if filename.endswith(".vtk"):
    ms, mesh = read_vtk(filename)
  else:
    mesh = meshio.read(filename)
    ms = MeshSpace(mesh, name=os.path.basename(filename), debug=debug)
  return ms, mesh

def from_meshio(points, cells):
  mesh = meshio.Mesh(points, cells)
  return MeshSpace(mesh, name="Meshio object")

def read_vtk(filename, debug=False):
  from vtkmodules.vtkIOLegacy import vtkPolyDataReader
  from vtk.util.numpy_support import vtk_to_numpy
  from vtkmodules.vtkCommonCore import vtkIdList
  
  reader = vtkPolyDataReader()
  reader.SetFileName(filename)
  reader.ReadAllScalarsOn()
  reader.ReadAllVectorsOn()
  reader.Update()
  polydata = reader.GetOutput()

  points = vtk_to_numpy(polydata.GetPoints().GetData())
  
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
  mesh = meshio.Mesh(points, numpy_cells)
  ms = MeshSpace(mesh, name=os.path.basename(filename))

  return ms, mesh

def from_vertices(points, elevation=None):
  tess = Delaunay(points)
  #triang = mtri.Triangulation(x=points[:, 0], y=points[:, 1], triangles=tri)
  if elevation is not None:
    nx, ny = points.shape
    _tp = np.zeros((nx, ny+1))
    _tp[:,:-1] = points 
    _tp[:,-1] = elevation
    points = _tp    

  cells = [("triangle", tess.simplices)]
  mesh = meshio.Mesh(points, cells)
  ms = MeshSpace(mesh)
  return ms


def from_meshgrid(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0, z=0.0, xp=10, yp=10):
  nx, ny = (xp, yp)
  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  xx, yy= np.meshgrid(x, y)
  points = np.vstack(list(map(np.ravel, [xx,yy] ))).T
  ms = from_vertices(points, z)
  #plt.triplot(points[:,0], points[:,1], tri.simplices)
  #plt.plot(xx, yy, "o")
  #plt.show()


  return ms, [xx, yy]   

