
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
from tqdm import tqdm
import vtk
from networkx.convert_matrix import from_scipy_sparse_matrix, to_scipy_sparse_matrix

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
  #adjacency = edges_face[edge_groups]

  # degenerate faces may appear in adjacency as the same value
  #nondegenerate = adjacency[:, 0] != adjacency[:, 1]
  #adjacency = adjacency[nondegenerate]

  # sort pairs in-place so we can search for indexes with ordered pairs
  np.sort(adjacency, axis=1)

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
  elif filename.endswith(".vtu"):
    ms, mesh = read_vtu(filename)
  else:
    mesh = meshio.read(filename)
    ms = MeshSpace(mesh, name=os.path.basename(filename), debug=debug)
  return ms, mesh

def from_meshio(points, cells):
  mesh = meshio.Mesh(points, cells)
  return MeshSpace(mesh, name="Meshio object")

def read_vtu(filein, debug=False, fileout=None):
  cells = []

  rd = vtk.vtkXMLUnstructuredGridReader()
  rd.SetFileName(filein)
  rd.Update()
  mesh = rd.GetOutput()
 
  g = nx.Graph()

  #Get point cells
  cc = vtk.vtkCellCenters()
  cc.SetInputData( mesh )
  cc.Update()
  centers = cc.GetOutput()

  points = []
  for pid in range(centers.GetNumberOfPoints()):
    points.append(centers.GetPoint(pid))

  #bulk = []; bulk_conn = []
  #bcs = []; bcs_conn = []
  #others = []; others_conn = []
  connectivity = []
  for cellid in tqdm(range(mesh.GetNumberOfCells())):
    cell = mesh.GetCell(cellid)
    if cell.GetCellType() != vtk.VTK_TETRA:
      continue

    g.add_node(cellid)    
    
    cell_ptids = vtk.vtkIdList()      
    mesh.GetCellPoints(cellid,cell_ptids)

    s_cell_ptids = set( 
      [ cell_ptids.GetId(k) for k in range(cell_ptids.GetNumberOfIds()) ] 
      )

    cells.append(list(s_cell_ptids)) #Create cell connectivity structure
    
    # Now 
    aset = set()
    for pti in s_cell_ptids:
      # for each of the points, check for confining cells
      cell_neigh_ids = vtk.vtkIdList()
      mesh.GetPointCells( pti, cell_neigh_ids )

      # For each confining cell
      for neigh_cell_i in range(cell_neigh_ids.GetNumberOfIds()):
        neigh_cell_id = cell_neigh_ids.GetId(neigh_cell_i)      
        # Check if it a valid one
        ncell = mesh.GetCell(neigh_cell_id)
        if ncell.GetCellType() != vtk.VTK_TETRA:
          continue
        
        # Get all its points
        ncell_ptids = vtk.vtkIdList()
        mesh.GetCellPoints( neigh_cell_id, ncell_ptids )
        neighb_cellpts = set( [ ncell_ptids.GetId(k) for k in range(ncell_ptids.GetNumberOfIds()) ] )

        #only cells connected to the face
        if len( s_cell_ptids.intersection(neighb_cellpts) )==3:
          aset.add( neigh_cell_id )

    # Store neighs cells as edges
    #g.add_edges_from([(cellid, ed) for ed in aset])

    #if len(aset) == 3:
    #  bcs.append(cellid) 
    #  bcs_conn.append(list(aset))
    #elif len(aset) == 4:
    #  bulk.append(cellid)
    #  bulk_conn.append(list(aset))
    #else:
    #  others.append(cellid)
    #  others_conn.append(list(aset))      

    connectivity.append( list(aset) ) #subtract middle cell
    for l in list(aset):
      g.add_edge( cellid, l )  

  #len(g.nodes) 2452707
  #1351940 bulk have conn == 4
  #bulk = np.asarray(bulk); bulk_conn = np.asarray(bulk_conn)
  # 548084 bcs have conn == 3
  #bcs = np.asarray(bcs); bcs_conn = np.asarray(bcs_conn)  
  #   1533 others have conn == 2
  #others = np.asarray(others); others_conn = np.asarray(others_conn)    
  #1901557 tot  
  #all_cells = np.hstack((bulk, bcs, others))     #1901557
  #ubulk_cells = np.unique(bulk_conn.reshape(-1)) #1878758
  #ubcs_cells = np.unique(bcs_conn.reshape(-1))   #1026020
  #uothers_cells = np.unique(others_conn.reshape(-1)) #3066
  #all_neighs = np.hstack((ubulk_cells, ubcs_cells, uothers_cells))  #2907844
  #uall_neighs = np.unique(all_neighs) #1901557
  
  #s_all_cells = set(all_cells)
  #s_all_neighs = set(uall_neighs)  
  
  #inter = s_all_cells.intersection(s_all_neighs) #len(inter) == 1901557
  #differen = s_all_cells.difference(s_all_neighs) # 0

  #print("All cells shape:", all_cells.shape)

  #g.add_nodes_from(all_cells)
  
  #s_all_nodes = set(g.nodes)
  #print("G NODES LEN:", len(g.nodes))
  #print(len(s_all_nodes.difference(s_all_cells)))
  #assert len(g.nodes)==1901557
  #for i, cid in enumerate(bulk):
  #  for k in bulk_conn[i]:
  #    assert i in g.nodes
  #    assert k in g.nodes

  #for i, cid in enumerate(bcs):
  #  for k in bcs_conn[i]:
  #    assert i in g.nodes
  #    assert k in g.nodes

  #for i, cid in enumerate(others):
  #  for k in bulk_conn[i]:
  #    assert i in g.nodes
  #    assert k in g.nodes
  
  #g.add_nodes_from(all_cells)
  
  if fileout is not None:
    lines = vtk.vtkCellArray()
    for i, kkk in enumerate(connectivity):
      for cellid in kkk:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint(cellid)
         
    pd = vtk.vtkPolyData()
    pd.SetPoints( centers.GetPoints() )
    pd.SetLines(lines)

    wr = vtk.vtkPolyDataWriter()
    wr.SetFileName(fileout)
    wr.SetInputData(pd)
    wr.Write()

  cells = [("tetra", np.asarray(cells))]
  mesh = meshio.Mesh(np.asarray(points), cells)
  ms = MeshSpace(mesh, name=os.path.basename(filein), 
    compute_conn=False, g3=g)
    
  ms._adj = to_scipy_sparse_matrix(g)

  #import ipdb
  #ipdb.set_trace()

  return ms, mesh


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

