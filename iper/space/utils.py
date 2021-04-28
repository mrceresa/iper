
import os, sys
import networkx as nx
import meshio
from iper.space.Space import MeshSpace
import logging
_log = logging.getLogger(__name__)
#import trimesh
#from matplotlib.tri import Triangulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_connectivity_3d_triangles(space, g):
  _log.info("Parsing 3d triangular surface...")
  _v = space._v
  x, y, z = _v[:,0], _v[:,1], _v[:,2]  
  
  #_surface = trimesh.Trimesh(vertices=space._v, faces=space._tri)
  #triang = Triangulation(x, y, triangles=space._tri)
  #space._plt_tri = triang; space._elev = z
  #space._surface = _surface

  _ad, _ed = face_adjacency(faces=space._tri, return_edges=True)
  space._adj = _ad
  _nodes = []
  for i, f in enumerate(space._tri):
    loop = (_v[f[0]], _v[f[1]], _v[f[2]])
    _c = np.mean(np.asarray(loop), axis=0)
    _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

  g.add_nodes_from(_nodes)
  g.add_edges_from(_ad)  
  
  space._adj2 = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))  
  return g

def parse_connectivity_3d_quads(space, g):
  _log.info("Parsing 3d triangular surface...")
  _v = space._v
  x, y, z = _v[:,0], _v[:,1], _v[:,2]  
  
  #_surface = trimesh.Trimesh(vertices=space._v, faces=space._tri)
  #triang = Triangulation(x, y, triangles=space._tri)
  #space._plt_tri = triang; space._elev = z
  #space._surface = _surface

  _ad, _ed = face_adjacency(faces=space._tri, return_edges=True)
  space._adj = _ad
  _nodes = []
  for i, f in enumerate(space._tri):
    loop = (_v[f[0]], _v[f[1]], _v[f[2]])
    _c = np.mean(np.asarray(loop), axis=0)
    _nodes.append( (i, {"vertices":loop, "centroid":_c}) )

  g.add_nodes_from(_nodes)
  g.add_edges_from(_ad)  
  
  space._adj2 = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))  
  return g

def face_adjacency(faces=None, return_edges=False):

    # first generate the list of edges for the current faces
    # also return the index for which face the edge is from
    edges, edges_face = faces_to_edges(faces, return_index=True)
    # make sure edge rows are sorted
    edges.sort(axis=1)

    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx:
    # edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows we discard edges which
    # don't occur twice
    edge_groups = group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.debug('No adjacent faces detected! Did you merge vertices?')

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and
    # self.faces[face_idx[*][1]] will share an edge
    adjacency = edges_face[edge_groups]

    # degenerate faces may appear in adjacency as the same value
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = adjacency[nondegenerate]

    # sort pairs in-place so we can search for indexes with ordered pairs
    adjacency.sort(axis=1)

    if return_edges:
        adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]
        assert len(adjacency_edges) == len(adjacency)
        return adjacency, adjacency_edges
    return adjacency


def faces_to_edges(faces, return_index=False):
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

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)),
                             (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges

def group_rows(data, require_count=None, digits=None):
    """
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]
    Note that using require_count allows numpy advanced
    indexing to be used in place of looping and
    checking hashes and is ~10x faster.
    Parameters
    ----------
    data : (n, m) array
      Data to group
    require_count : None or int
      Only return groups of a specified length, eg:
      require_count =  2
      [[1,2], [3,4], [1,2]] will return [[0,2]]
    digits : None or int
    If data is floating point how many decimals
    to consider, or calculated from tol.merge
    Returns
    ----------
    groups : sequence (*,) int
      Indices from in indicating identical rows.
    """

    # create a representation of the rows that can be sorted
    hashable = hashable_rows(data, digits=digits)
    # record the order of the rows so we can get the original indices back
    # later
    order = np.argsort(hashable)
    # but for now, we want our hashes sorted
    hashable = hashable[order]
    # this is checking each neighbour for equality, example:
    # example: hashable = [1, 1, 1]; dupe = [0, 0]
    dupe = hashable[1:] != hashable[:-1]
    # we want the first index of a group, so we can slice from that location
    # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    # if you wanted to use this one function to deal with non- regular groups
    # you could use: np.array_split(dupe_idx)
    # this is roughly 3x slower than using the group_dict method above.
    start_ok = np.diff(
        np.concatenate((dupe_idx, [len(hashable)]))) == require_count
    groups = np.tile(dupe_idx[start_ok].reshape((-1, 1)),
                     require_count) + np.arange(require_count)
    groups_idx = order[groups]
    if require_count == 1:
        return groups_idx.reshape(-1)
    return groups_idx

def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.
    Parameters
    ---------
    data : (n, m) array
      Input data
    digits : int or None
      How many digits to add to hash if data is floating point
      If None, tol.merge will be used
    Returns
    ---------
    hashable : (n,) array
      Custom data type which can be sorted
      or used as hash keys
    """
    # if there is no data return immediately
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already, return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is significantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # time for some righteous bitbanging
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))
        # if the max value is less than precision we can do this
        if np.abs(as_int).max() < 2**(precision - 1):
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.int64)
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(as_int.astype(np.int64).T):
                # will modify hashable in place
                np.bitwise_xor(hashable,
                               column << (offset * precision),
                               out=hashable)
            return hashable

    # reshape array into magical data type that is weird but hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def float_to_int(data, digits=None, dtype=np.int32):
    """
    Given a numpy array of float/bool/int, return as integers.
    Parameters
    -------------
    data :  (n, d) float, int, or bool
      Input data
    digits : float or int
      Precision for float conversion
    dtype : numpy.dtype
      What datatype should result be returned as
    Returns
    -------------
    as_int : (n, d) int
      Data as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)

    # if data is already an integer or boolean we're done
    # if the data is empty we are also done
    if data.dtype.kind in 'ib' or data.size == 0:
        return data.astype(dtype)
    elif data.dtype.kind != 'f':
        data = data.astype(np.float64)

    # populate digits from kwargs
    if digits is None:
        digits = util.decimal_to_digits(tol.merge)
    elif isinstance(digits, float) or isinstance(digits, np.float64):
        digits = util.decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warning('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')

    # data is float so convert to large integers
    data_max = np.abs(data).max() * 10**digits
    # ignore passed dtype if we have something large
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    as_int = np.round((data * 10 ** digits) - 1e-6).astype(dtype)

    return as_int

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
    ms, mesh = MeshSpace.read_vtk(filename)
  else:
    mesh = meshio.read(filename)
    ms = MeshSpace(mesh, name=filename, debug=debug)
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
  import ipdb
  ipdb.set_trace()
  # Drop first cell that is only the number of points
  numpy_cells = [("triangle", numpy_cells[::10,1:])] 
  mesh = meshio.Mesh(points, numpy_cells)
  ms = MeshSpace(mesh, name="VTK object")

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
  ms = MeshSpace.from_vertices(points, z)
  #plt.triplot(points[:,0], points[:,1], tri.simplices)
  #plt.plot(xx, yy, "o")
  #plt.show()


  return ms, [xx, yy]   

