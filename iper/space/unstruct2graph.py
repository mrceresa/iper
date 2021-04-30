import networkx as nx
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys, os
import pickle
from tqdm import tqdm

def read(fname):
  reader = vtk.vtkXMLUnstructuredGridReader()
  reader.SetFileName(fname)
  reader.Update()  
  ugrid = reader.GetOutput()
  
  return ugrid

def write(nodeCoords, nodes, edges = [], fileout = 'test'):
    """
    Store points and/or graphs as vtkPolyData or vtkUnstructuredGrid.
    Required argument:
    - nodeCoords is a list of node coordinates in the format [x,y,z]
    - nodes is a list of node ids
    Optional arguments:
    - edges is a list of edges in the format [nodeID1,nodeID2]
    - fileout is the output file name (will be given .vtp or .vtu extension)
    """

    print("Exporting network to polydata")
    pmin, pmax = min(nodes), max(nodes)+1
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(pmax)

    # First initialize to 0 to avoid weird nums
    for i in range(pmax):
      points.SetPoint(i, [0,0,0])  
    # Now fill only the one used
    for i, node in enumerate(nodes):
      points.SetPoint(node, coords[i])
    points.Modified()

    if edges:
      line = vtk.vtkCellArray()
      line.Allocate(len(edges))
      for edge in edges:
        line.InsertNextCell(2)
        line.InsertCellPoint(edge[0])
        line.InsertCellPoint(edge[1])   # line from point edge[0] to point edge[1]

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if edges:
      polydata.SetLines(line)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fileout+'.vtp')
    writer.SetInputData(polydata)
    writer.Write()

def points_conn(cid, cell, ugrid):
  neighs_cells = []
  edges = []
  cellPointIds = cell.GetPointIds()    
  for i in range(cellPointIds.GetNumberOfIds()):
    idList = vtk.vtkIdList()
    idList.InsertNextId(cellPointIds.GetId(i))

    # get the neighbors of the cell
    neighborCellIds = vtk.vtkIdList()

    ugrid.GetCellNeighbors(cid, idList, neighborCellIds)

    for j in range(neighborCellIds.GetNumberOfIds()):
      nid = neighborCellIds.GetId(j)
      ncell = ugrid.GetCell(nid)
      if ncell.GetCellType() == vtk.VTK_TETRA:
        neighs_cells.append(nid)
        edges.append( (cid, nid) )

    return neighs_cells, edges

def edges_conn(cid, cell, ugrid):
  neighs_cells = []
  edges = []
  for eid in range(cell.GetNumberOfEdges()):
    e = cell.GetEdge(eid)
    edgePointIds = e.GetPointIds()        

    idList = vtk.vtkIdList()
    idList.InsertNextId(edgePointIds.GetId(0))
    idList.InsertNextId(edgePointIds.GetId(1))    

    neighborCellIds = vtk.vtkIdList()
    
    ugrid.GetCellNeighbors(cid, idList, neighborCellIds)
      
    for j in range(neighborCellIds.GetNumberOfIds()):
      nid = neighborCellIds.GetId(j)
      ncell = ugrid.GetCell(nid)
      if ncell.GetCellType() == vtk.VTK_TETRA:
        neighs_cells.append(nid)
        edges.append( (cid, nid) )

    return neighs_cells, edges
  

if __name__ == '__main__':
  fname = sys.argv[1]
  foutname = "network-" + os.path.basename(fname)
  foutnamep = "network-" + os.path.basename(fname) + ".pkl" 
  print("Reading mesh", fname)
  print("Saving output in ", foutname)
  print("Saving obtained graph in ", foutnamep)      
  ugrid = read(fname)

  adj = []
  coords = []
  g = nx.Graph()
  
  tot_cells = ugrid.GetNumberOfCells()
  print("Starting traversal of %d..."%tot_cells)
  it = ugrid.NewCellIterator()
  it.InitTraversal()
  with tqdm(total=tot_cells) as pbar:
    while not it.IsDoneWithTraversal():
      cid = it.GetCellId()  
      cell = vtk.vtkGenericCell()
      it.GetCell(cell)

      if cell.GetCellType() != vtk.VTK_TETRA:
        it.GoToNextCell()
        pbar.update(1)
        continue

      cellPoints_a = vtk_to_numpy(cell.GetPoints().GetData())
      tetra_centroid = cellPoints_a.mean(axis=0)
      coords.append(tetra_centroid)
      g.add_node(cid, centroid=tetra_centroid)
          
      #neighs_cells, edges = points_conn(cid, cell, ugrid)
      neighs_cells, edges = edges_conn(cid, cell, ugrid)    
      
      g.add_edges_from(edges)
      it.GoToNextCell()
      pbar.update(1)
  
    adj.append(neighs_cells)

  coords = [list(c) for c in coords]
  print('nodes:', len(g.nodes()))
  print('edges:', len(g.edges()))

  write(coords, g.nodes, edges=g.edges, fileout=foutname)  
  
  with open(foutnamep, "wb") as fp:
    pickle.dump(g, fp)  



  
  
