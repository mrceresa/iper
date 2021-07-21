import unittest
from iper.space.geospacepandas import GeoSpaceComposer, GeoSpacePandas
from shapely.geometry import Polygon, LineString, Point
from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from random import uniform

from iper import XAgent
from mesa_geo import AgentCreator, GeoAgent
import geopandas as gpd
import pandas as pd
from random import random
import time

#bounds = [
#    ( uniform(x_min, x_Max), uniform(y_min, y_Max), uniform(x_min, x_Max), uniform(y_min, y_Max) ),
#    ( uniform(x_min, x_Max), uniform(y_min, y_Max), uniform(x_min, x_Max), uniform(y_min, y_Max) ),
#    ( uniform(x_min, x_Max), uniform(y_min, y_Max), uniform(x_min, x_Max), uniform(y_min, y_Max) ),
#    ( uniform(x_min, x_Max), uniform(y_min, y_Max), uniform(x_min, x_Max), uniform(y_min, y_Max) )
#]
#shells = [Polygon.from_bounds(*_s) for _s in shells]

def plotPol(pols, agents=[]):
  fig, axs = plt.subplots()
  axs.set_aspect('equal', 'datalim')
  
  for i, geom in enumerate(pols):    
    plt.plot(*geom.exterior.xy)
    c = geom.centroid
    plt.text(c.x, c.y,str(i))

  for a in agents: 
    plt.plot(*a.pos,"*")

class TestSpaceComposer(unittest.TestCase):

  def setUp(self):    
    x_min, y_min, x_Max, y_Max = (2.052, 4.317, 2.228, 4.467)
    sheet = Polygon.from_bounds( x_min, y_min, x_Max, y_Max )

    self.space = GeoSpaceComposer(extent=[x_min, x_Max, y_min, y_Max], N=1, M=1)


  def testInside(self):
    for _aid in range(100):
       a = XAgent(_aid)
       self.space.place_agent(a, self.space.getRandomPos())

    plotPol(self.space._grids, self.space._agents.values())
    #plt.show()
    plt.savefig("space_composer_inside.png")

    for a in self.space._agents.values():
      g = self.space._w(a.pos)
      gs = self.space._s(g)
      self.assertTrue(type(gs) is GeoSpacePandas)

  def testRemoval(self):
    for _aid in range(100):
       a = XAgent(_aid)
       self.space.place_agent(a, self.space.getRandomPos())

    agents_id = self.space._agents.copy()
    for _toDel in agents_id:
      self.space.remove_agent(_toDel)
      self.assertFalse(_toDel in self.space._agents)

    plotPol(self.space._grids, self.space._agents.values())
    #plt.show()
    plt.savefig("space_composer_remove.png")

  # def testUpdate(self):
  #   self.space.add_geo(self.agents)
  #   _toMove = self.agents[0]
  #   oShape = _toMove.shape
  #   newShape = Point(oShape.x + 1 , oShape.y + 1)
  #   self.space.update_shape(_toMove, newShape)

  #   _t = self.space._agents[id(_toMove)]
  #   self.assertTrue(newShape == _t.shape)

  def testUpdateEfficency(self):

     sp = self.space
     max_ag = 200000
     tic = time.perf_counter()
     for i in range(max_ag):
       a = XAgent(i)
       sp.place_agent(a, sp.getRandomPos())
     toc = time.perf_counter()
     _el1 = toc-tic
     print("Place took:", _el1)

     tic = time.perf_counter()
     positions=[]
     for i in range(max_ag):
       positions.append(sp.getRandomPos())
       sp.move_agent(sp.get_agent(i), positions[-1])
     toc = time.perf_counter()
     _el1 = toc-tic
     print("Move took:", _el1)

     tic = time.perf_counter()
     sp._create_gdf(use_ntrees=True)
     toc = time.perf_counter()
     _el1 = toc-tic
     print("To GDF took:", _el1)

     tic = time.perf_counter()
     for i in range(max_ag):
       res = sp.agents_at(positions[i], radius=2.0/111100)
       self.assertTrue(len(res) < max_ag, "All agents are returned, please check radius is in radiants not meters" )
     toc = time.perf_counter()
     _el1 = toc-tic
     print("Agents_at took:", _el1)

     tic = time.perf_counter()
     for i in range(max_ag):
       sp.remove_agent(sp.get_agent(i))
     toc = time.perf_counter()
     _el1 = toc-tic
     print("Remove took:", _el1)

  # def testGeoInterface(self):
  #   self.space.add_geo(self.agents)
  #   _g1 = self.space.__geo_interface__


if __name__ == '__main__':
    unittest.main()
