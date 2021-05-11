import unittest
from iper.space.Space import GeoSpacePandas
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

class MockModel:
  pass

def plotTest(sheet, agents, fname):
  fig = plt.figure(figsize=SIZE, dpi=90)

  # 3: invalid polygon, ring touch along a line
  ax = fig.add_subplot(121)
  plot_coords(ax, sheet.exterior)

  for _a in agents:
      patch = PolygonPatch(
          _a, 
          facecolor=color_isvalid(_a), 
          edgecolor=color_isvalid(_a, valid=BLUE), 
          alpha=0.5, zorder=2)
      ax.add_patch(patch)

  plt.savefig(fname)

class TestGeoSpace(unittest.TestCase):

  def setUp(self):
    self.spacePandas = GeoSpacePandas()
    
    x_min, y_min, x_Max, y_Max = (2.052, 4.317, 2.228, 4.467)
    sheet = Polygon.from_bounds( x_min, y_min, x_Max, y_Max )

    _ids = []; _lats = []; _longs = []
    for _aid in range(10):
      _ids.append(_aid)
      _lats.append(uniform(x_min, x_Max))
      _longs.append(uniform(y_min, y_Max))

    df = pd.DataFrame(
        {'index': _ids,
        'Latitude': _lats,
        'Longitude': _longs
        })

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)
        )


    gdf.crs = self.spacePandas.crs.to_string()

    model = MockModel()
    model.grid = self.spacePandas
    AC = AgentCreator(GeoAgent, {"model": model})
    self.agents = AC.from_GeoDataFrame(gdf)

  def testRemoval(self):

    self.spacePandas.add_geo(self.agents)
    _toDel = self.agents[0]
    self.spacePandas.remove_agent(_toDel)
    self.assertFalse(_toDel in self.spacePandas.agents)

  def testUpdate(self):
    self.spacePandas.add_geo(self.agents)
    _toMove = self.agents[0]
    oShape = _toMove.shape
    newShape = Point(oShape.x + 1 , oShape.y + 1)
    self.spacePandas.update_shape(_toMove, newShape)

    _t = self.spacePandas._agents[id(_toMove)]
    self.assertTrue(newShape == _t.shape)

  def testUpdateEfficency(self):
    sp = GeoSpacePandas()

    tic = time.perf_counter()
    for i in range(1000):
      a = XAgent(i)
      sp.place_agent(a, (i,i))
    toc = time.perf_counter()
    _el1 = toc-tic
    print("Place took:", _el1)

    tic = time.perf_counter()
    for i in range(1000):
      sp.move_agent(sp.get_agent(i), (i**2,i**2))
    toc = time.perf_counter()
    _el1 = toc-tic
    print("Move took:", _el1)

    tic = time.perf_counter()
    sp._create_gdf()
    print(sp._agdf)
    toc = time.perf_counter()
    _el1 = toc-tic
    print("To GDF took:", _el1)

    tic = time.perf_counter()
    for i in range(1000):
      sp.remove_agent(sp.get_agent(i))
    toc = time.perf_counter()
    _el1 = toc-tic
    print("Remove took:", _el1)

  def testGeoInterface(self):
    self.spacePandas.add_geo(self.agents)
    _g1 = self.spacePandas.__geo_interface__


if __name__ == '__main__':
    unittest.main()
