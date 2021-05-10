import unittest
from iper.space.Space import GeoSpacePandas, GeoSpaceQR
from shapely.geometry import Polygon, LineString, Point
from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from random import uniform

from mesa_geo import AgentCreator, GeoAgent
import geopandas as gpd
import pandas as pd

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
    self.spaceRT = GeoSpaceQR()
    
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

    self.spaceRT.add_agents(self.agents)

  def testInsertion(self):

    self.spacePandas.add_agents(self.agents)
    for _a in self.spacePandas.agents:
      self.assertTrue(_a in self.spaceRT.agents)

  def testRemoval(self):

    self.spacePandas.add_agents(self.agents)
    _toDel = self.agents[0]
    self.spacePandas.remove_agent(_toDel)
    self.assertFalse(_toDel in self.spacePandas.agents)

  def testUpdate(self):
    self.spacePandas.add_agents(self.agents)
    _toMove = self.agents[0]
    oShape = _toMove.shape
    newShape = Point(oShape.x + 1 , oShape.y + 1)
    self.spacePandas.update_shape(_toMove, newShape)

    _t = self.spacePandas._agents[id(_toMove)]
    self.assertTrue(newShape == _t.shape)

  def testGeoInterface(self):
    self.spacePandas.add_agents(self.agents)
    _g1 = self.spacePandas.__geo_interface__
    _g2 = self.spaceRT.__geo_interface__

    self.assertEqual(_g1, _g2)

if __name__ == '__main__':
    unittest.main()
