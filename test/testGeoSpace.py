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
    for _aid in range(1000):
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

    gdf.crs = self.spacePandas.crs.crs.to_string()
    AC = AgentCreator(GeoAgent, {"model":None})
    self.agents = AC.from_GeoDataFrame(gdf)

    self.spaceRT.add_agents(self.agents)

  def testInsertion(self):

    self.spacePandas.add_agents(self.agents)
    for _a in self.spacePandas.agents:
      self.assertTrue(_a in self.spaceRT.agents)

if __name__ == '__main__':
    unittest.main()
