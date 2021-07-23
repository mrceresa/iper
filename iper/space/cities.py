from .geospacepandas import GeoSpacePandas
import logging
import contextily as ctx
from shapely.geometry import Polygon, LineString, Point
import math
import os
import geopandas as gpd
from .mobility import Map_to_Graph

class CitySpace(GeoSpacePandas):
    def __init__(self, basemap, path_name, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.l = logging.getLogger(__name__)

      self._basemap = basemap

      self.l.info("Loading geodata")
      self._initGeo()
      self._loadGeoData()


      self._transport = {
        "walk": Map_to_Graph('Pedestrian',path_name),
        "drive": None,
        "metro": None,
        "bus": None,
      }

      self._nodes = {
        "walk": list(self._transport["walk"].G.nodes),
        "drive": None,
        "metro": None,
        "bus": None,
      }


      for k,v in self._transport.items():
        if v is not None: self.define_boundaries_from_graphs(v)

    def getRandomNode(self):
      return random.choice(self._nodes["walk"])

    def getNodePosition(self, node):
        return (self._transport[node]['lon'],
                self._transport[node]['lat'])


    def out_of_bounds(self, pos):
        xmin, ymin, xmax, ymax = self._xs["w"], self._xs["s"], self._xs["e"], self._xs["n"]

        if pos[0] < xmin or pos[0] > xmax: return True
        if pos[1] < ymin or pos[1] > ymax: return True
        return False

    def _initGeo(self):
        # Initialize geo data
        w,s,n,e = self._extent 
        zoom = ctx.tile._calculate_zoom(w, s, e, n)
        im2, ext = ctx.bounds2img(w,s,e,n,zoom,ll=True)
        # Print some metadata
        self._xs = {"w":w,"s":s,"n":n,"e":e,"image":im2,"ext":ext}
        #self._loc = ctx.Place(self._basemap, zoom_adjust=0)  # zoom_adjust modifies the auto-zoom
        # Print some metadata
        #self._xs = {}


        # Longitude w,e Latitude n,s

        #for attr in ["w", "s", "e", "n", "place", "zoom", "n_tiles"]:
        #    self._xs[attr] = getattr(self._loc, attr)
        #    self.l.debug("{}: {}".format(attr, self._xs[attr]))

        self._xs["centroid"] = LineString(
            (
                (self._xs["w"], self._xs["s"]),
                (self._xs["e"], self._xs["n"])
            )
        ).centroid

        self._xs["bbox"] = Polygon.from_bounds(
            self._xs["w"], self._xs["s"],
            self._xs["e"], self._xs["n"]
        )

        self._xs["dx"] = 111.32;  # One degree in longitude is this in KM
        self._xs["dy"] = 40075 * math.cos(self._xs["centroid"].y) / 360
        self._xs["ddx"] = 40.0/self._xs["dx"]*1000


        self.l.info("Arc amplitude at this latitude %f, %f" % (self._xs["dx"], self._xs["dy"]))

    def define_boundaries_from_graphs(self, map):
        self.boundaries = map.get_boundaries()

        self.boundaries['centroid'] = LineString(
            (
                (self.boundaries["w"], self.boundaries["s"]),
                (self.boundaries["e"], self.boundaries["n"])
            )).centroid

        self.boundaries["bbox"] = Polygon.from_bounds(
            self.boundaries["w"], self.boundaries["s"],
            self.boundaries["e"], self.boundaries["n"])

        self.boundaries["dx"] = 111.32;  # One degree in longitude is this in KM
        self.boundaries["dy"] = 40075 * math.cos(self.boundaries["centroid"].y) / 360
        self.l.info("Arc amplitude at this latitude %f, %f" % (self.boundaries["dx"], self.boundaries["dy"]))

    def _loadGeoData(self):
        path = os.getcwd()
        shpfilename = os.path.join(path, "shapefiles", "quartieriBarca1.shp")
        if not os.path.exists(shpfilename):
            shpfilename = os.path.join(path, "examples/bcn_multispace/shapefiles", "quartieriBarca1.shp")
        #print("Loading shapefile from", shpfilename)
        blocks = gpd.read_file(shpfilename)
        self._blocks = blocks