from mesa_geo.geoagent import GeoAgent
import pygtfs
import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import movingpandas as mpd
from datetime import datetime, timedelta
import numpy as np
from pyproj import CRS

import logging
_log = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

# Functions that may be needed
def load_merged_GTFS(self):
    self.merged_GTFS = pd.read_csv((ROOT_DIR + '/GTFS/merged_GTFS.csv'), dtype = str)
def save_merged_GTFS(self):
    self.merged_GTFS.to_csv(ROOT_DIR + '/GTFS/merged_GTFS.csv')
class pygtfs_Schedule():
    def __init__(self):
        sched = pygtfs.Schedule(":memory:")
        pygtfs.append_feed(sched, "GTFS/bus_metro")
        print(sched.agencies)

    def now(date, time):
        self.date = date
        self.time = time
# -----------------------------------------   
class TransportAgent(GeoAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

    def get_stops(self, trajectory):
        return trajectory.df.geometry.unique()

class Human(GeoAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        # Markov transition matrix
        self._vel1step = 0.4 #Km per hora

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

    def step(self):
        ox, oy = self.get_pos()
        newPos = oy + ny, ox + nx
        lat, lng = self.model.driveMap.get_lat_lng_from_point(newPos)
        newPosNode = Point(lng,lat)
        self.place_at(newPosNode)
        #neighbors = self.model.grid.get_neighbors(self)

class Tram(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        max_capacity = 100

    def step(self):
        pass

class Subway(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
        self.max_capacity = 500
        self.passangers = []
        self.trajectory = self.collection.get_trajectory(unique_id)
        self.stops = self.get_stops(self.trajectory)
        self.currentStation = 0

    def step(self):
        currentPos = self.get_pos()
        newPos = self.trajectory.get_pos(model.time)  #time from the main
        self.place_at(newPos)

        if self.stops[currentStation].x == newPos.x and self.stops[currentStation].y == newPos.y:
            #People can enter

        #Check whether the station has changed
        if self.stops[currentStation].x == currentPos.x and self.stops[currentStation].y == currentPos.y and 
        self.stops[currentStation].x != newPos.x and self.stops[currentStation].y != newPos.y:
            self.currentStation += 1
        else:
            # We are still in the station



class Bus(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)   
    def step(self):
        pass

class Car(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def step(self):
        pass

class Bike(TransportAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
    def step(self):
        pass


# DISCUSS
# Work done: 
    # Download GTFS
    # Transform GTFS to Trajectories.
    # Collect them in a collection
    # Get the starting and ending times of each trajectory
    # If the starting time of a trajectory coincides with the current time, create a new
    # instance of the correpsonding transport type.
    # Made the transport subway class 
    # Try to add humans to the transport.


# doubts: 
# Need to deal with more than 86.000.000 of trajectories
# Now working with steps, i need time to make trjectories work. 
# I init each trajectory inside model.step. Is it okay there? 
# Remove still think how to do it explenation in the function
# schedule random activation? First transports then humans. 
    # Maybe if we take into account seconds as time unit does not matter because we are in one stop for more 
    # than one step. 


# Is the structure good? 
# Add persons to transport_grid and therefore transport?
    # Get from the human_grid, the humans that are neigbours to the station location and then 
    # if they will, move them to the passangers list. How to add them into the tramsport grid?



# Check network grid .... tests!!!!


# HOW TIME WORKS
# TRAM FEED IS NOT VALID. SUGESTION: WORK FIRST WITH SUBWAY AND BUS AND THEN FIGURE OUT HOW TO WORK WITH TRAM
# WORKPLAN, NECESSARY TO BE IN THE PROJECT REPORT?    