from mesa import Agent
from mesa_geo.geoagent import GeoAgent
import pygtfs

class Agent(GeoAgent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def place_at(self, newPos):
        self.model.place_at(self, newPos)

    def get_pos(self):
        return (self.shape.x, self.shape.y)

class Human(Agent):
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

class Tram(Agent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def step(self):
        pass

class Subway(Agent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
    def step(self):
        pass

class Bus(Agent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)   
    def step(self):
        pass

class Car(Agent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)

    def step(self):
        pass

class Bike(Agent):
    def __init__(self, unique_id, model, shape):
        super().__init__(unique_id, model, shape)
    def step(self):
        pass
    
class Stops():
    pass

class Schedule():
    def __init__(self, date, time):
        self.date = date
        self.time = time

    def now(date, time):ç
        pass
    

if __name__ == "__main__":
    sched = pygtfs.Schedule(":memory:")
    pygtfs.append_feed(sched, "GTFS/bus_metro")
    print(sched.agencies)


    
# DISCUSS
# WORK AS TRAJECTORIES PRELOADED OR CHECK TIME AND READ FEED 
# HOW TIME WORKS
# TRAM FEED IS NOT VALID. SUGESTION: WORK FIRST WITH SUBWAY AND BUS AND THEN FIGURE OUT HOW TO WORK WITH TRAM
# WORKPLAN, NECESSARY TO BE IN THE PROJECT REPORT? 

    