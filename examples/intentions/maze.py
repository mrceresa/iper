
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from iper import MultiEnvironmentWorld, XAgent, PopulationRequest
import random
import numpy as np

import os

from xml.dom import minidom
import untangle


from iper import GeoSpacePandas
from iper import XAgent
from iper.behaviours.actions import MoveTo, Eat
from iper.brains import ScriptedBrain

import logging
_log = logging.getLogger(__name__)

#Random cat movement (respects obstacles)
class CatBrain(ScriptedBrain):
    def __init__(self, agent):
        super().__init__(agent)
        #self._model._nA = len(self._agent.getPossibleActions())
        #self._model._nS = len(self._agent.getPossibleStates())    
        #self._model.reset()

    def think(self, status, reward):


        cellmates = self._agent.model.grid.get_cell_list_contents([self._agent.pos])
        '''
        print("Cat is looking at the map:")
        for i in range(self._agent.model.grid.height):
            print(self._agent.model.grid[i])
        #for cell in cellmates:
        #    print(cell)
        print("----------------")
        '''
        if len(cellmates) > 1:
            food = [_c for _c in cellmates if type(_c) == FoodAgent ]
            if food:
                return [Eat(food[0])]
        
        print('Cat pos: ' +  str(self._agent.pos))
        self._actionsmap = {}
        auxPos = self._agent.pos
        
        self._actionsmap["Still"] = [MoveTo((0,0))]
        if auxPos[0] > 0:
            self._actionsmap["MoveUp"] = [MoveTo((-1,0))]
        if auxPos[0] < self._agent.model.grid.height-1:
            self._actionsmap["MoveDown"] = [MoveTo((1,0))]
        if auxPos[1] > 0:
            self._actionsmap["MoveLeft"] = [MoveTo((0,-1))]
        if auxPos[1] < self._agent.model.grid.width-1:
            self._actionsmap["MoveRight"] = [MoveTo((0,1))]
        
        
        if len(self._agent.model.grid.grid[auxPos[0]+1][auxPos[1]]) > 0:
            if type(self._agent.model.grid.grid[auxPos[0]+1][auxPos[1]][0]) == MazeBlock:
                self._actionsmap.pop("MoveDown", None)
        if len(self._agent.model.grid.grid[auxPos[0]][auxPos[1]+1]) > 0:
            if type(self._agent.model.grid.grid[auxPos[0]][auxPos[1]+1][0]) == MazeBlock:
                self._actionsmap.pop("MoveRight", None)
        if len(self._agent.model.grid.grid[auxPos[0]][auxPos[1]-1]) > 0:
            if type(self._agent.model.grid.grid[auxPos[0]][auxPos[1]-1][0]) == MazeBlock:
                self._actionsmap.pop("MoveLeft", None)
        if len(self._agent.model.grid.grid[auxPos[0]+1][auxPos[1]]) > 0:
            if type(self._agent.model.grid.grid[auxPos[0]+1][auxPos[1]][0]) == MazeBlock:
                self._actionsmap.pop("MoveUp", None)     
            
        _action = random.choice(list(self._actionsmap.keys()))
        print('Cat chose to ' + str(_action))
        return self._actionsmap.get(_action)

class GreedyCatBrain(ScriptedBrain):
    def __init__(self, agent):
        super().__init__(agent)
        self._buffer = np.zeros((self._agent.model.grid.width, self._agent.model.grid.height))
        #self._model._nA = len(self._agent.getPossibleActions())
        #self._model._nS = len(self._agent.getPossibleStates())    
        #self._model.reset()

    def think(self, status, reward):
        
        #get where all food positions are
        food_list = []
        walls = []
        for cell in self._agent.model.grid.coord_iter():
            cell_content, x, y = cell
            if len(cell_content) > 0:
                if type(cell_content[0]) == FoodAgent:
                    food_list.append((x,y))
                if type(cell_content[0]) == MazeBlock:
                    walls.append((x,y))

        #fill distances grid
        aux_grid = self._agent.model.grid
        for r in range(aux_grid.height):
            for c in range(aux_grid.width):
                if (r,c) in food_list:
                    self._buffer[r][c] = 0
                elif (r,c) in walls:
                    self._buffer[r][c] = 999
                else:
                    #find nearest food
                    mindist = 9999
                    for f in food_list:
                        dist = abs(r-f[0])
                        dist += abs(c-f[1])
                        if dist < mindist:
                            mindist = dist
                    self._buffer[r][c] = mindist
                    
        auxPos = self._agent.pos
        cell_content = self._agent.model.grid[auxPos[0]][auxPos[1]]
        for cont in cell_content:
            if type(cont) == FoodAgent:
                return [Eat(cont)]
        
        print("Cat is looking at the map:")
        for i in range(self._agent.model.grid.height):
            print(self._agent.model.grid[i])
        print("----------------")
        print(self._buffer)
        print("----------------")    
        print('Cat pos: ' +  str(self._agent.pos))
        self._actionsmap = {}
    
        dy = 0
        dx = 0
        minPos = 999
        if auxPos[0] > 0:
            if self._buffer[auxPos[0]-1][auxPos[1]] < minPos:
                dy = -1
                dx = 0
                minPos = self._buffer[auxPos[0]-1][auxPos[1]]
        if auxPos[0] < self._agent.model.grid.height-1:
            if self._buffer[auxPos[0]+1][auxPos[1]] < minPos:
                dy = 1
                dx = 0
                minPos = self._buffer[auxPos[0]+1][auxPos[1]]
        if auxPos[1] > 0:
            if self._buffer[auxPos[0]][auxPos[1]-1] < minPos:
                dy = 0
                dx = -1
                minPos = self._buffer[auxPos[0]][auxPos[1]-1]
        if auxPos[1] < self._agent.model.grid.width-1:
            if self._buffer[auxPos[0]][auxPos[1]+1] < minPos:
                dy = 0
                dx = 1
                minPos = self._buffer[auxPos[0]][auxPos[1]+1]
        
        
        _action = [MoveTo((dy,dx))]
        return _action

class SmartCatBrain(ScriptedBrain):
    def __init__(self, agent):
        super().__init__(agent)
        self._buffer = np.zeros((self._agent.model.grid.width, self._agent.model.grid.height))
        self._winPath = None
        #self._model._nA = len(self._agent.getPossibleActions())
        #self._model._nS = len(self._agent.getPossibleStates())    
        #self._model.reset()
    
    def rec(self, pos):
        if pos[0] > 0:
            if np.isnan(self._winPath[pos[0]-1][pos[1]]):
                self._winPath[pos[0]-1][pos[1]] = self._winPath[pos[0]][pos[1]] + 1 
                self.rec((pos[0]-1,pos[1]))
        if pos[0] < self._agent.model.grid.height-1:
            if np.isnan(self._winPath[pos[0]+1][pos[1]]):
                self._winPath[pos[0]+1][pos[1]] = self._winPath[pos[0]][pos[1]] + 1 
                self.rec((pos[0]+1,pos[1]))
        if pos[1] > 0:
            if np.isnan(self._winPath[pos[0]][pos[1]-1]):
                self._winPath[pos[0]][pos[1]-1] = self._winPath[pos[0]][pos[1]] + 1 
                self.rec((pos[0],pos[1]-1))
        if pos[1] < self._agent.model.grid.width-1:
            if np.isnan(self._winPath[pos[0]][pos[1]+1]):
                self._winPath[pos[0]][pos[1]+1] = self._winPath[pos[0]][pos[1]] + 1 
                self.rec((pos[0],pos[1]+1))
    
    def fillWinPath(self, food_list, walls, exits):
        
        self._winPath = np.full([self._agent.model.grid.width, self._agent.model.grid.height], np.nan)
        #self._winPath = np.empty((self._agent.model.grid.width, self._agent.model.grid.height))
        iters = self._agent.model.grid.width * self._agent.model.grid.height
        grid = self._agent.model.grid.grid
        
        for w in walls:
            self._winPath[w[0]][w[1]] = 999
        for e in exits:
            self._winPath[e[0]][e[1]] = 0
            self.rec(e)
            
    def think(self, status, reward):
        
        #get where all food positions are
        food_list = []
        walls = []
        exits = []
        for cell in self._agent.model.grid.coord_iter():
            cell_content, x, y = cell
            if len(cell_content) > 0:
                if type(cell_content[0]) == FoodAgent:
                    food_list.append((x,y))
                if type(cell_content[0]) == MazeBlock:
                    walls.append((x,y))
                if type(cell_content[0]) == MazeExit:
                    exits.append((x,y))
                    
        if self._winPath is None:
            self.fillWinPath(food_list,walls,exits)
            print(self._winPath)
        
        print('AgentEnergy     : ' + str(self._agent._envGetVal("BasalMetabolism","energy")))
        if int(self._agent._envGetVal("BasalMetabolism","energy")) > 0:
            auxPos = self._agent.pos
            dy = 0
            dx = 0
            minPos = 999
            if auxPos[0] > 0:
                if self._winPath[auxPos[0]-1][auxPos[1]] < minPos:
                    dy = -1
                    dx = 0
                    minPos = self._winPath[auxPos[0]-1][auxPos[1]]
            if auxPos[0] < self._agent.model.grid.height-1:
                if self._winPath[auxPos[0]+1][auxPos[1]] < minPos:
                    dy = 1
                    dx = 0
                    minPos = self._winPath[auxPos[0]+1][auxPos[1]]
            if auxPos[1] > 0:
                if self._winPath[auxPos[0]][auxPos[1]-1] < minPos:
                    dy = 0
                    dx = -1
                    minPos = self._winPath[auxPos[0]][auxPos[1]-1]
            if auxPos[1] < self._agent.model.grid.width-1:
                if self._winPath[auxPos[0]][auxPos[1]+1] < minPos:
                    dy = 0
                    dx = 1
                    minPos = self._winPath[auxPos[0]][auxPos[1]+1]
            _action = [MoveTo((dy,dx))]
            return _action
            
        else:
            #fill distances grid
            aux_grid = self._agent.model.grid
            for r in range(aux_grid.height):
                for c in range(aux_grid.width):
                    if (r,c) in food_list:
                        self._buffer[r][c] = 0
                    elif (r,c) in walls:
                        self._buffer[r][c] = 999
                    else:
                        #find nearest food
                        mindist = 9999
                        for f in food_list:
                            dist = abs(r-f[0])
                            dist += abs(c-f[1])
                            if dist < mindist:
                                mindist = dist
                        self._buffer[r][c] = mindist
                        
            auxPos = self._agent.pos
            cell_content = self._agent.model.grid[auxPos[0]][auxPos[1]]
            for cont in cell_content:
                if type(cont) == FoodAgent:
                    return [Eat(cont)]
            
            print("Cat is looking at the map:")
            for i in range(self._agent.model.grid.height):
                print(self._agent.model.grid[i])
            print("----------------")
            print(self._buffer)
            print("----------------")    
            print('Cat pos: ' +  str(self._agent.pos))
            self._actionsmap = {}
        
            dy = 0
            dx = 0
            minPos = 999
            if auxPos[0] > 0:
                if self._buffer[auxPos[0]-1][auxPos[1]] < minPos:
                    dy = -1
                    dx = 0
                    minPos = self._buffer[auxPos[0]-1][auxPos[1]]
            if auxPos[0] < self._agent.model.grid.height-1:
                if self._buffer[auxPos[0]+1][auxPos[1]] < minPos:
                    dy = 1
                    dx = 0
                    minPos = self._buffer[auxPos[0]+1][auxPos[1]]
            if auxPos[1] > 0:
                if self._buffer[auxPos[0]][auxPos[1]-1] < minPos:
                    dy = 0
                    dx = -1
                    minPos = self._buffer[auxPos[0]][auxPos[1]-1]
            if auxPos[1] < self._agent.model.grid.width-1:
                if self._buffer[auxPos[0]][auxPos[1]+1] < minPos:
                    dy = 0
                    dx = 1
                    minPos = self._buffer[auxPos[0]][auxPos[1]+1]
            
            
            _action = [MoveTo((dy,dx))]
            return _action


class MazeBlock(XAgent):
    def __init__(self, unique_id, model):
        XAgent.__init__(self, unique_id, model) 
        
    def _postInit(self, *argv, **kargv): 
        return

    def step(self):
        return

    def __repr__(self):
        return "Wall"
    
class MazeExit(XAgent):
    def __init__(self, unique_id, model):
        XAgent.__init__(self, unique_id, model) 
        
    def _postInit(self, *argv, **kargv): 
        return

    def step(self):
        return

    def __repr__(self):
        return "Exit"

class FoodAgent(XAgent):
    def __init__(self, unique_id, model):
        XAgent.__init__(self, unique_id, model) 
        
    def _postInit(self, *argv, **kargv): 
        self._envSetVal("FoodProduction","food", 1)

    def step(self):
        if (float(self._envGetVal("FoodProduction","food")) <= 0):
            self.removeAndClean("Eaten up!")

    def __repr__(self):
        return "Cheese"

class Cat(XAgent):
    def __init__(self, unique_id, model):
        XAgent.__init__(self, unique_id, model)

    def _postInit(self, *argv, **kargv):
        _log.debug("*** Agent %s postInit called"%self.id) 
        
    def step(self):
        self.updateState()  

class Maze(MultiEnvironmentWorld):
    def __init__(self, N, config={"size":{"width":2,"height":2,"torus":False}}):
        super().__init__(config)
        _log.info("Initalizing model")  
        self.schedule = RandomActivation(self)
        _size = self.config["size"]
        #self.grid = MultiGrid(_size["width"], _size["height"], _size["torus"])
        
        config = minidom.parse('config.xml')
        _width = eval(config.getElementsByTagName('width')[0].firstChild.data.replace('"',''))
        _height = eval(config.getElementsByTagName('height')[0].firstChild.data.replace('"',''))
        _torus = eval(config.getElementsByTagName('torus')[0].firstChild.data.replace('"',''))

        #print("Width =" + str(_width))
        self.grid = MultiGrid(_width, _height, _torus)
        
        #win = eval(config.getElementsByTagName('win')[0].firstChild.data.replace('"',''))
        #print(win)
        
        self.createAgents()

    def createAgents(self):
        self.l.info("Requesting agents for the simulation.")
        
        pr = PopulationRequest()
        
        config = minidom.parse('config.xml')
        
       
        obstacles = config.getElementsByTagName('obstacle')
        for _obs in obstacles:
            _v = {}
            for _att in range(_obs.attributes.length):
                _v[_obs.attributes.item(_att).name] = eval(_obs.attributes.item(_att).value.replace('"',''))
                #print(eval(_obs.attributes.item(_att).value.replace('"','')))
            pr._data[_obs.firstChild.data] = _v
        
        exit = config.getElementsByTagName('win')
        for _ex in exit:
            _v = {}
            for _att in range(_ex.attributes.length):
                _v[_ex.attributes.item(_att).name] = eval(_ex.attributes.item(_att).value.replace('"',''))
                #print(eval(_ex.attributes.item(_att).value.replace('"','')))
            pr._data[_ex.firstChild.data] = _v
        
        agents = config.getElementsByTagName('agent')
        for _ag in agents:
            _v = {}
            for _att in range(_ag.attributes.length):
                _v[_ag.attributes.item(_att).name] = eval(_ag.attributes.item(_att).value.replace('"',''))
                #print(eval(_ag.attributes.item(_att).value.replace('"','')))
            pr._data[_ag.firstChild.data] = _v
        '''

        pr._data = {
            "Cat": {"num": 1,
                        "type": Cat,
                        "templates":["BasicAnimal"],
                        "brain": CatBrain,
                        "inPos": (1,0)
                        },
            "Cheese": {"num": 2,
                        "type": FoodAgent,
                        "inPos": (self.grid.width-1,self.grid.height-1)
                        }                    
            }
        '''
        # Prepare agents for creation        
        _created = self.addPopulationRequest(pr)
        # Request their addition to the world
        super().createAgents()
        # Finish initialization of world-dependent data
        
    def plotAll(self):
        pass
        
    def run_model(self, n):
        self.info()
        self.currentStep = 0
        self.l.info("***** STARTING SIMULATION!!")
        for i in range(n):
            self.stepEnvironment() #update word state
            self.currentStep+=1
            self.schedule.step()
            _log.info("Step %d of %d"%(i, n))

        self.l.info("***** FINISHED SIMULATION!!")

        self.info()

    def info(self):
        self.l.info("*"*3 + "This is world %s",self.__class__.__name__ + "*"*3)

        self.l.info("*"*3 + "With %d agents: types and num %s"%(len(self._agentsById), str(self._agents)))
        self.l.info("*"*3 + "With %d environments: types and num %s"%(len(self._envs), str(self._envs)))

        for _a in self.getAgents():
            _log.debug("*"*3 + _a.info())    

        self.l.info("*"*3 + "...BYE!" + "*"*3)
