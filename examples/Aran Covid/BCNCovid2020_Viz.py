from BCNCovid2020 import BasicHuman, BCNCovid2020, State

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5}

    if agent.state == State.SUSC:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0
    elif agent.state == State.INF:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.4
    elif agent.state == State.REC:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.3
    elif agent.state == State.DEATH:
        portrayal["Color"] = "gray"
        portrayal["Layer"] = 3
        portrayal["r"] = 0.2
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)



server = ModularServer(BCNCovid2020,
                       [grid],
                       "Covid Model",
                       {"N":30, "basemap":"Barcelona, Spain", "width":10, "height":10})
server.port = 8521 # The default
server.launch()