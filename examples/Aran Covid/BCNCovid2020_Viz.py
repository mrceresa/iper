from BCNCovid2020 import BasicHuman, BCNCovid2020, State

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer


# {"N":30, "basemap":"Barcelona, Spain", "width":10, "height":10}
model_params = {
    "N": UserSettableParameter("slider", "Population size", 30, 10, 100, 10),
    "basemap": "Barcelona, Spain", "width": 10, "height": 10,
}


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
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.3
    elif agent.state == State.DEAD:
        portrayal["Color"] = "gray"
        portrayal["Layer"] = 3
        portrayal["r"] = 0.2
    return portrayal


grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
infected_chart = ChartModule(
    [
        {"Label": "INF", "Color": "Red"},
        {"Label": "SUSC", "Color": "Green"},
        {"Label": "REC", "Color": "Blue"},
        {"Label": "DEAD", "Color": "Black"},
    ]
)

server = ModularServer(BCNCovid2020,
                       [grid, infected_chart],
                       "Covid Model", model_params)

server.port = 8521  # The default
server.launch()
