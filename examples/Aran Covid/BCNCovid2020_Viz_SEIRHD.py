from BCNCovid2020SEIRHD import BCNCovid2020
from Covid_class import VirusCovid, State
from Human_Class import BasicHuman
from Hospital_class import Hospital, Workplace

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement

# {"N":30, "basemap":"Barcelona, Spain", "width":10, "height":10}
model_params = {
    "N": UserSettableParameter("slider", "Population size", 10, 10, 100, 10),
    "basemap": "Barcelona, Spain", "width": 50, "height": 50,
}


class TimeElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        time = getattr(model, "DateTime")
        return "Day: " + str(time.day) + ". Hour: " + str(time.hour) + ":" + str(time.minute)

        """day = model.get_day()  # get day from get_day function in model
        hour = model.get_hour
        hour = '{0:02.0f}:{1:02.0f}'.format(
            *divmod(model.get_hour() * 60, 60))  # get hour from get_hour function in model & transform it to hh:mm.
        return "Dia: " + str(day) + ". Hour: " + str(hour)"""


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5}

    if isinstance(agent, BasicHuman):
        if agent.state == State.SUSC:
            portrayal["Color"] = "green"
            portrayal["Layer"] = 0
        elif agent.state == State.EXP:
            portrayal["Color"] = "yellow"
            portrayal["Layer"] = 1
            portrayal["r"] = 0.4
        elif agent.state == State.INF:
            portrayal["Color"] = "red"
            portrayal["Layer"] = 1
            portrayal["r"] = 0.4
        elif agent.state == State.REC:
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 2
            portrayal["r"] = 0.3
        elif agent.state == State.HOSP:
            portrayal["Color"] = "gray"
            portrayal["Layer"] = 1
            portrayal["r"] = 0.4
        """elif agent.state == State.DEAD: #dont show dead agents
            portrayal["Color"] = "black"
            portrayal["Layer"] = 3
            portrayal["r"] = 0.2"""

    elif isinstance(agent, Hospital):
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "pink"
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif isinstance(agent, Workplace):
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 4
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal


grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)
infected_chart = ChartModule(
    [
        {"Label": "SUSC", "Color": "Green"},
        {"Label": "EXP", "Color": "Yellow"},
        {"Label": "INF", "Color": "Red"},
        {"Label": "REC", "Color": "Blue"},
        {"Label": "HOSP", "Color": "Gray"},
        {"Label": "DEAD", "Color": "Black"},
    ], data_collector_name="datacollector"
)

hospital_chart = ChartModule(
    [
        {"Label": "H-SUSC", "Color": "Green"},
        {"Label": "H-INF", "Color": "Red"},
        {"Label": "H-REC", "Color": "Blue"},
        {"Label": "H-HOSP", "Color": "Gray"},
        {"Label": "H-DEAD", "Color": "Black"},
    ], data_collector_name="hosp_collector"
)

show_time = TimeElement()

server = ModularServer(BCNCovid2020,
                       [show_time, grid, infected_chart, hospital_chart],
                       "Covid Model", model_params)

server.port = 8521  # The default
server.launch()
