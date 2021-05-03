from BCNCovid2020SEIRHD import BCNCovid2020
from Covid_class import VirusCovid, State
from Human_Class import BasicHuman
from Hospital_class import Hospital, Workplace

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement

# {"N":30, "basemap":"Barcelona, Spain", "width":10, "height":10}
model_params = {
    "N": UserSettableParameter("slider", "Population size", 50, 100, 1000, 100),
    "basemap": "Barcelona, Spain", "width": 100, "height": 100,
    "incubation_days": UserSettableParameter("slider", "Incubation days", 3, 1, 5, 1),
    "infection_days": UserSettableParameter("slider", "Infection days", 5, 1, 10, 1),
    "immune_days": UserSettableParameter("slider", "Immune days", 3, 1, 5, 1),
    "severe_days": UserSettableParameter("slider", "Severe days", 3, 1, 5, 1),
    "ptrans": UserSettableParameter("slider", "Transmission probability", 0.7, 0, 1, 0.1),
    "pSympt": UserSettableParameter("slider", "Symptomatic probability", 0.4, 0, 1, 0.1),
    "pTest": UserSettableParameter("slider", "Test probability", 0.9, 0, 1, 0.1),
    "pSympt": UserSettableParameter("slider", "Symptomatic probability", 0.4, 0, 1, 0.1),
    "N_hosp": UserSettableParameter("slider", "Number of Hospitals", 5, 1, 10, 1),
    "Hosp_capacity": UserSettableParameter("slider", "Hospital Capacity", 10, 2, 20, 2),
    "death_rate": UserSettableParameter("slider", "Death Rate", 0.02, 0, 1, 0.05),
    "severe_rate": UserSettableParameter("slider", "Severe Rate", 0.05, 0, 1, 0.05),




}


class TimeElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        time = getattr(model, "DateTime")
        return "Day: " + str(time.day) + ". Hour: " + str(time.hour) + ":" + str(time.minute)


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
            portrayal["Layer"] = 2
            portrayal["r"] = 0.4
        elif agent.state == State.REC:
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 3
            portrayal["r"] = 0.3
        elif agent.state == State.HOSP:
            portrayal["Color"] = "gray"
            portrayal["Layer"] = 4
            portrayal["r"] = 0.4
        elif agent.state == State.DEAD: #dont show dead agents
            portrayal["Color"] = "black"
            portrayal["Layer"] = 3
            portrayal["r"] = 0.2

    elif isinstance(agent, Hospital):
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "pink"
        portrayal["Layer"] = 5
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif isinstance(agent, Workplace):
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 5
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal


grid = CanvasGrid(agent_portrayal, 100, 100, 500, 500)
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

R0_chart = ChartModule(
    [
        {"Label": "R0", "Color": "Orange"},
        {"Label": "R0_Obs", "Color": "Green"},
    ], data_collector_name="datacollector"
)


show_time = TimeElement()

server = ModularServer(BCNCovid2020,
                       [show_time, grid, infected_chart, hospital_chart, R0_chart],
                       "Covid Model", model_params)

server.port = 8521  # The default
server.launch()
