import os
import agents


def get_incubation_time(model):
    """ Returns the incubation time (EXP state) following a normal distribution """
    return int(model.random.normalvariate(model.virus.incubation_days, model.virus.incubation_days_sd))


def get_infection_time(model):
    """ Returns the infeciton time (INF state) following a normal distribution """
    return int(model.random.normalvariate(model.virus.infection_days, model.virus.infection_days_sd))


def get_immune_time(model):
    """ Returns the immune time (REC state) following a normal distribution """
    return int(model.random.normalvariate(model.virus.immune_days, model.virus.immune_days_sd))


def get_severe_time(model):
    """ Returns the severe time (HOSP state) following a normal distribution """
    return int(model.random.normalvariate(model.virus.severe_days, model.virus.severe_days_sd))


def update_DC_table(model):
    """ Collects all statistics for the DC_Table """
    next_row = {'Day': model.DateTime.strftime('%d-%m'), 'Susceptible': get_susceptible_count(model),
                'Exposed': get_exposed_count(model), 'Infected': get_infected_count(model),
                'Recovered': get_recovered_count(model), 'Hospitalized': get_hosp_count(model),
                'Dead': get_dead_count(model), 'R0': get_R0(model), 'R0_Obs': get_R0_Obs(model)}
    model.datacollector.add_table_row("Model_DC_Table", next_row, ignore_missing=True)

    next_row2 = {'Day': model.DateTime.strftime('%d-%m'), 'Hosp-Susceptible': get_h_susceptible_count(model),
                 'Hosp-Infected': get_h_infected_count(model), 'Hosp-Recovered': get_h_recovered_count(model),
                 'Hosp-Hospitalized': get_hosp_count(model), 'Hosp-Dead': get_h_dead_count(model)}
    model.hosp_collector.add_table_row("Hosp_DC_Table", next_row2, ignore_missing=True)


def reset_counts(model):
    """ Sets to 0 the counts for the model datacollector """
    model.collector_counts = {"SUSC": 0, "EXP": 0, "INF": 0, "REC": 0, "HOSP": 0, "DEAD": 0, "R0": 0, "R0_Obs": 0, }

def update_stats(model):
    for human in [agent for agent in model.schedule.agents if isinstance(agent, agents.HumanAgent)]:
        """ Update Status dictionaries for data collector. """
        if human.machine.state == "S":
            model.collector_counts['SUSC'] += 1
        elif human.machine.state == "E":
            model.collector_counts['EXP'] += 1
        elif human.machine.state == "I" or human.machine.state == "A":
            model.collector_counts['INF'] += 1
        elif human.machine.state == "R":
            model.collector_counts['REC'] += 1
        elif human.machine.state == "H":
            model.collector_counts['HOSP'] += 1
        elif human.machine.state == "D":
            model.collector_counts['DEAD'] += 1


def reset_hosp_counts(model):
    """ Sets to 0 the counts for the hospital datacollector """
    model.hosp_collector_counts = {"H-SUSC": 0, "H-INF": 0, "H-REC": 0, "H-HOSP": 0, "H-DEAD": 0, }


""" Functions for the data collectors """


def get_susceptible_count(model):
    """ Returns the Susceptible human agents in the model """
    return model.collector_counts["SUSC"]


def get_exposed_count(model):
    """ Returns the Exposed human agents in the model """
    return model.collector_counts["EXP"]


def get_infected_count(model):
    """ Returns the Infected human agents in the model """
    return model.collector_counts["INF"]


def get_recovered_count(model):
    """ Returns the Recovered human agents in the model """
    return model.collector_counts["REC"]


def get_hosp_count(model):
    """ Returns the Hospitalized human agents in the model """
    return model.collector_counts["HOSP"]


def get_dead_count(model):
    """ Returns the Dead human agents in the model """
    return model.collector_counts["DEAD"]


def get_R0(model):
    """ Returns the R0 value of the model """
    return model.virus.R0


def get_R0_Obs(model):
    """ Returns the R0_obs value of the model """
    return model.virus.R0_obs


def get_h_susceptible_count(model):
    """ Returns the Susceptible human agents in the model recorded by Hospital """
    return model.hosp_collector_counts["H-SUSC"]


def get_h_infected_count(model):
    """ Returns the Infected human agents in the model recorded by Hospital"""
    return model.hosp_collector_counts["H-INF"]


def get_h_recovered_count(model):
    """ Returns the Recovered human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-REC"]


def get_h_hospitalized_count(model):
    """ Returns the Hospitalized human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-HOSP"]


def get_h_dead_count(model):
    """ Returns the Dead human agents in the model recorded by Hospitals """
    return model.hosp_collector_counts["H-DEAD"]
