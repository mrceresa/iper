from mesa import Agent
import numpy as np


class Hospital(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.place = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        self.total_capacity = int(len(self.model.schedule.agents) / 10)  # 10% of population
        self.list_pacients = set()


    def get_pacients(self):
        """ Returns hospital pacients """
        return len(self.list_pacients)

    def new_pacient(self, agent):
        """ Try to add a new pacient to the Hospital, otherwise return False """
        if self.get_pacients() < self.total_capacity:
            self.list_pacients.add(agent)
            return True
        else:
            return False

    def discharge_pacient(self, agent):
        """ Discharges a pacient """
        self.list_pacients.remove(agent)


class Workplace(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.place = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        self.total_capacity = 0
        self.workers = set()

    def set_capacity(self, mean, sd):
        """ Sets capacity of the workplace. """
        self.total_capacity = int(np.ceil(self.random.normalvariate(mean, sd)))
        #print("Capacity of workplace: " + str(self.unique_id) + " has a capacity of " + str(self.total_capacity))

    def add_worker(self, agent):
        """ Adds a worker to the workplace hospital pacients """
        self.workers.add(agent)

    def get_workers(self):
        """ Returns workplace workers.  """
        return self.workers

