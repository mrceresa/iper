from mesa import Agent
import Covid_class
from Covid_class import State, Mask
import numpy as np
from datetime import datetime, timedelta


class Hospital(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.place = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        self.total_capacity = int(len(self.model.schedule.agents) / 10)  # 10% of population
        self.list_pacients = set()  # patients in the hospital
        self.PCR_availables = 2  # self.random.randrange(3, 5)
        self.PCR_testing = {}  # patients waiting for the testing
        self.PCR_results = {}  # patients tested
        self.mask = Mask.FFP2

    def __str__(self):
        return "Hospital at " + str(self.place) + "with " + str(self.PCR_availables) + " available tests"

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

    def doTest(self, agent):
        """ Tests an agent.
        If result is positive agent contacts will be added to hospital PCR testing for future tests. """
        print(f"Agente {agent.unique_id} testeandose en hospital {self.unique_id}")

        today = self.model.DateTime.strftime('%Y-%m-%d')
        if not today in self.model.peopleTested:
            self.model.peopleTested[today] = {agent}
        else:
            self.model.peopleTested[today].add(agent)

        agentStatus = agent.state
        pTest = self.model.virus.pTest
        true_pos = np.random.choice([True, False], p=[pTest, 1 - pTest])
        if true_pos:  # test knows true state
            if agentStatus == State.EXP or agentStatus == State.INF:
                agentcontacts = agent.contacts
                print(f"Resulta que es positivo, da sus contactos {agentcontacts}")
                # save agent contacts for future tests
                for key in agentcontacts:
                    for elem in agentcontacts[key]:
                        # if not today in self.PCR_testing:
                        if not today in self.model.peopleToTest:
                            self.model.peopleToTest[today] = {elem}
                        else:
                            self.model.peopleToTest[today].add(elem)

        """if not today in self.PCR_results:
            self.PCR_results[today] = {agent}
        else:
            self.PCR_results[today].add(agent)"""
        # adds agents to list of people tested in model

    def decideTesting(self, patientsTested):
        """ Called by model function at midnight to decide which agents will be tested from the contact tracing set
        the next day with a 2-3 day delay."""
        PCRs = self.PCR_availables
        HospToTest = set()

        ThreeD_ago = (self.model.DateTime - timedelta(days=3)).strftime('%Y-%m-%d')

        if ThreeD_ago in self.model.peopleToTest and PCRs:
            for elem in self.model.peopleToTest[ThreeD_ago]:
                if elem not in patientsTested and PCRs > 0:
                    print(f"To test {elem.unique_id}")
                    PCRs -= 1
                    HospToTest.add(elem)
                    elem.quarantined = self.model.DateTime + timedelta(days=3)
                    elem.obj_place = self.place

            self.model.peopleToTest[ThreeD_ago] -= HospToTest

        print(patientsTested)
        TwoD_ago = (self.model.DateTime - timedelta(days=2)).strftime('%Y-%m-%d')
        if PCRs and TwoD_ago in self.model.peopleToTest:
            for elem in self.model.peopleToTest[TwoD_ago]:
                if elem not in (patientsTested | HospToTest) and PCRs > 0:
                    PCRs -= 1
                    HospToTest.add(elem)
                    elem.quarantined = self.model.DateTime + timedelta(days=3)
                    elem.obj_place = self.place

            self.model.peopleToTest[TwoD_ago] -= HospToTest

        print(f"Hoy el hospital {self.unique_id} testeara a {HospToTest} con tests {PCRs}")
        return patientsTested | HospToTest


class Workplace(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.place = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        self.total_capacity = 0
        self.workers = set()
        self.mask = Mask.RandomMask()

    def __str__(self):
        return "Workplace at " + str(self.place) + " with mask " + str(self.mask)

    def set_capacity(self, mean, sd):
        """ Sets capacity of the workplace. """
        self.total_capacity = int(np.ceil(self.random.normalvariate(mean, sd)))
        # print("Capacity of workplace: " + str(self.unique_id) + " has a capacity of " + str(self.total_capacity))

    def add_worker(self, agent):
        """ Adds a worker to the workplace hospital patients """
        self.workers.add(agent)

    def get_workers(self):
        """ Returns workplace workers.  """
        return self.workers
