from mesa import Agent
from SEAIHRD_class import SEAIHRD_covid, Mask
import numpy as np
import random
from datetime import datetime, timedelta

from iper import XAgent

class Hospital(XAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id)
        self.total_capacity = model.Hosp_capacity #int(random.normalvariate(model.Hosp_capacity, int(model.Hosp_capacity * 0.4)))  # 10% of population
        self.list_pacients = set()  # patients in the hospital
        self.PCR_availables = model.PCR_tests  # self.random.randrange(3, 5)
        #self.PCR_testing = {}  # patients waiting for the testing
        #self.PCR_results = {}  # patients tested
        self.mask = Mask.FFP2

    def __repr__(self):
        return "Agent " + str(self.id)

    def __str__(self):
        return "Hospital at " + str(self.pos) + "with " + str(self.PCR_availables) + " available tests"

    def _postInit(self):
        pass

    def add_patient(self, agent):
        """ Adds a patient """
        self.list_pacients.add(agent)

    def discharge_patient(self, agent):
        """ Discharges a patient """
        self.list_pacients.remove(agent)

    def doTest(self, agent):
        """ Tests an agent.
        If result is positive agent contacts will be added to hospital PCR testing for future tests. """
        # print(f"Agente {agent.unique_id} testeandose en hospital {self.unique_id}")

        today = self.model.DateTime.strftime('%Y-%m-%d')
        # if not today in self.model.peopleTested:
        #     self.model.peopleTested[today] = set()

        self.model.peopleTested[today].add(agent)

        agentStatus = agent.machine.state
        pTest = self.model.pTest
        true_pos = np.random.choice([True, False], p=[pTest, 1 - pTest])
        if true_pos:  # test knows true state
            if agentStatus in ["E", "I", "A"]:
                if not agent.HospDetected:
                    agent.HospDetected = True
                agentcontacts = agent.contacts
                # print(f"Resulta que es positivo, da sus contactos {agentcontacts}")
                # save agent contacts for future tests
                for key in agentcontacts:
                    for elem in agentcontacts[key]:
                        self.model.peopleToTest[today].add(elem)

        if self.model.quarantine_period == 0: agent.quarantined = None

    def decideTesting(self, patientsTested):
        """ Called by model function at midnight to decide which agents will be tested from the contact tracing set
        the next day with a 2-3 day delay."""

        PCRs = self.PCR_availables
        HospToTest = set()

        ThreeD_ago = (self.model.DateTime - timedelta(days=3)).strftime('%Y-%m-%d')

        quarantine_period = self.model.quarantine_period
        if quarantine_period == 0: quarantine_period = 1

        if ThreeD_ago in self.model.peopleToTest and PCRs:
            for elem in self.model.peopleToTest[ThreeD_ago]:
                agent = self.model.space.get_agent(elem)
                #if agent.quarantined is None: agent.quarantined = self.model.DateTime + timedelta(days=quarantine_period)
                if elem not in patientsTested and PCRs > 0 and (agent.machine.state not in ["H", "D"]):
                    PCRs -= 1
                    HospToTest.add(elem)
                    agent.quarantined = self.model.DateTime + timedelta(days=quarantine_period)
                    agent.obj_place = self.pos
                    agent.goal = "GO_TO_HOSPITAL"

            self.model.peopleToTest[ThreeD_ago] -= HospToTest

        # print(patientsTested)
        TwoD_ago = (self.model.DateTime - timedelta(days=2)).strftime('%Y-%m-%d')
        if PCRs and TwoD_ago in self.model.peopleToTest:
            for elem in self.model.peopleToTest[TwoD_ago]:
                agent = self.model.space.get_agent(elem)
                if elem not in (patientsTested | HospToTest) and PCRs > 0 and (agent.machine.state not in ["H", "D"]):
                    PCRs -= 1
                    HospToTest.add(elem)
                    agent.quarantined = self.model.DateTime + timedelta(days=quarantine_period)
                    agent.obj_place = self.pos
                    agent.goal = "GO_TO_HOSPITAL"

            self.model.peopleToTest[TwoD_ago] -= HospToTest

        return patientsTested | HospToTest


class Workplace(XAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #self.pos = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
        self.total_capacity = 0
        self.workers = set()
        self.mask = random.choice([Mask.HYGIENIC, Mask.FFP2])

    def __str__(self):
        return "Workplace at " + str(self.pos) + " with mask " + str(self.mask)

    def __repr__(self):
        return "Agent " + str(self.id)

    def _postInit(self):
        pass

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
