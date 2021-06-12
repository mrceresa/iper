import numpy as np
import pandas as pd
import random, math
import datetime 
import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa_geo import GeoSpace, GeoAgent, AgentCreator


class EmploymentManager (object):
    def __init__(self, city):
        self.jobs = []
        self.applicants = []
        self.model = city

    def job_request(self, agent):
        self.applicants.append(agent)

    def job_offer(self, business):
        self.jobs.append(business)

    def match(self, _b, _a):
        _a.searching = False
        _a.employed = True
        _a.work_place = (_b.unique_id, _b.shape)
        _a.work_start = _b.work_start
        _a.work_end = _b.work_end

        _b.employees.append(_a.unique_id)
        _b.searching = False
        # print('Matching ' + str(_a.unique_id) + ' and ' + str(_b.unique_id))

    def resolve(self):
        if (len(self.applicants) != 0) and (len(self.jobs) != 0):
            for _j in self.jobs:
              if len(self.applicants) == 0:
                  return
              else:
                  self.match(_j, self.applicants[0])
                  self.applicants = self.applicants[1:]
                  self.jobs = self.jobs[1:]
    
    def pay_employee (self, _b, agent_id):
        _a = self.agents[agent_id]
        _b.funds -= 1000
        _a.inventory["funds"] = _a.inventory["funds"] + 1000 
        return