import math
import numpy as np

class COVID2019Infection(object):
    def __init__(self,config_agents):
      #L'istituto RKI, Robert Koch Institut calcola Rt come 
      # rapporto tra la somma del numero di contagiati negli ultimi 4 giorni e la somma del numero dei contagiati nei 4 giorni precedenti
      self.days_for_R0_RKI=3 #these are the days for the computation of the coefficient Rt according to the RKI method
      self.Infected_detects_for_RKI_today=0
      self.Infected_detects_for_RKI=[]
      self.Infected_type_for_RKI=["I","A"]
      self.Infected_for_RKI_today=0
      self.Infected_for_RKI=[]
      self.totalInStatesForDay=[] 

      self.agents_in_states={"S":0, "E":0,"A":0,"I":0,"H":0,"R":0,"D":0}
      self.E_today=0
      self.today1=0
      self.Hospitalized_total=0
      self.perc_vacc_day= 0.01
      self.vaccinations_on_day=math.ceil(self.perc_vacc_day * config_agents)
      self.tobevaccinatedtoday=self.vaccinations_on_day
      self.pTest = 0.95
      self.R0 = 0
      self.R0_obs = 0
      self.R0_observed = {}
      self.contact_count = [0, 0]
      self._contact_agents = set()

    def calculate_R0(self):

        if self.count<self.days_for_R0_RKI:
            self.R0=0
        else:

            if self.agents_in_states["I"]+self.agents_in_states["A"]==0:
                self.R0=0
            else:
                self.R0=(self.E_today*10)/(self.agents_in_states["I"]+self.agents_in_states["A"])

        
        self.E_today=0

        if len(self.Infected_detects_for_RKI)< 2*self.days_for_R0_RKI or sum(self.Infected_detects_for_RKI[-2*self.days_for_R0_RKI:-self.days_for_R0_RKI])==0:
            self.R0_obs = 0
        else:
            self.R0_obs =sum(self.Infected_detects_for_RKI[-self.days_for_R0_RKI:len(self.Infected_detects_for_RKI)])/sum(self.Infected_detects_for_RKI[-2*self.days_for_R0_RKI:-self.days_for_R0_RKI])


    def startInfection(self):
            # INFECTION
            infected = np.random.choice(["S", "E","A","I"],  p=[0.995, 0, 0 ,0.005])#p=[0.985, 0, 0.015]) p=[0.970, 0.015,0 ,0.015])#
            self.agents_in_states[infected]+= 1
            if infected == "I":
                self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(agentsToBecreated - i, "I",
                                                                                 age_(), agent=self._agentsToAdd[agentsToBecreated - i])
                self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                    list(range(1, 11)))
                self.collector_counts["SUSC"] -= 1
                self.collector_counts["INF"] += 1  # Adjust initial counts
                self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0,
                                                                                                            round(
                                                                                                                1 /
                                                                                                                self._agentsToAdd[
                                                                                                                    agentsToBecreated - i].machine.rate[
                                                                                                                    'rIR']) -
                                                                                                            self._agentsToAdd[
                                                                                                                agentsToBecreated - i].machine.time_in_state,
                                                                                                            0]
            elif infected == "E":
                self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(agentsToBecreated - i, "E",
                                                                                 age_(), agent=self._agentsToAdd[agentsToBecreated - i])
                self._agentsToAdd[agentsToBecreated - i].machine.time_in_state = random.choice(
                    list(range(1, 5)))

                self.collector_counts["SUSC"] -= 1
                self.collector_counts["EXP"] += 1  # Adjust initial counts
                self._agentsToAdd[agentsToBecreated - i].R0_contacts[self.getTime().strftime('%Y-%m-%d')] = \
                    [0, round(1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rEI']) + round(
                        1 / self._agentsToAdd[agentsToBecreated - i].machine.rate['rIR']) - self._agentsToAdd[
                         agentsToBecreated - i].machine.time_in_state, 0]

            else:
                self._agentsToAdd[agentsToBecreated - i].machine = SEAIHRD_covid(
                    self._agentsToAdd[agentsToBecreated - i].id, "S", age_(), agent=self._agentsToAdd[agentsToBecreated - i])

            # EMPLOYMENT

            if np.random.choice([True, False], p=[self.employment_rate, 1 - self.employment_rate]) and 5 < \
                    self._agentsToAdd[agentsToBecreated - i].machine.age < 65:
                workplaces = random.sample(
                    list(range(len(self._agentsToAdd) - Workplaces, len(self._agentsToAdd))), Workplaces)
                for workplace in workplaces:
                    if self._agentsToAdd[workplace].total_capacity > len(
                            self._agentsToAdd[workplace].get_workers()):
                        self._agentsToAdd[agentsToBecreated - i].workplace = self._agentsToAdd[workplace].id
                        self._agentsToAdd[workplace].add_worker(self._agentsToAdd[agentsToBecreated - i].id)
                    break

            agentsToBecreated -= family_dist[index]
            family_dist[index] = 0

            self.totalInStatesForDay.append(self.agents_in_states)       