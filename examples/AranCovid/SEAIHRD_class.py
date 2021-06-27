from transitions import Machine
from enum import Enum
import random
import math
import numpy as np

class SEAIHRD_covid(object):
    # recovery time with gaussian distribution:{'19':(13.43,5.8), '29':(13.98,5.85),'39':(14.31,6.7),'49':(14.78,5.8),'59':(14.78,5.8),'99':(14.8,6.2)}
    # A single center Chinese study of 221 discharged COVID-19
    # patients observed an average time to recovery of 10.63±1.93 days
    # for mild to moderate patients, compared with 18.70±2.50 for severe patients.
    states = ['S', 'E', 'A', 'I', 'R', 'H', 'D']
    transitions = [
        {'trigger': 'nothing', 'source': '*', 'dest': '='},
        {'trigger': 'vaccination', 'source': 'S', 'dest': 'R'},
        {'trigger': 'contact', 'source': 'S', 'dest': 'E', 'conditions': 'prob_infection'},
        {'trigger': 'contact', 'source': 'S', 'dest': '='},
        {'trigger': 'end_encubation', 'source': 'E', 'dest': 'I', 'conditions': 'prob_sintomatic'},
        {'trigger': 'end_encubation', 'source': 'E', 'dest': 'A'},
        {'trigger': 'severity', 'source': 'I', 'dest': 'H', 'conditions': 'prob_severity'},
        {'trigger': 'severity', 'source': 'I', 'dest': '='},
        {'trigger': 'recovered', 'source': ['A', 'I', 'H'], 'dest': 'R'},
        {'trigger': 'dead', 'source': 'H', 'dest': 'D', 'conditions': 'prob_to_die', }
    ]

    def __init__(self, name, state, age, agent=None):
        # self.actual_state=actual_state
        self.time_in_state = 0
        self.pAI= {"0-9": 0.45, "10-19": 0.55, "20-29": 0.73,"30-39": 0.73, "40-49": 0.75,
                   "50-59": 0.85,"60-69": 0.85,"70-79": 0.90,"80-89": 0.95,"90+": 0.95}
        #self.pAI = {"0-9": 0.2, "10-19": 0.2, "20-29": 0.2, "30-39": 0.2, "40-49": 0.2,
        #            "50-59": 0.2, "60-69": 0.2, "70-79": 0.2, "80-89": 0.2, "90+": 0.2}
        self.pIH = {"0-9": 0.06, "10-19": 0.06, "20-29": 0.08, "30-39": 0.09, "40-49": 0.10,
                    "50-59": 0.15, "60-69": 0.18, "70-79": 0.20, "80-89": 0.50, "90+": 0.60}
        self.pHD = {"0-9": 0.03, "10-19": 0.20, "20-29": 0.20, "30-39": 0.20, "40-49": 0.25,
                    "50-59": 0.30, "60-69": 0.30, "70-79": 0.40, "80-89": 0.80, "90+": 0.99}
        self.rate={}
        # self.prob_inf=0.9
        rm = {'rEI': 1 / 5, 'rIR': 0.05, 'rIH': 0.1, 'rHR': 0.05, 'rHD': 0.1}
        rs = {'rEI':0.05, 'rIR': 0.01, 'rIH': 0.03, 'rHR': 0.01, 'rHD': 0.03}
        for k,v in rm.items():
            self.rate[k] = np.absolute(np.random.normal(v,rs[k]))

        self.name = name
        self.age = age
        self.machine = Machine(model=self, states=SEAIHRD_covid.states, transitions=SEAIHRD_covid.transitions,
                               initial=state)
        self.prob_inf = 0.8
        self._agent = agent

    def roundup(self, x):
        return int(math.ceil(x / 10.0)) * 10

    def recovery_time(self):
        pass

    def prob_infection(self, Mask1, Mask2):
        """ probability of infection after a risk contact. """
        pMask1 = Mask1.maskPtrans(Mask1)
        pMask2 = Mask2.maskPtrans(Mask2)
        pbi = self.prob_inf * pMask1 * pMask2
        res = random.random() < pbi

        if res: self.time_in_state = 0
        # print("CONTACT", self.prob_inf, pMask1, pMask2, prob_inf)
        return res

    def prob_sintomatic(self):
        """ probability to become symptomatic. """
        p = list(self.pAI.values())[int(self.roundup(self.age) / 10) - 1]
        return random.random() < p

    def prob_to_die(self, H_collapse=False):
        """ probability to died"""
        p = list(self.pHD.values())[int(self.roundup(self.age) / 10) - 1]
        if H_collapse:
            return True
        else:
            return random.random() < p

    def prob_severity(self):
        """ probability to become hospitalized. """
        p = list(self.pIH.values())[int(self.roundup(self.age) / 10) - 1]
        return random.random() < p

    def check_state(self):

        # time = self.time_in_state

        inizial_state = self.state
        transition_for_E = self.nothing
        transition_for_A = self.nothing
        transition_for_I = self.nothing
        transition_for_H = self.nothing

        if inizial_state == 'E':
            if self.time_in_state == round(1 / self.rate['rEI']):  # 5 days on end_encubation
                #transition_for_E = self.end_encubation
                self.end_encubation()

        elif inizial_state == 'A':
            if self.time_in_state == int(round(1 / self.rate['rIR'])):  # 20 days -
                #transition_for_A = self.recovered
                self.recovered()

        elif inizial_state == 'I':
            if self.time_in_state == round(1 / self.rate['rIR']):  # 20 days -
                #transition_for_I = self.recovered
                 self.recovered()

            elif self.time_in_state == round(1 / self.rate['rIH']):  # 10 days on prob_severity
                #transition_for_I = self.severity
                self.severity()



        elif inizial_state == 'H':
            if self.time_in_state == round(1 / self.rate['rHR']):  # 25 days -
                #transition_for_H = self.recovered
                self.recovered()

            elif self.time_in_state == round(1 / self.rate['rHD']) :  # 10 days with p_to_die
                #transition_for_H = self.dead
                self.dead()




        # transition = {'S': self.nothing,
        #               'E': transition_for_E,
        #               'A': transition_for_A,
        #               'I': transition_for_I,
        #               'H': transition_for_H,
        #               'R': self.nothing,
        #               'D': self.nothing,
        #               }
        # result_state = transition[self.state]()

        if inizial_state == self.state:
            self.time_in_state += 1
        else:
            self.time_in_state = 0

        #return True #result_state



        # self.machine.add_transition(trigger='infected',source='S',dest='E')
        # self.machine.add_transition('recovered','E', 'S')


# covid_state=SEAIHRD_covid("Agent_01","E",90)
# covid_state.prob_inf=0.6
# print(covid_state.state)
# covid_state.contact()

class Mask(Enum):
    NONE = 0
    HYGIENIC = 1
    FFP2 = 2

    def __str__(self):
        return self.name

    @classmethod
    def maskPtrans(self, mask):
        if mask == self.NONE:
            return 1
        elif mask == self.HYGIENIC:
            return float(1-0.70)
        elif mask == self.FFP2:
            return float(1-0.90)

    @classmethod
    def RandomMask(self, probs):
        a = random.choices([0, 1, 2], weights=probs, k=1)
        return Mask(a[0])
