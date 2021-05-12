from transitions import Machine
import random
import math

class SEAIHRD_covid(object):
#recovery time with gaussian distribution:{'19':(13.43,5.8), '29':(13.98,5.85),'39':(14.31,6.7),'49':(14.78,5.8),'59':(14.78,5.8),'99':(14.8,6.2)} 
#A single center Chinese study of 221 discharged COVID-19
# patients observed an average time to recovery of 10.63±1.93 days
# for mild to moderate patients, compared with 18.70±2.50 for severe patients.

   
    def roundup(self,x):
        return int(math.ceil(x / 10.0)) * 10

    def recovery_time(self):
        pass
   
    def prob_infection(self):
        """ probability of infection after a risk contact. """
        return random.random() < self.prob_inf
            
    def prob_sintomatic(self):
        """ probability to become symptomatic. """
        p=list(self.pAI.values())[int(self.roundup(self.age)/10)-1]
        return random.random() < p
    
    def prob_to_die(self):
        """ probability to died"""
        p=list(self.pHD.values())[int(self.roundup(self.age)/10)-1]
        return random.random() < p
    
    def prob_severity(self):
        """ probability to become hospitalized. """
        p=list(self.pIH.values())[int(self.roundup(self.age)/10)-1]
        return random.random() < p
    
    states = ['S', 'E', 'A', 'I', 'R','H', 'D']
    transitions=[
        {'trigger':'vaccination','source':'S','dest':'R'},
        {'trigger':'contact','source':'S','dest':'E', 'conditions':'prob_infection'},
        {'trigger':'contact','source':'S','dest':'S'},
        {'trigger':'end_encubation','source':'E','dest':'I', 'conditions':'prob_sintomatic'},
        {'trigger':'end_encubation','source':'E','dest':'A'},
        {'trigger':'severity','source':'I','dest':'H', 'conditions':'prob_severity'},
        {'trigger':'severity','source':'I','dest':'R'},
        {'trigger':'recovered','source':['A','I','H'],'dest':'R'},
        {'trigger':'dead','source':'H','dest':'D','conditions':'prob_to_die'}
        ]
    def __init__(self, name, actual_state, time_in_state,age):
        self.actual_state=actual_state
        self.time_in_state=time_in_state
        self.pAI={"0-9": 0.45, "10-19": 0.55, "20-29": 0.73,"30-39": 0.73, "40-49": 0.75,
                  "50-59": 0.85,"60-69": 0.85,"70-79": 0.90,"80-89": 0.95,"90+": 0.95}
        self.pIH={"0-9": 0.06, "10-19": 0.06, "20-29": 0.08, "30-39": 0.09, "40-49": 0.10,
               "50-59": 0.15, "60-69": 0.18, "70-79": 0.20, "80-89": 0.50, "90+": 0.60}
        self.pHD= {"0-9": 0.03, "10-19": 0.20, "20-29": 0.20,"30-39": 0.20, "40-49": 0.25, 
               "50-59": 0.30,"60-69": 0.30,"70-79": 0.40,"80-89": 0.80,"90+": 0.99}
        
        self.prob_inf=0.9
        self.name = name
        self.age=age
        self.machine = Machine(model=self, states=SEAIHRD_covid.states,transitions=SEAIHRD_covid.transitions, initial=self.actual_state)   
        # self.machine.add_transition(trigger='infected',source='S',dest='E')
        # self.machine.add_transition('recovered','E', 'S')
        
        
#covid_state=SEAIHRD_covid("Agent_01","S",20,90)
#print(covid_state.state)
#covid_state.contact()
#print(covid_state.state)  
#covid_state.end_encubation()
#print(covid_state.state)

