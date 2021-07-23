class PolicyIntervention(object):
  def __init__(self):
    # alarm state characteristics
    self.alarm_state = config["alarm_state"]
    self.lockdown_total = False
    self.quarantine_period = config["quarantine"]
    self.night_curfew = 24
    self.masks_probs = [1, 0, 0]  # [0.01, 0.66,0.33]      
    self.peopleInMeeting = config["peopleMeeting"]  # max people to meet with
    self.peopleInMeetingSd = config["peopleMeeting"] * 0.2

  def activate_alarm_state(self):
      self.alarm_state['inf_threshold'] = self.getTime().strftime("%Y-%m-%d")

      if 'night_curfew' in self.alarm_state.keys():
          self.night_curfew = self.alarm_state['night_curfew']

      if 'masks' in self.alarm_state.keys():
          self.masks_probs = self.alarm_state['masks']

      if 'quarantine' in self.alarm_state.keys():
          self.quarantine_period = self.alarm_state['quarantine']

      if 'meeting' in self.alarm_state.keys():
          self.peopleInMeeting = self.alarm_state['meeting']
          self.peopleInMeetingSd = self.alarm_state['meeting'] * 0.2

      if 'remote-working' in self.alarm_state.keys() and self.alarm_state['remote-working'] < self.employment_rate:
          fire_employees = round(self.alarm_state['remote-working'] / self.employment_rate, 2)
          for human in self.schedule.agents:
              if isinstance(human, HumanAgent):
                  if human.workplace is not None and np.random.choice([False, True],
                                                                      p=[fire_employees, 1 - fire_employees]):
                      human.workplace = None

      if 'total_lockdown' in self.alarm_state.keys():
          self.lockdown_total = self.alarm_state['total_lockdown']


class Vaccination(object):
  pass


class InfectionTesting():
  def __init__(self):
    self.peopleTested = {}
    self.peopleToTest = {}
    self.PCR_tests = config["tests"] / config["hospitals"]



class ContactTracing():
    def clean_contact_list(self, Adays, Hdays, Tdays):
      """ Function for deleting past day contacts sets and arrange today's tests"""
      date = self.getTime().strftime('%Y-%m-%d')
      Atime = (self.getTime() - timedelta(days=Adays)).strftime('%Y-%m-%d')
      Htime = (self.getTime() - timedelta(days=Hdays)).strftime('%Y-%m-%d')

      Ttime = (self.getTime() - timedelta(days=Tdays)).strftime('%Y-%m-%d')
      if Ttime in self.peopleTested: del self.peopleTested[Ttime]

      # People tested in the last recorded days
      peopleTested = set()
      for key in self.peopleTested:
          for elem in self.peopleTested[key]:
              peopleTested.add(elem)

      # print(f"Lista total de agentes a testear a primera hora: {self.peopleToTest}")

      # shuffle agent list to distribute to test agents among the hospitals
      agents_list = self.schedule.agents.copy()
      random.shuffle(agents_list)
      for a in agents_list:
          # delete contacts of human agents of Adays time
          if isinstance(a, HumanAgent):
              if Atime in a.contacts: del a.contacts[Atime]
              if Atime in a.R0_contacts: del a.R0_contacts[Atime]
              # create dict R0 for infected people in case it is not updated during the day
              if a.machine.state in ["I", "A"]:
                  a.R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0, round(
                      1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]
              elif a.machine.state == "E":
                  a.R0_contacts[self.getTime().strftime('%Y-%m-%d')] = [0, round(1 / a.machine.rate['rEI']) + round(
                      1 / a.machine.rate['rIR']) - a.machine.time_in_state, 0]

          elif isinstance(a, Hospital):
              peopleTested = a.decideTesting(peopleTested)
              """if date in a.PCR_results:
                  if not date in self.peopleTested: self.peopleTested[date] = a.PCR_results[date]
                  else:
                      for elem in a.PCR_results[date]: self.peopleTested[date].add(elem)"""
              # delete contact tracing of Hdays time
              #if Htime in a.PCR_testing: del a.PCR_testing[Htime]
              #if Htime in a.PCR_results: del a.PCR_results[Htime]

              #print(f"Lista de contactos de hospital {a.unique_id} es {a.PCR_testing}. Con {peopleTested}")

      today = self.getTime().strftime('%Y-%m-%d')
      #if not today in self.peopleTested:
      self.peopleTested[today] = set()
      self.peopleToTest[today] = set()

      # print(f"Lista total de testeados: {self.peopleTested}")
      # print(f"Lista total de agentes a testear: {self.peopleToTest}")