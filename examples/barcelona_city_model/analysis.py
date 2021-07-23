
class CityStatistics():
  def __init__(self):
    # variables for model data collector
    self.collector_counts = None
    dc.reset_counts(self)
    self.collector_counts["SUSC"] = config["agents"]

    self.datacollector = DataCollector(
        {"SUSC": dc.get_susceptible_count, "EXP": dc.get_exposed_count, "INF": dc.get_infected_count,
         "REC": dc.get_recovered_count, "HOSP": dc.get_hosp_count, "DEAD": dc.get_dead_count, "R0": dc.get_R0,
         "R0_Obs": dc.get_R0_Obs
         # , "Mcontacts": dc.get_R0_Obs0, "Quarantined": dc.get_R0_Obs1,"Contacts": dc.get_R0_Obs2,
         },
        tables={"Model_DC_Table": {"Day": [], "Susceptible": [], "Exposed": [], "Infected": [], "Recovered": [],
                                   "Hospitalized": [], "Dead": [], "R0": [], "R0_Obs": []
                                   # , "Mcontacts": [],"Quarantined": [], "Contacts": []
                                   }}
    )

    # variables for hospital data collector
    self.hosp_collector_counts = None
    dc.reset_hosp_counts(self)
    self.hosp_collector_counts["H-SUSC"] = config["agents"]
    self.hosp_collector = DataCollector(
        {"H-SUSC": dc.get_h_susceptible_count, "H-INF": dc.get_h_infected_count, "H-REC": dc.get_h_recovered_count,
         "H-HOSP": dc.get_h_hospitalized_count, "H-DEAD": dc.get_h_dead_count, },
        tables={"Hosp_DC_Table": {"Day": [], "Hosp-Susceptible": [], "Hosp-Infected": [], "Hosp-Recovered": [],
                                  "Hosp-Hospitalized": [], "Hosp-Dead": []}}
    )

    self.datacollector.collect(self)
    self.hosp_collector.collect(self)		