from enum import IntEnum


class VirusCovid(object):
    def __init__(self, len_timetable):
        self.r0 = 2.5

        # from EXP to INF
        self.incubation_days = 2 * len_timetable  # by multiplying the length of the timetable, we get the total number of days in such state
        self.incubation_days_sd = 1 * len_timetable

        # from INF to REC - HOSP
        self.infection_days = 21 * len_timetable
        self.infection_days_sd = 7 * len_timetable

        # from REC to SUSC
        self.immune_days = 30 * len_timetable
        self.immune_days_sd = 10 * len_timetable

        # from HOSP to REC - DEATH
        self.severe_days = 20 * len_timetable
        self.severe_days_sd = 10 * len_timetable

        self.ptrans = 0.5
        self.death_rate = 0.02
        self.severe_rate = 0.05


class State(IntEnum):
    SUSC = 0
    EXP = 1
    INF = 2
    REC = 3
    HOSP = 4
    DEAD = 5