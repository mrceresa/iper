from enum import Enum


class VirusCovid(object):
    def __init__(self):
        self.r0 = 2.5

        # from EXP to INF
        self.incubation_days = 2
        self.incubation_days_sd = 1

        # from INF to REC - HOSP
        self.infection_days = 5#1
        self.infection_days_sd = 1#7

        # from REC to SUSC
        self.immune_days = 3#0
        self.immune_days_sd = 1#0

        # from HOSP to REC - DEATH
        self.severe_days = 2#0
        self.severe_days_sd = 1#0

        self.ptrans = 0.7
        self.pSympt = 0.4  # probability to present the symptoms
        self.pTest = 0.8  # probability of test of true positive
        self.death_rate = 0.002 / (24 * 4)
        self.severe_rate = 0.005 / (24 * 4)


class State(Enum):
    SUSC = 0
    EXP = 1
    INF = 2
    REC = 3
    HOSP = 4
    DEAD = 5

    def __str__(self):
        return self.name


class Mask(Enum):
    NONE = 0
    HYGIENIC = 1
    FFP2 = 2

    def __str__(self):
        return self.name

    def maskPtrans(self):
        if Mask.NONE:return 1
        elif Mask.HYGIENIC: return 0.90
        elif Mask.FFP2: return 0.80

    def RandomMask(self):
        return Mask.FFP2
