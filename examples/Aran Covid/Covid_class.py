from enum import Enum
import random


class VirusCovid(object):
    def __init__(self, incubation_days, incubation_days_sd, infection_days, infection_days_sd, immune_days,
                 immune_days_sd, severe_days, severe_days_sd, ptrans, pSympt, pTest, death_rate,severe_rate):
        # self.r0 = 2.5

        # from EXP to INF
        self.incubation_days = incubation_days
        self.incubation_days_sd = incubation_days_sd

        # from INF to REC - HOSP
        self.infection_days = infection_days
        self.infection_days_sd = infection_days_sd

        # from REC to SUSC
        self.immune_days = immune_days
        self.immune_days_sd = immune_days_sd

        # from HOSP to REC - DEATH
        self.severe_days = severe_days
        self.severe_days_sd = severe_days_sd

        self.ptrans = ptrans
        self.pSympt = pSympt  # probability to present the symptoms
        self.pTest = pTest # probability of test of true positive
        self.death_rate = death_rate
        self.severe_rate = severe_rate

    def pTrans(self, Mask1, Mask2):
        pMask1 = Mask1.maskPtrans(Mask1)
        pMask2 = Mask2.maskPtrans(Mask2)
        return self.ptrans * pMask1 * pMask2


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

    @classmethod
    def maskPtrans(self, mask):
        if mask == self.NONE:
            return 1
        elif mask == self.HYGIENIC:
            return 0.90
        elif mask == self.FFP2:
            return 0.80

    @classmethod
    def RandomMask(self):
        mask_list = [Mask.NONE, Mask.HYGIENIC, Mask.FFP2]
        return random.choice(mask_list)
