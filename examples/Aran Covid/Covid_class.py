from enum import Enum
import random


class VirusCovid(object):
    def __init__(self):
        self.r0 = 2.5

        # from EXP to INF
        self.incubation_days = 3
        self.incubation_days_sd = 1

        # from INF to REC - HOSP
        self.infection_days = 5  # 1
        self.infection_days_sd = 1  # 7

        # from REC to SUSC
        self.immune_days = 3  # 0
        self.immune_days_sd = 1  # 0

        # from HOSP to REC - DEATH
        self.severe_days = 3  # 0
        self.severe_days_sd = 1  # 0

        self.ptrans = 0.7
        self.pSympt = 0.4  # probability to present the symptoms
        self.pTest = 1  # probability of test of true positive
        self.death_rate = 0.002 / (24 * 4)
        self.severe_rate = 0.005 / (24 * 4)

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
        if mask == self.NONE: return 1
        elif mask == self.HYGIENIC: return 0.90
        elif mask == self.FFP2: return 0.80

    @classmethod
    def RandomMask(self):
        mask_list = [Mask.NONE, Mask.HYGIENIC, Mask.FFP2]
        return random.choice(mask_list)
