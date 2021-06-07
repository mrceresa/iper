import sys, os

import numpy as np
import matplotlib.pyplot as plt

import logging
import logging.config

from bcn_ms import CityModel
import argparse
import geopy

geopy.geocoders.options.default_user_agent = "iper-social"

from datetime import datetime
from iper import PopulationRequest

from agents import HumanAgent, RandomWalk
from Hospital_class import Workplace, Hospital
import DataCollector_functions as dc
import contextily as ctx
import time


def main(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    ctx.set_cache_dir(args.cache_dir)
    # Set log level
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sdt = datetime.today().strftime('%Y%m%d-%H%M%S')
    fname = os.path.join(args.output_dir, "abm_debug_%s.log" % sdt)

    logconfig_fname = os.path.join(os.getcwd(), 'logging.conf')
    print("Trying to load config file from", logconfig_fname)
    logging.config.fileConfig(
        logconfig_fname,
        defaults={'logfilename': fname}
    )

    l = logging.getLogger(__name__)

    # Start model
    l.info("Starting City simulator with params %s" % str(args))
    args.lockdown['inf_threshold'] = int(args.lockdown['inf_threshold'] * args.agents)
    config = {
        "logger": l,
        "basemap": args.basemap,
        "agents": args.agents,
        "family": args.family,
        "age": args.age,
        "hospitals": args.hospitals,
        "tests": args.tests,
        "quarantine": args.quarantine,
        "alarm_state": args.lockdown,
        "peopleMeeting": args.meeting
    }
    city = CityModel(config)
    pr = PopulationRequest()

    pr._data = {
        "Human": {"num": args.agents,
                  "type": HumanAgent,
                  "defaultBehaviours": [RandomWalk],
                  "gridSize": (100,)
                  },
        "Hospital": {"num": args.hospitals,
                     "type": Hospital,
                     "gridSize": (100,)
                     },
        "Workplace": {"num": args.workplaces,
                      "type": Workplace,
                      "gridSize": (100,)
                      }
    }

    city.addPopulationRequest(pr)
    city.createAgents(args.agents, args.workplaces)

    city.plotAll(args.output_dir, "pre.png")
    tic = time.perf_counter()
    city.run(args.steps)
    toc = time.perf_counter()
    l.info("Simulation lasted %.1f seconds" % (toc - tic))

    l.info("Saving results to %s" % args.output_dir)
    city.plot_results(args.output_dir)
    city.plotAll(args.output_dir, "res.png")

    return city



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="results-%s" % datetime.today().strftime('%Y%m%d-%H'),
                        help="Output dir")  # %Y%m%d-%H%M%S
    parser.add_argument('-c', '--cache_dir', type=str, default="ctx-cache", help="Dir to cache maps")
    parser.add_argument('-v', '--verbose', action="store_true", help="Print additional information")
    parser.add_argument('-s', '--steps', type=int, default=30, help="Timesteps to run the model for")
    parser.add_argument('-n', '--agents', type=int, default=10000, help="Numer of starting agents")
    parser.add_argument('-H', '--hospitals', type=int, default=10, help="Numer of hospitals")
    parser.add_argument('-t', '--tests', type=int, default=10, help="Number of tests applied daily")
    parser.add_argument('-q', '--quarantine', type=int, default=10, help="Number of self-quarantine days")
    parser.add_argument('-l', '--lockdown', type= dict, default={'inf_threshold':0.0, 'night_curfew': 23, 'masks': [0.01, 0.64, 0.35], 'quarantine': 10, 'meeting': 5, 'remote-working': 0.7, 'total_lockdown': False}, help="Number of detected infected people to apply health measures")
    parser.add_argument('-w', '--workplaces', type=int, default=20, help="Numer of workplaces")
    parser.add_argument('-m', '--meeting', type=int, default=8, help="Numer of People on Meetings")
    parser.add_argument('-b', '--basemap', type=str, default="Barcelona, Spain",
                        help="Basemap for geo referencing the model")
    parser.add_argument('-f', '--family', type=list, default=[19.9, 23.8, 20.4, 24.8, 8.9, 2.2],
                        help="distribution listeach term in the distr list represents the probability of generating a familywith a number of individuals equal to the index of that element of distr")
    parser.add_argument('-j', '--job', type=dict,
                        default={"unemployed": 6.0, "type1": 14.00, "type2": 10.00, "type3": 10.00, "type4": 10.00,"type5": 10.00, "type6": 40.00, }, help="it is a dictionary containing workgroups")
    parser.add_argument('-a', '--age', type=dict,
                        default={"00-10": 8.89, "11-20": 8.58, "21-30": 13.04, "31-40": 15.41, "41-50": 15.34,
                                 "51-60": 13.06, "61-70": 10.53, "71-80": 8.41, "81-90": 5.46, "91-99": 1.28},
                        help="it is a dictionary containing age groups")
    parser.add_argument('-V', '--virus', type=dict,
                        default={"incubation_days": 3, "infection_days": 5, "immune_days": 3, "severe_days": 3,
                                 "pTrans": 0.7, "pSympt": 0.8, "pTest": 0.9, "death_rate": 0.002, "severe_rate": 0.005},
                        help="it is a dictionary containing virus characteristics")
    parser.set_defaults(func=main)

    args = parser.parse_args()
    model = args.func(args)
