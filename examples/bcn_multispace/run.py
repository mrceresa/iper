import sys,os

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
  fname = os.path.join(args.output_dir, "abm_debug_%s.log"%sdt)

  logconfig_fname = os.path.join(os.getcwd(), 'logging.conf')
  print("Trying to load config file from", logconfig_fname)
  logging.config.fileConfig(
    logconfig_fname, 
    defaults={'logfilename': fname}
    )

  l = logging.getLogger(__name__)


  # Start model
  l.info("Starting City simulator with params %s"%str(args))
  config = {
    "logger":l,
    "basemap":args.basemap
  }
  city = CityModel(config)
  pr = PopulationRequest()

  pr._data = {
      "Human":{"num":args.agents, 
          "type":HumanAgent, 
          "defaultBehaviours":[RandomWalk],
          "gridSize": (100,)
          }            
        }

  city.addPopulationRequest(pr)
  city.createAgents()    

  city.plotAll(args.output_dir, "pre.png")  
  tic = time.perf_counter()  
  city.run(args.steps)
  toc = time.perf_counter()        
  l.info("Simulation lasted %.1f seconds"%(toc-tic))    
  
  l.info("Saving results to %s"%args.output_dir)
  city.plotAll(args.output_dir, "res.png")
  
  return city


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o','--output_dir', type=str, default="results-%s"%datetime.today().strftime('%Y%m%d-%H'), help="Output dir" ) #%Y%m%d-%H%M%S
  parser.add_argument('-c','--cache_dir', type=str, default="ctx-cache", help="Dir to cache maps")
  parser.add_argument('-v','--verbose', action="store_true", help="Print additional information" )
  parser.add_argument('-s','--steps', type=int, default=10, help="Timesteps to run the model for" )          
  parser.add_argument('-n','--agents', type=int, default=10000, help="Numer of starting agents" )
  parser.add_argument('-b','--basemap', type=str, default="Barcelona, Spain", help="Basemap for geo referencing the model" )
  parser.add_argument('-f','--family', type=list, default=[19.9 ,23.8 ,20.4, 24.8, 8.9, 2.2], help="distribution listeach term in the distr list represents the probability of generating a familywith a number of individuals equal to the index of that element of distr" ) 
  parser.add_argument('-j','--job', type=dict, default={"unemployed":6.0,"type1":14.00,"type2":10.00,"type3":10.00,"type4":10.00,"type5":10.00,"type6":40.00,} , help="it is a dictionary containing workgroups" )
  parser.add_argument('-a','--age', type=dict, default={"00-10":8.89,"11-20":8.58,"21-30":13.04,"31-40":15.41,"41-50":15.34,"51-60":13.06,"61-70":10.53,"71-80":8.41,"81-90":5.46,"91-99":1.28}, help="it is a dictionary containing age groups" )
  parser.set_defaults(func=main)  
  
  args = parser.parse_args()  
  model = args.func(args)  


	
	
