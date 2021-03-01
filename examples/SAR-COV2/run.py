import sys

sys.path.append('/home/enrico/iper-social-simulations')
import numpy as np
import matplotlib.pyplot as plt

import coloredlogs, logging
_log = logging.getLogger(__name__)

from BCNCovid2020 import BCNCovid2020
import argparse
import geopy
geopy.geocoders.options.default_user_agent = "iper-social"

def main(args):

  # Set log level
  loglevel = 'DEBUG' if args.verbose else 'INFO'
  coloredlogs.install(level=loglevel)
  
  # Start model
  _log.info("Started BCN Mobility simulator with params %s"%str(args))
  model = BCNCovid2020(args.agents, args.basemap, args.family, args.job, args.age)
  
  model.plotAll("start.png")
  
  model.run_model(args.steps)
  model.plotAll("end.png")
 
  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-v','--verbose', action="store_true", help="Print additional information" )
  parser.add_argument('-s','--steps', type=int, default=10, help="Timesteps to run the model for" )          
  parser.add_argument('-n','--agents', type=int, default=1000, help="Numer of starting agents" )
  parser.add_argument('-b','--basemap', type=str, default="Barcelona, Spain", help="Basemap for geo referencing the model" )
  parser.add_argument('-f','--family', type=list, default=[19.9 ,23.8 ,20.4, 24.8, 8.9, 2.2], help="distribution listeach term in the distr list represents the probability of generating a familywith a number of individuals equal to the index of that element of distr" ) 
  parser.add_argument('-j','--job', type=dict, default={"unemployed":6.0,"type1":14.00,"type2":10.00,"type3":10.00,"type4":10.00,"type5":10.00,"type6":40.00,} , help="it is a dictionary containing workgroups" )
  parser.add_argument('-a','--age', type=dict, default={"0-10":8.89,"11-20":8.58,"21-30":13.04,"31-40":15.41,"41-50":15.34,"51-60":13.06,"61-70":10.53,"71-80":8.41,"81-90":5.46,"91-100":1.28}, help="it is a dictionary containing age groups" )
  parser.set_defaults(func=main)  
  
  args = parser.parse_args()  
  model = args.func(args)  


	
	
