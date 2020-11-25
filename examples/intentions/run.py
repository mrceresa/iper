
import numpy as np
import matplotlib.pyplot as plt

import coloredlogs, logging
_log = logging.getLogger(__name__)

from maze import Maze
import argparse
import geopy
geopy.geocoders.options.default_user_agent = "iper-social"

def main(args):

  # Set log level
  loglevel = 'DEBUG' if args.verbose else 'INFO'
  coloredlogs.install(level=loglevel)
  
  # Start model
  _log.info("Started Maze simulator with params %s"%str(args))
  model = Maze(args.agents)
  
  model.plotAll()
  plt.savefig("start.png")
  model.run_model(args.steps)
  model.plotAll()
  plt.savefig("end.png")
  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-v','--verbose', action="store_true", help="Print additional information" )
  parser.add_argument('-s','--steps', type=int, default=10, help="Timesteps to run the model for" )          
  parser.add_argument('-n','--agents', type=int, default=1000, help="Numer of starting agents" )
  parser.set_defaults(func=main)  
  
  args = parser.parse_args()  
  model = args.func(args)  


	
	
