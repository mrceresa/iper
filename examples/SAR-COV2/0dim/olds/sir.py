#!/usr/bin/python

import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

import seaborn as sns
sns.set_style("darkgrid")

from plotdf import plotdf

import argparse
from random import random
import gillespy2

import csv

# The original SIR model differential equations with mass action effect
def derivOrig(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I 
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# SIR model with corrected incidence:
# b*N^v*S*I/N, con v=0
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N 
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Adimensional SIR model (remember to use tau instead of t).
def deriv_adim(y, tau, N, mu):
    S, I, R = y
    dSdt = -S * I / N
    dIdt = (S / N - mu) * I 
    dRdt = mu * I 
    return dSdt, dIdt, dRdt

def solveSIRdet(y0, tau, N, mu, beta, gamma):
  # Integrate the SIR equations over the time grid, t
  #ret = scint.odeint(deriv_adim, y0, tau, args=(N, mu))
  ret = scint.odeint(deriv, y0, tau, args=(N, beta, gamma))
  s, i, r = ret.T

  return {"S":s, "I":i, "R":r}

def solveSIRstoch(y0, t, N, mu, beta, gamma, niter=5):
  SS=[]; II=[]; RR=[]
  for _ in range(niter):
    s0, i0, r0 = y0  #suscettibili iniziali
    i=[i0]; r=[r0]; s=[s0]

    for _t in range(len(t)):
      print("_t:",_t)
      
      _s = int(s[_t]); _i = int(i[_t]);
      pInfection = beta*_i/(N-1)
      #pInfection = _s*_i*beta/(math.factorial(N)/(2*math.factorial(N-2)))
      print("\tProbability of getting infected this round:",pInfection)
      print("\tProbability of recovering:",gamma)
      print("\tSimulating transition S->I for %d agents:"%_s)

      infected=np.random.binomial(_s,pInfection)
      recovered=np.random.binomial(_i,gamma)
      
      print("\tSimulating transition I->R for %d agents:"%_i)    
      
      print("\t New infected: %d  New recovered: %d"%(infected, recovered))  
      s.append(s[_t]-infected)
      i.append(i[_t]+infected-recovered)
      r.append(r[_t]+recovered)
    
    # Normalization
    s = np.asarray(s[1:]); SS.append(s)
    i = np.asarray(i[1:]); II.append(i)
    r = np.asarray(r[1:]); RR.append(r)

  return {"S":np.asarray(SS).T, "I":np.asarray(II).T, "R":np.asarray(RR).T}

def solveGillespie(y0, t, N, mu, beta, gamma, niter=5):
  s0, i0, r0 = y0  #suscettibili iniziali

  class sSIR(gillespy2.Model):
    def __init__(self, t, s0, i0, r0, beta, gamma):
      # First call the gillespy2.Model initializer.
      gillespy2.Model.__init__(self, name='StochasticSIR')

      # Define parameters for the rates of creation and dissociation.
      k_i = gillespy2.Parameter(name='k_i', expression=beta/N)
      k_r = gillespy2.Parameter(name='k_r', expression=gamma)
      self.add_parameter([k_i, k_r])

      # Define variables for the molecular species representing M and D.
      s = gillespy2.Species(name='S', initial_value=s0)
      i = gillespy2.Species(name='I', initial_value=i0)         
      r = gillespy2.Species(name='R', initial_value=r0)   
      self.add_species([s, i, r])

      # The list of reactants and products for a Reaction object are each a
      # Python dictionary in which the dictionary keys are Species objects
      # and the values are stoichiometries of the species in the reaction.
      r_i = gillespy2.Reaction(name="infection", rate=k_i, reactants={s:1,i:1}, products={i:2})
      r_r = gillespy2.Reaction(name="recovery", rate=k_r, reactants={i:1}, products={r:1})
      self.add_reaction([r_i, r_r])

      # Set the timespan for the simulation.
      self.timespan(t)

  SS=[]; II=[]; RR=[]
  print("Solving with gillespies for timesteps:",t)
  model = sSIR(t, s0, i0, r0, beta, gamma)
  results = model.run(number_of_trajectories=niter)
  for trajectory in results:
    SS.append(trajectory["S"]) 
    II.append(trajectory["I"])
    RR.append(trajectory["R"])

  return {"S":np.asarray(SS).T, "I":np.asarray(II).T, "R":np.asarray(RR).T}


def main(args):
  from utils import plotPhase, plotTrajectories, plotAlls

  #plt.rc('text', usetex=True)

  # How many simul for stochastic methods:
  nits = args.nits
  nitg = args.nitg
  # Total population, N.
  N = args.agents
  # Initial number of infected and recovered individuals, I0 and R0.
  I0, R0 = args.i0, args.r0
  # Everyone else, S0, is susceptible to infection initially.
  S0 = N - I0 - R0
  # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
  beta, gamma = args.beta, args.gamma 
  # A grid of time points (in days)
  t = np.linspace(0, args.days, args.days)

  tau = t*beta*N
  mu = gamma/(beta*N)
  print("beta:",beta)
  print("gamma:",gamma)
  print("Adimentional virulence mu:",mu)

  # Initial conditions vector
  y0 = 1.0*S0, 1.0*I0, 1.0*R0
  sir_det = solveSIRdet(y0, t, N, mu, beta, gamma)
  sir_stoch = solveSIRstoch(y0, t, N, mu, beta, gamma, nits)
  sir_gill = solveGillespie(y0, t, N, mu, beta, gamma, nitg)

  np.savetxt('sir_results.csv', np.column_stack( 
    (t, sir_det["S"], sir_det["I"], sir_det["R"])
  ), delimiter=', ' )

  xbound, ybound = (0.0, 1.0), (0.0, 1.0)
  gridsteps = 30; tmax=2e3; nsteps=1e2; tdir="forward"

  axes = plotPhase(tau, xbound, ybound, gridsteps, tmax, nsteps, N, mu)

  inits=[(0.9, 0.7, 0.0),(0.9, 0.6, 0.0),
         (0.9, 0.4, 0.0), (0.9, 0.2, 0.0)]  
  plotTrajectories(inits, axes, tmax*4, nsteps*4, tdir, N, mu)

  plotAlls(t, sir_det, sir_stoch, sir_gill, nits, nitg)


if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  parser.add_argument('-d','--days', type=int, default=500, help="Timesteps to run the model for" )          
  parser.add_argument('-n','--agents', type=int, default=1000, help="Initial population" )  
  parser.add_argument('--i0', type=int, default=25, help="Initial infected" ) 
  parser.add_argument('--r0', type=int, default=0, help="Initial recovered" )     
  parser.add_argument('--beta', type=float, default=0.09, help="Contact rate" )
  parser.add_argument('--gamma', type=float, default=0.05, help="Mean recovery rate" )   
  parser.add_argument('--nits', type=int, default=50, help="# of simulations for stochastic solver" ) 
  parser.add_argument('--nitg', type=int, default=5, help="# of simulations for gillespie solver" )      
  parser.set_defaults(func=main)  
  
  args = parser.parse_args()  
  args.func(args)  
