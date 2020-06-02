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
from matplotlib.gridspec import GridSpec

def plotSEIRD(tau, sd, sdfname='sir.png', sdstyle='-'):
  # Plot the data on three separate curves for S(t), E(t), I(t) and R(t)
  fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='w')
  gs = GridSpec(2, 2, figure=fig)
  ax1 = fig.add_subplot(gs[0, :])
  ax3 = fig.add_subplot(gs[-1, 0])
  ax4 = fig.add_subplot(gs[-1, -1])

  ax1.plot(tau, sd["S"], 'b'+sdstyle, alpha=0.5, lw=2, label='Susceptible')
  ax1.plot(tau, sd["E"], 'y'+sdstyle, alpha=0.5, lw=2, label='Exposed')
  ax1.plot(tau, sd["I"], 'r'+sdstyle, alpha=0.5, lw=2, label='Infected')
  ax1.plot(tau, sd["R"], 'g'+sdstyle, alpha=0.5, lw=2, label='Recovered')
  ax1.plot(tau, sd["D"], 'k'+sdstyle, alpha=0.5, lw=2, label='Death')
  fig.suptitle("SEIRD model")
  ax1.set_xlabel('Time (days)')
  ax1.set_ylabel('Population')
  #ax.set_ylim(0,1.2)
  ax1.yaxis.set_tick_params(length=0)
  ax1.xaxis.set_tick_params(length=0)
  ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax1.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(False)
  
  ax3.set_xlabel("Time (days)")
  ax3.set_ylabel(r"$R_0$")
  ax3.plot(tau, sd["R_0"], 'g'+sdstyle, alpha=0.5, lw=2, label='$R_0$')
  ax3.legend()

  ax4.set_xlabel("Time (days)")
  ax4.set_ylabel("Alpha")
  ax4.plot(tau, sd["alpha"], 'k'+sdstyle, alpha=0.5, lw=2, label='alpha')
  ax4.legend()

  plt.savefig(sdfname)


# SEIR model with corrected incidence:
def deriv(y, t, N, beta, gamma, delta, alpha, rho):
    S, E, I, R, D = y
    dSdt = -beta(t) * I * S / N
    dEdt = beta(t) * I * S / N   - delta * E
    dIdt = delta * E    - (1 - alpha(t, I, N) ) * gamma * I      - alpha(t, I, N) * rho * I
    dRdt = (1 - alpha(t, I, N)) * gamma * I
    dDdt = alpha(t, I, N) * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def solveSIRdet(y0, t, N, beta, gamma, delta, alpha, rho):
  # Integrate the SEIR equations over the time grid, t
  ret = scint.odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
  s, e, i, r, d = ret.T

  return {"S":s, "E":e, "I":i, "R":r, "D":d}

def doSim(args):
  print(args)
  # Total population, N.
  N = args.agents
  # Initial number of infected and recovered individuals, I0 and R0.
  E0, I0, R0, D0 = args.e0, args.i0, args.r0, 0
  # Everyone else, S0, is susceptible to infection initially.
  S0 = N - I0 - R0 - E0
  # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
  gamma, delta, rho = args.gamma, args.delta, args.rho 
  # A grid of time points (in days)
  t = np.linspace(0, args.days, args.days)

  # Lockdown effect
  MAXR0 = 6.0; minR0 = 0.9
  k = 0.5 # Transition parameter from max to min R0
  # starting day of hard lockdown
  startLockdown = args.lock if args.lock > 0 else args.days 
  def R_0(tau):
    return ( MAXR0 - minR0 ) / (1 + np.exp(-k*(-tau+startLockdown))) + minR0
  def beta(tau):
    return R_0(tau)*gamma

  # Effect of age on death rate
  alpha_age = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.3}
  demographic = {"0-29": 0.05, "30-59": 0.25, "60-89": 0.45, "89+": 0.25}
  age_effect = 1.0
  alpha_av = sum(alpha_age[i] * demographic[i] for i in list(alpha_age.keys()))
  def alpha(tau, I, N):
    return age_effect*I/N + alpha_av

  # Initial conditions vector
  y0 = S0, E0, I0, R0, D0
  sir_det = solveSIRdet(y0, t, N, beta, gamma, delta, alpha, rho)
  sir_det["R_0"] = list(map(R_0, t)) 
  sir_det["alpha"] = [alpha(tau, sir_det["I"][tau], N) for tau in range(len(t))]

  np.savetxt('seird_results.csv', np.column_stack( 
    (t, sir_det["S"], sir_det["E"], sir_det["I"], sir_det["R"], sir_det["D"])
  ), delimiter=', ' )

  plotSEIRD(t, sir_det, sdfname="seird.png")

def doFit(args):
  print(args)


if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers()
  sim = subparsers.add_parser('simulate', aliases=['sim'])
  sim.add_argument('-d','--days', type=int, default=150, help="Timesteps to run the model for" )          
  sim.add_argument('-n','--agents', type=int, default=50000000, help="Initial population" )  
  sim.add_argument('--e0', type=int, default=0, help="Initial exposed" )  
  sim.add_argument('--i0', type=int, default=1, help="Initial infected" ) 
  sim.add_argument('--r0', type=int, default=0, help="Initial recovered" )    
  sim.add_argument('--gamma', type=float, default=1.0/10.0, help="Mean recovery rate" ) 
  sim.add_argument('--delta', type=float, default=1.0/2.0, help="Inverse of the incubation period" )     
  sim.add_argument('--rho', type=float, default=1.0/8.0, help="Days from critical to death" )  
  sim.add_argument('--lock', type=int, default=0, help="When to start the lockdown" )  
  sim.set_defaults(func=doSim)  

  fit = subparsers.add_parser('fit')
  fit.add_argument("--data", type=str, help="csv with data for fit")
  fit.set_defaults(func=doFit)  

  args = parser.parse_args()  
  args.func(args)  
