#!/usr/bin/python

import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

import seaborn as sns
sns.set_style("darkgrid")

import pandas as pd
import argparse
from random import random

import csv
from matplotlib.gridspec import GridSpec
from lmfit import Parameters, Model
from functools import partial


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


def solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, gamma, delta, rho):
  # Lockdown effect
  def R_0(tau):
    return ( r0_max - r0_min ) / (1 + np.exp(-k*(-tau+startLockdown))) + r0_min
  def beta(tau):
    return R_0(tau)*gamma

  # Effect of age on death rate
  alpha_age = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.3}
  demographic = {"0-29": 0.05, "30-59": 0.25, "60-89": 0.45, "89+": 0.25}
  alpha_av = sum(alpha_age[i] * demographic[i] for i in list(alpha_age.keys()))
  def alpha(tau, I, N):
    return age_effect*I/N + alpha_av

  # Integrate the SEIR equations over the time grid, t
  ret = scint.odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
  s, e, i, r, d = ret.T

  res = {"S":s, "E":e, "I":i, "R":r, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
  res["alpha"] = [alpha(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

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
  r0_max = args.r0_max; r0_min = args.r0_min
  k = args.k # Transition parameter from max to min R0
  # starting day of hard lockdown
  startLockdown = args.lock if args.lock > 0 else args.days 

  # Strenght of the age effect
  age_effect = 1.0

  # Initial conditions vector
  y0 = S0, E0, I0, R0, D0
  print("Average recovery time %.3f days"%(1/gamma))
  print("Average incubation time %.3f days"%(1/delta))
  print("average survival of criticals %.3f days"%(1/rho))
  sir_det = solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, gamma, delta, rho)

  np.savetxt('seird_results.csv', np.column_stack( 
    (t, sir_det["S"], sir_det["E"], sir_det["I"], sir_det["R"], sir_det["D"])
  ), delimiter=', ' )

  plotSEIRD(t, sir_det, sdfname="seird.png")

def doFit(args):
  print(args)
  data = pd.read_csv(args.data)
  regione = 'P.A. Trento'
  r1 = data[data['denominazione_regione']==regione]
  N = args.agents
  I=np.array(r1['totale_positivi'])
  D=np.array(r1['deceduti'])
  R=np.array(r1['dimessi_guariti'])

  days = args.shift + len(r1["data"])
  if args.shift:
    I = np.concatenate((np.zeros(args.shift), I))
    R = np.concatenate((np.zeros(args.shift), R))
    D = np.concatenate((np.zeros(args.shift), D))

  S = N-I-D-R
  t = np.linspace(0, days-1, days, dtype=int)

  i0=1; e0=0; r0=0; s0=N-i0-r0; d0=0     
  y0=s0,e0,i0,r0,d0

  age_effect = 1.0

  def covid_deaths(t, r0_max, r0_min, k, startLockdown, gamma, delta, rho):   
    res = solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, gamma, delta, rho)
    return res["D"]

  mod = Model(covid_deaths)

  mod.set_param_hint('r0_max',value=3.0,min=2.0,max=5.0)
  mod.set_param_hint('r0_min',value=0.9,min=0.3,max=3.5)
  mod.set_param_hint('k',value=2.5,min=0.01,max=5.0)
  mod.set_param_hint('startLockdown',value=90,min=0,max=days)
  mod.set_param_hint('gamma',value=0.1,min=0.01,max=1.0)
  mod.set_param_hint('delta',value=1.0/3.0,min=0.1,max=1.0)
  mod.set_param_hint('rho',value=0.5,min=0.1,max=1.0)

  params=mod.make_params()

  result = mod.fit(D, params, method="least_squares", t=t)
  print(result.fit_report())
  result.plot_fit(datafmt="-")
  plt.savefig("best_fit.png")
  print("**** Estimated parameters:")
  print(result.best_values)
  

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
  sim.add_argument('--rho', type=float, default=1.0/8.0, help="Inverse of days from critical to death" )  
  sim.add_argument('--lock', type=int, default=0, help="When to start the lockdown" )  
  sim.add_argument('--r0_max', type=float, default=5.0, help="Maximum of the transmission parameter" )  
  sim.add_argument('--r0_min', type=float, default=0.9, help="Minimum of the transmission parameter" )  
  sim.add_argument('-k', type=float, default=2.5, help="Transition parameter of the lockdown")  

  sim.set_defaults(func=doSim)  

  fit = subparsers.add_parser('fit')
  fit.add_argument("--data", type=str, default="dpc-covid19-ita-regioni.csv", help="csv with data for fit")
  fit.add_argument('--shift', type=int, default=0, help="How many days before the outbrek started" )  
  fit.add_argument('-n','--agents', type=int, help="Initial population", required=True ) 
  fit.set_defaults(func=doFit)  

  args = parser.parse_args()  
  args.func(args)  
