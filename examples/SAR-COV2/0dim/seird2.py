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


def plotSEIRHD(tau, sd, sdfname='sir.png', sdstyle='-'):
  # Plot the data on three separate curves for S(t), E(t), I(t) and R(t)
  fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='w')
  gs = GridSpec(2, 2, figure=fig)
  ax1 = fig.add_subplot(gs[0, :])
  ax3 = fig.add_subplot(gs[-1, 0])
  ax4 = fig.add_subplot(gs[-1, -1])

  ax1.plot(tau, sd["S"], sdstyle, color="black", alpha=0.5, lw=2, label='Susceptible')
  ax1.plot(tau, sd["E"], sdstyle, color="yellow", alpha=0.5, lw=2, label='Exposed')
  ax1.plot(tau, sd["I"], sdstyle, color="orange", alpha=0.5, lw=2, label='Infected')
  ax1.plot(tau, sd["R"], sdstyle, color="green", alpha=0.5, lw=2, label='Recovered')
  ax1.plot(tau, sd["H"], sdstyle, color="red", alpha=0.5, lw=2, label='Hospitalized')
  ax1.plot(tau, sd["D"], sdstyle, color="black", alpha=0.5, lw=2, label='Death')
  ax1.plot(tau, sd["S"] + sd["E"] + sd["I"] + sd["R"] + sd["H"] + sd["D"], "--", color="blue", alpha=0.5, lw=2, label='Death')  
  fig.suptitle("SEIRHD model")
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
def deriv(y, t, N, beta, alpha, rates,pHD=0.8):
    S, E, I, R, H, D = y
    dSdt = -rates["rse"]* S/N* beta(t)*I
    dEdt =  rates["rse"]* S/N* beta(t)*I - rates["rei"]*1.0*E
    dIdt =  rates["rei"]*1.0*E  - rates["rir"]*(1 - alpha(t, I, N) )*I  - rates["rih"]*alpha(t, I, N)*I
    dHdt =  rates["rih"]*alpha(t, I, N)*I - rates["rhd"]*pHD*H - rates["rhr"]*(1-pHD)*H
    dRdt =  rates["rih"]*(1 - alpha(t, I, N))*I + rates["rhr"]*(1-pHD)*H
    dDdt =  rates["rhd"]*pHD*H
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt


def solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, rates):
  # Lockdown effect
  def R_0(tau):
    return ( r0_max - r0_min ) / (1 + np.exp(-k*(-tau+startLockdown))) + r0_min
  def beta(tau):
    return R_0(tau)*rates["rir"]

  # Effect of age on death rate
  alpha_age = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.3}
  demographic = {"0-29": 0.2, "30-59": 0.4, "60-89": 0.35, "89+": 0.05}
  alpha_av = sum(alpha_age[i] * demographic[i] for i in list(alpha_age.keys()))
  print("Average alpha:", alpha_av)
  def alpha(tau, I, N):
    #return age_effect*I/N + alpha_av
    return alpha_av

  # Integrate the SEIR equations over the time grid, t
  ret = scint.odeint(deriv, y0, t, args=(N, beta, alpha, rates))
  s, e, i, r, h, d = ret.T

  res = {"S":s, "E":e, "I":i, "R":r, "H":h, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
  res["alpha"] = [alpha(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

def doSim(args):
  print(args)
  # Total population, N.
  N = args.agents
  # Initial number of infected and recovered individuals, I0 and R0.
  E0, I0, R0, H0, D0 = args.e0, args.i0, args.r0, args.h0, 0
  # Everyone else, S0, is susceptible to infection initially.
  S0 = N - I0 - R0 - E0 - H0 - D0
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
  y0 = S0, E0, I0, R0, H0, D0
  print("Average recovery time %.3f days"%(1/args.rates["rir"]))
  print("Average incubation time %.3f days"%(1/args.rates["rei"]))
  print("average survival of criticals %.3f days"%(1/args.rates["rhd"]))
  sir_det = solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, args.rates)

  np.savetxt('seird_results.csv', np.column_stack( 
    (t, sir_det["S"], sir_det["E"], sir_det["I"], sir_det["R"], sir_det["H"], sir_det["D"])
  ), delimiter=', ' )

  plotSEIRHD(t, sir_det, sdfname="seirhd.png")

def doFit(args):
  print(args)
  data = pd.read_csv(args.data)
  regione = 'P.A. Trento'
  r1 = data[data['denominazione_regione']==regione]
  N = args.agents
  I=np.array(r1['totale_positivi'])
  H=np.array(r1['totale_ospedalizzati'])
  D=np.array(r1['deceduti'])
  R=np.array(r1['dimessi_guariti'])

  days = args.shift + len(r1["data"])
  if args.shift:
    I = np.concatenate((np.zeros(args.shift), I))
    R = np.concatenate((np.zeros(args.shift), R))
    H = np.concatenate((np.zeros(args.shift), H))
    D = np.concatenate((np.zeros(args.shift), D))

  S = N-I-D-R-H
  t = np.linspace(0, days-1, days, dtype=int)

  i0=1; e0=0; r0=0; h0=0; s0=N-i0-r0-h0; d0=0
  y0=s0,e0,i0,r0,h0,d0

  age_effect = 1.0

  def covid_deaths(t, r0_max, r0_min, k, startLockdown, rse, rei, rih, rir, rhr, rhd):   
    res = solveSIRdet(y0, t, N, r0_max, r0_min, k, startLockdown, age_effect, {
        "rse":1.0, "rei":1.0/2.0, "rih":1.0/10.0, "rir":1.0/10.0, "rhr":1.0/7.0, "rhd":1.0/8.0       
      })
    return res["D"]

  mod = Model(covid_deaths)

  mod.set_param_hint('r0_max',value=3.0,min=2.0,max=5.0)
  mod.set_param_hint('r0_min',value=0.9,min=0.3,max=3.5)
  mod.set_param_hint('k',value=2.5,min=0.01,max=5.0)
  mod.set_param_hint('startLockdown',value=90,min=0,max=days)
  mod.set_param_hint('rse',value=1.0,min=0.9,max=1.0)
  mod.set_param_hint('rei',value=1.0/5.0,min=0.01,max=1.0)  
  mod.set_param_hint('rir',value=1.0/15.0,min=0.01,max=1.0)
  mod.set_param_hint('rih',value=1.0/10.0,min=0.01,max=1.0)
  mod.set_param_hint('rhd',value=1.0/10.0,min=0.01,max=1.0)
  mod.set_param_hint('rhr',value=1.0/10.0,min=0.01,max=1.0)

  params=mod.make_params()

  result = mod.fit(D, params, method="least_squares", t=t)
  print(result.fit_report())
  result.plot_fit(datafmt="-")
  plt.savefig("best_fit.png")
  print("**** Estimated parameters:")
  print(result.best_values)
  
class StoreDictKeyPair(argparse.Action):
 def __call__(self, parser, namespace, values, option_string=None):
   my_dict = {}
   for kv in values.split(","):
     k,v = kv.split("=")
     my_dict[k] = float(v)
   setattr(namespace, self.dest, my_dict)

if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers()
  sim = subparsers.add_parser('simulate', aliases=['sim'])
  sim.add_argument('-d','--days', type=int, default=150, help="Timesteps to run the model for" )          
  sim.add_argument('-n','--agents', type=int, default=50000000, help="Initial population" )  
  sim.add_argument('--e0', type=int, default=0, help="Initial exposed" )  
  sim.add_argument('--i0', type=int, default=1, help="Initial infected" ) 
  sim.add_argument('--r0', type=int, default=0, help="Initial recovered" )     
  sim.add_argument('--h0', type=int, default=0, help="Initial hospitalized" )    
  sim.add_argument('--lock', type=int, default=0, help="When to start the lockdown" )  
  sim.add_argument('--r0_max', type=float, default=5.0, help="Maximum of the transmission parameter" )  
  sim.add_argument('--r0_min', type=float, default=0.9, help="Minimum of the transmission parameter" )  
  sim.add_argument('-k', type=float, default=2.5, help="Transition parameter of the lockdown")  
  sim.add_argument("--rates", dest="rates", action=StoreDictKeyPair, default={
    "rse":1.0, "rei":1.0/2.0, "rih":1.0/10.0, "rir":1.0/10.0, "rhr":1.0/7.0, "rhd":1.0/8.0 
    }, metavar="rse=V1,rei=V2,...")

  sim.set_defaults(func=doSim)  

  fit = subparsers.add_parser('fit')
  fit.add_argument("--data", type=str, default="dpc-covid19-ita-regioni.csv", help="csv with data for fit")
  fit.add_argument('--shift', type=int, default=0, help="How many days before the outbrek started" )  
  fit.add_argument('-n','--agents', type=int, help="Initial population", required=True ) 
  fit.set_defaults(func=doFit)  

  args = parser.parse_args()  
  args.func(args)  
