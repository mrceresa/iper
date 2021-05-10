# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:16:26 2020

@author: Enrico
"""

from lmfit import minimize,Parameters,fit_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import os
import inspect
import argparse
import time
from utils import StoreDictKeyPair
from covid19dh import covid19

from utils import plotSAIRHD, plotSAIRHDfit
from datetime import date

mainVars={}

#========================================================================================================
#======================================================================================================== 


def deriv(y, t, N, beta,pAI, pIH, rates,pHD):
    
    S, E, A, I, R, Ra, H, D = y
    dSdt = -rates["rse"]* S/N* beta(t,S,N)*(I+A)
   # dSdt = -S*(1-(1- (I+A)/N* beta(t))**rates["rsa"])
    dEdt =  rates["rse"]* S/N* beta(t,S,N)*(I+A) - rates["rea"]*E
    dAdt =  rates["rea"]*E - rates["rai"]*pAI(t,I,N)*A-rates["rar"]*(1-pAI(t,I,N))*A
    dIdt =  rates["rai"]*pAI(t,I,N)*A  - rates["rir"]*(1 - pIH(t, I, N) )*I  - rates["rih"]*pIH(t, I, N)*I
    dHdt =  rates["rih"]*pIH(t, I, N)*I - rates["rhd"]*pHD(t, I, N)*H - rates["rhr"]*(1-pHD(t, I, N))*H    
    dRdt =  rates["rir"]*(1 - pIH(t, I, N))*I + rates["rhr"]*(1-pHD(t, I, N))*H 
    dRadt=  rates["rar"]*(1-pAI(t,I,N))*A
    dDdt =  rates["rhd"]*pHD(t, I, N)*H
    
    return dSdt, dEdt, dAdt, dIdt, dRdt,dRadt ,dHdt, dDdt

def solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown, rates,demographic, icu_beds):
  #======================================================================================================  
#pEI probabilità di passare da EaI, pIH da I a H, pHD da H a D 
  
  #pAI_age   = {"0-29": 0.15, "30-59": 0.25, "60-89": 0.35, "89+": 0.45}  
  pAI_age   = {"0-9": 0.45, "10-19": 0.55, "20-29": 0.73,"30-39": 0.73, "40-49": 0.75, "50-59": 0.85,"60-69": 0.85,"70-79": 0.9,"80-89": 0.95,"90+": 0.95} 
  pAI_av    = sum(pAI_age[i] * demographic[i] for i in list(pAI_age.keys()))
  print("Average pAI:", pAI_av)
  def pAI(tau, I, N):
    return pAI_av     
  
  pIH_age   = {"0-9": 0.06, "10-19": 0.06, "20-29": 0.08,"30-39": 0.09, "40-49": 0.10, "50-59": 0.15,"60-69": 0.18,"70-79": 0.2,"80-89": 0.5,"90+": 0.6}
  pIH_av    = sum(pIH_age[i] * demographic[i] for i in list(pIH_age.keys()))
  print("Average pIH:", pIH_av)
  def pIH(tau, I, N):
    #return age_effect*I/N + pIH_av    
    return pIH_av
    #return 0.10
  pHD_age = {"0-9": 0.03, "10-19": 0.2, "20-29": 0.2,"30-39": 0.2, "40-49": 0.25, "50-59": 0.30,"60-69": 0.3,"70-79": 0.40,"80-89": 0.8,"90+": 0.99}
  pHD_av = sum(pHD_age[i] * demographic[i] for i in list(pHD_age.keys()))
  print("Average pHD:", pHD_av)
  def pHD(tau, I, N):
    return pHD_av
    #return 0.20
    #return age_effect*I/N + pHD_av

#=======================================================================================================  
    
  # Lockdown effect
  def R_0(tau):
    return ( r0_max - r0_min ) / (1 + np.exp(-k*(-tau+startLockdown))) + r0_min
  def beta(tau,S,N):
    #return R_0(tau)*rates["rir"]*N/S
    #return R_0(tau)*rates["rir"]/rates["rse"]
    #return (R_0(tau)/rates["rse"])*((1/rates["rir"])+(1/rates["rai"]))**(-1)*pIH_av 
    return (R_0(tau)/(rates["rse"]))*(N/S)*(pAI_av*(pIH_av*(1/rates["rih"]+1/rates["rai"])+(1-pIH_av)*(1/rates["rir"]+1/rates["rai"]))+(1-pAI_av)*(1/(rates["rar"])))**(-1)
  
#=======================================================================================================    
# Integrate the SEIHR equations over the time grid, t
    
  ret = odeint(deriv, y0, t, args=(N, beta,pAI, pIH, rates,pHD))
  s, e, a, i, r, ra, h, d = ret.T

  res = {"S":s,"E":e, "A":a, "I":i, "R":r, "Ra":ra, "H":h, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
  res["pIH"] = [pIH(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

#========================================================================================================
#========================================================================================================

def doSim(args):
  args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d%H%M"))
  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)    

  print(args)
  # Total population, N.
  N = args.agents
  # Initial number of infected and recovered individuals, I0 and R0.
  A0, I0, R0, H0, D0 = args.a0, args.i0, args.r0, args.h0, 0
  # Everyone else, S0, is susceptible to infection initially.
  S0 = N - I0 - R0 - A0 - H0 - D0
  # A grid of time points (in days)
  t = np.linspace(0, args.days, args.days)

  # Lockdown effect
  r0_max = args.r0_max; r0_min = args.r0_min
  k = args.k # Transition parameter from max to min R0
  # starting day of hard lockdown
  startLockdown = args.lock if args.lock > 0 else args.days 

  # Strenght of the age effect
  age_effect = 1.0
  demographic = {"0-9": 0.084, "10-19": 0.096, "20-29": 0.103,"30-39": 0.117, "40-49": 0.153, "50-59": 0.155,"60-69": 0.122,"70-79": 0.099,"80-89": 0.059,"90+": 0.01}

  # Initial conditions vector
  y0 = S0, A0, I0, R0, H0, D0

  icu_beds = pd.read_csv(args.data_icu, header=0)
  icu_beds = dict(zip(icu_beds["Country"], icu_beds["ICU_Beds"]))
  icu_beds = icu_beds["Italy"] * N / 100000 # Emergency life support beds per 100k citizens
  print("Maximum icy beds", icu_beds)  
  print("Average recovery time %.3f days"%(1/args.rates["rir"]))
  print("average survival of criticals %.3f days"%(1/args.rates["rhd"]))
  sir_det = solveSIRdet(y0, t, N, age_effect, r0_max, r0_min, k, startLockdown, args.rates, demographic, icu_beds)

  fname = os.path.join(args.output_dir, 'seird_results.csv')
  np.savetxt(fname, np.column_stack( 
    (t, sir_det["S"], sir_det["A"], sir_det["I"], sir_det["R"], sir_det["H"], sir_det["D"])
  ), delimiter=', ' )

  plotSAIRHD(t, sir_det, sdfname=os.path.join(args.output_dir,"sairhd.png"))


def doFit(args):
  args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d%H%M"))
  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)    
  global mainVars
  
  #if args.country=="ESP":
    #df=pd.read_csv('/home/enrico/iper-social-simulations/examples/SAR-COV2/0dim/data/dfESP.csv')
  #else:
  df, _ = covid19(args.country)
  #df, _ = covid19(args.country)
  #import ipdb; ipdb.set_trace()
  df = df[df["date"]>args.start]
  df = df[df["date"]<args.stop]
  df = df[df["confirmed"]>0]#Starts from first infections
  #dati_regione1=dati_regione_[dati_regione_['totale_positivi']>25] #per stabilire il t0

  N=df["population"].mean()
  icu_beds = pd.read_csv(args.data_icu, header=0)
  icu_beds = dict(zip(icu_beds["Country"], icu_beds["ICU_Beds"]))
  icu_beds = icu_beds["Italy"] * N / 100000 # Emergency life support beds per 100k citizens
  print("Maximum icy beds", icu_beds)
  #dati_regione1=pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv')
  #regione='ITA' 
  
  age_effect=1
  demographic = {"0-9": 0.084, "10-19": 0.096, "20-29": 0.103,"30-39": 0.117, "40-49": 0.153, "50-59": 0.155,"60-69": 0.122,"70-79": 0.099,"80-89": 0.059,"90+": 0.01}

  
  #I=np.array(dati_regione1['totale_positivi'])
  #D=np.array(dati_regione1['deceduti'])
  #R=np.array(dati_regione1['dimessi_guariti'])
  #H=np.array(dati_regione1['totale_ospedalizzati'])
  #S=N-I-D-H-R

  df.plot(x="date", y=["tests"])
  plt.savefig(os.path.join(args.output_dir, "tests-%s.png"%args.country))

  df.plot(x="date", y=["confirmed","recovered", "deaths"])
  plt.savefig(os.path.join(args.output_dir, "confirmed-recovered-%s.png"%args.country))

  df.plot(x="date", y=["deaths","hosp","vent","icu"])        
  plt.savefig(os.path.join(args.output_dir, "severity-%s.png"%args.country))

  print("Maximum people in vent", df["vent"].max())
  print("Maximum people in icu", df["icu"].max())

  #x1=df.copy(deep=True)
  #Iday=x1.iloc[:,3]
  #Iday.fillna(0,inplace=True,axis=0)
  #Iday=Iday.values

  #for i in range(len(Iday)-1,1,-1): 
    #Iday[i]=Iday[i]-Iday[i-1]

  #if args.country=="ESP":
   # df1=pd.read_csv('https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/nacional_covid19.csv')
   # df1 = df1[df1["fecha"]>args.start]
   # df1 = df1[df1["fecha"]<args.stop]
   # x1=df1.copy(deep=True)
   # Hday=x1.iloc[:,3]
   # Hday.fillna(0,inplace=True,axis=0)
   # Hday=Hday.values

  if args.country=='ESP':
    I = df["confirmed"]# in questo caso rappresenta il totale degli individui che si sono infettati
    R = df["recovered"]# totale recovered
    D = df["deaths"]   # Totale morti
    #H = df["hosp"]
    S = N - I 

    t=np.array([i+1 for i in range(len(I))])
    i0=df["confirmed"].iloc[0]  #  35708    #infetti a t0 
    e0=0
    a0=1.0*i0
    r0=0
    ra0=0.2*r0
    h0=0                                           
    d0=df["deaths"].iloc[0] # 35587      #deceduti  t0
    s0=N-i0#-a0                   #suscettibili a t0
    y0=s0,e0,a0,i0,r0,ra0,h0,d0 



  else:
    I = df["confirmed"]-df["recovered"]-df["hosp"]-df["deaths"]
    R = df["recovered"]
    D = df["deaths"]
    H = df["hosp"]
    S = N - df["confirmed"]

    t=np.array([i+1 for i in range(len(I))])
    i0=df["confirmed"].iloc[0]- df["recovered"].iloc[0]-df["deaths"].iloc[0]  #  35708    #infetti a t0
    e0=0 
    a0=1.0*i0                            #asintomatici a t0
    r0=df["recovered"].iloc[0] # 211885      #recovered a t0
    ra0=0.2*r0
    h0=df["hosp"].iloc[0]   # 2000    #ospedaliz. a t0                         
    d0=df["deaths"].iloc[0] # 35587      #deceduti  t0
    s0=N-df["confirmed"].iloc[0]#-i0-r0 -h0 -d0  -a0 -ra0                  #suscettibili a t0
    y0=s0,e0,a0,i0,r0,ra0,h0,d0 
  
  def resid_ESP(params,y0, t, N,age_effect,S,I,D,icu_beds):
      print(params)
      rates={"rse":0,"rea":0 ,"rai":0, "rih":0, "rir":0, "rhr":0, "rhd":0 }
         
      r0_max = params['r0_max']
      r0_min = params['r0_min']
      k = params['k']
      startLockdown = params['startLockdown']
      seihrd_det=solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,  params, demographic, icu_beds)
      #return sum((seihrd_det['S']-S)**2+(seihrd_det['I']-I)**2+(seihrd_det['H']-H)**2+(seihrd_det['R']-R)**2+(seihrd_det['D']-D)**2)
      #er=np.array((((seihrd_det['S']-S)**2),((seihrd_det['I']-I)**2),((seihrd_det['H']-H)**2),((seihrd_det['R']-R)**2),((seihrd_det['D']-D)**2)))
      a=np.array((S-(seihrd_det['S']+seihrd_det['A']+seihrd_det['E']+seihrd_det['Ra']))**2) 
      #a=np.array((S-(seihrd_det['S']+seihrd_det['A']+seihrd_det['E']))**2)

      #b=np.array((I-(seihrd_det['I']+seihrd_det['H']+seihrd_det['D']))**2)
      b=np.array((I-(seihrd_det['I']+seihrd_det['R']+seihrd_det['H']+seihrd_det['D']))**2)
       #b=np.array((Iday-seihrd_det['I'])**2)
       #c=np.array((H-seihrd_det['H'])**2)
      #d=np.array((R-seihrd_det['R'])**2)
      e=np.array((1.2*D-1.2*seihrd_det['D'])**2)

      tot=np.concatenate((b,e,a))

      
      #return er
      #er=er.flatten()
      #er2= np.array((seihrd_det['S']-S)**2,(seihrd_det['I']-I)**2,(seihrd_det['H']-H)**2,(seihrd_det['R']-R)**2,(seihrd_det['D']-D)**2)
      return  tot

  def resid_ALL(params,y0, t, N,age_effect,S,I,H,R,D,icu_beds):
      print(params)
      rates={"rse":0,"rea":0 , "rai":0, "rih":0, "rir":0, "rhr":0, "rhd":0 }
         
      r0_max = params['r0_max']
      r0_min = params['r0_min']
      k = params['k']
      startLockdown = params['startLockdown']
      seihrd_det=solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,  params, demographic, icu_beds)
      #return sum((seihrd_det['S']-S)**2+(seihrd_det['I']-I)**2+(seihrd_det['H']-H)**2+(seihrd_det['R']-R)**2+(seihrd_det['D']-D)**2)
      #er=np.array((((seihrd_det['S']-S)**2),((seihrd_det['I']-I)**2),((seihrd_det['H']-H)**2),((seihrd_det['R']-R)**2),((seihrd_det['D']-D)**2)))
      #a=np.array(S-(seihrd_det['S'])**2)
      a=np.array((S-(seihrd_det['S']+seihrd_det['A']+seihrd_det['E']+seihrd_det['Ra']))**2) 
      b=np.array((I-seihrd_det['I'])**2)
      c=np.array((H-seihrd_det['H'])**2)
      d=np.array((R-seihrd_det['R'])**2)
      e=np.array(1.44*(D-seihrd_det['D'])**2)#mettere fuori 1.2

      tot=np.concatenate((a,b,e,c,d))
   
      #return er
      #er=er.flatten()
      #er2= np.array((seihrd_det['S']-S)**2,(seihrd_det['I']-I)**2,(seihrd_det['H']-H)**2,(seihrd_det['R']-R)**2,(seihrd_det['D']-D)**2)
      return  tot

  if args.country=='ESP':

    params2=Parameters()
    params2.add('r0_max',value=5.0,min=0.1,max=15.0,vary=True)
    params2.add('r0_min',value=0.9,min=0.1,max=3.5,vary=True)
    params2.add('k',value=2.5,min=0.01,max=5.0,vary=True)
    params2.add('startLockdown',value=31.0, min=0,max=100, vary=True)
    params2.add('rse',value=1.0, min=0.01 ,max=2.0, vary=False)
    params2.add('rea',value=0.2, min=0.01 ,max=2.0, vary=False)
    params2.add('rai',value=0.5,min=0.01,max=2,vary=True)  
    params2.add('rar',value=0.03,min=0.01,max=2,vary=True)  
    params2.add('rih',value=0.2,min=0.01,max=2.0,vary=True)
    params2.add('rir',value=0.03,min=0.01,max=2,vary=True)
    params2.add('rhr',value=0.26,min=0.01,max=2,vary=True)#0.0743636663768294
    params2.add('rhd',value=0.125,min=0.01,max=1,vary=True)#

    out = minimize(resid_ESP, params2, args=(y0, t, N,age_effect,S,I,D,icu_beds),
     method='differential_evolution')#method='differential_evolution',method='leastsq'

  else:

    params2=Parameters()
    params2.add('r0_max',value=5.0,min=0.1,max=15.0,vary=True)
    params2.add('r0_min',value=0.9,min=0.1,max=15,vary=True)
    params2.add('k',value=2.5,min=0.01,max=5.0,vary=True)
    params2.add('startLockdown',value=31.0, min=0,max=100, vary=True)
    params2.add('rse',value=1, min=0.01 ,max=10.0, vary=False)
    params2.add('rea',value=0.2, min=0.01 ,max=2.0, vary=False)
    params2.add('rai',value=0.5,min=0.01,max=2,vary=True)  
    params2.add('rar',value=0.1,min=0.01,max=2,vary=True)  
    params2.add('rih',value=0.1,min=0.01,max=2.0,vary=True)
    params2.add('rir',value=0.1,min=0.01,max=2,vary=True)
    params2.add('rhr',value=0.142,min=0.01,max=2,vary=True)
    params2.add('rhd',value=0.125,min=0.01,max=2,vary=True)

    
   
  
  #res3=resid_(params2,y0, t, N,age_effect,S,I,H,R,D,icu_beds)
    out = minimize(resid_ALL, params2, args=(y0, t, N,age_effect,S,I,H,R,D,icu_beds),
     method='differential_evolution')#method='differential_evolution',method='leastsq'

  print(fit_report(out))


  print(out.params.pretty_print())
  #out.plot_fit(datafmt="-")
  #plt.savefig(os.path.join(args.output_dir, "best_fit.png"))
  #import ipdb; ipdb.set_trace()

  print("**** Estimated parameters:")
  print([out.params.items()])
  
  rates_fit={"rse":out.params['rse'].value, 
             "rea":out.params['rea'].value,
             "rai":out.params['rai'].value, 
             "rar":out.params['rar'].value, 
             "rih":out.params['rih'].value, 
             "rir":out.params['rir'].value, 
             "rhr":out.params['rhr'].value, 
             "rhd":out.params['rhd'].value }
  
  seihrd_det_fit=solveSIRdet(y0, t, N,age_effect, out.params['r0_max'].value, out.params['r0_min'].value, out.params['k'].value, out.params['startLockdown'].value,  rates_fit,demographic,icu_beds)
  
  
  
  plt.figure(figsize=(15, 12))
  plt.suptitle(args.country)

  if args.country=='ESP':
    I_tot=seihrd_det_fit['I']+seihrd_det_fit['R']+seihrd_det_fit['D']+seihrd_det_fit['H']
    ax3=plt.subplot(331)
    ax3.plot(t,I_tot,color="r")
    ax3.plot(t,I,label='Infected-cumulative',color="r",ls="--")
    plt.xlabel('day')
    plt.ylabel('Infected')
  
    ax4=plt.subplot(332)        
    ax4.plot(t,seihrd_det_fit['D'],color="black")
    ax4.plot(t,D,label='Dead',color="black",ls="--")
    plt.xlabel('day')
    plt.ylabel('Dead')
  
    ax5=plt.subplot(333)
    ax5.plot(t,R,label='Recovered',color="g",ls="--")        
    ax5.plot(t,seihrd_det_fit['R'],color="g")
    plt.xlabel('day')
    plt.ylabel('Recovered-seihrd')
  
    #ax6=plt.subplot(334)        
    #ax6.plot(t,seihrd_det_fit['S'],color="y")
    #ax6.plot(t,S,label='Susceptible',color="y",ls="--")
    #plt.xlabel('day')
    #plt.ylabel('Susceptible')
  
    ax7=plt.subplot(335)        
    ax7.plot(t,seihrd_det_fit['H'],color="orange")
    plt.xlabel('day')
    plt.ylabel('Hosp-seihrd')
  
  
    ax8=plt.subplot(336)        
    #ax8.plot(t,I_tot,color="r")
    ax8.plot(t,seihrd_det_fit['I'],color="r")
    ax8.plot(t,I-seihrd_det_fit['R']-D-seihrd_det_fit['H'],label='Infected-cumulative',color="r",ls="--")
    #ax8.plot(t,seihrd_det_fit['R'],color="g")
    #ax8.plot(t,R,label='Recovered',color="g",ls="--")
    ax8.plot(t,seihrd_det_fit['D'],color="black")
    ax8.plot(t,D,label='Dead',color="black",ls="--")
    #ax8.plot(t,seihrd_det_fit['H'],color="orange")
    plt.xlabel('day')
    plt.ylabel('people')
  
  else:
    ax3=plt.subplot(331)
    ax3.plot(t,seihrd_det_fit['I'],color="r")
    ax3.plot(t,I,label='Infected',color="r",ls="--")
    plt.xlabel('day')
    plt.ylabel('Infected')
  
    ax4=plt.subplot(332)        
    ax4.plot(t,seihrd_det_fit['D'],color="black")
    ax4.plot(t,D,label='Dead',color="black",ls="--")
    plt.xlabel('day')
    plt.ylabel('Dead')
  
    ax5=plt.subplot(333)        
    ax5.plot(t,seihrd_det_fit['R'],color="g")
    ax5.plot(t,R,label='Recovered',color="g",ls="--")
    plt.xlabel('day')
    plt.ylabel('Recovered')
  
    ax6=plt.subplot(334)
    ax6.plot(t,(seihrd_det_fit['S']),color="y")        
    #ax6.plot(t,((seihrd_det_fit['S']+seihrd_det_fit['A']+seihrd_det_fit['E']+seihrd_det_fit['Ra'])),color="y")
    ax6.plot(t,S,label='Susceptible',color="y",ls="--")
    plt.xlabel('day')
    plt.ylabel('Susceptible')
  
    ax7=plt.subplot(335)        
    ax7.plot(t,seihrd_det_fit['H'],color="orange")
    ax7.plot(t,H,label='Hosp',color="orange",ls="--")
    plt.xlabel('day')
    plt.ylabel('Hosp.')
  
  
    ax8=plt.subplot(336)        
    ax8.plot(t,seihrd_det_fit['I'],color="r")
    ax8.plot(t,I,label='Infected',color="r",ls="--")
    ax8.plot(t,seihrd_det_fit['R'],color="g")
    ax8.plot(t,R,label='Recovered',color="g",ls="--")
    ax8.plot(t,seihrd_det_fit['D'],color="black")
    ax8.plot(t,D,label='Dead',color="black",ls="--")
    ax8.plot(t,seihrd_det_fit['H'],color="orange")
    ax8.plot(t,H,label='Hosp',color="orange",ls="--")
    plt.xlabel('day')
    plt.ylabel('people')
  
  #df_fitSAIHRD=pd.read_csv(os.path.join('fit','df_fitSAIHRD.csv'))
  
  risult=np.array([[args.country,
                    out.params['rse'].value,
                    out.params['rai'].value,
                    out.params['rih'].value,
                    out.params['rir'].value,
                    out.params['rhr'].value,
                    out.params['rhd'].value,
                    out.params['r0_max'].value,
                    out.params['r0_min'].value,
                    out.params['k'].value,
                    out.params['startLockdown'].value]])
  
  
  df_fitSAIHRD = pd.DataFrame(risult,columns=["regione", "rse", "rai","rih","rir","rhr","rhd","r0_max","r0_min","k","startLockdown"])
  fname = "df_fitSAIHRD.csv"
  df_fitSAIHRD.to_csv(os.path.join(args.output_dir,fname),index=False)
  saveFIG=args.country+'.png'
  plt.savefig(os.path.join(args.output_dir, saveFIG))
  
  mainVars=inspect.currentframe().f_locals

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers()
  sim = subparsers.add_parser('simulate', aliases=['sim'])
  sim.add_argument('-d','--days', type=int, default=150, help="Timesteps to run the model for" )          
  sim.add_argument('-n','--agents', type=int, default=50000000, help="Initial population" )  
  sim.add_argument('--a0', type=int, default=0, help="Initial exposed" )  
  sim.add_argument('--i0', type=int, default=1, help="Initial infected" ) 
  sim.add_argument('--r0', type=int, default=0, help="Initial recovered" )     
  sim.add_argument('--h0', type=int, default=0, help="Initial hospitalized" )    
  sim.add_argument('--lock', type=int, default=0, help="When to start the lockdown" )  
  sim.add_argument('--r0_max', type=float, default=5.0, help="Maximum of the transmission parameter" )  
  sim.add_argument('--r0_min', type=float, default=0.9, help="Minimum of the transmission parameter" )  
  sim.add_argument('-k', type=float, default=2.5, help="Transition parameter of the lockdown")  
  sim.add_argument("--rates", dest="rates", action=StoreDictKeyPair, default={
    "rsa":1.0, "rai":1.0/2.0, "rar":1.0/10.0, "rih":1.0/10.0, "rir":1.0/10.0, "rhr":1.0/7.0, "rhd":1.0/8.0 
    }, metavar="rse=V1,rei=V2,...")
  sim.add_argument('--output_dir', type=str, default=os.path.join("results","seihrd"), help="Output directory" )        
  sim.add_argument("--data-icu", type=str, default=os.path.join('data','beds.csv'), help="csv with data for fit")  
  sim.set_defaults(func=doSim)  

  fit = subparsers.add_parser('fit')
  fit.add_argument("--country", type=str, default="ITA", help="Country code to fit")
  fit.add_argument("--data-icu", type=str, default=os.path.join('data','beds.csv'), help="csv with data for fit")
  fit.add_argument('--shift', type=int, default=0, help="How many days before the outbrek started" )  
  fit.add_argument('--output_dir', type=str, default=os.path.join("results","seihrd"), help="Output directory" ) 
  fit.add_argument('--start', type=str, default="2020-03-04", help="day start fit")
  fit.add_argument('--stop', type=str, default="2020-06-01", help="day stop fit")      
  fit.set_defaults(func=doFit)  

  args = parser.parse_args()
  args.func(args)



