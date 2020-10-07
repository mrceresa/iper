# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:16:26 2020

@author: Enrico
"""

from lmfit import minimize,Parameters
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import os

#========================================================================================================
#======================================================================================================== 


def deriv(y, t, N, beta,pAI, pIH, pHC, rates,pCD):
    
    S, A, I, R1, R2, H, C, D = y
    dSdt = -rates["rsa"]* S/N* beta(t)*(I+A)
    dAdt =  rates["rsa"]* S/N* beta(t)*(I+A) - rates["rai"]*pAI*A-rates["rar"]*(1-pAI)*A
    dIdt =  rates["rai"]*pAI*A  - rates["rir"]*(1 - pIH )*I  - rates["rih"]*pIH*I
    dHdt =  rates["rih"]*pIH*I - rates["rhc"]*pHC*H - rates["rhr"]*(1-pHC)*H
    dCdt =  rates["rhc"]*pHC*H - rates["rcd"]*pCD*C - rates["rcr"]*(1-pCD)*C
    dR1dt=  rates["rar"]*(1 - pAI)*A
    dR2dt=  rates["rir"]*(1 - pIH)*I + rates["rhr"]*(1-pHC)*H + rates["rcr"]*(1-pCD)*C    
    dDdt =  rates["rcd"]*pCD*C
    
    return dSdt, dAdt, dIdt, dR1dt, dR2dt, dHdt, dCdt, dDdt

def solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,pAI,pIH,pHC,pCD, rates,demographic, icu_beds):
    
  # Lockdown effect
  def R_0(tau):
    return ( r0_max - r0_min ) / (1 + np.exp(-k*(-tau+startLockdown))) + r0_min
  def beta(tau):
    return R_0(tau)*rates["rir"]  
  
#======================================================================================================  
#pEI probabilità di passare da EaI, pIH da I a H, pHD da H a D 
  
#  pAI_age   = {"0-29": 0.15, "30-59": 0.25, "60-89": 0.35, "89+": 0.45}   
#  pAI_av    = sum(pAI_age[i] * demographic[i] for i in list(pAI_age.keys()))
#  print("Average pAI:", pAI_av)
#  def pAI(tau, I, N):
#    return pAI_av     
#  
#  pIH_age   = {"0-29": 0.1, "30-59": 0.2, "60-89": 0.3, "89+": 0.4}
#  pIH_av    = sum(pIH_age[i] * demographic[i] for i in list(pIH_age.keys()))
#  print("Average pIH:", pIH_av)
#  def pIH(tau, I, N):
#    return pIH_av # + age_effect*I/N +  
#
#  pHC_age   = {"0-29": 0.05, "30-59": 0.10, "60-89": 0.5, "89+": 0.8}   
#  pHC_av    = sum(pHC_age[i] * demographic[i] for i in list(pHC_age.keys()))
#  print("Average pHC:", pHC_av)
#  def pHC(tau, I, N):
#    return pHC_av 
#  
#  pCD_age = {"0-29": 0.15, "30-59": 0.25, "60-89": 0.35, "89+": 0.45}
#  pCD_av = sum(pCD_age[i] * demographic[i] for i in list(pCD_age.keys()))
#  print("Average pCD:", pCD_av)
#  def pCD(tau, I, N):
#    return  pCD_av#+age_effect*I/N +

#=======================================================================================================    
# Integrate the SEIHR equations over the time grid, t
    
  ret = odeint(deriv, y0, t, args=(N, beta,pAI, pIH, pHC, rates, pCD))
  s, a, i, h, r1, r2, c, d = ret.T

  res = {"S":s, "A":a, "I":i, "H":h, "R1":r1, "R2":r2, "C":c, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
 # res["pIH"] = [pIH(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

#========================================================================================================
#========================================================================================================
  
#def main():
    
    
    
dati_regione=pd.read_csv('dpc-covid19-ita-regioni.csv')
regione='P.A. Trento'
TRUEregione=dati_regione['denominazione_regione']==regione
dati_regione_=dati_regione[TRUEregione]
dati_regione1=dati_regione_[dati_regione_['totale_positivi']>25] #per stabilire il t0

N=500000
icu_beds = pd.read_csv("data/beds.csv", header=0)
icu_beds = dict(zip(icu_beds["Country"], icu_beds["ICU_Beds"]))
icu_beds = icu_beds["Italy"] * N / 100000 # Emergency life support beds per 100k citizens
print(icu_beds)
#dati_regione1=pd.read_csv('dpc-covid19-ita-andamento-nazionale.csv')
#regione='ITA' 

age_effect=1
demographic = {"0-29": 0.2, "30-59": 0.4, "60-89": 0.35, "89+": 0.05}
delay=0

I=np.array(dati_regione1['totale_positivi'])
D=np.array(dati_regione1['deceduti'])
R2=np.array(dati_regione1['dimessi_guariti'])
H=np.array(dati_regione1['ricoverati_con_sintomi'])
C=np.array(dati_regione1['terapia_intensiva'])
S=N-I-D-H-C-R2

t_=np.array([i+1 for i in range(len(I))])
delay_=np.array([delay for i in range(len(I))])
t=t_+delay

i0=dati_regione1.iloc[0,10]        #infetti a t0 
a0=4*i0
r10=0                            #asintomatici a t0
r20=dati_regione1.iloc[0,12]        #recovered a t0
h0=dati_regione1.iloc[0,6]         #ospedaliz con sintomi. a t0 
c0=dati_regione1.iloc[0,7]                       
d0=dati_regione1.iloc[0,13]        #deceduti  t0
s0=N-i0-r20 -h0 -c0 -d0  -a0                   #suscettibili a t0
#============================================================================ 
#per l' Italia
#    i0=dati_regione1.iloc[0,6]        #infetti a t0 
#    e0=4*i0                            #asintomatici a t0
#    r0=dati_regione1.iloc[0,9]        #recovered a t0
#    h0=dati_regione1.iloc[0,4]         #ospedaliz. a t0                     
#    d0=dati_regione1.iloc[0,10]        #deceduti  t0
#    s0=N-i0-r0 -h0  -d0                   #suscettibili a t0
#     
#============================================================================    
y0=s0,a0,i0,r10,r20,h0,c0,d0 


def resid_(params,y0, t, N,age_effect,S,I,H,R,C,D,icu_beds):
    print(params)
    rates={"rsa":0, "rai":0,"rar":0, "rih":0, "rir":0, "rhr":0,"rhc":0,"rcr":0, "rcd":0 }
       
    r0_max = params['r0_max']
    r0_min = params['r0_min']
    k = params['k']
    startLockdown = params['startLockdown']
    pAI=params['pAI']
    pIH=params['pIH']
    pHC=params['pHC'] 
    pCD=params['pCD']
    rates['rsa']=params['rsa']
    rates['rai']=params['rai']
    rates['rar']=params['rar']
    rates['rih']=params['rih']
    rates['rir']=params['rir'] 
    rates['rhr']=params['rhr']
    rates['rhc']=params['rhc']
    rates['rcr']=params['rcr']
    rates['rcd']=params['rcd']
    
    seihrd_det=solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,pAI,pIH,pHC,pCD,  rates,demographic, icu_beds)
    #return sum((seihrd_det['S']-S)**2+(seihrd_det['I']-I)**2+(seihrd_det['H']-H)**2+(seihrd_det['R']-R)**2+(seihrd_det['D']-D)**2)
    #er=np.array((((seihrd_det['S']-S)**2),((seihrd_det['I']-I)**2),((seihrd_det['H']-H)**2),((seihrd_det['R']-R)**2),((seihrd_det['D']-D)**2)))
    
    #a=np.array((S-seihrd_det['S'])**2)
    b=np.array((I-seihrd_det['I'])**2)
    c=np.array((H-seihrd_det['H'])**2)
    d=np.array((R-seihrd_det['R2'])**2)
    e=np.array((1.2*D-1.2*seihrd_det['D'])**2)
    f=np.array((C-seihrd_det['C'])**2)
    
    tot=np.concatenate((b,c,d,e,f))
    
    #return er
    #er=er.flatten()
    #er2= np.array((seihrd_det['S']-S)**2,(seihrd_det['I']-I)**2,(seihrd_det['H']-H)**2,(seihrd_det['R']-R)**2,(seihrd_det['D']-D)**2)
    return  tot


params2=Parameters()
params2.add('r0_max',value=3.8,vary=True,min=2.0,max=5.0)
params2.add('r0_min',value=1,vary=True,min=0.3,max=3)
params2.add('k',value=4,vary=True,min=0.01,max=5.0)
params2.add('startLockdown',value=31.673,vary=True,min=0,max=100)
params2.add('pAI',value=0.1,vary=True,min=0.01,max=0.99)
params2.add('pIH',value=0.1,vary=True,min=0.01,max=0.99)
params2.add('pHC',value=0.1,vary=True,min=0.01,max=0.99)
params2.add('pCD',value=0.1,vary=True,min=0.01,max=0.99)
params2.add('rsa',value=1.49,vary=True,min=0.01,max=2)
params2.add('rai',value=1.2,vary=True,min=0.01,max=2)  
params2.add('rar',value=0.25,vary=True,min=0.01,max=2.0)
params2.add('rih',value=1/12,vary=False,min=0.01,max=2.0)
params2.add('rir',value=0.019,vary=True,min=0.01,max=2)
params2.add('rhr',value=0.2,vary=True,min=0.01,max=2)
params2.add('rhc',value=0.1,vary=True,min=0.01,max=2)
params2.add('rcr',value=1/6.5,vary=False,min=0.01,max=2.0)
params2.add('rcd',value=1/7.5,vary=False,min=0.01,max=2.0)


res3=resid_(params2,y0, t, N,age_effect,S,I,H,R2,C,D,icu_beds)
out = minimize(resid_, params2, args=(y0, t, N,age_effect,S,I,H,R2,C,D,icu_beds),method='differential_evolution',verbose=False)#method='differential_evolution',method='leastsq'


rates_fit={"rsa":out.params['rsa'].value, "rai":out.params['rai'].value, "rar":out.params['rar'].value, "rih":out.params['rih'].value, "rir":out.params['rir'].value, "rhr":out.params['rhr'].value, "rhc":out.params['rhc'].value, "rcr":out.params['rcr'].value, "rcd":out.params['rcd'].value}
prob_fit={"pAI":out.params['pAI'].value,"pIH":out.params['pIH'].value,"pHC":out.params['pHC'].value,"pCD":out.params['pCD'].value }
seihrd_det_fit=solveSIRdet(y0, t, N,age_effect, out.params['r0_max'].value, out.params['r0_min'].value, out.params['k'].value, out.params['startLockdown'].value,out.params['pAI'].value,out.params['pIH'].value,out.params['pHC'].value,out.params['pCD'].value,  rates_fit,demographic,icu_beds)

S_=seihrd_det_fit['S']+seihrd_det_fit['A']+seihrd_det_fit['R1']

plt.figure(figsize=(15, 12))
plt.suptitle(regione)

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
ax5.plot(t,seihrd_det_fit['R2'],color="g")
ax5.plot(t,R2,label='Recovered',color="g",ls="--")
plt.xlabel('day')
plt.ylabel('Recovered')

ax6=plt.subplot(334)        
ax6.plot(t,S_,color="y")#seihrd_det_fit['S']
ax6.plot(t,S,label='Susceptible',color="y",ls="--")
plt.xlabel('day')
plt.ylabel('Susceptible')

ax7=plt.subplot(335)        
ax7.plot(t,seihrd_det_fit['H'],color="orange")
ax7.plot(t,H,label='Hosp',color="orange",ls="--")
plt.xlabel('day')
plt.ylabel('Hosp.')


#ax8=plt.subplot(336)        
#ax8.plot(t,seihrd_det_fit['I'],color="r")
#ax8.plot(t,I,label='Infected',color="r",ls="--")
#ax8.plot(t,seihrd_det_fit['R2'],color="g")
#ax8.plot(t,R2,label='Recovered',color="g",ls="--")
#ax8.plot(t,seihrd_det_fit['D'],color="black")
#ax8.plot(t,D,label='Dead',color="black",ls="--")
#ax8.plot(t,seihrd_det_fit['H'],color="orange")
#ax8.plot(t,H,label='Hosp',color="orange",ls="--")
#plt.xlabel('day')
#plt.ylabel('people')

ax8=plt.subplot(336)        
ax8.plot(t,seihrd_det_fit['C'],color="orange")
ax8.plot(t,C,label='Crit.',color="orange",ls="--")
plt.xlabel('day')
plt.ylabel('Critic.')

#    
#    
#    
#    df_fitSEIHRD=pd.read_csv('df_fitSAIHRD.csv')
#    
risult=np.array([[regione,
                  out.params['rsa'].value,
                  out.params['rai'].value,
                  out.params['rar'].value,
                  out.params['rih'].value,
                  out.params['rir'].value,
                  out.params['rhc'].value,
                  out.params['rhr'].value,
                  out.params['rcd'].value,
                  out.params['rcr'].value,
                  out.params['r0_max'].value,
                  out.params['r0_min'].value,
                  out.params['k'].value,
                  out.params['startLockdown'].value,
                  out.params['pAI'].value,
                  out.params['pIH'].value,
                  out.params['pHC'].value,
                  out.params['pCD'].value,]])


df_fitSEIHRD = pd.DataFrame(risult,columns=["regione", "rsa", "rai","rar","rih","rir","rhc","rhr","rcd","rcr","r0_max","r0_min","k","startLockdown","pAI","pIH","pHC","pCD"])#.append(df_fitSEIHRD, ignore_index=True)
fname = "df_fitSAIHRD.csv"
df_fitSEIHRD.to_csv(fname,index=False)
saveFIG=regione+'.png'
plt.savefig(saveFIG)

#if __name__ == "__main__":
#    main()


