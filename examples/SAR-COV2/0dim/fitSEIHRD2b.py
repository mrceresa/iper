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



dati_regione=pd.read_csv('dpc-covid19-ita-regioni.csv')
regione='P.A. Trento'
TRUEregione=dati_regione['denominazione_regione']==regione
dati_regione_=dati_regione[TRUEregione]
dati_regione1=dati_regione_[dati_regione_['totale_positivi']>25] #per stabilire il t0
 

N=500000
age_effect=1
demographic = {"0-29": 0.2, "30-59": 0.4, "60-89": 0.35, "89+": 0.05}

I=np.array(dati_regione1['totale_positivi'])
D=np.array(dati_regione1['deceduti'])
R=np.array(dati_regione1['dimessi_guariti'])
H=np.array(dati_regione1['totale_ospedalizzati'])
S=N-I-D-H-R

t=np.array([i+1 for i in range(len(I))])

i0=dati_regione1.iloc[0,10]        #infetti a t0 
e0=4*i0                            #asintomatici a t0
r0=dati_regione1.iloc[0,12]        #recovered a t0
h0=dati_regione1.iloc[0,8]         #ospedaliz. a t0                     
s0=N-i0-r0 -h0                     #suscettibili a t0
d0=dati_regione1.iloc[0,13]        #deceduti  t0

y0=s0,e0,i0,r0,h0,d0



#========================================================================================================
#======================================================================================================== 


def deriv(y, t, N, beta,pEI, pIH, rates,pHD):
    
    S, E, I, R, H, D = y
    dSdt = -rates["rse"]* S/N* beta(t)*(I+E)
    dEdt =  rates["rse"]* S/N* beta(t)*(I+E) - rates["rei"]*pEI(t,I,N)*E-rates["rir"]*(1-pEI(t,I,N))*E
    dIdt =  rates["rei"]*pEI(t,I,N)*E  - rates["rir"]*(1 - pIH(t, I, N) )*I  - rates["rih"]*pIH(t, I, N)*I
    dHdt =  rates["rih"]*pIH(t, I, N)*I - rates["rhd"]*pHD(t, I, N)*H - rates["rhr"]*(1-pHD(t, I, N))*H
    dRdt =  rates["rir"]*(1 - pIH(t, I, N))*I + rates["rhr"]*(1-pHD(t, I, N))*H + rates["rir"]*(1-pEI(t,I,N))*E
    dDdt =  rates["rhd"]*pHD(t, I, N)*H
    
    return dSdt, dEdt, dIdt, dRdt, dHdt, dDdt

def solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown, rates,demographic):
    
  # Lockdown effect
  def R_0(tau):
    return ( r0_max - r0_min ) / (1 + np.exp(-k*(-tau+startLockdown))) + r0_min
  def beta(tau):
    return R_0(tau)*rates["rir"]


  
  
#======================================================================================================  
#pEI probabilità di passare da EaI, pIH da I a H, pHD da H a D 
  
  pEI_age   = {"0-29": 0.15, "30-59": 0.25, "60-89": 0.35, "89+": 0.45}   
  pEI_av    = sum(pEI_age[i] * demographic[i] for i in list(pEI_age.keys()))
  print("Average pEI:", pEI_av)
  def pEI(tau, I, N):
    return pEI_av    
  
  #pIH_age   = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.3}
  #pIH_age   = {"0-29": 0.05, "30-59": 0.08, "60-89": 0.2, "89+": 0.3} 
  pIH_age   = {"0-29": 0.1, "30-59": 0.2, "60-89": 0.3, "89+": 0.4}
  pIH_av    = sum(pIH_age[i] * demographic[i] for i in list(pIH_age.keys()))
  print("Average pIH:", pIH_av)
  def pIH(tau, I, N):
    return age_effect*I/N + pIH_av    
  
  pHD_age = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.4}  
  pHD_av = sum(pHD_age[i] * demographic[i] for i in list(pHD_age.keys()))
  print("Average pHD:", pHD_av)
  def pHD(tau, I, N):
    return age_effect*I/N + pHD_av

#=======================================================================================================    
# Integrate the SEIHR equations over the time grid, t
    
  ret = odeint(deriv, y0, t, args=(N, beta,pEI, pIH, rates,pHD))
  s, e, i, r, h, d = ret.T

  res = {"S":s, "E":e, "I":i, "R":r, "H":h, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
  res["pIH"] = [pIH(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

#========================================================================================================
#======================================================================================================== 
  


def resid_(params,y0, t, N,age_effect,S,I,H,R,D):
    print(params)
    rates={"rse":0, "rei":0, "rih":0, "rir":0, "rhr":0, "rhd":0 }
       
    r0_max = params['r0_max']
    r0_min = params['r0_min']
    k = params['k']
    startLockdown = params['startLockdown']
    rates['rse']=params['rse']
    rates['rei']=params['rei']
    rates['rir']=params['rir']
    rates['rih']=params['rih'] 
    rates['rhd']=params['rhd']
    rates['rhr']=params['rhr']
    seihrd_det=solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,  rates,demographic)
    #return sum((seihrd_det['S']-S)**2+(seihrd_det['I']-I)**2+(seihrd_det['H']-H)**2+(seihrd_det['R']-R)**2+(seihrd_det['D']-D)**2)
    #er=np.array((((seihrd_det['S']-S)**2),((seihrd_det['I']-I)**2),((seihrd_det['H']-H)**2),((seihrd_det['R']-R)**2),((seihrd_det['D']-D)**2)))
    
    #a=np.array((S-seihrd_det['S'])**2)
    b=np.array((I-seihrd_det['I'])**2)
    c=np.array((H-seihrd_det['H'])**2)
    d=np.array((R-seihrd_det['R'])**2)
    e=np.array((1.2*D-1.2*seihrd_det['D'])**2)
    
    tot=np.concatenate((b,c,d,e))
    
    #return er
    #er=er.flatten()
    #er2= np.array((seihrd_det['S']-S)**2,(seihrd_det['I']-I)**2,(seihrd_det['H']-H)**2,(seihrd_det['R']-R)**2,(seihrd_det['D']-D)**2)
    return  tot



params2=Parameters()
params2.add('r0_max',value=3.8,vary=True,min=2.0,max=5.0)
params2.add('r0_min',value=2.3,vary=True,min=0.3,max=3.5)
params2.add('k',value=4,vary=True,min=0.01,max=5.0)
params2.add('startLockdown',value=31.673,vary=True,min=0,max=100)
params2.add('rse',value=1.49,vary=True,min=0.9,max=1.5)
params2.add('rei',value=1.2,vary=True,min=0.01,max=1.2)  
params2.add('rih',value=0.25,vary=True,min=0.01,max=2.0)
params2.add('rir',value=0.019,vary=True,min=0.01,max=1.2)
params2.add('rhd',value=0.2,vary=True,min=0.01,max=1.2)
params2.add('rhr',value=0.1,vary=True,min=0.01,max=1.2)


res3=resid_(params2,y0, t, N,age_effect,S,I,H,R,D)
out = minimize(resid_, params2, args=(y0, t, N,age_effect,S,I,H,R,D),method='differential_evolution')#method='differential_evolution',method='leastsq'




rates_fit={"rse":out.params['rse'].value, "rei":out.params['rei'].value, "rih":out.params['rih'].value, "rir":out.params['rir'].value, "rhr":out.params['rhr'].value, "rhd":out.params['rhd'].value }

seihrd_det_fit=solveSIRdet(y0, t, N,age_effect, out.params['r0_max'].value, out.params['r0_min'].value, out.params['k'].value, out.params['startLockdown'].value,  rates_fit,demographic)



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
ax5.plot(t,seihrd_det_fit['R'],color="g")
ax5.plot(t,R,label='Recovered',color="g",ls="--")
plt.xlabel('day')
plt.ylabel('Recovered')

ax6=plt.subplot(334)        
ax6.plot(t,seihrd_det_fit['S'],color="y")
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
#ax7.plot(t,seihrd_det_fit['H'],color="orange")
#ax7.plot(t,H,label='Hosp',color="orange",ls="--")
plt.xlabel('day')
plt.ylabel('people')

#crea un dataframe e lo salva in un file csv con i dati del modello differenziale
#fitSEIHRD=pd.DataFrame({"rse":out.params['rse'].value, "rei":out.params['rei'].value, "rih":out.params['rih'].value, "rir":out.params['rir'].value, "rhr":out.params['rhr'].value, "rhd":out.params['rhd'].value,'r0_max':out.params['r0_max'].value,'r0_min': out.params['r0_min'].value,'k': out.params['k'].value,'startLockdown': out.params['startLockdown'].value })
#fitSEIHRD.to_csv(r'C:\Users\Enrico\Desktop\SIR\dfSIRD.csv',index=False)

df_fitSEIHRD=pd.read_csv('df_fitSEIHRD.csv')

risult=np.array([[regione,
                  out.params['rse'].value,
                  out.params['rei'].value,
                  out.params['rih'].value,
                  out.params['rir'].value,
                  out.params['rhr'].value,
                  out.params['rhd'].value,
                  out.params['r0_max'].value,
                  out.params['r0_min'].value,
                  out.params['k'].value,
                  out.params['startLockdown'].value]])


df_fitSEIHRD = pd.DataFrame(risult,columns=["regione", "rse", "rei","rih","rir","rhr","rhd","r0_max","r0_min","k","startLockdown"]).append(df_fitSEIHRD, ignore_index=True)
df_fitSEIHRD.to_csv(r'C:\Users\Enrico\Desktop\SIR\df_fitSEIHRD.csv',index=False)
saveFIG=regione+'.png'
plt.savefig(saveFIG)



