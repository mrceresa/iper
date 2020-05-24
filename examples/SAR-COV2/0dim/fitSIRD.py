# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:53:07 2020

@author: Enrico 
"""
from lmfit import Parameters
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from functools import partial
from SIRD import modello

#dati_regione=pd.read_csv('C:\\Users\Enrico\\Desktop\\Dati_Covid\\dati_Covid_Italia\\dati-regioni\\dpc-covid19-ita-regioni.csv')
dati_regione=pd.read_csv('dpc-covid19-ita-regioni.csv')
regione='P.A. Trento'
TRUEregione=dati_regione['denominazione_regione']==regione
dati_regione1=dati_regione[TRUEregione]
 

N=5000

I=np.array(dati_regione1['totale_positivi'])
D=np.array(dati_regione1['deceduti'])
R=np.array(dati_regione1['dimessi_guariti'])
S=N-I-D-R

# =============================================================================
# N=1000
# dati_SIRD=pd.read_csv(r'C:\Users\Enrico\Desktop\SIR\dfSIRD.csv')
# I=np.array(dati_SIRD['I'])
# R=np.array(dati_SIRD['R'])
# D=np.array(dati_SIRD['D'])
# S=np.array(dati_SIRD['S'])
# 
# =============================================================================

t=np.array([i+1 for i in range(len(I))])

i0=25                              #infetti inziali                             
r0=0                               #immuni iniziali
s0=N-i0-r0                         #suscettibili iniziali
d0=0     
y0=s0,i0,r0,d0

# =============================================================================
# def modello(z,t,N, a, beta,l):
#     dSdt = -a * z[0] * z[1]/(N)                      
#     dIdt =  a * z[0] * z[1]/(N) - beta * z[1]-l*z[1] 
#     dRdt = beta * z[1] 
#     dDdt = l*z[1]
#     dzdt = [dSdt,dIdt,dRdt,dDdt]
#     return dzdt
# =============================================================================

def residual(S, I, R, t, y0, N, param):   
    a, beta, l = param
    Z = odeint(modello, y0, t, args=( N, a, beta, l)) 
    return (sum((Z[:,0]-S)**2+(Z[:,1]-I)**2+(Z[:,2]-R)**2))

res=residual(S, I, R, t, y0, N, [0.01,0.01,0.01])

params=Parameters()
params.add('a1',value=0.01,min=0,max=1)
params.add('b1',value=0.01,min=0,max=1)
params.add('c1',value=0.01,min=0,max=1)

residual2=partial(residual,S, I, R, t, y0, N)
msol=minimize(residual2,params,method='Nelder-Mead')

Zo=odeint(modello, y0, t, args=( N, msol.x[0], msol.x[1], msol.x[2])) 


ax=plt.subplot()          
ax.plot(t,I,label='Infected',color="r",ls="--")
ax.plot(t,D,label='Dead',color="black",ls="--")
ax.plot(t,R,label='Recovered',color="g",ls="--")
ax.plot(t,S,label='subs',color="y",ls="--")

ax.plot(t,Zo[:,0],label='S diff',color="y",ls="-")
ax.plot(t,Zo[:,1],label='I diff',color="r",ls="-")
ax.plot(t,Zo[:,2],label='R diff',color="g",ls="-")
ax.plot(t,Zo[:,3],label='D diff',color="black",ls="-")


ax.legend(loc="upper right")         
plt.title("Covid19_ITA")
plt.xlabel("day")
plt.ylabel("population")  
plt.show()