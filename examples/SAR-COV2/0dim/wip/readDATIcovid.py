# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:30:02 2020

@author: Enrico
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


N=60000000


#=============================================================================
dati_regione1=pd.read_table(os.path.join('data','Spain.tsv'),header=3)
dati_regione=dati_regione1[dati_regione1['cases']>25] #per stabilire il t0
I=dati_regione['cases']-dati_regione['recovered']
D=dati_regione['deaths']
R=dati_regione['recovered']
C=dati_regione['icu']
H=dati_regione['hospitalized']
S==N-I-D-H-R-C


i0=dati_regione.iloc[0,1]        #infetti a t0 
a0=4*i0                            #asintomatici a t0
r0=dati_regione.iloc[0,5]        #recovered a t0
h0=dati_regione.iloc[0,3]         #ospedaliz. a t0
c0=dati_regione.iloc[0,4]- dati_regione.iloc[0,5]       #ospedaliz.crit a t0             
d0=dati_regione.iloc[0,2]        #deceduti  t0
s0=N-i0-r0 -h0 -d0 -c0 -a0                     #suscettibili a t0
#=============================================================================





t=[i+1 for i in range(len(I))]

dati_regione.head()
dati_regione.info()

ax=plt.subplot()          
ax.plot(t,I,label='Infected',color="r",ls="--")
ax.plot(t,H,label='Hospitalized',color="y",ls="--")
ax.plot(t,C,label='Icu',color="orange",ls="--")
ax.plot(t,D,label='Dead',color="black",ls="--")
ax.plot(t,R,label='Recovered',color="g",ls="--")
#ax.plot(t,S,label='Recovered',color="g",ls="--")

ax.legend(loc="upper right")         
plt.title("Covid19_ITA")
plt.xlabel("day")
plt.ylabel("population")  
plt.show()