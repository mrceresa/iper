# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:40:08 2020

@author: Enrico
"""

import pandas as pd
import matplotlib.pyplot as plt

#dati_regione=pd.read_csv('C:\\Users\Enrico\\Desktop\\Dati_Covid\\dati_Covid_Italia\\dati-regioni\\dpc-covid19-ita-regioni.csv')
dati_regione=pd.read_csv('dpc-covid19-ita-regioni.csv')
regione='P.A. Bolzano'
TRUEregione=dati_regione['denominazione_regione']==regione
dati_regione1=dati_regione[TRUEregione]

#N=5000

I=dati_regione1['totale_positivi']
D=dati_regione1['deceduti']
R=dati_regione1['dimessi_guariti']
#S=N-I-D-R
t=[i+1 for i in range(len(I))]

ax=plt.subplot()          
ax.plot(t,I,label='Infected',color="r",ls="--")
ax.plot(t,D,label='Dead',color="black",ls="--")
ax.plot(t,R,label='Recovered',color="g",ls="--")
#ax.plot(t,S,label='Recovered',color="g",ls="--")

ax.legend(loc="upper right")         
plt.title("Covid19_ITA")
plt.xlabel("day")
plt.ylabel("population")  
plt.show()