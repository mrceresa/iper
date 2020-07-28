# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:59:04 2020

@author: Enrico
"""

#from fitSEIHRD2b import solveSIRdet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitSEIHRD2b import solveSIRdet

age_effect=1
N=10000000
days=100
i0=25       
e0=4*i0                            
r0=0       
h0=0
d0=0   

i0=166     
e0=4*i0                            
r0=0       
h0=95
d0=6 
                     
s0=N-i0-r0 -h0-d0                   
       

y0=s0,e0,i0,r0,h0,d0

demographic = {"0-29": 0.2, "30-59": 0.4, "60-89": 0.35, "89+": 0.05}
t=np.array([i+1 for i in range(days)])

datiFIT=pd.read_csv('df_fitSEIHRD.csv')


    

r0_max=datiFIT['r0_max'].mean()
r0_min=datiFIT['r0_min'].mean()
k=datiFIT['k'].mean()
lenDF=len(datiFIT.columns)
y=datiFIT.iloc[:,3].mean()

par=np.array([i+0.1 for i in range(lenDF-1)])
for i in range(lenDF-1):
    par[i]=datiFIT.iloc[:,i+1].mean()
par=[2,2,0.088,0.0175,0.043489,0.306,5,0.64,0.1587,28.15]  
rates_fit={"rse":par[0], "rei":par[1], "rih":par[2], "rir":par[3], "rhr":par[4], "rhd":par[5] }    
seihrd_det_fit=solveSIRdet(y0, t, N,age_effect, par[6], par[7], par[8], par[9],  rates_fit,demographic)



plt.figure(figsize=(8, 6))
plt.suptitle('one')

ax8=plt.subplot()
ax8.plot(t,seihrd_det_fit['I'],color="r")
ax8.plot(t,seihrd_det_fit['R'],color="g")
ax8.plot(t,seihrd_det_fit['D'],color="black")
ax8.plot(t,seihrd_det_fit['H'],color="orange")

plt.xlabel('day')
plt.ylabel('people')