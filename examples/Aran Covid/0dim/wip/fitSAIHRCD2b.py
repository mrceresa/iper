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
import inspect

#========================================================================================================
#======================================================================================================== 


def deriv(y, t, N, beta,pAI, pIH, pHC, pCD, rates):
    
    S, A, I, R, H, C, D = y
    dSdt = -rates["rsa"]* S/N* beta(t)*(I+A)
    dAdt =  rates["rsa"]* S/N* beta(t)*(I+A) - rates["rai"]*pAI*A-rates["rar"]*(1-pAI)*A
    dIdt =  rates["rai"]*pAI*A  - rates["rir"]*(1 - pIH)*I  - rates["rih"]*pIH*I
    dRdt =  rates["rir"]*(1 - pIH)*I + rates["rhr"]*(1-pHC)*H +rates["rir"]*(1-pCD)*C #+ rates["rar"]*(1-pAI(t,I,N))*A
    dHdt =  rates["rih"]*pIH*I - rates["rhc"]*pHC*H - rates["rhr"]*(1-pHC)*H 
    dCdt =  rates["rhc"]*pHC*H - rates["rcd"]*pCD*C -rates["rcr"]*(1-pCD)*C
    dDdt =  rates["rcd"]*pCD*C
    
    return dSdt, dAdt, dIdt, dRdt, dHdt,dCdt, dDdt

def solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown, pAI,pIH,pHC,pCD,rates,demographic, icu_beds):
    
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
#    return age_effect*I/N + pIH_av    
#  
#  pHC_age = {"0-29": 0.05, "30-59": 0.10, "60-89": 0.4, "89+": 0.6}  
#  pHC_av = sum(pHC_age[i] * demographic[i] for i in list(pHC_age.keys()))
#  print("Average pHC:", pHC_av)
#  def pHC(tau, I, N):
#    return age_effect*I/N + pHC_av
#
#  pCD_age = {"0-29": 0.01, "30-59": 0.05, "60-89": 0.2, "89+": 0.4}  
#  pCD_av = sum(pCD_age[i] * demographic[i] for i in list(pCD_age.keys()))
#  print("Average pCD:", pCD_av)
#  def pCD(tau, I, N):
#    return age_effect*I/N + pCD_av

#=======================================================================================================    
# Integrate the SEIHR equations over the time grid, t
    
  ret = odeint(deriv, y0, t, args=(N, beta,pAI, pIH, pHC, pCD, rates))
  s, a, i, r, h, c, d = ret.T

  res = {"S":s, "A":a, "I":i, "R":r, "H":h, "C":c, "D":d} 
  res["R_0"] = list(map(R_0, t)) 
  #res["pIH"] = [pIH(tau, res["I"][tau], N) for tau in range(len(t))]

  return res

#========================================================================================================
#========================================================================================================
  
def main():
    
    global mainVars
    
    N=350000
         
    age_effect=1
    demographic = {"0-29": 0.2, "30-59": 0.4, "60-89": 0.35, "89+": 0.05}
    
    icu_beds = pd.read_csv("data/beds.csv", header=0)
    icu_beds = dict(zip(icu_beds["Country"], icu_beds["ICU_Beds"]))
    icu_beds = icu_beds["Italy"] * N / 100000 # Emergency life support beds per 100k citizens
    print(icu_beds)
    
#=================================================================================================  
    #per le regioni
#    dati_regione=pd.read_csv(os.path.join('data','dpc-covid19-ita-regioni.csv'))
#    regione='Veneto'
#    TRUEregione=dati_regione['denominazione_regione']==regione
#    dati_regione_=dati_regione[TRUEregione]
#    dati_regione1=dati_regione_[dati_regione_['totale_positivi']>25] #per stabilire il t0
#    
#    
#    I=np.array(dati_regione1['totale_positivi'])
#    D=np.array(dati_regione1['deceduti'])
#    R=np.array(dati_regione1['dimessi_guariti'])
#    H=np.array(dati_regione1['ricoverati_con_sintomi'])
#    C=np.array(dati_regione1['terapia_intensiva'])
#    S=N-I-D-H-R-C
#    
#   
#    
#    i0=dati_regione1.iloc[0,10]        #infetti a t0 
#    a0=4*i0                            #asintomatici a t0
#    r0=dati_regione1.iloc[0,12]        #recovered a t0
#    h0=dati_regione1.iloc[0,6]
#    c0=dati_regione1.iloc[0,7]         #ospedaliz.crit a t0                         
#    d0=dati_regione1.iloc[0,13]        #deceduti  t0
#    s0=N-i0-r0 -h0 -d0 -c0 -a0                   #suscettibili a t0
    
#======================================================================================================    
    
    
    
#============================================================================ 
#per le nazioni
    regione='Iceland'
    dati_regione1=pd.read_table(os.path.join('data','Iceland.tsv'),header=3)
    dati_regione=dati_regione1[dati_regione1['cases']>25] #per stabilire il t0
    I=dati_regione['cases']-dati_regione['recovered']
    D=dati_regione['deaths']
    R=dati_regione['recovered']
    C=dati_regione['icu']
    H=dati_regione['hospitalized']
    S=N-I-D-H-R-C
    
    
    i0=dati_regione.iloc[0,1]        #infetti a t0 
    a0=4*i0                            #asintomatici a t0
    r0=dati_regione.iloc[0,5]        #recovered a t0
    h0=dati_regione.iloc[0,3]         #ospedaliz. a t0
    c0=dati_regione.iloc[0,4]- dati_regione.iloc[0,5]       #ospedaliz.crit a t0             
    d0=dati_regione.iloc[0,2]        #deceduti  t0
    s0=N-i0-r0 -h0 -d0 -c0 -a0                     #suscettibili a t0
#     
#============================================================================  

    
    t=np.array([i+1 for i in range(len(I))])
    y0=s0,a0,i0,r0,h0,c0,d0 
    
    
    def resid_(params,y0, t, N,age_effect,S,I,H,R,C,D,icu_beds):
        print(params)
        rates={"rsa":0, "rai":0, "rih":0, "rir":0, "rhr":0, "rhc":0,"rcd":0,"rar":0,"rcr":0 }
           
        r0_max = params['r0_max']
        r0_min = params['r0_min']
        k = params['k']
        startLockdown = params['startLockdown']
        rates['rsa']=params['rsa']
        rates['rai']=params['rai']
        rates['rir']=params['rir']
        rates['rih']=params['rih'] 
        rates['rhc']=params['rhc']
        rates['rhr']=params['rhr']
        rates['rcd']=params['rcd']
        rates['rar']=params['rar']
        rates['rcr']=params['rcr']
        pAI=params['pAI']
        pIH=params['pIH']
        pHC=params['pHC']
        pCD=params['pCD']
        seihrd_det=solveSIRdet(y0, t, N,age_effect, r0_max, r0_min, k, startLockdown,  pAI,pIH,pHC,pCD, rates,demographic, icu_beds)
        #return sum((seihrd_det['S']-S)**2+(seihrd_det['I']-I)**2+(seihrd_det['H']-H)**2+(seihrd_det['R']-R)**2+(seihrd_det['D']-D)**2)
        #er=np.array((((seihrd_det['S']-S)**2),((seihrd_det['I']-I)**2),((seihrd_det['H']-H)**2),((seihrd_det['R']-R)**2),((seihrd_det['D']-D)**2)))
        
        a=np.array((S-seihrd_det['S'])**2)
        b=np.array((I-seihrd_det['I'])**2)
        c=np.array((H-seihrd_det['H'])**2)
        d=np.array((R-seihrd_det['R'])**2)
        e=np.array((1.2*D-1.2*seihrd_det['D'])**2)
        f=np.array((1.5*C-1.5*seihrd_det['C'])**2)
        
        tot=np.concatenate((a,b,d,c,e,f))
        
        #return er
        #er=er.flatten()
        #er2= np.array((seihrd_det['S']-S)**2,(seihrd_det['I']-I)**2,(seihrd_det['H']-H)**2,(seihrd_det['R']-R)**2,(seihrd_det['D']-D)**2)
        return  tot
    
    
    params2=Parameters()
    params2.add('r0_max',value=4.9,vary=True,min=2.0,max=5.0)
    params2.add('r0_min',value=0.3,vary=True,min=0.3,max=3.5)
    params2.add('k',value=0.168,vary=True,min=0.01,max=2.0)
    params2.add('startLockdown',value=31.673,vary=True,min=0,max=100)
    params2.add('pAI',value=0.59,vary=True,min=0.01,max=1)
    params2.add('pIH',value=0.99,vary=True,min=0.01,max=1)
    params2.add('pHC',value=0.2,vary=True,min=0.01,max=1)
    params2.add('pCD',value=0.2,vary=True,min=0.01,max=1)
    params2.add('rsa',value=1.169,vary=True,min=0.9,max=2)
    params2.add('rai',value=0.735,vary=True,min=0.01,max=2)  
    params2.add('rih',value=0.051,vary=True,min=0.01,max=2.0)
    params2.add('rir',value=0.026,vary=True,min=0.01,max=2)
    params2.add('rhc',value=0.2,vary=True,min=0.01,max=2)
    params2.add('rcd',value=0.2,vary=True,min=0.01,max=2)
    params2.add('rhr',value=0.1,vary=True,min=0.01,max=2)
    params2.add('rar',value=0.1,vary=True,min=0.01,max=2)
    params2.add('rcr',value=0.1,vary=True,min=0.01,max=2)
    
    res3=resid_(params2,y0, t, N,age_effect,S,I,H,R,C,D,icu_beds)
    out = minimize(resid_, params2, args=(y0, t, N,age_effect,S,I,H,R,C,D,icu_beds),method='differential_evolution', verbose=False)#method='differential_evolution',method='leastsq'
    
    
    rates_fit={"rsa":out.params['rsa'].value, "rai":out.params['rai'].value, "rih":out.params['rih'].value, "rir":out.params['rir'].value, "rhr":out.params['rhr'].value, "rhc":out.params['rhc'].value, "rcd":out.params['rcd'].value,"rar":out.params['rar'].value,"rcr":out.params['rcr'].value }
    
    seihrd_det_fit=solveSIRdet(y0, t, N,age_effect, out.params['r0_max'].value, out.params['r0_min'].value, out.params['k'].value, out.params['startLockdown'],out.params["pAI"].value,out.params["pIH"].value,out.params["pHC"].value,out.params["pCD"].value,  rates_fit,demographic,icu_beds)
    
    
    
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
    
    
    #    ax8=plt.subplot(336)        
    #    ax8.plot(t,seihrd_det_fit['I'],color="r")
    #    ax8.plot(t,I,label='Infected',color="r",ls="--")
    #    ax8.plot(t,seihrd_det_fit['R'],color="g")
    #    ax8.plot(t,R,label='Recovered',color="g",ls="--")
    #    ax8.plot(t,seihrd_det_fit['D'],color="black")
    #    ax8.plot(t,D,label='Dead',color="black",ls="--")
    #    ax8.plot(t,seihrd_det_fit['H'],color="orange")
    #    ax8.plot(t,H,label='Hosp',color="orange",ls="--")
    #    plt.xlabel('day')
    #    plt.ylabel('people')
    
    ax8=plt.subplot(336)        
    ax8.plot(t,seihrd_det_fit['C'],color="orange")
    ax8.plot(t,C,label='Crit.',color="orange",ls="--")
    plt.xlabel('day')
    plt.ylabel('Critic.')
    
    df_fitSAIHRD=pd.read_csv('df_fitSAIHRD.csv')
    
    risult=np.array([[regione,
                      out.params['pAI'].value,
                      out.params['pIH'].value,
                      out.params['pHC'].value,
                      out.params['pCD'].value,
                      out.params['rsa'].value,
                      out.params['rai'].value,
                      out.params['rih'].value,
                      out.params['rir'].value,
                      out.params['rhr'].value,
                      out.params['rhc'].value,
                      out.params['rcd'].value,
                      out.params['rar'].value,
                      out.params['rcr'].value,
                      out.params['r0_max'].value,
                      out.params['r0_min'].value,
                      out.params['k'].value,
                      out.params['startLockdown'].value]])
    
    
    df_fitSAIHRD = pd.DataFrame(risult,columns=["regione", "pAI","pIH","pHC","pCD", "rsa", "rai","rih","rir","rhr","rhc","rcd","rar","rcr","r0_max","r0_min","k","startLockdown"]).append(df_fitSAIHRD, ignore_index=True)
    fname = "df_fitSAIHRD.csv"
    df_fitSAIHRD.to_csv(fname,index=False)
    saveFIG=regione+'.png'
    plt.savefig(saveFIG)
    
    mainVars=inspect.currentframe().f_locals
    
if __name__ == "__main__":
    main()


