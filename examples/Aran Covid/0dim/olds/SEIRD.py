# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:27:17 2020

@author: Enrico
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



N=1000                             #popolazione totale
i0=25                              #infetti inziali
e0=0                               
r0=0                               #immuni iniziali
s0=N-i0-r0                         #suscettibili iniziali
d0=0                               
T=1000                             #tempo totale di ossevarvazione
dt=1                               #intervallo di discretizzazione
numero_cicli=100                   #numero di simulazioni stocastiche 
a=0.09                             #probabilità di infezione di un individuo in un incontro a rischio
beta=0.04                          #probabilità di guarigione di un individuo nel tempo dt 
eta=0.1                            #probabilità di passare da E ad I
ro=0.01                            #probabilità di passare da E ad R
l=0.01                             #probabilità di passare d I a D


#definiamo le funct. dove: z[0]=S;z[1]=E z[2]=I; z[3]=R; z[4]=D
def modello(z,t,N, a, beta, eta ,ro, l):
  
     dSdt = -a * z[0] * z[2]/(N) 
     dEdt=a * z[0] * z[2]/(N)-eta*z[1] -ro*z[1]                       
     dIdt = eta*z[1]  - beta * z[2]-l*z[2]
     dRdt = beta * z[2]+ro*z[1]  
     dDdt= l*z[2]
     dzdt=[dSdt,dEdt,dIdt,dRdt,dDdt]
     return dzdt
 
n_rilevazioni=int(T/dt)
t=np.linspace(0,T,n_rilevazioni+1)
y0=s0,e0,i0,r0,d0

Z=odeint(modello,y0,t,args=(N,a,beta,eta,ro,l))



S=[]
E=[]
I=[]
R=[]
D=[]
#inizializzo le variabili che serviranno per calcolare il valore medio.gli elementi saranno la somma di ogni elemento in ogni simulaz.
sommaS=np.zeros(n_rilevazioni+1)
sommaE=np.zeros(n_rilevazioni+1)
sommaI=np.zeros(n_rilevazioni+1)
sommaR=np.zeros(n_rilevazioni+1)
sommaD=np.zeros(n_rilevazioni+1)
#questo ciclo for ha il compinto di effettuare la simulazione stocastica un "numero_cicli" di volte
for k in range(numero_cicli):
    
    e=[e0]
    i=[i0]
    r=[r0]
    s=[s0]
    d=[d0]

    for y in range(n_rilevazioni):



    #supponiamo che ogni dt ogni individuo incontri in media un individuo, per N individui ci saranno N(N-1)/2 incontri
    #per S suscettibili ed I malati ci saranno S*I incontri a rischio
    #per ciascun individuo la probabilità di avere un incontro a richio è (S*I)/(N(N-1)/2)
    #per ciascun individuo la probabilità di contagio è  a*(S*I)/(N(N-1)/2)------
    #(per N individui ci aspettiamo ogni dt a*2*(S*I)/(N-1) nuovi contagi)
    
        
        exposed=np.random.binomial(s[y],a*i[y]/(N-1))
        
        infected=np.random.binomial(e[y],eta)
        if e[y]>0 and e[y]!=infected:    
            Erecovered=np.random.binomial(e[y]-infected,ro*e[y]/(e[y]-infected))#e[y]-eta*e[y]    
        else:
            Erecovered=0
            
            
       
        recovered=np.random.binomial(i[y],beta)
        if i[y]>0 and i[y]!=recovered:    
            Dead=np.random.binomial(i[y]-recovered,l*i[y]/(i[y]-recovered))#e[y]-eta*e[y]    
        else:
            Dead=0
               
    
        s.append(s[y]-exposed)
        e.append(e[y]+exposed-infected-Erecovered)
        i.append(i[y]+infected-recovered-Dead)
        r.append(r[y]+recovered+Erecovered)
        d.append(d[y]+Dead)
        
        
    S.append(s)
    E.append(e)
    I.append(i)
    R.append(r)
    D.append(d)
    
    #questo ciclo costruisce i vettori i cui elementi sono la somma degli elementi di S,E,I,R,D di ogni simulazione.
    for y in range(n_rilevazioni+1): 
        sommaS[y]=S[k][y]+sommaS[y]
        sommaE[y]=E[k][y]+sommaE[y]
        sommaI[y]=I[k][y]+sommaI[y]
        sommaR[y]=R[k][y]+sommaR[y]
        sommaD[y]=D[k][y]+sommaD[y]
        #sommaS.append(y-y)
        #sommaS[y]=sommaS[y]+S[k][y]
        
    #vettori S,I,R, mediati in "numero_cicli" simulazioni
Smed=sommaS/numero_cicli
Emed=sommaE/numero_cicli
Imed=sommaI/numero_cicli
Rmed=sommaR/numero_cicli
Dmed=sommaD/numero_cicli
    
#function implementata per il calcolo dello scarto quadratico medio
def devStand(a):
    
    lun=len(a[0])
    a_std=np.zeros(lun)
    for y in range(lun):
        a_=[row[y] for row in a]
        a_std[y]=np.std(a_)
    return a_std   


#vettori delle deviazioni standard   
S_std1=devStand(S)
E_std1=devStand(E)
I_std1=devStand(I)
R_std1=devStand(R)
D_std1=devStand(D)



ax=plt.subplot()          
ax.plot(t,Smed,label='S stoc',color="y",ls="--")
ax.plot(t,Emed,label='E stoc',color="orange",ls="--")
ax.plot(t,Imed,label='I stoc',color="r",ls="--") 
ax.plot(t,Rmed,label='R stoc',color="g",ls="--")
ax.plot(t,Dmed,label='D stoc',color="black",ls="--")

ax.plot(t,Z[:,0],label='S diff',color="y",ls="-")
ax.plot(t,Z[:,1],label='E diff',color="orange",ls="-")
ax.plot(t,Z[:,2],label='I diff',color="r",ls="-")
ax.plot(t,Z[:,3],label='R diff',color="g",ls="-")
ax.plot(t,Z[:,4],label='D diff',color="black",ls="-")
       
ax.legend(loc="upper right")
         
plt.title("SEIRD")
plt.xlabel("day")
plt.ylabel("population")  


plt.show()
