#!/usr/bin/python

import os, sys

import numpy as np
from scipy import integrate 
import matplotlib.pyplot as plt

k1 = 0.8 # Fertility rate
k2 = 0.1 # Death rate
CarryingCapacity = 1e4 # Limiting
hup1 = 0.33 # Conversion rate the productores 1
hup2 = 0.33 # Conversion rate the productores 2
dp1 = 0.01 # Exit of P1
dp2 = 0.01 # Exit of P1
da = 0.1 # Exit of A
db = 0.1 # Exit of B
pa = 0.1 # Production rate for A by P1
pb = 0.1 # Production rate for B by P2
cab = 0.1 # Conversion rate for a in b
lh1=1.0
meanSalary=10

def model(z, t, H):
  U, P1, P2, A, B, Ih = z
  
  dUdt = k1*U*( CarryingCapacity - U ) / CarryingCapacity -k2*U - hup1*U
  dP1dt = hup1*U -dp1*P1
  dP2dt = hup2*U -dp2*P2
  dAdt =  pa*P1 - cab*A*P2 - da*A
  dBdt =  cab*A*P2 - db*B
  dIhdt = meanSalary*lh1*H*P1
  
  dzdt = [dUdt, dP1dt, dP2dt, dAdt, dBdt, dIhdt]
  return dzdt

if __name__ == '__main__':
  # Initial conditions
  U_0 = 1e3;   P1_0 = 0; P2_0 = 0; A_0 = 0; B_0 = 0; Ih_0 = 0;
  z0 = [U_0, P1_0, P2_0, A_0, B_0, Ih_0]
   
  # Timepoints
  n = 401; t = np.linspace(0,40,n)
  
  # Input
  H = np.empty_like(t);
  H[0] = U_0
  # Store
  x_U = np.empty_like(t); x_P1 = np.empty_like(t); x_P2 = np.empty_like(t);
  x_A = np.empty_like(t); x_B = np.empty_like(t);
  x_Ih = np.empty_like(t);
  x_U[0] = z0[0]; x_P1[0] = z0[1]; x_P1[0] = z0[2];
  x_A[0] = z0[3]; x_B[0] = z0[4];
  x_Ih[0] = z0[5];
  # solve ODE
  for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = integrate.odeint(model, z0, tspan, args=(H[i],))
    # Store solution
    x_U[i] = z[1][0]; x_P1[i] = z[1][1]; x_P2[i] = z[1][2]; 
    x_A[i] = z[1][3]; x_B[i] = z[1][4];
    x_Ih[i] = z[1][5];
    # Update inputs
    H[i] = x_U[i] - x_P1[i] - x_P2[i] 
    # next initial condition
    z0 = z[1]
  
  fig, ax = plt.subplots(4, sharex=True, figsize=(8,8))
  fig.suptitle("Populations")  
  ax[0].plot(t, x_U, label="U(t)")
  ax[0].set_ylabel('Population [#]')
  ax[0].set_xlabel('Time [days]')
  ax[0].legend(loc='best')
  ax[1].plot(t, x_P1, label="P1(t)")  
  ax[1].set_ylabel('Population [#]')
  ax[1].set_xlabel('Time [days]')
  ax[1].legend(loc='best')  
  ax[2].plot(t, x_P2, label="P2(t)")  
  ax[2].set_ylabel('Population [#]')
  ax[2].set_xlabel('Time [days]')
  ax[2].legend(loc='best')
  ax[3].plot(t, H, label="H(t)")  
  ax[3].set_ylabel('Population [#]')
  ax[3].set_xlabel('Time [days]')
  ax[3].legend(loc='best')      
  plt.savefig("population.png")
  
  fig, ax = plt.subplots(2, sharex=True, figsize=(8,8))
  fig.suptitle("Products")  
  ax[0].plot(t, x_A, label="A(t)")
  ax[0].set_ylabel('Quantity [#]')
  ax[0].set_xlabel('Time [days]')
  ax[0].legend(loc='best')
  ax[1].plot(t, x_B, label="B(t)")  
  ax[1].set_ylabel('Quantity [#]')
  ax[1].set_xlabel('Time [days]')
  ax[1].legend(loc='best')  
  plt.savefig("products.png")
  
  fig, ax = plt.subplots(2, sharex=True, figsize=(8,8))
  fig.suptitle("Incomes")  
  ax[0].plot(t, x_Ih, label="Ih(t)")
  ax[0].set_ylabel('Money [$]')
  ax[0].set_xlabel('Time [days]')
  ax[0].legend(loc='best')

  plt.savefig("incomes.png")
  
