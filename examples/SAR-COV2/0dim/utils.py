import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import scipy.integrate as scint

import seaborn as sns
sns.set_style("darkgrid")

from plotdf import plotdf
import numpy as np
import os
import argparse
from matplotlib.gridspec import GridSpec

class StoreDictKeyPair(argparse.Action):
 def __call__(self, parser, namespace, values, option_string=None):
   my_dict = {}
   for kv in values.split(","):
     k,v = kv.split("=")
     my_dict[k] = float(v)
   setattr(namespace, self.dest, my_dict)

def plotPhase(tau, xbound, ybound, gridsteps, tmax, nsteps, N, mu, outdir="./"):

  from sir import deriv_adim

  parameters = {"N":N,"mu":mu}
  fig = plt.figure(figsize=(8,8), facecolor='w')
  axes = fig.add_subplot(111, axisbelow=True)

  x = np.linspace(xbound[0],xbound[1],gridsteps)
  y = np.linspace(ybound[0],ybound[1],gridsteps)
  xx,yy = np.meshgrid(x,y)
  uu = np.empty_like(xx)
  vv = np.empty_like(yy)
  for i in range(gridsteps):
    for j in range(gridsteps):
      res = deriv_adim(np.array([xx[i,j],yy[i,j], 0.0]), tau, **parameters)
      uu[i,j] = res[0]
      vv[i,j] = res[1]

  artists = []
  EE = np.hypot(uu, vv)

  #I = plt.imshow(EE,extent=[np.min(xx),np.max(xx),np.min(yy),np.max(yy)],cmap='coolwarm')
  axes.quiver(
      xx,yy,uu,vv,EE,
      cmap="autumn", 
      #norm=colors.LogNorm(vmin=M.min(),vmax=M.max()),
      width=0.002)

  plt.title(r"2D Phase space for SIR model")
  axes.grid(b=True, which='major', c='w', lw=2, ls='-')
  axes.axvline(x=mu*N, color="k")
  axes.set_xlabel(r'S')
  axes.set_ylabel(r'I')
  plt.xlim(xbound)
  plt.ylim(ybound)
  plt.savefig(os.path.join(outdir, "ps.png"))

  return axes

def plotTrajectories(inits, axes, tmax, nsteps, tdir, N, mu, outdir="./"):
  from sir import deriv_adim

  parameters = {"N":N,"mu":mu}
  # Plot some trajectories over the phase space in axes
  def g(x,t):
    return np.array(deriv_adim(x, t, **parameters))

  def bg(x,t):
    return -1.0*np.array(deriv_adim(x, t, **parameters))

  t = np.linspace(0, int(tmax), int(nsteps))
  for y0 in inits:
    traj_f = np.empty((0,2))
    traj_b = np.empty((0,2))
    if tdir in ["forward","both"]:
      traj_f = scint.odeint(g,y0,t)
    if tdir in ["backward","both"]:
      traj_b = scint.odeint(bg,y0,t)
    
    if tdir != "both":
      traj = traj_f
    else:
      traj = np.vstack((np.flipud(traj_b),traj_f))
    axes.plot(traj[:,0],traj[:,1])
  
  plt.savefig(os.path.join(outdir,"ps_traj.png"))

def plotSAIRHDfit(country, t, df, S, A, I, H, R, D, fitImage, resImage ):
  plt.figure(figsize=(15, 12))
  plt.suptitle(country)
 
  ax6=plt.subplot(231)        
  ax6.plot(t,df['S'],color="y")
  ax6.plot(t,S,label='Susceptible',color="y",ls="--")
  plt.xlabel('day')
  plt.ylabel('Susceptible')
 
  ax3=plt.subplot(232)
  ax3.plot(t,df['I'],color="r")
  ax3.plot(t,I,label='Infected',color="r",ls="--")
  plt.xlabel('day')
  plt.ylabel('Infected')
  
  ax7=plt.subplot(233)        
  ax7.plot(t,df['H'],color="orange")
  ax7.plot(t,H,label='Hosp',color="orange",ls="--")
  plt.xlabel('day')
  plt.ylabel('Hosp.')

  ax5=plt.subplot(234)        
  ax5.plot(t,df['R'],color="g")
  ax5.plot(t,R,label='Recovered',color="g",ls="--")
  plt.xlabel('day')
  plt.ylabel('Recovered')

  ax4=plt.subplot(235)        
  ax4.plot(t,df['D'],color="black")
  ax4.plot(t,D,label='Dead',color="black",ls="--")
  plt.xlabel('day')
  plt.ylabel('Dead')
  
  ax8=plt.subplot(236)        
  ax8.plot(t,df['I'],color="r")
  ax8.plot(t,I,label='Infected',color="r",ls="--")
  ax8.plot(t,df['R'],color="g")
  ax8.plot(t,R,label='Recovered',color="g",ls="--")
  ax8.plot(t,df['D'],color="black")
  ax8.plot(t,D,label='Dead',color="black",ls="--")
  ax8.plot(t,df['H'],color="orange")
  ax8.plot(t,H,label='Hosp',color="orange",ls="--")
  plt.xlabel('day')
  plt.ylabel('people')

  plt.savefig(fitImage)

  plt.figure(figsize=(15, 12))
  plt.suptitle(country)
  
  ax6=plt.subplot(231)        
  ax6.plot(t,S-df['S'],label='Susceptible',color="y",ls="--")
  plt.xlabel('day')
  plt.ylabel('Susceptible')

  ax3=plt.subplot(232)
  ax3.plot(t,I-df['I'],label='Infected',color="r",ls="--")
  plt.xlabel('day')
  plt.ylabel('Infected')

  ax7=plt.subplot(233)        
  ax7.plot(t,H-df['H'],label='Hosp',color="orange",ls="--")
  plt.xlabel('day')
  plt.ylabel('Hosp.') 
  
  ax5=plt.subplot(234)        
  ax5.plot(t,R-df['R'],label='Recovered',color="g",ls="--")
  plt.xlabel('day')
  plt.ylabel('Recovered')

  ax4=plt.subplot(235)        
  ax4.plot(t,D-df['D'],label='Dead',color="black",ls="--")
  plt.xlabel('day')
  plt.ylabel('Dead')
  

  
  plt.savefig(resImage) 



def plotSAIRHD(tau, sd, sdfname='sairhd.png', sdstyle='-', logy=False):
  # Plot the data on three separate curves for S(t), E(t), I(t) and R(t)
  fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='w')
  gs = GridSpec(2, 2, figure=fig)
  ax1 = fig.add_subplot(gs[0, :])
  ax3 = fig.add_subplot(gs[-1, 0])
  ax4 = fig.add_subplot(gs[-1, -1])

  ax1.plot(tau, sd["S"], sdstyle, color="black", alpha=0.5, lw=2, label='Susceptible')
  ax1.plot(tau, sd["A"], sdstyle, color="yellow", alpha=0.5, lw=2, label='Asymptomatics')
  ax1.plot(tau, sd["I"], sdstyle, color="orange", alpha=0.5, lw=2, label='Infected')
  ax1.plot(tau, sd["R"], sdstyle, color="green", alpha=0.5, lw=2, label='Recovered')
  ax1.plot(tau, sd["H"], sdstyle, color="red", alpha=0.5, lw=2, label='Hospitalized')
  ax1.plot(tau, sd["D"], sdstyle, color="black", alpha=0.5, lw=2, label='Death')
  ax1.plot(tau, sd["S"] + sd["A"] + sd["I"] + sd["R"] + sd["H"] + sd["D"], "--", color="blue", alpha=0.5, lw=2, label='TOTAL')  
  fig.suptitle("SAIRHD model")
  ax1.set_xlabel('Time (days)')
  ax1.set_ylabel('Population')
  #ax.set_ylim(0,1.2)
  ax1.yaxis.set_tick_params(length=0)
  ax1.xaxis.set_tick_params(length=0)
  ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
  if logy: ax1.set_yscale('log')
  legend = ax1.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(False)
  
  ax3.set_xlabel("Time (days)")
  ax3.set_ylabel(r"$R_0$")
  ax3.plot(tau, sd["R_0"], 'g'+sdstyle, alpha=0.5, lw=2, label='$R_0$')
  ax3.legend()

  ax4.set_xlabel("Time (days)")
  ax4.set_ylabel("pIH")
  ax4.plot(tau, sd["pIH"], 'k'+sdstyle, alpha=0.5, lw=2, label='pIH')
  ax4.legend()

  plt.savefig(sdfname)

def plotAlls(tau, sd, ss=None, sg=None, nits=100, nitg=100, sdfname='sir.png', sdstyle='-', outdir="./"):
  # Plot the data on three separate curves for S(t), I(t) and R(t)
  fig = plt.figure(figsize=(10, 6), dpi=300, facecolor='w')
  ax = fig.add_subplot(111, axisbelow=True)

  ax.plot(tau, sd["S"], 'b'+sdstyle, alpha=0.5, lw=2, label='Susceptible')
  ax.plot(tau, sd["I"], 'r'+sdstyle, alpha=0.5, lw=2, label='Infected')
  ax.plot(tau, sd["R"], 'g'+sdstyle, alpha=0.5, lw=2, label='Recovered')
  plt.title(r"Adimensional SIR model")
  ax.set_xlabel(r'$\tau$')
  ax.set_ylabel(r'Ratio \(\%\)')
  #ax.set_ylim(0,1.2)
  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
  plt.savefig(os.path.join(outdir,sdfname))

  if ss:
    Sm = np.mean(ss["S"], axis=1)
    Ss = np.std(ss["S"], axis=1)
    S_err = 1.96*Ss/np.sqrt(nits)

    Im = np.mean(ss["I"], axis=1)
    Is = np.std(ss["I"], axis=1)
    I_err = 1.96*Is/np.sqrt(nits)

    Rm = np.mean(ss["R"], axis=1)
    Rs = np.std(ss["R"], axis=1)
    R_err = 1.96*Rs/np.sqrt(nits)

    plt.plot(tau, Sm, 'b--', alpha=0.5, lw=2, label='Susceptible Stoch')
    plt.fill_between(tau, Sm - S_err, Sm + S_err, alpha=0.2)

    plt.plot(tau, Im, 'r--', alpha=0.5, lw=2, label='Infected Stoch')
    plt.fill_between(tau, Im - I_err, Im + I_err, alpha=0.2)

    plt.plot(tau, Rm, 'g--', alpha=0.5, lw=2, label='Recovered Stoch')
    plt.fill_between(tau, Rm - R_err, Rm + R_err, alpha=0.2)

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
    plt.savefig(os.path.join(outdir,"sir_alls.png"))


  if sg:
    Sm = np.mean(sg["S"], axis=1)
    Ss = np.std(sg["S"], axis=1)
    S_err = 1.96*Ss/np.sqrt(nitg)

    Im = np.mean(sg["I"], axis=1)
    Is = np.std(sg["I"], axis=1)
    I_err = 1.96*Is/np.sqrt(nitg)

    Rm = np.mean(sg["R"], axis=1)
    Rs = np.std(sg["R"], axis=1)
    R_err = 1.96*Rs/np.sqrt(nitg)

    plt.plot(tau, Sm, 'b-.', alpha=0.5, lw=2, label='Susceptible Gill')
    plt.fill_between(tau, Sm - S_err, Sm + S_err, alpha=0.2)

    plt.plot(tau, Im, 'r-.', alpha=0.5, lw=2, label='Infected Gill')
    plt.fill_between(tau, Im - I_err, Im + I_err, alpha=0.2)

    plt.plot(tau, Rm, 'g-.', alpha=0.5, lw=2, label='Recovered Gill')
    plt.fill_between(tau, Rm - R_err, Rm + R_err, alpha=0.2)


    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
    plt.savefig(os.path.join(outdir,"sir_gill.png"))
