import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

from sir import deriv

from utils import plotAlls
from scipy.optimize import minimize
from functools import partial

def residual(S, I, R, t, y0, param):

    """
    compute the residual between actual data and fitted data
    """

    N, beta, gamma = param
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    s, i, r = ret.T

    return ( sum( (s - S)**2 + (i - I)**2 + (r - R)**2 ) )


if __name__ == "__main__":
  _d = np.loadtxt('sir_results.csv', delimiter=', ' )

  print(_d)
  t = _d[:, 0]; 

  N = 1000
  I0, R0 = 25, 0
  S0 = N - I0 - R0
  y0 = [S0, I0, R0]

  _residual = partial(residual, _d[:, 1], _d[:, 2], _d[:, 3], t, y0)
  msol = minimize(_residual, [1000, 0.01, 0.01], method='Nelder-Mead')

  print(msol)
  N_e, beta_e, gamma_e = msol.x

  print("**** Estimated parameters N: %.3f, beta: %.3f, gamma: %.3f"%(N_e, beta_e, gamma_e )) 

  #Plot data
  #plotAlls(t, data, sdfname='fit_data.png',sdstyle='*')
