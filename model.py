
"""
This script implements the equations 1-7 given in paper Hertag 2020.

Variables and parameters are represented in the matrix form. "ci" in the
paper corresponds to "c" here. Also, "c" in the paper corresponds to "constant"
here. The external input "x" to the inhibitory neurons are expressed as "x_ext_PV"
and "x_ext_SOM".

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from util import connection_probability

N_PC = 160
N_PV = 20
N_SOM = 20


tau_E = 0.06 # s
tau_I = 0.002 # s
theta = np.ones(N_PC)*14 # 1/s


lambda_D = 0.27
lambda_E = 0.31

# w_ij is from j to i
w_EP = connection_probability(np.ones((N_PC,N_PV)), 0.60, 2.8)
w_DS = connection_probability(np.ones((N_PC,N_SOM)), 0.55, 3.5)
w_DE = connection_probability(np.ones((N_PC,N_PC)), 0.1, 0.42)
w_PE = connection_probability(np.ones((N_PV,N_PC)), 0.45, 1.5)
w_PS = connection_probability(np.ones((N_PV,N_SOM)), 0.6, 0.7)
w_SE = connection_probability(np.ones((N_SOM,N_PC)), 0.35, 1)

theta_c = 28 # 1/s
constant = 7 # 1/s

# random assigned
x_E = np.ones(N_PC)
x_D = np.ones(N_PC)
x_ext_PV  = 2*np.ones(N_PV)
x_ext_SOM = 2*np.ones(N_SOM)


args = (tau_E,tau_I,theta,lambda_D,lambda_E,
        w_EP,w_DS,w_DE,w_PE,w_PS,w_SE,x_E,x_D,x_ext_PV,x_ext_SOM)

# functions, equations 2-6 in the paper
def I(I_D, I_E, c=0):
    return lambda_D*(np.maximum(0, I_D + c) + (1-lambda_E)*I_E)

def I_E(x_E, rP):
    return x_E - np.matmul(w_EP, rP)

def I_D(x_D, rS, rE):
    return x_D - np.matmul(w_DS,rS) + np.matmul(w_DE, rE)

def c(I_D0):
    return constant*np.heaviside(I_D0-theta_c, 0)

def I_D0(I_E, I_D):
    return lambda_E*I_E + (1-lambda_D)*I_D

# ODEs in 1D array form
def dxdt(x, t, *args):
    rE = x[0:N_PC]
    rP = x[N_PC:N_PC+N_PV]
    rS = x[N_PC+N_PV:]

    tau_E,tau_I,theta,lambda_D,lambda_E,w_EP,w_DS,w_DE,w_PE,w_PS,w_SE,x_E,x_D,x_ext_PV,x_ext_SOM = args

    I_E_param = I_E(x_E,rP)
    I_D_param = I_D(x_D, rS, rE)
    I_param = I(I_D_param, I_E_param)

    drEdt = (1/tau_E)*(-rE + np.maximum(0,I_param - theta))
    drPdt = (1/tau_I)*\
           ( -rP + np.matmul(w_PE,rE) - np.matmul(w_PS, rS) + x_ext_PV )
    drSdt = (1/tau_I)*\
           ( -rS + np.matmul(w_SE,rE) + x_ext_SOM )

    dxdt1d = np.concatenate((drEdt,drPdt,drSdt))
    return dxdt1d


t = np.linspace(0, 0.1, int(0.1*(1/0.01)))
E0 = np.random.rand(N_PC)
P0 = np.random.rand(N_PV)
S0 = np.random.rand(N_SOM)

x0 = np.concatenate((E0, P0, S0))
sol = odeint(dxdt, x0, t, args=(tau_E,tau_I,theta,lambda_D,lambda_E,w_EP,
        w_DS,w_DE,w_PE,w_PS,w_SE,x_E,x_D,x_ext_PV,x_ext_SOM))

rE = np.mean(sol[:, :N_PC], axis=1)
rP = np.mean(sol[:, N_PC:N_PC+N_PV], axis=1)
rS = np.mean(sol[:, N_PC+N_PV:], axis=1)
plt.plot(t, rE, 'r', label='PC')
plt.plot(t, rP, 'g', label='PV')
plt.plot(t, rS, 'b', label='SOM/SST')
plt.legend(loc='best')
plt.xlabel('time [ms]')
plt.ylabel('Firing rates [Hz]')
plt.grid()
plt.show()