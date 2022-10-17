
"""
This script implements the equations 1-7 given in paper Hertag 2020.

Variables and parameters are represented in the matrix form. "ci" in the
paper corresponds to "c" here. Also, "c" in the paper corresponds to "constant"
here. The external input "x" to the inhibitory neurons are expressed as "x_P"
and "x_S".

"""

import numpy as np
from numba import cuda, jit
import matplotlib.pyplot as plt
from util import *

sim_duration = 5  # s
delta_t = 0.00001  # s
sim_timepoints = int(sim_duration * (1 / delta_t))

rE = np.zeros(sim_timepoints, dtype=np.float64)
rP = np.zeros(sim_timepoints, dtype=np.float64)
rS = np.zeros(sim_timepoints, dtype=np.float64)
t = np.float16(np.linspace(0, sim_duration, sim_timepoints))


#@jit(target_backend='cpu')
def simulation(rE,rP,rS,t,delta_t):
    N_PC = 160
    N_PV = 20
    N_SOM = 20

    stim_start = 26.5
    stim_strength = 0
    np.random.seed(124)

    tau_E = 0.06 # s
    tau_I = 0.002 # s
    theta = np.ones(N_PC)*14 # 1/s


    lambda_D = 0.27
    lambda_E = 0.31

    # w_ij is from j to i
    w_EP = connection_probability(1 * np.random.rand(N_PC, N_PV) + .5 * np.ones((N_PC, N_PV)), 0.60, 1)
    w_DS = connection_probability(1 * np.random.rand(N_PC, N_SOM) + .5 * np.ones((N_PC, N_SOM)), 0.55, 0.35)
    w_PE = connection_probability(1 * np.random.rand(N_PV, N_PC) + .5 * np.ones((N_PV, N_PC)), 0.45, 0.5)
    w_PP = connection_probability(1 * np.random.rand(N_PV, N_PV) + .5 * np.ones((N_PV, N_PV)), 0.50, 2)
    w_PS = connection_probability(1 * np.random.rand(N_PV, N_SOM) + .5 * np.ones((N_PV, N_SOM)), 0.60, 1.5)
    w_DE = connection_probability(1 * np.random.rand(N_PC, N_PC) + .5 * np.ones((N_PC, N_PC)), 0.10, 1.5)
    w_SE = connection_probability(1 * np.random.rand(N_SOM, N_PC) + .5 * np.ones((N_SOM, N_PC)), 0.35, 0.5)

    # inhibiting S, disinhibiting P
    w_SP = connection_probability(1 * np.random.rand(N_PV, N_PV) + .5 * np.ones((N_PV, N_PV)), 0.50, 0)
    w_SS = connection_probability(1 * np.random.rand(N_PV, N_PV) + .5 * np.ones((N_PV, N_PV)), 0.50, 2)

    # random assigned # you can use 1, 0, 2, 2 config as well
    x_E = 28*np.ones(N_PC)
    x_D = 0*np.ones(N_PC)
    x_P  = 2*np.ones(N_PV)
    x_S = 2*np.ones(N_SOM)

    E0 = np.random.rand(N_PC) + 3*np.ones(N_PC)
    P0 = np.random.rand(N_PV)
    S0 = np.random.rand(N_SOM)

    step = 0
    for i in t:
        # averages are calculated
        rE[step] = np.mean(E0)
        rP[step] = np.mean(P0)
        rS[step] = np.mean(S0)

        I_E = x_E - np.matmul(w_EP, P0)
        I_D = x_D - np.matmul(w_DS,S0) + np.matmul(w_DE, E0)
        I = lambda_D*I_D + (1-lambda_E)*I_E

        if i < stim_start:
            stimulus = 0
        else:
            stimulus = stim_strength


        E = E0 + delta_t*(1/tau_E)*( -E0 + np.maximum(0,I - theta) + stimulus) #+ np.random.rand(N_PC))
        P = P0 + delta_t*(1/tau_I)*( -P0 + np.matmul(w_PE,E0) - np.matmul(w_PS, S0) - np.matmul(w_PP, P0) + x_P) # + np.random.rand(N_PV))
        S = S0 + delta_t*(1/tau_I)*( -S0 + np.matmul(w_SE,E0) - np.matmul(w_SP, P0) - np.matmul(w_SS, S0) + x_S) # + np.random.rand(N_SOM))

        E[E < 0] = 0
        P[P < 0] = 0
        S[S < 0] = 0

        # placeholder parameters are freed
        E0 = E
        P0 = P
        S0 = S

        # counter is updated
        step=step+1
    return rE,rP,rS

    """plt.figure()
    plt.plot(t[:], rE[:], 'r', label='PC,  saturates at '+ str(np.round(np.mean(rE[-80:]),3)) + 'Hz')
    plt.plot(t[:], rP[:], 'g', label='PV,  saturates at '+ str(np.round(np.mean(rP[-80:]),3)) + 'Hz')
    plt.plot(t[:], rS[:], 'b', label='SST, saturates at '+ str(np.round(np.mean(rS[-80:]),3)) + 'Hz')
    plt.legend(loc='best')
    plt.xlabel('time [ms]')
    plt.ylabel('Firing rates [Hz]')
    plt.grid()
    plt.show()"""

rE_sol,rP_sol,rS_sol = simulation(rE, rP, rS, t, delta_t)
show_plot(rE_sol, rP_sol, rS_sol, t)