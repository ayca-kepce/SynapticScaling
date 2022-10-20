
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


@jit(nopython=True)
def euler_loop(delta_t,vars,t,num_neurons,weights,noises,back_inputs,stimulus,taus,exc_params,flags):
    (rE1, rE2, rP, rS) = vars
    (N_PC, N_PV, N_SOM) = num_neurons
    (w_DE1, w_DE12, w_DS1, w_EP1, w_PE1, w_SE1, w_DE2, w_DE21, w_DS2, w_EP2, w_PE2, w_SE2, w_PS, w_PP) = weights
    (noise_E1,noise_E2,noise_P,noise_S) = noises
    (x_E, x_D, x_P, x_S) = back_inputs
    (tau_E, tau_I) = taus
    (theta, lambda_D, lambda_E) = exc_params
    (plasticity, synaptic_scaling) = flags

    np.random.seed(124)
    # setting up initial conditions
    E01 = np.random.rand(N_PC)
    E02 = np.random.rand(N_PC)
    P0 = np.random.rand(N_PV)
    S0 = np.random.rand(N_SOM)

    step = 0
    for i in t:
        # averages are calculated
        rE1[step] = np.mean(E01)
        rE2[step] = np.mean(E02)
        rP[step] = np.mean(P0)
        rS[step] = np.mean(S0)

        I_E1 = x_E - w_EP1 @ P0
        I_D1 = x_D - w_DS1 @ S0 + w_DE1 @ E01 + w_DE12 @ E02
        I1 = lambda_D*I_D1 + (1-lambda_E)*I_E1

        I_E2 = x_E - w_EP2 @ P0
        I_D2 = x_D - w_DS2 @ S0 + w_DE2 @ E02 + w_DE21 @ E01
        I2 = lambda_D*I_D2 + (1-lambda_E)*I_E2



        term00 = delta_t * (1 / tau_E) * np.mean(-E01)
        term01 = delta_t * (1 / tau_E) * np.mean(np.maximum(0, I1 - theta))

        term02 = delta_t * (1 / tau_E) * stimulus[step]
        #terms_sum1 = np.sum(term00 + term01 + term02)
        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - theta)+ stimulus[step] + noise_E1[:,step])

        term03 = delta_t * (1 / tau_E) * np.mean(-E02)
        term04 = delta_t * (1 / tau_E) * np.mean(np.maximum(0, I2 - theta))
        term05 = delta_t * (1 / tau_E) * np.mean(noise_E2[:,step])
        #terms_sum2 = np.sum(term03 + term04 + term05)
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - theta) + noise_E2[:,step])

        term06 = delta_t * (1 / tau_I) * np.mean(w_PE1@ E01)
        term07 = delta_t * (1 / tau_I) * np.mean(w_PE1@ E02)
        term08 = delta_t * (1 / tau_I) * np.mean(-w_PS@ S0)
        term09 = delta_t * (1 / tau_I) * np.mean(-w_PP@ P0)
        term10 = delta_t * (1 / tau_I) * np.mean(x_P)
        #terms_sum3 = np.sum(term06 + term07 + term08 + term09 + term10)
        P = P0 + delta_t*(1/tau_I)*( -P0 + w_PE1 @ E01 + w_PE2 @ E02 - w_PS @ S0 - w_PP @ P0 + x_P + stimulus[step] + noise_P[:,step])

        term11 = delta_t * (1 / tau_I) * np.mean(-S0)
        term12 = delta_t * (1 / tau_I) * np.mean(w_SE1@ E01)
        term13 = delta_t * (1 / tau_I) * np.mean(w_SE2@ E02)
        term14 = delta_t * (1 / tau_I) * np.mean(x_S)
        term15 = delta_t * (1 / tau_I) * np.mean(noise_S[:,step])
        #terms_sum4 = np.sum(term11 + term12 + term13 + term14 + term15)
        S = S0 + delta_t*(1/tau_I)*( -S0 + w_SE1 @ E01 + w_SE2 @ E02 + x_S  + noise_S[:,step])

        E1[E1 < 0] = 0
        E2[E2 < 0] = 0
        P[P < 0] = 0
        S[S < 0] = 0

        # placeholder parameters are freed
        E01 = E1
        E02 = E2
        P0 = P
        S0 = S

        # counter is updated
        step=step+1
        if not np.mod(step, 10000):
            print("I1",term01)
            print("I2",term04)
            print("e1->p",term06)
            print("e2->p",term07)
            print("s->p",term08)
            print("p->p",term09)
            print("e1->s",term12)
            print("e2->s",term13)

