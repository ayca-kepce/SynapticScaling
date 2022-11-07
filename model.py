
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


#@jit(nopython=True)
def model_2PC(delta_t,vars,t,num_neurons,weights,noises,back_inputs,stimulus,taus,exc_params,flags):
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
        I_D1 = x_D - w_DS1 @ S0 + w_DE1 @ E01 + w_DE12 @ E02 + noise_E1[:,step] + stimulus[step]
        I1 = lambda_D*I_D1 + (1-lambda_E)*I_E1

        I_E2 = x_E - w_EP2 @ P0
        I_D2 = x_D - w_DS2 @ S0 + w_DE2 @ E02 + w_DE21 @ E01 + noise_E2[:,step]
        I2 = lambda_D*I_D2 + (1-lambda_E)*I_E2

        """term00 = delta_t * (1 / tau_E) * np.mean(-E01)
        term01 = delta_t * (1 / tau_E) * np.mean(np.maximum(0, I1 - theta))
        term02 = delta_t * (1 / tau_E) * stimulus[step]
        #terms_sum1 = np.sum(term00 + term01 + term02)"""
        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - theta))

        """term03 = delta_t * (1 / tau_E) * np.mean(-E02)
        term04 = delta_t * (1 / tau_E) * np.mean(np.maximum(0, I2 - theta))
        term05 = delta_t * (1 / tau_E) * np.mean(noise_E2[:,step])
        #terms_sum2 = np.sum(term03 + term04 + term05)"""
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - theta))

        """term06 = delta_t * (1 / tau_I) * np.mean(w_PE1@ E01)
        term07 = delta_t * (1 / tau_I) * np.mean(w_PE1@ E02)
        term08 = delta_t * (1 / tau_I) * np.mean(-w_PS@ S0)
        term09 = delta_t * (1 / tau_I) * np.mean(-w_PP@ P0)
        term10 = delta_t * (1 / tau_I) * np.mean(x_P)
        #terms_sum3 = np.sum(term06 + term07 + term08 + term09 + term10)"""
        P = P0 + delta_t*(1/tau_I)*( -P0 + w_PE1 @ E01 + w_PE2 @ E02 - w_PS @ S0 - w_PP @ P0 + x_P + stimulus[step] + noise_P[:,step])

        """term11 = delta_t * (1 / tau_I) * np.mean(-S0)
        term12 = delta_t * (1 / tau_I) * np.mean(w_SE1@ E01)
        term13 = delta_t * (1 / tau_I) * np.mean(w_SE2@ E02)
        term14 = delta_t * (1 / tau_I) * np.mean(x_S)
        term15 = delta_t * (1 / tau_I) * np.mean(noise_S[:,step])
        #terms_sum4 = np.sum(term11 + term12 + term13 + term14 + term15)"""
        S = S0 + delta_t*(1/tau_I)*( -S0 + w_SE1 @ E01 + w_SE2 @ E02 + x_S  + noise_S[:,step])

        if not np.mod(step, 10000):
            print("I1", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I1 - theta)))
            print("I2", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I2 - theta)))
            print("e1->p", delta_t * (1 / tau_I) * np.mean(w_PE1@ E01))
            print("e2->p", delta_t * (1 / tau_I) * np.mean(w_PE1@ E02))
            print("s->p",  delta_t * (1 / tau_I) * np.mean(-w_PS@ S0))
            print("p->p", delta_t * (1 / tau_I) * np.mean(-w_PP@ P0))
            print("e1->s", delta_t * (1 / tau_I) * np.mean(w_SE1@ E01))
            print("e2->s", delta_t * (1 / tau_I) * np.mean(w_SE2@ E02))

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


#@jit(nopython=True)
def model_2D(delta_t,vars,t,num_neurons,weights,noises,back_inputs,stimulus,stim_start,taus,exc_params,indv_neurons=None,flag_save_rates=0):
    (rE1, rE2, rP1, rP2, rS1, rS2) = vars
    (N_PC, N_PV, N_SOM) = num_neurons
    (w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
     w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22) = weights
    (noise_E1,noise_E2,noise_P1,noise_P2,noise_S1,noise_S2) = noises
    (x_E, x_D, x_P, x_S) = back_inputs
    (tau_E, tau_P, tau_S) = taus
    (theta, lambda_D, lambda_E) = exc_params

    np.random.seed(124)
    # setting up initial conditions
    E01 = np.random.rand(N_PC); E02 = np.random.rand(N_PC);
    P01 = np.random.rand(N_PV); P02 = np.random.rand(N_PV);
    S01 = np.random.rand(N_SOM); S02 = np.random.rand(N_SOM)

    step = 0
    for i in t:
        # averages are calculated
        rE1[step] = np.mean(E01); rE2[step] = np.mean(E02); rP1[step] = np.mean(P01); rP2[step] = np.mean(P02); rS1[step] = np.mean(S01); rS2[step] = np.mean(S02)

        """indv_neurons[:20,step] = E01[:20]
        indv_neurons[20:40,step] = E02[:20]"""

        I_E1 = x_E - w_EP11 @ P01 - w_EP12 @ P02
        I_D1 = x_D - w_DS11 @ S01 - w_DS12 @ S02 + w_DE11 @ E01 + w_DE12 @ E02 + noise_E1[:,step] + stimulus[step]
        I1 = lambda_D*I_D1 + (1-lambda_E)*I_E1

        I_E2 = x_E - w_EP21 @ P01 - w_EP22 @ P02
        I_D2 = x_D - w_DS21 @ S01 - w_DS22 @ S02 + w_DE22 @ E02 + w_DE21 @ E01 + noise_E2[:,step]
        I2 = lambda_D*I_D2 + (1-lambda_E)*I_E2

        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - theta) )
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - theta) )

        P1 = P01 + delta_t*(1/tau_P)*( -P01 + w_PE11 @ E01 + w_PE12 @ E02 - w_PS11 @ S01 - w_PS12 @ S02 - w_PP11 @ P01 - w_PP12 @ P02 + x_P + stimulus[step] + noise_P1[:,step])
        P2 = P02 + delta_t*(1/tau_P)*( -P02 + w_PE21 @ E01 + w_PE22 @ E02 - w_PS21 @ S01 - w_PS22 @ S02 - w_PP21 @ P01 - w_PP22 @ P02 + x_P + stimulus[step] + noise_P2[:,step])

        S1 = S01 + delta_t*(1/tau_S)*( -S01 + w_SE11 @ E01 + w_SE12 @ E02 + x_S  + noise_S1[:,step])
        S2 = S02 + delta_t*(1/tau_S)*( -S02 + w_SE21 @ E01 + w_SE22 @ E02 + x_S  + noise_S2[:,step])

        """if not np.mod(step, 10000):
            print("I1", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I1 - theta)))
            print("I2", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I2 - theta)))
            print("e1->p", delta_t * (1 / tau_I) * np.mean(w_PE11 @ E01))
            print("e2->p", delta_t * (1 / tau_I) * np.mean(w_PE12 @ E02))
            print("s->p", delta_t * (1 / tau_I) * np.mean(-w_PS11 @ S01))
            print("p->p", delta_t * (1 / tau_I) * np.mean(-w_PP11 @ P01))
            print("e1->s", delta_t * (1 / tau_I) * np.mean(w_SE11 @ E01))
            print("e2->s", delta_t * (1 / tau_I) * np.mean(w_SE22 @ E02))"""

        # limit rates to go below 0
        E1[E1 < 0] = 0; E2[E2 < 0] = 0; P1[P1 < 0] = 0; P2[P2 < 0] = 0; S1[S1 < 0] = 0; S2[S2 < 0] = 0
        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        # counter is updated
        step=step+1

    if flag_save_rates:
        if np.max(stimulus) != 0:
            print("The average rates can only be calculated without the stimulus.")
            #quit()
        E1[E1 == 0] = 1e-20
        E2[E2 == 0] = 1e-20
        P1[P1 == 0] = 1e-20
        P2[P2 == 0] = 1e-20
        S1[S1 == 0] = 1e-20
        S2[S2 == 0] = 1e-20
        pickle.dump((E1, E2, P1, P2, S1, S2, theta), open(r'params\average_rates_2D.pkl', 'wb'))



@jit(nopython=True)
def model_2D_plasticity_scaling(delta_t,vars,initial_values,t,num_neurons,weights,noises,back_inputs,\
                                stim_strength,stim_start,stim_stop,taus,lambdas,theta_E_default,upper_bound,indv_neurons=None,flags=(0,0,0)):
    (rE1, rE2, rP1, rP2, rS1, rS2,
     J_EE11, J_EE12, J_EE21, J_EE22,
     J_EP11, J_EP12, J_EP21, J_EP22,
     J_ES11, J_ES12, J_ES21, J_ES22,
     av_theta_E1, av_theta_E2,
     hebb11,ss_EE11,hebb21,ss_EE21,ss_EP21,ss_ES21) = vars

    (N_PC, N_PV, N_SOM) = num_neurons
    (_, _, _, _, w_PE11, w_SE11, w_PS11, w_PP11, _, _, w_PE12, w_SE12, w_PS12, w_PP12,
     _, _, _, _, w_PE21, w_SE21, w_PS21, w_PP21, _, _, w_PE22, w_SE22, w_PS22, w_PP22) = weights

    (noise_E1,noise_E2,noise_P1,noise_P2,noise_S1,noise_S2) = noises
    (x_E, x_D, x_P, x_S) = back_inputs
    (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta) = taus
    (lambda_D, lambda_E) = lambdas
    theta_E1 = np.ones(N_PC);theta_E2 = np.ones(N_PC)
    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag, BCM_flag)= flags

    np.random.seed(124)
    # setting up initial conditions
    (E01, E02, P01, P02, S01, S02,
     EE110, EE120, EE210, EE220,
     EP110, EP120, EP210, EP220,
     ES110, ES120, ES210, ES220) = initial_values

    stimulus = 0
    heb_plas_mask = 0
    exc_scal_mask = 0
    inh_scal_mask = 0
    BCM_mask = 0
    dif_E1 = np.zeros(N_PC); dif_E2 = np.zeros(N_PC); ratio_E1 = np.ones(N_PC); ratio_E2 = np.ones(N_PC)

    step = 0
    for i in t:
        # averages are calculated
        rE1[step] = np.mean(E01); rE2[step] = np.mean(E02); rP1[step] = np.mean(P01); rP2[step] = np.mean(P02); rS1[step] = np.mean(S01); rS2[step] = np.mean(S02)
        hebb11[step] = np.mean( heb_plas_mask * delta_t * (1 / tau_plas) * dif_E1.reshape(N_PC, 1) @ E01.reshape(1, N_PC))
        ss_EE11[step] = np.mean(exc_scal_mask*delta_t*(1/tau_scaling) * (EE110.T * (1 - ratio_E1)).T)
        hebb21[step] = np.mean(heb_plas_mask*delta_t*(1 / tau_plas) * dif_E1.reshape(N_PC,1) @ E02.reshape(1,N_PC))
        ss_EE21[step] = np.mean(exc_scal_mask*delta_t*(1/tau_scaling) * (EE120.T * (1 - ratio_E1)).T)
        ss_EP21[step] = np.mean(-inh_scal_mask*delta_t*(1 / tau_scaling) * (EP210.T * (1 - ratio_E2)).T)
        ss_ES21[step] = np.mean(inh_scal_mask*delta_t*(1 / tau_scaling) * (ES210.T * (1 - ratio_E2)).T)

        J_EE11[step] = np.mean(EE110); J_EE12[step] = np.mean(EE120); J_EE21[step] = np.mean(EE210); J_EE22[step] = np.mean(EE220)
        J_EP11[step] = np.mean(EP110); J_EP12[step] = np.mean(EP120); J_EP21[step] = np.mean(EP210); J_EP22[step] = np.mean(EP220)
        J_ES11[step] = np.mean(ES110); J_ES12[step] = np.mean(ES120); J_ES21[step] = np.mean(ES210); J_ES22[step] = np.mean(ES220)
        av_theta_E1[step] = np.mean(theta_E1); av_theta_E2[step] = np.mean(theta_E2)

        # the initial condition of the pasticity threshold is defined right before the stimulus, when the system is at steady state
        if step == int(stim_start * (1 / delta_t)):
            (theta_E1, theta_E2) = (E01,E02)
            heb_plas_mask = hebbian_plasticity_flag
            exc_scal_mask = exc_scaling_flag
            inh_scal_mask = inh_scaling_flag
            BCM_mask = (heb_plas_mask or exc_scal_mask or inh_scal_mask) and BCM_flag
            stimulus = stim_strength
        elif step == int(stim_stop * (1 / delta_t)):
            stimulus = 0

        """indv_neurons[:20,step] = E01[:20]
        indv_neurons[20:40,step] = E02[:20]"""

        I_E1 = x_E - EP110 @ P01 - EP120 @ P02
        I_D1 = x_D - ES110 @ S01 - ES120 @ S02 + EE110 @ E01 + EE120 @ E02 + noise_E1[:,step] + stimulus
        I1 = lambda_D*I_D1 + (1-lambda_E)*I_E1

        I_E2 = x_E - EP210 @ P01 - EP220 @ P02
        I_D2 = x_D - ES210 @ S01 - ES220 @ S02 + EE220 @ E02 + EE210 @ E01 + noise_E2[:,step]
        I2 = lambda_D*I_D2 + (1-lambda_E)*I_E2

        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - theta_E_default) )
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - theta_E_default) )

        P1 = P01 + delta_t*(1/tau_P)*( -P01 + w_PE11 @ E01 + w_PE12 @ E02 - w_PS11 @ S01 - w_PS12 @ S02 - w_PP11 @ P01 - w_PP12 @ P02 + x_P + .25*stimulus + noise_P1[:,step])
        P2 = P02 + delta_t*(1/tau_P)*( -P02 + w_PE21 @ E01 + w_PE22 @ E02 - w_PS21 @ S01 - w_PS22 @ S02 - w_PP21 @ P01 - w_PP22 @ P02 + x_P + noise_P2[:,step])

        S1 = S01 + delta_t*(1/tau_S)*( -S01 + w_SE11 @ E01 + w_SE12 @ E02 + x_S  + noise_S1[:,step])
        S2 = S02 + delta_t*(1/tau_S)*( -S02 + w_SE21 @ E01 + w_SE22 @ E02 + x_S  + noise_S2[:,step])

        theta_E1 = theta_E1 + BCM_mask*delta_t * (1 / tau_theta) * (E01 - theta_E1)
        theta_E2 = theta_E2 + BCM_mask*delta_t * (1 / tau_theta) * (E02 - theta_E2)

        # in order to avoid zero division
        theta_E1[theta_E1 == 0] = 1e-323; theta_E2[theta_E2 == 0] = 1e-323
        E01[E01 == 0] = 1e-323; E02[E02 == 0] = 1e-323

        # round the plasticity mechanisms to activate the plasticity when there is significant change
        round_array(E01 - theta_E1, 2, dif_E1); round_array(E02 - theta_E2, 2, dif_E2)
        round_array(E01 / theta_E1, 2, ratio_E1); round_array(E02 / theta_E2, 2, ratio_E2)

        EE11 = EE110 + heb_plas_mask*delta_t*(1 / tau_plas) * dif_E1.reshape(N_PC,1) @ E01.reshape(1,N_PC) + exc_scal_mask*delta_t*(1/tau_scaling) * (EE110.T * (1 - ratio_E1)).T
        EE12 = EE120 + heb_plas_mask*delta_t*(1 / tau_plas) * dif_E1.reshape(N_PC,1) @ E02.reshape(1,N_PC) + exc_scal_mask*delta_t*(1/tau_scaling) * (EE120.T * (1 - ratio_E1)).T
        EE21 = EE210 + heb_plas_mask*delta_t*(1 / tau_plas) * dif_E2.reshape(N_PC,1) @ E01.reshape(1,N_PC) + exc_scal_mask*delta_t*(1/tau_scaling) * (EE210.T * (1 - ratio_E2)).T
        EE22 = EE220 + heb_plas_mask*delta_t*(1 / tau_plas) * dif_E2.reshape(N_PC,1) @ E02.reshape(1,N_PC) + exc_scal_mask*delta_t*(1/tau_scaling) * (EE220.T * (1 - ratio_E2)).T

        EP11 = EP110 - inh_scal_mask*delta_t*(1 / tau_scaling) * (EP110.T * (1 - ratio_E1)).T
        EP12 = EP120 - inh_scal_mask*delta_t*(1 / tau_scaling) * (EP120.T * (1 - ratio_E1)).T
        EP21 = EP210 - inh_scal_mask*delta_t*(1 / tau_scaling) * (EP210.T * (1 - ratio_E2)).T
        EP22 = EP220 - inh_scal_mask*delta_t*(1 / tau_scaling) * (EP220.T * (1 - ratio_E2)).T

        ES11 = ES110 + inh_scal_mask*delta_t*(1 / tau_scaling) * (ES110.T * (1 - ratio_E1)).T
        ES12 = ES120 + inh_scal_mask*delta_t*(1 / tau_scaling) * (ES120.T * (1 - ratio_E1)).T
        ES21 = ES210 + inh_scal_mask*delta_t*(1 / tau_scaling) * (ES210.T * (1 - ratio_E2)).T
        ES22 = ES220 + inh_scal_mask*delta_t*(1 / tau_scaling) * (ES220.T * (1 - ratio_E2)).T

        """if not np.mod(step, 10000):
            print("I1", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I1 - theta)))
            print("I2", delta_t * (1 / tau_E) * np.mean(np.maximum(0, I2 - theta)))
            print("e1->p", delta_t * (1 / tau_I) * np.mean(w_PE11 @ E01))
            print("e2->p", delta_t * (1 / tau_I) * np.mean(w_PE12 @ E02))
            print("s->p", delta_t * (1 / tau_I) * np.mean(-w_PS11 @ S01))
            print("p->p", delta_t * (1 / tau_I) * np.mean(-w_PP11 @ P01))
            print("e1->s", delta_t * (1 / tau_I) * np.mean(w_SE11 @ E01))
            print("e2->s", delta_t * (1 / tau_I) * np.mean(w_SE22 @ E02))"""

        # rates and plasticity thresholds cannot go below 0
        E1[E1 < 0] = 0; E2[E2 < 0] = 0; P1[P1 < 0] = 0; P2[P2 < 0] = 0; S1[S1 < 0] = 0; S2[S2 < 0] = 0
        theta_E1[theta_E1 < 0] = 0; theta_E2[theta_E2 < 0] = 0

        # hard bounds are applied to the weights
        EE11 = apply_hard_bound(EE11,0,upper_bound);EE12 = apply_hard_bound(EE12,0,upper_bound)
        EE21 = apply_hard_bound(EE21,0,upper_bound);EE22 = apply_hard_bound(EE22,0,upper_bound)
        EP11 = apply_hard_bound(EP11,0,upper_bound);EP12 = apply_hard_bound(EP12,0,upper_bound)
        EP21 = apply_hard_bound(EP21,0,upper_bound);EP22 = apply_hard_bound(EP22,0,upper_bound)
        ES11 = apply_hard_bound(ES11,0,upper_bound);ES12 = apply_hard_bound(ES12,0,upper_bound)
        ES21 = apply_hard_bound(ES21,0,upper_bound);ES22 = apply_hard_bound(ES22,0,upper_bound)

        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # counter is updated
        step=step+1


