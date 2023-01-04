
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

#@jit((float16[:], float16[:], float16[:], float16[:], int16, float[:]),nopython=True)
@jit(nopython=True)
def model_plasticity_based_on_current_all_mass(delta_t, vars, plas_terms,t, weights, back_inputs,
                                stim_strengths, stim_start, stim_stop, taus, lambdas, rheobases, upper_bounds,
                                learning_rates, adaptive_LR_method = "sum", synaptic_scaling_method = "subtractive",
                                synaptic_scaling_update_method = "every_timestep",
                                synaptic_scaling_compare_method = "individual",
                                BCM_p=1, beta_ss=1, ss_exponential=1,flags=(0,0,0,0,0)):
    (rE1, rE2, rP1, rP2, rS1, rS2, av_I1, av_I2,
     J_EE11, J_EE12, J_EE21, J_EE22, J_EP11, J_EP12, J_EP21, J_EP22, J_ES11, J_ES12, J_ES21, J_ES22) = vars
    (hebEE11, hebEE12, hebEE21, hebEE22,
     ss1_list, ss2_list, av_theta_I1, av_theta_I2,
     LR_EE11,LR_EE12,LR_EE21,LR_EE22) = plas_terms
    theta_I1,theta_I2 = 1,1
    (LR_E01,LR_E02) = learning_rates
    learning_rate_EE11,learning_rate_EE12,learning_rate_EE21,learning_rate_EE22=LR_E01,LR_E02,LR_E01,LR_E02

    (w_EE11, w_EE12, w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11,
     w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
     w_EE22, w_EE21, w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21,
     w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22) = weights
    (x_E, x_P, x_S) = back_inputs
    (stim_strength_E,stim_strength_P) = stim_strengths
    (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta, tau_LR) = taus
    (rheobase_E,rheobase_P,rheobase_S) = rheobases
    (upper_bound_E, upper_bound_P, upper_bound_S) = upper_bounds

    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag,
     adaptive_threshold_flag, adaptive_LR_flag)= flags

    # setting up initial conditions
    E01, E02, P01, P02, S01, S02 = 1,1,1,1,1,1
    EE110, EE120, EE210, EE220 = w_EE11, w_EE12, w_EE21, w_EE22
    EP110, EP120, EP210, EP220 = w_EP11, w_EP12, w_EP21, w_EP22
    ES110, ES120, ES210, ES220 = w_DS11, w_DS12, w_DS21, w_DS22
    E1, E2 = 0,0
    stimulus_E, stimulus_P = 0, 0
    heb_plas_mask = 0; exc_scal_mask = 0; inh_scal_mask = 0; adaptive_threshold_mask = 0; adaptive_LR_mask = 0

    heb_term_EE11, heb_term_EE12, heb_term_EE21, heb_term_EE22 = 0, 0, 0, 0
    initial_sum_of_EE_weights = w_EE11+w_EE12+w_EE21+w_EE22
    flag_every_timestep = 0
    flag_threshold_exceeded = 0
    flag_at_every_x_second = 0

    step = 0
    for i in t:
        # the initial condition of the pasticity threshold is defined right before the stimulus, when the system is at steady state
        if step == int(stim_start * (1 / delta_t)):
            (theta_I1, theta_I2) = (E1,E2)
            initial_theta_mean = (theta_I1 + theta_I2) / 2
            if synaptic_scaling_update_method == "every_timestep":
                flag_every_timestep = 1
            elif synaptic_scaling_update_method == "threshold_exceeded":
                flag_threshold_exceeded = 1
            elif synaptic_scaling_update_method == "at_every_x_second":
                flag_at_every_x_second = 1
            heb_plas_mask = hebbian_plasticity_flag
            exc_scal_mask = exc_scaling_flag
            inh_scal_mask = inh_scaling_flag
            adaptive_threshold_mask = (heb_plas_mask or exc_scal_mask or inh_scal_mask) and adaptive_threshold_flag
            adaptive_LR_mask = heb_plas_mask and adaptive_LR_flag
            stimulus_E = stim_strength_E
            stimulus_P = stim_strength_P
            print("Stimulus started.")
        elif step == int(stim_stop * (1 / delta_t)):
            if adaptive_LR_method == "3-factor":
                learning_rate_EE11 = 0
                learning_rate_EE12 = 0
                learning_rate_EE21 = 0
                learning_rate_EE22 = 0
            stimulus_E, stimulus_P = 0, 0
            print("Stimulus ended.")

        #if step > int(stim_start * (1 / delta_t))-5:
            #breakpoint()

        I1 = x_E - EP110 * P01 - EP120 * P02 - ES110 * S01 - ES120 * S02 + EE110 * E01 + EE120 * E02 + stimulus_E
        I2 = x_E - EP210 * P01 - EP220 * P02 - ES210 * S01 - ES220 * S02 + EE210 * E01 + EE220 * E02

        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - rheobase_E))
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - rheobase_E))

        P1 = P01 + delta_t*(1/tau_P)*( -P01 + np.maximum(0, w_PE11 * E01 + w_PE12 * E02 - w_PS11 * S01 - w_PS12 * S02
                                                         - w_PP11 * P01 - w_PP12 * P02 + x_P - rheobase_P + stimulus_P))
        P2 = P02 + delta_t*(1/tau_P)*( -P02 + np.maximum(0, w_PE21 * E01 + w_PE22 * E02 - w_PS21 * S01 - w_PS22 * S02
                                                         - w_PP21 * P01 - w_PP22 * P02 + x_P - rheobase_P))

        S1 = S01 + delta_t*(1/tau_S)*( -S01 + np.maximum(0, w_SE11 * E01 + w_SE12 * E02 + x_S - rheobase_S))
        S2 = S02 + delta_t*(1/tau_S)*( -S02 + np.maximum(0, w_SE21 * E01 + w_SE22 * E02 + x_S - rheobase_S))

        theta_mean = (theta_I1+theta_I2)/2
        theta_I1 = theta_I1 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * (-(theta_mean - initial_theta_mean) + (E1 - theta_I1))
        theta_I2 = theta_I2 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * (-(theta_mean - initial_theta_mean) + (E2 - theta_I2))

        """theta_I1 = theta_I1 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * (E1**BCM_p - theta_I1)
        theta_I2 = theta_I2 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * (E2**BCM_p - theta_I2)"""

        # rates and plasticity thresholds cannot go below 0 (exc ones cannot go below 1e-323, in order to avoid zero division)
        theta_I1=max(theta_I1,1e-323); theta_I2=max(theta_I2, 1e-323)
        E1=max(E1, 1e-323); E2=max(E2, 1e-323); P1 = max(0, P1); P2 = max(0, P2); S1 = max(0, S1); S2 = max(0, S2)

        if heb_plas_mask:
            heb_term_EE11 = learning_rate_EE11 * delta_t * (1 / tau_plas) * ((E1 - theta_I1) * E1)
            heb_term_EE12 = learning_rate_EE12 * delta_t * (1 / tau_plas) * ((E1 - theta_I1) * E2)
            heb_term_EE21 = learning_rate_EE21 * delta_t * (1 / tau_plas) * ((E2 - theta_I2) * E1)
            heb_term_EE22 = learning_rate_EE22 * delta_t * (1 / tau_plas) * ((E2 - theta_I2) * E2)

        """if heb_term_EE21<0:
            print(heb_term_EE21,step)
        if heb_term_EE22<0:
            print(heb_term_EE22,step)"""

        sum_of_EE_weights = EE110+EE120+EE210+EE220
        apply_ss = ((sum_of_EE_weights > initial_sum_of_EE_weights * (1 + beta_ss)) or
                    (sum_of_EE_weights < initial_sum_of_EE_weights * (1 - beta_ss)))

        if synaptic_scaling_compare_method == "individual":
            dif_E1   = E1 - theta_I1; dif_E2   = E2 - theta_I2
            ratio_E1 = E1 / theta_I1; ratio_E2 = E2 / theta_I2
        elif synaptic_scaling_compare_method == "all":
            dif_E1 = sum_of_EE_weights - initial_sum_of_EE_weights; ratio_E1 = sum_of_EE_weights/initial_sum_of_EE_weights
            dif_E2 = sum_of_EE_weights - initial_sum_of_EE_weights; ratio_E2 = sum_of_EE_weights/initial_sum_of_EE_weights

        if synaptic_scaling_method == "multiplicative":
            if (flag_every_timestep or (flag_threshold_exceeded and apply_ss)):  # or (flag_at_every_x_second and not np.mod(i,1000))):
                ss1 = exc_scal_mask*delta_t * (1 / tau_scaling) * (1 - ratio_E1**ss_exponential)
                ss2 = exc_scal_mask*delta_t * (1 / tau_scaling) * (1 - ratio_E2**ss_exponential)
            else:
                ss1, ss2 = 0, 0

            EE110 = (1+ss1)*EE110
            EE120 = (1+ss1)*EE120
            EE210 = (1+ss2)*EE210
            EE220 = (1+ss2)*EE220

            EP11 = (1-ss1)*EP110
            EP12 = (1-ss1)*EP120
            EP21 = (1-ss2)*EP210
            EP22 = (1-ss2)*EP220

            ES11 = (1+ss1)*ES110
            ES12 = (1+ss1)*ES120
            ES21 = (1+ss2)*ES210
            ES22 = (1+ss2)*ES220

        elif synaptic_scaling_method == "subtractive":
            if (flag_every_timestep or (flag_threshold_exceeded and apply_ss)):# or (flag_at_every_x_second and not np.mod(i,1000))):
                ss1 = exc_scal_mask * delta_t * (1 / tau_scaling) * (1 - ratio_E1**ss_exponential)
                ss2 = exc_scal_mask * delta_t * (1 / tau_scaling) * (1 - ratio_E2**ss_exponential)
            else:
                ss1, ss2 = 0, 0

            EE110 = EE110 + ss1
            EE120 = EE120 + ss1
            EE210 = EE210 + ss2
            EE220 = EE220 + ss2

            EP11 = EP110 - ss1
            EP12 = EP120 - ss1
            EP21 = EP210 - ss2
            EP22 = EP220 - ss2

            ES11 = ES110 + ss1
            ES12 = ES120 + ss1
            ES21 = ES210 + ss2
            ES22 = ES220 + ss2

        elif synaptic_scaling_method == "subtractive_dif":
            if (flag_every_timestep or (flag_threshold_exceeded and apply_ss)):# or (flag_at_every_x_second and not np.mod(i,1000))):
                ss1 = exc_scal_mask * delta_t * (1 / tau_scaling) * dif_E1
                ss2 = exc_scal_mask * delta_t * (1 / tau_scaling) * dif_E2
            else:
                ss1, ss2 = 0, 0

            EE110 = EE110 - ss1
            EE120 = EE120 - ss1
            EE210 = EE210 - ss2
            EE220 = EE220 - ss2

            EP11 = EP110 + ss1
            EP12 = EP120 + ss1
            EP21 = EP210 + ss2
            EP22 = EP220 + ss2

            ES11 = ES110 - ss1
            ES12 = ES120 - ss1
            ES21 = ES210 - ss2
            ES22 = ES220 - ss2

        # in order to have hebbian plasticity in the absence of synaptic scaling, it is defined here
        EE11 = EE110 + heb_term_EE11
        EE12 = EE120 + heb_term_EE12
        EE21 = EE210 + heb_term_EE21
        EE22 = EE220 + heb_term_EE22

        if adaptive_LR_method == "ode":
            learning_rate_EE11 = learning_rate_EE11 - adaptive_LR_mask*delta_t * (1 / tau_LR) * heb_term_EE11
            learning_rate_EE12 = learning_rate_EE12 - adaptive_LR_mask*delta_t * (1 / tau_LR) * heb_term_EE12
            learning_rate_EE21 = learning_rate_EE21 - adaptive_LR_mask*delta_t * (1 / tau_LR) * heb_term_EE21
            learning_rate_EE22 = learning_rate_EE22 - adaptive_LR_mask*delta_t * (1 / tau_LR) * heb_term_EE22

        elif ((adaptive_LR_method == "sum") and (adaptive_LR_mask)):
            learning_rate_EE11 = learning_rate_EE11 / (tau_LR*np.abs(np.mean(hebEE11[step-1000:step]))+1)
            learning_rate_EE12 = learning_rate_EE12 / (tau_LR*np.abs(np.mean(hebEE12[step-1000:step]))+1)
            learning_rate_EE21 = learning_rate_EE21 / (tau_LR*np.abs(np.mean(hebEE21[step-1000:step]))+1)
            learning_rate_EE22 = learning_rate_EE22 / (tau_LR*np.abs(np.mean(hebEE22[step-1000:step]))+1)

            """learning_rate_EE11=learning_rate_EE11*(learning_rate_EE11>0.01)
            learning_rate_EE12=learning_rate_EE12*(learning_rate_EE12>0.01)
            learning_rate_EE21=learning_rate_EE21*(learning_rate_EE21>0.01)
            learning_rate_EE22=learning_rate_EE22*(learning_rate_EE22>0.01)"""

        elif (adaptive_LR_method == "decay_after_stim"):
            learning_rate_EE11 = learning_rate_EE11 + adaptive_LR_mask * (1/tau_LR) * delta_t * (-learning_rate_EE11)
            learning_rate_EE12 = learning_rate_EE11
            learning_rate_EE21 = learning_rate_EE11
            learning_rate_EE22 = learning_rate_EE11

        """learning_rate_EE11 = np.round(learning_rate_EE11, 3);learning_rate_EE12 = np.round(learning_rate_EE12, 3)
        learning_rate_EE21 = np.round(learning_rate_EE21, 3);learning_rate_EE22 = np.round(learning_rate_EE22, 3)"""


        # hard bounds are applied to the weights
        EE11 = max(0,min(EE11,upper_bound_E));EE12 = max(0,min(EE12,upper_bound_E))
        EE21 = max(0,min(EE21,upper_bound_E));EE22 = max(0,min(EE22,upper_bound_E))
        EP11 = max(0,min(EP11,upper_bound_P));EP12 = max(0,min(EP12,upper_bound_P))
        EP21 = max(0,min(EP21,upper_bound_P));EP22 = max(0,min(EP22,upper_bound_P))
        ES11 = max(0,min(ES11,upper_bound_S));ES12 = max(0,min(ES12,upper_bound_S))
        ES21 = max(0,min(ES21,upper_bound_S));ES22 = max(0,min(ES22,upper_bound_S))

        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # values are assigned to the lists
        rE1[step] = E01; rE2[step] = E02; rP1[step] = P01; rP2[step] = P02; rS1[step] = S01; rS2[step] = S02
        av_I1[step] = I1; av_I2[step] = I2
        J_EE11[step] = EE110; J_EE12[step] = EE120; J_EE21[step] = EE210; J_EE22[step] = EE220
        J_EP11[step] = EP110; J_EP12[step] = EP120; J_EP21[step] = EP210; J_EP22[step] = EP220
        J_ES11[step] = ES110; J_ES12[step] = ES120; J_ES21[step] = ES210; J_ES22[step] = ES220
        hebEE11[step] = heb_term_EE11; hebEE12[step] = heb_term_EE12; hebEE21[step] = heb_term_EE21; hebEE22[step] = heb_term_EE22
        ss1_list[step] = ss1; ss2_list[step] = ss2
        av_theta_I1[step] = theta_I1; av_theta_I2[step] = theta_I2
        LR_EE11[step] = learning_rate_EE11; LR_EE12[step] = learning_rate_EE12
        LR_EE21[step] = learning_rate_EE21; LR_EE22[step] = learning_rate_EE22

        """if np.mod(step, 1000)*rE1 > 1e2:
            print("Break since it goes to infinity.")
            quit()"""

        # counter is updated
        step=step+1

    print("E rates", E1, E2)
    print("weights EE ",EE11,EE12,EE21,EE22)
    print("weights EP ",EP11,EP12,EP21,EP22)
    print("weights ES ",ES11,ES12,ES21,ES22)
    print("final hebb terms ", heb_term_EE11,heb_term_EE12,heb_term_EE21,heb_term_EE22)
    print("min hebb terms ", np.min(hebEE11),np.min(hebEE12),np.min(hebEE21),np.min(hebEE22))
    print("min hebb terms ", np.argmin(hebEE11)*delta_t,np.argmin(hebEE12)*delta_t,
                             np.argmin(hebEE21)*delta_t,np.argmin(hebEE22)*delta_t)
    print("min ss terms ", np.min(ss1_list), np.argmin(ss1_list)*delta_t,
                           np.min(ss2_list), np.argmin(ss2_list)*delta_t)
    print("max ss terms ", np.max(ss1_list), np.argmax(ss1_list)*delta_t,
                           np.max(ss2_list), np.argmax(ss2_list)*delta_t)
    print("thetas", theta_I1, theta_I2)
    print("learning rates", learning_rate_EE11,learning_rate_EE12,learning_rate_EE21,learning_rate_EE22)