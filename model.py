

import numpy as np
from numba import cuda, jit
import matplotlib.pyplot as plt
from util import *


@jit(nopython=True)
def model(delta_t, hold_every, res_rates, res_weights, sim_duration, weights, back_inputs,
          stim_trains, stim_times, taus, beta_K, rheobases,
          flags=(0, 0, 0, 0, 0, 0), flags_theta = (1,1)):

    (hold_every_stimuli, hold_every_simulation) = hold_every
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights
    theta1,theta2 = 1,1
    learning_rate = 1

    (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii,
     w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij) = weights
    (g_E, g_P, g_S, g_top_down_to_S) = back_inputs
    (stim_train_E,stim_train_P, stim_train_S) = stim_trains
    (stim_start, stim_stop) = stim_times[0]
    (tau_E, tau_P, tau_S, tau_plas,
     tau_scaling_E, tau_scaling_P, tau_scaling_S,
     tau_theta, tau_beta) = taus
    (rheobase_E,rheobase_P,rheobase_S) = rheobases

    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag_p, inh_scaling_flag_s,
     adaptive_threshold_flag, adaptive_LR_flag)= flags
    (flag_theta_shift, flag_theta_local) = flags_theta

    # setting up initial conditions
    E01, E02, P01, P02, S01, S02 = 1,1,1,1,1,1 #10,10,10,10,10,10
    EE110, EE120, EE210, EE220 = w_EEii, w_EEij, w_EEij, w_EEii
    EP110, EP120, EP210, EP220 = w_EPii, w_EPij, w_EPij, w_EPii
    ES110, ES120, ES210, ES220 = w_ESii, w_ESij, w_ESij, w_ESii
    E1, E2 = 0,0
    max_E[0] = 0
    stimulus_E1, stimulus_P1, stimulus_S1 = 0, 0, 0
    stimulus_E2, stimulus_P2, stimulus_S2 = 0, 0, 0
    heb_plas_mask = 0; exc_scal_mask = 0; inh_scal_mask_p = 0; inh_scal_mask_s = 0
    adaptive_threshold_mask = 0; adaptive_LR_mask = 0

    heb_term_EE11, heb_term_EE12, heb_term_EE21, heb_term_EE22 = 0, 0, 0, 0
    initial_theta_mean = 0
    beta1 = 1
    beta2 = 1

    # info-holder counter (counter) and index (idx) to fill the arrays. np.mod doesn't
    # work in numba, thus we need counter to hold data at every "hold_every" step
    phase1 = 0
    phase3 = 0
    counter1, counter2 , counter3 = 0, 0, 0
    i_1, i_2, i_3 = 0, 0, 0
    stim_applied = 0

    for step in range(sim_duration):
        if step == int((stim_start + 2) * (1 / delta_t)): # if stim on
            if stim_applied == 0: # if first stim
                (theta1, theta2) = (E1,E2)
                initial_theta_mean = (theta1 + theta2) / 2
                beta1 = initial_theta_mean - beta_K
                beta2 = initial_theta_mean - beta_K

                heb_plas_mask = hebbian_plasticity_flag
                exc_scal_mask = exc_scaling_flag
                inh_scal_mask_p = inh_scaling_flag_p
                inh_scal_mask_s = inh_scaling_flag_s

                adaptive_threshold_mask = (heb_plas_mask or exc_scal_mask or inh_scal_mask_p) and adaptive_threshold_flag
                adaptive_LR_mask = heb_plas_mask and adaptive_LR_flag
                if adaptive_LR_mask:
                    learning_rate = 1

            if stim_applied == 1:
                counter2 = hold_every_simulation + 5  # stop the data-holder counter

            stimulus_E1, stimulus_E2 = stim_train_E[stim_applied]
            stimulus_P1, stimulus_P2 = stim_train_P[stim_applied]
            stimulus_S1, stimulus_S2 = stim_train_S[stim_applied]

            # if there is more than 1 stimulation, it increases the index number in the stim_times list
            stim_applied = stim_applied + 1 # increase the no stim applied

        if step == int((stim_stop + 2)*(1/delta_t)): # if stim off
            g_S = 3 + g_top_down_to_S

            if stim_applied == 1:
                counter2 = hold_every_simulation  # start the data-holder counter

            # hebbian is turned off for the testing
            if adaptive_LR_mask:
                learning_rate = 0

            stimulus_E1, stimulus_E2 = 0, 0
            stimulus_P1, stimulus_P2 = 0, 0
            stimulus_S1, stimulus_S2 = 0, 0

            if stim_times.shape[0] > stim_applied: # set the new timing for the next stim if exists
                (stim_start, stim_stop) = stim_times[stim_applied]

        # setting the counters for phase 1 and 3 with 5 seconds of margin before and after stimulation
        if step == int(2*(1/delta_t)):
            counter1 = hold_every_stimuli  # start the data-holder counter1
            phase1 = 1
        elif step == int((stim_times[0][1] + 5 + 2) * (1 / delta_t)):
            phase1 = 0

        elif step == int((stim_times[1][0] - 5 + 2) * (1 / delta_t)):
            counter3 = hold_every_stimuli  # start the data-holder counter3
            phase3 = 1
        elif step == int((stim_times[1][1] + 5 + 2) * (1 / delta_t)):
            phase3 = 0


        # values are assigned to the lists
        if phase1 and counter1 == hold_every_stimuli:
            r_phase1[:,i_1] = [E01, E02, P01, P02, S01, S02]
            J_EE_phase1[:,i_1] = [EE110, EE120, EE210, EE220]

            i_1 = i_1 + 1
            counter1 = 0  # restart

        elif phase3 and counter3 == hold_every_stimuli:
            r_phase3[:,i_3] = [E01, E02, P01, P02, S01, S02]

            i_3 = i_3 + 1
            counter3 = 0  # restart

        if stim_applied == 1 and counter2 == hold_every_simulation:
            r_phase2[:,i_2] = [E01, E02, P01, P02, S01, S02, theta1, theta2, beta1, beta2]
            J_phase2[:,i_2] = [EE110, EE120, EE210, EE220, EP110, EP120, EP210, EP220, ES110, ES120, ES210, ES220]

            i_2 = i_2 + 1
            counter2 = 0  # restart

        if E01 > max_E[0]:
            max_E[0] = E01

        # if the system explodes, stop the simulation
        if E01 > 100:
            break

        I1 = g_E - EP110 * P01 - EP120 * P02 - ES110 * S01 - ES120 * S02 + EE110 * E01 + EE120 * E02 + stimulus_E1
        I2 = g_E - EP210 * P01 - EP220 * P02 - ES210 * S01 - ES220 * S02 + EE210 * E01 + EE220 * E02 + stimulus_E2

        E1 = E01 + delta_t*(1/tau_E)*(-E01 + np.maximum(0,I1 - rheobase_E))
        E2 = E02 + delta_t*(1/tau_E)*(-E02 + np.maximum(0,I2 - rheobase_E))

        P1 = P01 + delta_t*(1/tau_P)*(-P01 + np.maximum(0, w_PEii * E01 + w_PEij * E02 - w_PSii * S01 - w_PSij * S02
                                                         -w_PPii * P01 - w_PPij * P02 + g_P - rheobase_P + stimulus_P1))
        P2 = P02 + delta_t*(1/tau_P)*(-P02 + np.maximum(0, w_PEij * E01 + w_PEii * E02 - w_PSij * S01 - w_PSii * S02
                                                         -w_PPij * P01 - w_PPii * P02 + g_P - rheobase_P + stimulus_P2))

        S1 = S01 + delta_t*(1/tau_S)*(-S01 + np.maximum(0, w_SEii * E01 + w_SEij * E02 + g_S - rheobase_S + stimulus_S1))
        S2 = S02 + delta_t*(1/tau_S)*(-S02 + np.maximum(0, w_SEij * E01 + w_SEii * E02 + g_S - rheobase_S + stimulus_S2))


        beta1 = beta1 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E1 - beta1)
        beta2 = beta2 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E2 - beta2)

        theta1 = theta1 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(theta1 - beta1) + flag_theta_local*(E1 - theta1))
        theta2 = theta2 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(theta2 - beta2) + flag_theta_local*(E2 - theta2))

        # rates and plasticity thresholds cannot go below 0 (boundary set to 1e-10 in order to avoid very small numbers
        # leading rE/theta to a very large value)
        beta1=max(beta1,1e-10); beta2=max(beta2, 1e-10)
        theta1=max(theta1,1e-10); theta2=max(theta2, 1e-10)
        E1 = max(E1, 1e-10); E2 = max(E2, 1e-10)
        P1 = max(P1, 1e-10); P2 = max(P2, 1e-10)
        S1 = max(S1, 1e-10); S2 = max(S2, 1e-10)

        if heb_plas_mask:
            heb_term_EE11 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E1)
            heb_term_EE12 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E2)
            heb_term_EE21 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E1)
            heb_term_EE22 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E2)

        # preventing ratios getting very large when thetas approach to zero
        ratio_E1 = max(E1, 1e-2) / max(theta1,1e-2); ratio_E2 = max(E2, 1e-2) / max(theta2,1e-2)
        p_e = 1; p_p = 1; p_s = 1

        ss1_e = exc_scal_mask * delta_t * (1 / tau_scaling_E) * ((1 - ratio_E1)**p_e)
        ss2_e = exc_scal_mask * delta_t * (1 / tau_scaling_E) * ((1 - ratio_E2)**p_e)

        ss1_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * ((1 - ratio_E1)**p_p)
        ss2_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * ((1 - ratio_E2)**p_p)

        ss1_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * ((1 - ratio_E1)**p_s)
        ss2_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * ((1 - ratio_E2)**p_s)

        EE110 = EE110 + ss1_e*EE110
        EE120 = EE120 + ss1_e*EE120
        EE210 = EE210 + ss2_e*EE210
        EE220 = EE220 + ss2_e*EE220

        EP11 = EP110 - ss1_p*EP110
        EP12 = EP120 - ss1_p*EP120
        EP21 = EP210 - ss2_p*EP210
        EP22 = EP220 - ss2_p*EP220

        ES11 = ES110 + ss1_s*ES110
        ES12 = ES120 + ss1_s*ES120
        ES21 = ES210 + ss2_s*ES210
        ES22 = ES220 + ss2_s*ES220

        # in order to have hebbian plasticity in the absence of synaptic scaling, it is defined here
        EE11 = EE110 + heb_term_EE11
        EE12 = EE120 + heb_term_EE12
        EE21 = EE210 + heb_term_EE21
        EE22 = EE220 + heb_term_EE22

        # lower bond is applied to the weights
        EE11 = max(0,EE11);EE12 = max(0,EE12)
        EE21 = max(0,EE21);EE22 = max(0,EE22)
        EP11 = max(0,EP11);EP12 = max(0,EP12)
        EP21 = max(0,EP21);EP22 = max(0,EP22)
        ES11 = max(0,ES11);ES12 = max(0,ES12)
        ES21 = max(0,ES21);ES22 = max(0,ES22)

        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # update the data-holder counters
        counter1 = counter1 + 1; counter2 = counter2 + 1; counter3 = counter3 + 1



@jit(nopython=True)
def model_VIP(delta_t, hold_every, res_rates, res_weights, sim_duration, weights,back_inputs,
          stim_trains, stim_times, taus, alpha, beta_K, rheobases,
          flags=(0, 0, 0, 0, 0, 0), flags_theta = (1,1)):

    (hold_every_stimuli, hold_every_simulation) = hold_every
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_phase1, J_phase2) = res_weights
    theta1,theta2 = 1,1
    learning_rate = 1

    (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii, w_SVii, w_VEii, w_VPii, w_VSii,
     w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij, w_SVij, w_VEij, w_VPij, w_VSij) = weights
    (g_E, g_P, g_S, x_V) = back_inputs
    (stim_train_E,stim_train_P, stim_train_S) = stim_trains
    (stim_start, stim_stop) = stim_times[0]
    (tau_E, tau_P, tau_S, tau_plas,
     tau_scaling_E, tau_scaling_P, tau_scaling_S,
     tau_theta, tau_beta) = taus
    (rheobase_E,rheobase_P,rheobase_S) = rheobases

    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag_p, inh_scaling_flag_s,
     adaptive_threshold_flag, adaptive_LR_flag)= flags
    (flag_theta_shift, flag_theta_local) = flags_theta

    # setting up initial conditions
    E01, E02, P01, P02, S01, S02, V01, V02 = 1,1,1,1,1,1,1,1 #10,10,10,10,10,10
    EE110, EE120, EE210, EE220 = w_EEii, w_EEij, w_EEij, w_EEii
    EP110, EP120, EP210, EP220 = w_EPii, w_EPij, w_EPij, w_EPii
    ES110, ES120, ES210, ES220 = w_ESii, w_ESij, w_ESij, w_ESii
    E1, E2 = 0,0
    max_E[0] = 0
    stimulus_E1, stimulus_P1, stimulus_S1 = 0, 0, 0
    stimulus_E2, stimulus_P2, stimulus_S2 = 0, 0, 0
    heb_plas_mask = 0; exc_scal_mask = 0; inh_scal_mask_p = 0; inh_scal_mask_s = 0
    adaptive_threshold_mask = 0; adaptive_LR_mask = 0

    heb_term_EE11, heb_term_EE12, heb_term_EE21, heb_term_EE22 = 0, 0, 0, 0
    initial_theta_mean = 0
    beta1 = 1
    beta2 = 1

    # info-holder counter (counter) and index (idx) to fill the arrays. np.mod doesn't
    # work in numba, thus we need counter to hold data at every "hold_every" step
    phase1 = 0
    phase3 = 0
    counter1, counter2 , counter3 = 0, 0, 0
    i_1, i_2, i_3 = 0, 0, 0
    stim_applied = 0

    for step in range(sim_duration):
        if step == int((stim_start + 2) * (1 / delta_t)): # if stim on
            if stim_applied == 0: # if first stim
                (theta1, theta2) = (E1,E2)
                initial_theta_mean = (theta1 + theta2) / 2

                heb_plas_mask = hebbian_plasticity_flag
                exc_scal_mask = exc_scaling_flag
                inh_scal_mask_p = inh_scaling_flag_p
                inh_scal_mask_s = inh_scaling_flag_s

                adaptive_threshold_mask = (heb_plas_mask or exc_scal_mask or inh_scal_mask_p) and adaptive_threshold_flag
                adaptive_LR_mask = heb_plas_mask and adaptive_LR_flag
                if adaptive_LR_mask:
                    learning_rate = 1

            if stim_applied == 1:
                counter2 = hold_every_simulation + 5  # stop the data-holder counter

            stimulus_E1, stimulus_E2 = stim_train_E[stim_applied]
            stimulus_P1, stimulus_P2 = stim_train_P[stim_applied]
            stimulus_S1, stimulus_S2 = stim_train_S[stim_applied]

            # if there is more than 1 stimulation, it increases the index number in the stim_times list
            stim_applied = stim_applied + 1 # increase the no stim applied

        if step == int((stim_stop + 2)*(1/delta_t)): # if stim off
            if stim_applied == 1:
                counter2 = hold_every_simulation  # start the data-holder counter
                beta1 = initial_theta_mean - beta_K
                beta2 = initial_theta_mean - beta_K

            # hebbian is turned off for the testing
            if adaptive_LR_mask:
                learning_rate = 0

            stimulus_E1, stimulus_E2 = 0, 0
            stimulus_P1, stimulus_P2 = 0, 0
            stimulus_S1, stimulus_S2 = 0, 0

            if stim_times.shape[0] > stim_applied: # set the new timing for the next stim if exists
                (stim_start, stim_stop) = stim_times[stim_applied]

        # setting the counters for phase 1 and 3
        if step == int(2*(1/delta_t)):
            counter1 = hold_every_stimuli  # start the data-holder counter1
            phase1 = 1
        elif step == int((stim_times[0][1] + 5 + 2) * (1 / delta_t)):
            phase1 = 0

        elif step == int((stim_times[1][0] - 5 + 2) * (1 / delta_t)):
            counter3 = hold_every_stimuli  # start the data-holder counter3
            phase3 = 1
        elif step == int((stim_times[1][1] + 5 + 2) * (1 / delta_t)):
            phase3 = 0


        # values are assigned to the lists
        if phase1 and counter1 == hold_every_stimuli:
            r_phase1[:,i_1] = [E01, E02, P01, P02, S01, S02, V01, V02]
            J_phase1[:,i_1] = [EE110, EE120, EE210, EE220, EP110, EP120, EP210, EP220, ES110, ES120, ES210, ES220]

            i_1 = i_1 + 1
            counter1 = 0  # restart

        elif phase3 and counter3 == hold_every_stimuli:
            r_phase3[:,i_3] = [E01, E02, P01, P02, S01, S02, V01, V02]

            i_3 = i_3 + 1
            counter3 = 0  # restart

        if stim_applied == 1 and counter2 == hold_every_simulation:
            r_phase2[:,i_2] = [E01, E02, P01, P02, S01, S02, V01, V02]
            J_phase2[:,i_2] = [EE110, EE120, EE210, EE220, EP110, EP120, EP210, EP220, ES110, ES120, ES210, ES220]

            i_2 = i_2 + 1
            counter2 = 0  # restart

        if E01 > max_E[0]:
            max_E[0] = E01

        # if the system explodes, stop the simulation
        if E01 > 1000:
            break

        I1 = g_E - EP110 * P01 - EP120 * P02 - ES110 * S01 - ES120 * S02 + EE110 * E01 + EE120 * E02 + stimulus_E1
        I2 = g_E - EP210 * P01 - EP220 * P02 - ES210 * S01 - ES220 * S02 + EE210 * E01 + EE220 * E02 + stimulus_E2

        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - rheobase_E))
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - rheobase_E))

        P1 = P01 + delta_t*(1/tau_P)*( -P01 + np.maximum(0, w_PEii * E01 + w_PEij * E02 - w_PSii * S01 - w_PSij * S02
                                                         - w_PPii * P01 - w_PPij * P02 + g_P - rheobase_P + stimulus_P1))
        P2 = P02 + delta_t*(1/tau_P)*( -P02 + np.maximum(0, w_PEij * E01 + w_PEii * E02 - w_PSij * S01 - w_PSii * S02
                                                         - w_PPij * P01 - w_PPii * P02 + g_P - rheobase_P + stimulus_P2))

        S1 = S01 + delta_t*(1/tau_S)*( -S01 + np.maximum(0, w_SEii * E01 + w_SEij * E02 + w_SVii * V01 + w_SVij * V02 + g_S - rheobase_S + stimulus_S1))
        S2 = S02 + delta_t*(1/tau_S)*( -S02 + np.maximum(0, w_SEij * E01 + w_SEii * E02 + w_SVij * V01 + w_SVii * V02 + g_S - rheobase_S + stimulus_S2))

        V1 = V01 + delta_t * (1 / tau_S) * (-V01 + np.maximum(0, w_VEii * E01 + w_VEij * E02 - w_VSii * S01 - w_VSij * S02
                                      - w_VPii * P01 - w_VPij * P02 + x_V - rheobase_P))
        V2 = V02 + delta_t * (1 / tau_S) * (-V02 + np.maximum(0, w_VEij * E01 + w_VEii * E02 - w_VSij * S01 - w_VSii * S02
                                      - w_VPij * P01 - w_VPii * P02 + x_V - rheobase_P))

        beta1 = beta1 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E1 - beta1)
        beta2 = beta2 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E2 - beta2)

        theta1 = theta1 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(1-alpha)*(theta1 - beta1) + flag_theta_local*alpha*(E1 - theta1))
        theta2 = theta2 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(1-alpha)*(theta2 - beta2) + flag_theta_local*alpha*(E2 - theta2))

        # rates and plasticity thresholds cannot go below 0 (exc ones cannot go below 1e-323, in order to avoid zero division)
        theta1=max(theta1,1e-10); theta2=max(theta2, 1e-10)
        E1 = max(E1, 1e-10); E2 = max(E2, 1e-10)
        P1 = max(P1, 1e-10); P2 = max(P2, 1e-10)
        S1 = max(S1, 1e-10); S2 = max(S2, 1e-10)
        V1 = max(V1, 1e-10); V2 = max(V2, 1e-10)

        if heb_plas_mask:
            heb_term_EE11 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E1)
            heb_term_EE12 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E2)
            heb_term_EE21 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E1)
            heb_term_EE22 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E2)

        # preventing ratios getting very large when thetas approach to zero
        ratio_E1 = max(E1, 1e-2) / max(theta1,1e-2); ratio_E2 = max(E2, 1e-2) / max(theta2,1e-2)

        ss1 = exc_scal_mask*delta_t * (1 / tau_scaling_E) * (1 - ratio_E1)
        ss2 = exc_scal_mask*delta_t * (1 / tau_scaling_E) * (1 - ratio_E2)

        ss1_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * (1 - ratio_E1)
        ss2_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * (1 - ratio_E2)

        ss1_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * (1 - ratio_E1)
        ss2_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * (1 - ratio_E2)

        EE110 = EE110 + ss1*EE110
        EE120 = EE120 + ss1*EE120
        EE210 = EE210 + ss2*EE210
        EE220 = EE220 + ss2*EE220

        EP11 = EP110 - ss1_p*EP110
        EP12 = EP120 - ss1_p*EP120
        EP21 = EP210 - ss2_p*EP210
        EP22 = EP220 - ss2_p*EP220

        ES11 = ES110 + ss1_s*ES110
        ES12 = ES120 + ss1_s*ES120
        ES21 = ES210 + ss2_s*ES210
        ES22 = ES220 + ss2_s*ES220

        # in order to have hebbian plasticity in the absence of synaptic scaling, it is defined here
        EE11 = EE110 + heb_term_EE11
        EE12 = EE120 + heb_term_EE12
        EE21 = EE210 + heb_term_EE21
        EE22 = EE220 + heb_term_EE22

        # lower bond is applied to the weights
        EE11 = max(0,EE11);EE12 = max(0,EE12)
        EE21 = max(0,EE21);EE22 = max(0,EE22)
        EP11 = max(0,EP11);EP12 = max(0,EP12)
        EP21 = max(0,EP21);EP22 = max(0,EP22)
        ES11 = max(0,ES11);ES12 = max(0,ES12)
        ES21 = max(0,ES21);ES22 = max(0,ES22)

        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2; V01 = V1; V02 = V2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # update the data-holder counters
        counter1 = counter1 + 1; counter2 = counter2 + 1; counter3 = counter3 + 1


@jit(nopython=True)
def old_model(delta_t, hold_every, vars, plas_terms, sim_duration, weights, back_inputs,
          stim_trains, stim_times, taus, alpha, beta_K, rheobases, flags, flags_theta=(1,1)):
    (rE1, rE2, rP1, rP2, rS1, rS2,
     J_EE11, J_EE12, J_EE21, J_EE22, J_EP11, J_EP12, J_EP21, J_EP22, J_ES11, J_ES12, J_ES21, J_ES22) = vars
    (hebEE11, hebEE12, hebEE21, hebEE22, ss1_list, ss2_list,
     av_theta1, av_theta2, beta1_list, beta2_list) = plas_terms
    theta1,theta2 = 1,1
    learning_rate = 1

    (w_EE11, w_EE12, w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11,
     w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
     w_EE22, w_EE21, w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21,
     w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22) = weights
    (g_E, g_P, g_S) = back_inputs
    (stim_train_E,stim_train_P, stim_train_S) = stim_trains
    (stim_start, stim_stop) = stim_times[0]
    stim_applied = 0
    (tau_E, tau_P, tau_S, tau_plas,
     tau_scaling_E, tau_scaling_P, tau_scaling_S,
     tau_theta, tau_LR, tau_beta) = taus
    (rheobase_E,rheobase_P,rheobase_S) = rheobases

    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag_p, inh_scaling_flag_s,
     adaptive_threshold_flag, adaptive_LR_flag) = flags

    (flag_theta_shift, flag_theta_local) = flags_theta

    # setting up initial conditions
    E01, E02, P01, P02, S01, S02 = 1,1,1,1,1,1
    EE110, EE120, EE210, EE220 = w_EE11, w_EE12, w_EE21, w_EE22
    EP110, EP120, EP210, EP220 = w_EP11, w_EP12, w_EP21, w_EP22
    ES110, ES120, ES210, ES220 = w_DS11, w_DS12, w_DS21, w_DS22
    E1, E2 = 0,0
    stimulus_E1, stimulus_P1, stimulus_S1 = 0, 0, 0
    stimulus_E2, stimulus_P2, stimulus_S2 = 0, 0, 0
    heb_plas_mask = 0; exc_scal_mask = 0; inh_scal_mask_p = 0; inh_scal_mask_s = 0;
    adaptive_threshold_mask = 0; adaptive_LR_mask = 0

    heb_term_EE11, heb_term_EE12, heb_term_EE21, heb_term_EE22 = 0, 0, 0, 0
    initial_theta_mean = 0
    beta1 = 1
    beta2 = 1

    # info-holder counter (counter) and index (idx) to fill the arrays. np.mod doesn't
    # work in numba, thus we need counter to hold data at every "hold_every" step
    counter = 0
    i=0

    for step in range(sim_duration):
        if step == int(stim_start * (1 / delta_t)): # if stim on
            if stim_applied == 0: # if first stim
                (theta1, theta2) = (E1,E2)
                initial_theta_mean = (theta1 + theta2) / 2
                beta1 = initial_theta_mean - beta_K
                beta2 = initial_theta_mean - beta_K

                heb_plas_mask = hebbian_plasticity_flag
                exc_scal_mask = exc_scaling_flag
                inh_scal_mask_p = inh_scaling_flag_p
                inh_scal_mask_s = inh_scaling_flag_s

                adaptive_threshold_mask = (heb_plas_mask or exc_scal_mask or inh_scal_mask_p) and adaptive_threshold_flag
                adaptive_LR_mask = heb_plas_mask and adaptive_LR_flag
                if adaptive_LR_mask:
                    learning_rate = 1

            stimulus_E1, stimulus_E2 = stim_train_E[stim_applied]
            stimulus_P1, stimulus_P2 = stim_train_P[stim_applied]
            stimulus_S1, stimulus_S2 = stim_train_S[stim_applied]
            #print("Stimulus started.")

            # if there is more than 1 stimulation, it increases the index number in the stim_times list
            stim_applied = stim_applied + 1 # increase the no stim applied


        elif step == int(stim_stop*(1/delta_t)): # if stim off
            # hebbian is turned off for the testing
            if adaptive_LR_mask:
                learning_rate = 0

            stimulus_E1, stimulus_E2 = 0, 0
            stimulus_P1, stimulus_P2 = 0, 0
            stimulus_S1, stimulus_S2 = 0, 0

            if stim_times.shape[0] > stim_applied: # set the new timing for the next stim if exists
                (stim_start, stim_stop) = stim_times[stim_applied]
            #print('reset EPS Scaling after phase 1, model - line 740')
            #exc_scal_mask   = 0
            #inh_scal_mask_p = 0
            #inh_scal_mask_s = 0

            """print('reset EPS Scaling during phase 1, model - line 712 and line 751')
            exc_scal_mask   =  exc_scaling_flag
            inh_scal_mask_p = inh_scaling_flag_p
            inh_scal_mask_s = inh_scaling_flag_s"""

            """print('introducing E scaling at Phase 2, line 758')
            exc_scal_mask = 1"""


        I1 = g_E - EP110 * P01 - EP120 * P02 - ES110 * S01 - ES120 * S02 + EE110 * E01 + EE120 * E02 + stimulus_E1
        I2 = g_E - EP210 * P01 - EP220 * P02 - ES210 * S01 - ES220 * S02 + EE210 * E01 + EE220 * E02 + stimulus_E2

        E1 = E01 + delta_t*(1/tau_E)*( -E01 + np.maximum(0,I1 - rheobase_E))
        E2 = E02 + delta_t*(1/tau_E)*( -E02 + np.maximum(0,I2 - rheobase_E))

        P1 = P01 + delta_t*(1/tau_P)*( -P01 + np.maximum(0, w_PE11 * E01 + w_PE12 * E02 - w_PS11 * S01 - w_PS12 * S02
                                                         - w_PP11 * P01 - w_PP12 * P02 + g_P - rheobase_P + stimulus_P1))
        P2 = P02 + delta_t*(1/tau_P)*( -P02 + np.maximum(0, w_PE21 * E01 + w_PE22 * E02 - w_PS21 * S01 - w_PS22 * S02
                                                         - w_PP21 * P01 - w_PP22 * P02 + g_P - rheobase_P + stimulus_P2))

        S1 = S01 + delta_t*(1/tau_S)*( -S01 + np.maximum(0, w_SE11 * E01 + w_SE12 * E02 + g_S - rheobase_S + stimulus_S1))
        S2 = S02 + delta_t*(1/tau_S)*( -S02 + np.maximum(0, w_SE21 * E01 + w_SE22 * E02 + g_S - rheobase_S + stimulus_S2))


        beta1 = beta1 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E1 - beta1)
        beta2 = beta2 + adaptive_threshold_mask*delta_t * (1 / tau_beta) * (E2 - beta2)
        print(beta1)
        theta1 = theta1 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(1-alpha)*(theta1 - beta1) + flag_theta_local*alpha*(E1 - theta1))
        theta2 = theta2 + adaptive_threshold_mask*delta_t * (1 / tau_theta) * \
                   (-flag_theta_shift*(1-alpha)*(theta2 - beta2) + flag_theta_local*alpha*(E2 - theta2))

        # rates and plasticity thresholds cannot go below 0 (exc ones cannot go below 1e-323, in order to avoid zero division)
        theta1=max(theta1,1e-10); theta2=max(theta2, 1e-10)
        E1 = max(E1, 1e-10); E2 = max(E2, 1e-10)
        #E1 = min(E1, 4)
        P1 = max(P1, 1e-10); P2 = max(P2, 1e-10)
        S1 = max(S1, 1e-10); S2 = max(S2, 1e-10)

        if heb_plas_mask:
            heb_term_EE11 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E1)
            heb_term_EE12 = learning_rate * delta_t * (1 / tau_plas) * ((E1 - initial_theta_mean) * E2)
            heb_term_EE21 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E1)
            heb_term_EE22 = learning_rate * delta_t * (1 / tau_plas) * ((E2 - initial_theta_mean) * E2)


        ratio_E1 = max(E1, 1e-2) / max(theta1,1e-2); ratio_E2 = max(E2, 1e-2) / max(theta2,1e-2)
        #ratio_E1 = E1 / theta1; ratio_E2 = E2 / theta2

        ss1 = exc_scal_mask*delta_t * (1 / tau_scaling_E) * (1 - ratio_E1)
        ss2 = exc_scal_mask*delta_t * (1 / tau_scaling_E) * (1 - ratio_E2)

        ss1_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * (1 - ratio_E1)
        ss2_p = inh_scal_mask_p*delta_t * (1 / tau_scaling_P) * (1 - ratio_E2)

        ss1_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * (1 - ratio_E1)
        ss2_s = inh_scal_mask_s*delta_t * (1 / tau_scaling_S) * (1 - ratio_E2)


        EE110 = EE110 + ss1*EE110
        EE120 = EE120 + ss1*EE120
        EE210 = EE210 + ss2*EE210
        EE220 = EE220 + ss2*EE220

        EP11 = EP110 - ss1_p*EP110
        EP12 = EP120 - ss1_p*EP120
        EP21 = EP210 - ss2_p*EP210
        EP22 = EP220 - ss2_p*EP220

        ES11 = ES110 + ss1_s*ES110
        ES12 = ES120 + ss1_s*ES120
        ES21 = ES210 + ss2_s*ES210
        ES22 = ES220 + ss2_s*ES220

        # in order to have hebbian plasticity in the absence of synaptic scaling, it is defined here
        EE11 = EE110 + heb_term_EE11
        EE12 = EE120 + heb_term_EE12
        EE21 = EE210 + heb_term_EE21
        EE22 = EE220 + heb_term_EE22

        if E1 > 100:
            break

        # placeholder parameters are freed
        E01 = E1; E02 = E2; P01 = P1; P02 = P2; S01 = S1; S02 = S2
        EE110=EE11; EE120=EE12; EE210=EE21; EE220=EE22
        EP110=EP11; EP120=EP12; EP210=EP21; EP220=EP22
        ES110=ES11; ES120=ES12; ES210=ES21; ES220=ES22

        # values are assigned to the lists
        if counter == hold_every:
            rE1[i] = E01; rE2[i] = E02; rP1[i] = P01; rP2[i] = P02; rS1[i] = S01; rS2[i] = S02
            J_EE11[i] = EE110; J_EE12[i] = EE120; J_EE21[i] = EE210; J_EE22[i] = EE220
            J_EP11[i] = EP110; J_EP12[i] = EP120; J_EP21[i] = EP210; J_EP22[i] = EP220
            J_ES11[i] = ES110; J_ES12[i] = ES120; J_ES21[i] = ES210; J_ES22[i] = ES220
            hebEE11[i] = heb_term_EE11; hebEE12[i] = heb_term_EE12; hebEE21[i] = heb_term_EE21; hebEE22[i] = heb_term_EE22
            ss1_list[i] = ss1; ss2_list[i] = ss2
            av_theta1[i] = theta1; av_theta2[i] = theta2
            beta1_list[i] = beta1; beta2_list[i] = beta2
            i = i + 1
            counter = 0 # start the data-holder counter again

        counter = counter + 1 # update the data-holder counter

