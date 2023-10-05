import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
from model import *
import os
import math


def analyze_model(dk, x_VIP_to_S=0.5, run_model=1, plot_results=1, save_results = 0):
    directory = os.getcwd()
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\\'
    dir_data = r'E:\data\\'

    stim_duration = 15
    sim_duration = int((dk) * 60 + stim_duration + 10 + 2)  # [s]
    delta_t = np.float32(0.0001)  # time step [s]
    hold_every_stimuli = 20  # register data during phase 1 and 3 (conditioning and testing) at every x step
    hold_every_simulation = 200000 # register data during phase 2 (in between conditioning and testing) at every x step
    hold_every = (hold_every_stimuli, hold_every_simulation)

    # total number of timepoints for stimulation
    stim_no_time_points = int((stim_duration + 10) * (1 / delta_t) * (1 / hold_every_stimuli))
    phase2_no_time_points = int((dk * 60 - 20) * (1 / delta_t) * (1 / hold_every_simulation)) + 1 # total no the rest

    t_stimuli = np.linspace(0, stim_duration + 10, stim_no_time_points)
    t_simulation = np.linspace(0, int(dk/60), phase2_no_time_points)

    stim_times_for_threshold = np.array([[0, 0 + stim_duration],
                                         [int(dk * 60) + 5, int(dk * 60) + 5 + stim_duration]])
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(dk * 60) + 5, int(dk * 60) + 5 + stim_duration]]).reshape(2, 2)

    stim_train_E = np.array([(1, 0), (0, 1)])
    stim_train_P = np.array([(0.5, 0), (0, 0.5)])
    stim_train_S = np.array([(0, 0), (0, 0)])
    stim_trains = (stim_train_E, stim_train_P, stim_train_S)

    tau_E = 0.02  # s
    tau_P = 0.005 # s
    tau_S = 0.01  # s
    tau_plas = 90 # s
    tau_beta = 30 * (60 * 60)
    tau_scaling_E = 15 * (60 * 60)  # s
    tau_scaling_P = 15 * (60 * 60)  # s
    tau_scaling_S = 15 * (60 * 60)  # s
    tau_theta = 12 * (60 * 60)  # s
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)


    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # background inputs
    x_E = 4.5
    x_P = 3.2
    x_S = 3.5 - x_VIP_to_S
    back_inputs = (x_E, x_P, x_S)

    # alpha and K parameters
    alpha = 0.5
    beta_K = 0.5

    # initial weights
    w_EPii = .7
    w_ESii = .7
    ij_ratio = 0.3
    w_EPij = np.round(w_EPii * ij_ratio, 3)
    w_ESij = np.round(w_ESii * ij_ratio, 3)

    # Pfeffer & Kumar
    """w_EEii = 0.6; w_EEij = np.round(w_EEii*ij_ratio, 3)
    w_EPii = 0.2; w_EPij = np.round(w_EPii*ij_ratio, 3)
    w_ESii = 0.11; w_ESij = np.round(w_ESii*ij_ratio, 3)
    w_PEii = 1; w_PEij = np.round(w_PEii*ij_ratio, 3)
    w_PPii = 0.1; w_PPij = np.round(w_PPii*ij_ratio, 3)
    w_PSii = 0.06; w_PSij = np.round(w_PSii*ij_ratio, 3)
    w_SEii = 1; w_SEij = np.round(w_SEii*ij_ratio, 3)"""

    # Weights
    w_EEii = 0.4; w_EEij = 0.3
    w_PEii = 0.3; w_PEij = 0.1
    w_PPii = 0.2; w_PPij = 0.1
    w_PSii = 0.3; w_PSij = 0.1
    w_SEii = 0.4; w_SEij = 0.1

    weights = (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii,
               w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij)


    # arrays created to hold data
    r_phase1 = np.zeros((6, stim_no_time_points), dtype=np.float32)
    J_EE_phase1 = np.zeros((4, stim_no_time_points), dtype=np.float32)
    r_phase2 = np.zeros((6, phase2_no_time_points), dtype=np.float32)
    J_phase2 = np.zeros((12, phase2_no_time_points), dtype=np.float32)
    r_phase3 = np.zeros((6, stim_no_time_points), dtype=np.float32)
    max_E = np.zeros(1, dtype=np.float32)
    res_rates = (r_phase1, r_phase2, r_phase3, max_E)
    res_weights = (J_EE_phase1, J_phase2)

    flags_list = [(1, 1, 1, 1, 1, 1)] #, (1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1),
                  #(1, 1, 0, 0, 1, 1), (1, 0, 0, 1, 1, 1), (1, 0, 1, 0, 1, 1)]
    flags_theta = (1, 1)

    for flags in flags_list:
        id, name, title = determine_name(flags)
        name = 'Case' + id + str(dk/60) + 'h'
        print('*****', title, '*****')

        if run_model:
            print('Calculating the perception threshold.')
            model(delta_t, hold_every, res_rates, res_weights, int((5) * (1 / delta_t)), weights,
                           back_inputs, stim_trains, stim_times_for_threshold, taus, alpha, beta_K,
                           rheobases, flags=(0,0,0,0,0,0), flags_theta=(0,0))

            delta_tt = (hold_every_stimuli) * delta_t
            p_threshold = res_rates[0][0][int((5 - 2) * (1 / delta_tt)) - 10]
            print('\n')

            print('Simulation started.')
            model(delta_t, hold_every, res_rates, res_weights,
                  int(sim_duration * (1 / delta_t)), weights, back_inputs,
                  stim_trains, stim_times, taus, alpha, beta_K, rheobases,
                  flags=flags, flags_theta=flags_theta)

            if save_results:
                res_list = [t_stimuli, t_simulation, delta_t, hold_every, res_rates, res_weights,
                            p_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(res_list, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                res_list = pickle.load(file)
            print('Data is read.')

            [t_stimuli, t_simulation, delta_t, hold_every, res_rates, res_weights,
             p_threshold, stim_times, stim_duration, sim_duration] = res_list

        if plot_results:
            print('Plotting the results.')
            plot_all([t_stimuli, t_simulation], res_rates, res_weights,
                     p_threshold, stim_times, dir_plot + name, dk, format='.png')



def analyze_model_VIP(dk, x_VIP_to_S=0.5, run_model=1, plot_results=1, save_results = 0):
    directory = os.getcwd()
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\\'
    dir_data = r'E:\data\VIP\\'

    stim_duration = 15
    sim_duration = int((dk) * 60 + stim_duration + 10 + 2)  # [s]
    delta_t = np.float32(0.0001)  # time step [s]
    hold_every_stimuli = 20  # register data during phase 1 and 3 (conditioning and testing) at every x step
    hold_every_simulation = 200000 # register data during phase 2 (in between conditioning and testing) at every x step
    hold_every = (hold_every_stimuli, hold_every_simulation)

    # total number of timepoints for stimulation
    stim_no_time_points = int((stim_duration + 10) * (1 / delta_t) * (1 / hold_every_stimuli))
    phase2_no_time_points = int((dk * 60 - 20) * (1 / delta_t) * (1 / hold_every_simulation)) + 1 # total no the rest

    t_stimuli = np.linspace(0, stim_duration + 10, stim_no_time_points)
    t_simulation = np.linspace(0, int(dk/60), phase2_no_time_points)

    stim_times_for_threshold = np.array([[0, 0 + stim_duration],
                                         [int(dk * 60) + 5, int(dk * 60) + 5 + stim_duration]])
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(dk * 60) + 5, int(dk * 60) + 5 + stim_duration]]).reshape(2, 2)

    stim_train_E = np.array([(1, 0), (0, 1)])
    stim_train_P = np.array([(0.5, 0), (0, 0.5)])
    stim_train_S = np.array([(0, 0), (0, 0)])
    stim_trains = (stim_train_E, stim_train_P, stim_train_S)

    tau_E = 0.02  # s
    tau_P = 0.005 # s
    tau_S = 0.01  # s
    tau_plas = 60*25 # s
    tau_beta = 3.0 * (60 * 60)
    tau_scaling_E = 1.5 * (60 * 60)  # s
    tau_scaling_P = 1.5 * (60 * 60)  # s
    tau_scaling_S = 1.5 * (60 * 60)  # s
    tau_theta = 3.0 * (60 * 60)  # s
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)


    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # background inputs
    x_E = 3; x_P = 3; x_S = 3; x_V = 1.7
    x_E = 30; x_P = 30; x_S = 30; x_V = 20
    back_inputs = (x_E, x_P, x_S, x_V)

    # alpha and K parameters
    alpha = 0.5
    beta_K = 10

    # initial weights
    ij_ratio = 0.3

    # Kumar
    w_EEii = 0.30; w_EEij = np.round(w_EEii*.8, 3)
    w_EPii = 0.10; w_EPij = np.round(w_EPii*ij_ratio, 3)
    w_ESii = 0.50; w_ESij = np.round(w_ESii*ij_ratio, 3)
    w_PEii = 0.50; w_PEij = np.round(w_PEii*ij_ratio, 3)
    w_PPii = 0.10; w_PPij = np.round(w_PPii*ij_ratio, 3)
    w_PSii = 0.07; w_PSij = np.round(w_PSii*ij_ratio, 3)
    w_SEii = 0.35; w_SEij = np.round(w_SEii*ij_ratio, 3)
    w_SVii = 0.10; w_SVij = np.round(w_SVii*ij_ratio, 3)
    w_VEii = 0.50; w_VEij = np.round(w_SVii*ij_ratio, 3)
    w_VPii = 0.15; w_VPij = np.round(w_SVii*ij_ratio, 3)
    w_VSii = 0.05; w_VSij = np.round(w_SVii*ij_ratio, 3)

    # Pfeffer & Kumar
    """w_EEii = 0.6; w_EEij = np.round(w_EEii*ij_ratio, 3)
    w_EPii = 0.2; w_EPij = np.round(w_EPii*ij_ratio, 3)
    w_ESii = 0.11; w_ESij = np.round(w_ESii*ij_ratio, 3)
    w_PEii = 1; w_PEij = np.round(w_PEii*ij_ratio, 3)
    w_PPii = 0.2; w_PPij = np.round(w_PPii*ij_ratio, 3)
    w_PSii = 0.06; w_PSij = np.round(w_PSii*ij_ratio, 3)
    w_SEii = 1; w_SEij = np.round(w_SEii*ij_ratio, 3)
    w_SVii = 0.1; w_SVij = 0.5"""

    # Weights
    """w_EEii = 0.4; w_EEij = 0.3
    w_PEii = 0.3; w_PEij = 0.1
    w_PPii = 0.2; w_PPij = 0.1
    w_PSii = 0.3; w_PSij = 0.1
    w_SEii = 0.4; w_SEij = 0.1
    w_SVii = 0.2; w_SVij = 0.1
    
    x_E = 4.5
    x_P = 3.2
    x_S = 2.7
    x_V = 1.6
    back_inputs = (x_E, x_P, x_S, x_V)"""

    weights = (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii, w_SVii, w_VEii, w_VPii, w_VSii,
               w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij, w_SVij, w_VEij, w_VPij, w_VSij)
    # arrays created to hold data
    r_phase1 = np.zeros((8, stim_no_time_points), dtype=np.float32)
    J_EE_phase1 = np.zeros((4, stim_no_time_points), dtype=np.float32)
    r_phase2 = np.zeros((8, phase2_no_time_points), dtype=np.float32)
    J_phase2 = np.zeros((12, phase2_no_time_points), dtype=np.float32)
    r_phase3 = np.zeros((8, stim_no_time_points), dtype=np.float32)
    max_E = np.zeros(1, dtype=np.float32)
    res_rates = (r_phase1, r_phase2, r_phase3, max_E)
    res_weights = (J_EE_phase1, J_phase2)

    flags_list = [(1, 1, 1, 1, 1, 1)] #, (1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1),
                  #(1, 1, 0, 0, 1, 1), (1, 0, 0, 1, 1, 1), (1, 0, 1, 0, 1, 1)]
    flags_theta = (1, 1)

    for flags in flags_list:
        id, name, title = determine_name(flags)
        name = 'Case' + id + str(dk/60) + 'h'
        print('*****', title, '*****')

        if run_model:
            print('Calculating the perception threshold.')
            model_VIP(delta_t, hold_every, res_rates, res_weights, int((5) * (1 / delta_t)), weights,
                           back_inputs, stim_trains, stim_times_for_threshold, taus, alpha, beta_K,
                           rheobases, flags=(0,0,0,0,0,0), flags_theta=(0,0))

            delta_tt = (hold_every_stimuli) * delta_t
            p_threshold = res_rates[0][0][int((5 - 2) * (1 / delta_tt)) - 10]
            print('\n')

            print('Simulation started.')
            model_VIP(delta_t, hold_every, res_rates, res_weights,
                  int(sim_duration * (1 / delta_t)), weights, back_inputs,
                  stim_trains, stim_times, taus, alpha, beta_K, rheobases,
                  flags=flags, flags_theta=flags_theta)

            if save_results:
                res_list = [t_stimuli, t_simulation, delta_t, hold_every, res_rates, res_weights,
                            p_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(res_list, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                res_list = pickle.load(file)
            print('Data is read.')

            [t_stimuli, t_simulation, delta_t, hold_every, res_rates, res_weights,
             p_threshold, stim_times, stim_duration, sim_duration] = res_list

        if plot_results:
            print('Plotting the results.')
            plot_all_VIP([t_stimuli, t_simulation], res_rates, res_weights,
                     p_threshold, stim_times, dir_plot + name, dk, format='.png')





def span_initial_conditions(dk, x_VIP_to_S=0.5, run_search=0, plot_results=1):
    directory = os.getcwd()
    dir_data = r'E:\data_domain\new_params\\'
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\\'

    n = 21

    flags_list = [(1, 1, 1, 1, 1, 1), (1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 1, 1),
                  (1, 1, 1, 0, 1, 1), (1, 0, 0, 1, 1, 1), (1, 1, 0, 0, 1, 1), (1, 0, 1, 0, 1, 1)]
    flags_theta = (1, 1)


    for flags in flags_list:
        id, _, title = determine_name(flags)
        name = r'Case' + str(id) + "_" + str(dk) + 'dk' + "_" + str(n) + 'n'

        wES_iis = np.linspace(0, 1, n)
        wEP_iis = np.linspace(0, 1, n)

        if run_search:
            print('Span of initial conditions started.')

            stim_duration = 15
            sim_duration = int((dk + 1) * 60)  # s
            delta_t = np.float32(0.0001)  # s
            hold_every_stimuli = 20
            hold_every_simulation = 200000
            hold_every = (hold_every_stimuli, hold_every_simulation)
            stim_no_time_points = int(stim_duration * (1 / delta_t) * (1 / hold_every_stimuli))
            phase2_no_time_points = int((dk * 60 - 20) * (1 / delta_t) * (1 / hold_every_simulation)) + 6

            t_stimuli = np.linspace(0, stim_duration, stim_no_time_points)
            t_simulation = np.linspace(0, dk * 60* 60 - 20 , phase2_no_time_points)

            stim_times = np.array([[5, 5 + stim_duration],
                                   [int(dk * 60) + 5, int(dk * 60) + 5 + stim_duration]]).reshape(2, 2)

            stim_train_E = np.array([(1, 0), (0, 1)])
            stim_train_P = np.array([(0.5, 0), (0, 0.5)])
            stim_train_S = np.array([(0, 0), (0, 0)])
            stim_trains = (stim_train_E, stim_train_P, stim_train_S)

            tau_E = 0.02  # s
            tau_P = 0.005  # s
            tau_S = 0.01  # s
            tau_plas = 120  # s
            tau_beta = 3.0 * (60 * 60)
            tau_scaling_E = 1.5 * (60 * 60)  # s
            tau_scaling_P = 1.5 * (60 * 60)  # s
            tau_scaling_S = 1.5 * (60 * 60)  # s
            tau_theta = 1.2 * (60 * 60)  # s
            taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

            rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
            rheobases = (rheobase_E, rheobase_P, rheobase_S)

            # background inputs
            x_E = 4.5
            x_P = 3.2
            x_S = 3.5 - x_VIP_to_S
            back_inputs = (x_E, x_P, x_S)

            # some parameters
            ij_ratio = 0.3
            alpha = 0.5
            beta_K = 0.5

            # data arrays are created
            r_phase1 = np.zeros((6, stim_no_time_points), dtype=np.float32)
            J_EE_phase1 = np.zeros((4, stim_no_time_points), dtype=np.float32)

            r_phase2 = np.zeros((6, phase2_no_time_points), dtype=np.float32)
            J_phase2 = np.zeros((12, phase2_no_time_points), dtype=np.float32)

            r_phase3 = np.zeros((6, stim_no_time_points), dtype=np.float32)

            max_E = np.zeros(1, dtype=np.float32)

            res_rates = (r_phase1, r_phase2, r_phase3, max_E)
            res_weights = (J_EE_phase1, J_phase2)

            results_list = [[], []]

            for w_ESii in wES_iis:
                for w_EPii in wEP_iis:


                    # Pfeffer & Kumar
                    """w_EEii = 0.6; w_EEij = np.round(w_EEii*ij_ratio, 3)
                    w_EPii = 0.2; w_EPij = np.round(w_EPii*ij_ratio, 3)
                    w_ESii = 0.11; w_ESij = np.round(w_ESii*ij_ratio, 3)
                    w_PEii = 1; w_PEij = np.round(w_PEii*ij_ratio, 3)
                    w_PPii = 0.1; w_PPij = np.round(w_PPii*ij_ratio, 3)
                    w_PSii = 0.06; w_PSij = np.round(w_PSii*ij_ratio, 3)
                    w_SEii = 1; w_SEij = np.round(w_SEii*ij_ratio, 3)"""

                    # Weights
                    w_EEii = 0.4; w_EEij = 0.3
                    w_PEii = 0.3; w_PEij = 0.1
                    w_PPii = 0.2; w_PPij = 0.1
                    w_PSii = 0.3; w_PSij = 0.1
                    w_SEii = 0.4; w_SEij = 0.1

                    w_EPij = np.round(w_EPii * ij_ratio, 3)
                    w_ESij = np.round(w_ESii * ij_ratio, 3)

                    weights = (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii,
                               w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij)

                    model(delta_t, hold_every, res_rates, res_weights,
                          int(sim_duration * (1 / delta_t)), weights,back_inputs,
                          stim_trains, stim_times, taus, alpha, beta_K, rheobases,
                          flags=flags, flags_theta=flags_theta)

                    results_list[0].append([r_phase1.copy(), r_phase2.copy(), r_phase3.copy(), max_E.copy()])
                    results_list[1].append([J_EE_phase1.copy(), J_phase2.copy()])

            # Open a file and use dump()
            with open(dir_data + name + '.pkl', 'wb') as file:
                # A new file will be created
                pickle.dump(results_list, file)

            print("Case" + id + " over.")

        if plot_results:
            with open(dir_data + name + '.pkl', 'rb') as file:
                results_list = pickle.load(file)

            with open(dir_data + 'Thresholds.pkl', 'rb') as file:
                thresholds_list = pickle.load(file)
            print('Data is read')


            res_rates, res_weights = results_list[0], results_list[1]
            rE1_phase1 = np.zeros((n ** 2, res_rates[0][0].shape[1]))
            rE1_phase2 = np.zeros((n ** 2, res_rates[0][1].shape[1]-5))
            rE2_phase3 = np.zeros((n ** 2, res_rates[0][2].shape[1]))
            max_E = np.zeros((n ** 2, 1))
            min_r = np.zeros((n ** 2, 1))

            for idx, i in enumerate(res_rates):
                rE1_phase1[idx], rE1_phase2[idx], rE2_phase3[idx], max_E[idx] = i[0][0], i[1][0][:-5], i[2][1], i[3]
                min_r[idx] = min(min(np.min(i[0]), np.min(i[1][:,:-5])), np.min(i[2]))

            dot = 0
            res_mem_spec = []

            with open(dir_data + r'Case4_2dk_21n.pkl', 'rb') as file:
                results_op_region = pickle.load(file)
            op_region_binary = np.zeros((n ** 2, 1))

            for idx, i in enumerate(results_op_region[1]):
                op_region_binary[idx] = (i[0][:, -1] > [.4, .3, .3, .4]).all()

            for w_ES_ii in wES_iis:
                for w_EP_ii in wEP_iis:
                    degeneracy = not op_region_binary[dot] or min_r[dot] < 0.005
                    p_threshold = thresholds_list.pop(0)

                    if max_E[dot] < 50:
                        specific_memory = (rE2_phase3[dot] < p_threshold).all()

                        if not degeneracy:
                            if specific_memory:
                                res_mem_spec.append(1)
                            else:
                                res_mem_spec.append(0)
                        else:
                            res_mem_spec.append(2)

                    else:
                        res_mem_spec.append(np.nan)
                    dot = dot + 1

            plot_span_init_conds(res_mem_spec, wEP_iis, wES_iis, dir_plot, name, n,
                              plot_bars=0, plot_legends=0, format='.svg', title=title)



def span_initial_conditions_thresholds():
    n = 21

    upper_bound_E = 1e6
    upper_bound_P = 1e6
    upper_bound_S = 1e6

    upper_bounds = (upper_bound_E, upper_bound_P, upper_bound_S)

    tau_E = 0.02  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    tau_plas = 120  # s
    tau_beta = 30 * (60 * 60)
    tau_scaling_E = 15 * (60 * 60)   # s
    tau_scaling_P = 15 * (60 * 60)   # s
    tau_scaling_S = 15 * (60 * 60)   # s
    tau_theta = 12 * (60 * 60)   # s
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # background inputs
    x_E = 4.5
    x_P = 3.5
    x_S = 4
    back_inputs = (x_E, x_P, x_S)

    # some parameters
    ij_ratio = 0.3
    alpha = 0.5
    beta_K = 0.5


    flags_list = [(0,0,0,0,0,0)]

    wES_iis = np.linspace(0, 1, n)
    wEP_iis = np.linspace(0, 1, n)

    for flags in flags_list:
        dir =  r'E:\data_domain\new_params\VIP_effect\x_VIP_to_S_1\\'

        name = dir + 'Thresholds'

        stim_duration = 15
        sim_duration = 2  # s
        delta_t = np.float32(0.0001)  # s
        hold_every = 20
        timepoints = int(sim_duration * (1 / delta_t) * (1 / hold_every))

        stim_times_for_threshold = np.array([[0, 0 + stim_duration]])
        stim_train_E = np.array([(1, 0)])
        stim_train_P = np.array([(0.5, 0)])
        stim_train_S = np.array([(0, 0)])
        stim_trains = (stim_train_E, stim_train_P, stim_train_S)

        rE1 = np.zeros(timepoints, dtype=np.float32); rE2 = np.zeros(timepoints, dtype=np.float32)
        rP1 = np.zeros(timepoints, dtype=np.float32); rP2 = np.zeros(timepoints, dtype=np.float32)
        rS1 = np.zeros(timepoints, dtype=np.float32); rS2 = np.zeros(timepoints, dtype=np.float32)

        J_EE11 = np.zeros(timepoints, dtype=np.float32); J_EE12 = np.zeros(timepoints, dtype=np.float32)
        J_EE21 = np.zeros(timepoints, dtype=np.float32); J_EE22 = np.zeros(timepoints, dtype=np.float32)

        J_EP11 = np.zeros(timepoints, dtype=np.float32); J_EP12 = np.zeros(timepoints, dtype=np.float32)
        J_EP21 = np.zeros(timepoints, dtype=np.float32); J_EP22 = np.zeros(timepoints, dtype=np.float32)

        J_ES11 = np.zeros(timepoints, dtype=np.float32); J_ES12 = np.zeros(timepoints, dtype=np.float32)
        J_ES21 = np.zeros(timepoints, dtype=np.float32); J_ES22 = np.zeros(timepoints, dtype=np.float32)

        vars = (rE1, rE2, rP1, rP2, rS1, rS2,
                J_EE11, J_EE12, J_EE21, J_EE22,
                J_EP11, J_EP12, J_EP21, J_EP22,
                J_ES11, J_ES12, J_ES21, J_ES22)

        hebEE11, hebEE12 = np.zeros(timepoints, dtype=np.float32), np.zeros(timepoints, dtype=np.float32)
        hebEE21, hebEE22 = np.zeros(timepoints, dtype=np.float32), np.zeros(timepoints, dtype=np.float32)
        ss1_list, ss2_list = np.zeros(timepoints, dtype=np.float32), np.zeros(timepoints, dtype=np.float32)
        av_theta_I1, av_theta_I2 = np.zeros(timepoints, dtype=np.float32), np.zeros(timepoints, dtype=np.float32)
        beta1_list, beta2_list = np.zeros(timepoints, dtype=np.float32), np.zeros(timepoints, dtype=np.float32)
        plas_terms = (hebEE11, hebEE12, hebEE21, hebEE22, ss1_list, ss2_list,
                      av_theta_I1, av_theta_I2, beta1_list, beta2_list)

        results_list = []

        for w_ESii in wES_iis:
            for w_EPii in wEP_iis:

                # Pfeffer & Kumar
                """w_EEii = 0.6; w_EEij = np.round(w_EEii*ij_ratio, 3)
                w_EPii = 0.2; w_EPij = np.round(w_EPii*ij_ratio, 3)
                w_ESii = 0.11; w_ESij = np.round(w_ESii*ij_ratio, 3)
                w_PEii = 1; w_PEij = np.round(w_PEii*ij_ratio, 3)
                w_PPii = 0.1; w_PPij = np.round(w_PPii*ij_ratio, 3)
                w_PSii = 0.06; w_PSij = np.round(w_PSii*ij_ratio, 3)
                w_SEii = 1; w_SEij = np.round(w_SEii*ij_ratio, 3)"""

                # Weights
                w_EEii = 0.4; w_EEij = 0.3
                w_PEii = 0.3; w_PEij = 0.1
                w_PPii = 0.2; w_PPij = 0.1
                w_PSii = 0.3; w_PSij = 0.1
                w_SEii = 0.4; w_SEij = 0.1

                w_EPij = np.round(w_EPii * ij_ratio, 3)
                w_ESij = np.round(w_ESii * ij_ratio, 3)

                weights = (w_EEii, w_EPii, w_ESii, w_PEii, w_PPii, w_PSii, w_SEii,
                           w_EEij, w_EPij, w_ESij, w_PEij, w_PPij, w_PSij, w_SEij)

                model(delta_t, hold_every, vars, plas_terms, int((sim_duration) * (1 / delta_t)), weights,
                      back_inputs, stim_trains, stim_times_for_threshold, taus, alpha, beta_K,
                      rheobases, flags=(0, 0, 0, 0, 0, 0), flags_theta=(0, 0))


                if not np.isnan(rE1).any():  # and not rE1[-100]==0:
                    results_list.append([rE1[-10].copy()])
                else:
                    results_list.append([np.nan])

        print('Data saving started.')

        # Open a file and use dump()
        with open(name + '.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(results_list, file)



#analyze_model_VIP(4 * 60, run_model=1, save_results=1, plot_results=1)
#analyze_model(24 * 60, run_model=1, save_results=1, plot_results=1)
analyze_model_VIP(4.8 * 60, run_model=1, save_results=0, plot_results=1)

