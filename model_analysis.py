import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
from model import *
from plotting_functions import *
import os
from parameters import *
import pickle

def analyze_model(hour_sim, flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                  K=0.25, g_top_down_to_S=0, flag_only_S_on=False, run_simulation=True, save_results = False, plot_results=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    stim_duration = 15  # stimulation duration in seconds
    # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
    sim_duration = int((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)
    delta_t = 0.0001  # time step in seconds (0.1 ms)
    sampling_rate_stim = 20  # register data at every 20 step during phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = 200000  # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Total number of timepoints for stimulation and simulation
    n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
    n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1 # total no the rest

    l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
    l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    # The stimuli are given as inputs to the populations.
    g_stim_E = np.array([(1, 0), (0, 1)])
    g_stim_P = np.array([(0.5, 0), (0, 0.5)])
    g_stim_S = np.array([(0, 0), (0, 0)])
    g_stim = (g_stim_E, g_stim_P, g_stim_S)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_hebb = 240  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
    tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
    tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
    taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    # Rheobases (minimum input needed for firing rates to be above zero)
    rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # Background inputs
    g_E = 4.5
    g_P = 3.2
    g_S = 3
    back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

    # Initial conditions for plastic weights
    w_EP_within = 0.7; w_EP_cross = w_EP_within*0.3
    w_ES_within = 0.7; w_ES_cross = w_ES_within*0.3
    w_EE_within = 0.5; w_EE_cross = 0.4

    # Weights
    w_PE_within = 0.3; w_PE_cross = 0.1
    w_PP_within = 0.2; w_PP_cross = 0.1
    w_PS_within = 0.3; w_PS_cross = 0.1
    w_SE_within = 0.4; w_SE_cross = 0.1

    weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
               w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

    # Arrays created to hold data
    r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22
    r_phase2 = np.zeros((10, n_time_points_phase2),
                        dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
    J_phase2 = np.zeros((12, n_time_points_phase2),
                        dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
    r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    max_E = np.zeros(1, dtype=np.float32)

    # Lists to hold data arrays
    l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
    l_res_weights = (J_EE_phase1, J_phase2)

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_' + str(hour_sim) + 'h' + '_k' + str(K) + '_td' + str(g_top_down_to_S)

        print('*****', title, '*****')

        if run_simulation:

            print('Simulation started.')
            print('\n')

            model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights,
                  back_inputs, g_stim, stim_times, taus, K, rheobases, flags=(0,0,0,0,0,0), flags_theta=flags_theta)

            idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
            av_threshold = r_phase1[1][idx_av_threshold] * 1.15

            model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights,
                  back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

            if save_results:
                l_results = [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates,
                             l_res_weights,
                             av_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates, l_res_weights,
             av_threshold, stim_times, stim_duration, sim_duration] = l_results

        if plot_results:
            print('Plotting the results.')
            time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                       stim_times, dir_plot + name, hour_sim, flag_only_S_on=flag_only_S_on, format='.png')
            time_plots([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights, av_threshold,
                       stim_times, dir_plot + name, hour_sim, flag_only_S_on=flag_only_S_on, format='.svg')




def plot_testing_at_regular_intervals(flags_list, flags_theta=(1,1), dir_data=r'\figures\data\\', dir_plot=r'\figures\\',
                                      K=0.25, g_top_down_to_S=0, flag_only_S_on=False, run_simulation=True,
                                      save_results = False, plot_results=False):

    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    hour_sims = np.arange(48) + 1
    l_delta_rE1 = []

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_test_every_h' + '_k' + str(K) + '_td' + str(g_top_down_to_S)

        print('*****', title, '*****')

        if run_simulation:
            print('Simulation started.')
            print('\n')
            for hour_sim in hour_sims:
                stim_duration = 15 # stimulation duration in seconds
                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
                sim_duration = int((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)
                delta_t = 0.0001 # time step in seconds (0.1 ms)
                sampling_rate_stim = 20 # register data at every 20 step during phase 1 and 3 (conditioning and testing)
                sampling_rate_sim = 200000 # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
                sampling_rate = (sampling_rate_stim, sampling_rate_sim)

                # Total number of timepoints for stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1 # total no the rest

                l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
                l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

                # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
                stim_times = np.array([[5, 5 + stim_duration],
                                       [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005 # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 240 # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60) # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60) # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Initial conditions for plastic weights
                w_EP_within = 0.7; w_EP_cross = 0.21
                w_ES_within = 0.7; w_ES_cross = 0.21
                w_EE_within = 0.5 ; w_EE_cross = 0.4

                # Weights
                w_PE_within = 0.3; w_PE_cross = 0.1
                w_PP_within = 0.2; w_PP_cross = 0.1
                w_PS_within = 0.3; w_PS_cross = 0.1
                w_SE_within = 0.4; w_SE_cross = 0.1

                weights = (w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                           w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32) # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((10, n_time_points_phase2), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.zeros((12, n_time_points_phase2),dtype=np.float32) # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32) # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(30 * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=(0,0,0,0,0,0), flags_theta=flags_theta)

                idx_av_threshold = int(15 * (1 / delta_t) * (1 / sampling_rate_stim))
                av_threshold = r_phase1[1][idx_av_threshold] * 1.15

                model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)), weights,
                      back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                l_delta_rE1.append(np.max(r_phase3[0][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                               int(stim_times[0][1] * (1 / (delta_t * sampling_rate_stim)))]).copy())

                print('Simulation of ' + str(hour_sim) + ' hours is completed')
            if save_results:
                l_results = [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [r_phase1, l_time_points_phase2, r_phase2, l_delta_rE1, av_threshold, delta_t, sampling_rate_sim] = l_results

        if plot_results:
            print('Plotting the results.')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.png')
            change_in_reactivation_every_h(l_time_points_phase2, hour_sims, l_delta_rE1, av_threshold,
                                           dir_plot + name, flag_only_S_on=flag_only_S_on, format='.svg')





def plot_all_cases_CIR(dir_data=r'\figures\data\\', dir_plot=r'\figures\\', K=0.25, g_top_down_to_S=0):

    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param flags_list
    :param flags_theta
    :param dir_data
    :param dir_plot
    :param K: Tunes the steady state value of target activity and its regulator
    :param g_top_down_to_S: Represents the top-down signal to SST neurons triggered by the hyperexcitation. It reaches
    SST neurons at the offset of the conditioning
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """
    directory = os.getcwd()

    hour_sims = np.arange(48) + 1
    l_all_delta_rE1 = []
    name_plot = 'CIR_all_cases'

    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1), (1, 1, 1, 1, 1, 0),
                  (1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 0), (1, 1, 1, 0, 0, 1), (1, 1, 1, 0, 0, 0)]

    for flags in flags_list:
        id, title = determine_name(flags)
        name_data = 'Case' + id + '_test_every_h' + '_k' + str(K) + '_td' + str(g_top_down_to_S)

        # Open the file and read
        with open(directory+dir_data + name_data + '.pkl', 'rb') as file:
            l_results = pickle.load(file)
        print('Data is read.')

        [_, l_time_points_phase2, _, l_delta_rE1, av_threshold, _, _] = l_results
        l_all_delta_rE1.append(l_delta_rE1)

    print('Plotting the results.')
    all_cases_CIR(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold, directory+dir_plot+name_plot, format='.png')
    all_cases_CIR(l_time_points_phase2, hour_sims, l_all_delta_rE1, av_threshold, directory+dir_plot+name_plot, format='.svg')





def span_initial_conditions(flag_case, hour_sim_case, g_top_down_to_S=0, K=0.5,
                            search_wEP_wES=0, search_wEE_wEP=0, search_wEE_wES=0,
                            run_search=0, plot_results=1):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param run_search: True to run the numerical simulation
    :param plot_results: True to read the saved data and plot

    This function spans the initial conditions in pairs of W_ES, W_EP and W_EE weights if the corresponding flag for
    each combination is true (search_wEP_wES, search_wEE_wEP, search_wEE_wES). It saves the results (if run_search is
    True) or reads the saved data (if run_search is False). The memory specificity/overgeneralization result for every
    dot in the initial condition space can be plotted (if plot_results is True).
    """

    # Data and plotting directories are defined
    directory = r'C:\Users\AYCA\PycharmProjects\SynapticScaling'  # os.getcwd()
    dir_data = r'E:\data_domain\wEPwES_yedek\\'
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\span_initial_cond\new\\'

    n = 21  # the number of different values for each weight
    stim_duration = 15  # stimulation duration in seconds
    delta_t = np.float32(0.0001)  # time step in seconds (0.1 ms)
    sampling_rate_stim = 20  # register data at every 20 step in phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = (
                                    1 / delta_t) * 10 * 60 * 60  # register data at every 10 min in phase 2 (between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim_case * 60 * 60) + 5,
                            int(hour_sim_case * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    flags_list = [(1, 1, 1, 1, 1, 1), flag_case]
    flags_theta = (1, 1)
    id, title = determine_name(flag_case)

    # the name to identify the current case under investigation
    name = 'Case' + id + '_' + str(hour_sim_case) + 'h' + '_td' + str(g_top_down_to_S) + '_k' + str(K)

    # Lists of the names and simulation times for two simulations. The first simulation determines the operating region
    # whereas the second simulation runs the search
    names_list = [r'Op_reg', name]
    l_hour_sim = [int((stim_duration + 10 + 3) / (60 * 60)),
                  # for determining only LTP cases, only conditioning phase needed
                  hour_sim_case]  # case under investigation

    ### Following 3 if conditions provide the search for the 3 pairs of wEE, wEP and wES depending on the flags.

    # Spanning the initial conditions for wEP and wES
    if search_wEP_wES == 1:
        # loop for two simulations (operating region and initial conditions search)
        for flags, name, hour_sim in zip(flags_list, names_list, l_hour_sim):
            # The initial conditions of the weights
            wES_withins = np.linspace(0, 1, n)
            wEP_withins = np.linspace(0, 1, n)

            # Empty lists are created to hold data
            l_res_rates = ()
            l_res_weights = ()

            # Simulation will be run if the run_search flag is set to True
            if run_search:
                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 15 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 15 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 15 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Total number of timepoints in stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1

                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each,
                # 2 extra seconds to reach steady state initially
                sim_duration = int(hour_sim * 60 * 60 + (stim_duration + 10) * 2 + 2)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((10, n_time_points_phase2),
                                    dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.zeros((12, n_time_points_phase2),
                                    dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)
                av_threshold = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)
                results_list = [[], []]

                # Initial conditions are spanned in the loops
                print('Span of initial conditions started.')
                print('...')
                for w_ES_within in wES_withins:
                    for w_EP_within in wEP_withins:
                        # Weights
                        w_EE_within = 0.4;
                        w_EE_cross = 0.3
                        w_PE_within = 0.3;
                        w_PE_cross = 0.1
                        w_PP_within = 0.2;
                        w_PP_cross = 0.1
                        w_PS_within = 0.3;
                        w_PS_cross = 0.1
                        w_SE_within = 0.4;
                        w_SE_cross = 0.1

                        # Cross-connections for the weights that are spanned
                        cross_within_ratio = 0.3  # Cross connection to within connection ratio
                        w_EP_cross = np.round(w_EP_within * cross_within_ratio, 3)
                        w_ES_cross = np.round(w_ES_within * cross_within_ratio, 3)

                        weights = (
                        w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                        w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                        # The model is run for the given parameters
                        model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)),
                              weights,
                              back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                        # The aversion threshold is calculated, see the find_baseline_reactivation() function for details
                        idx_av_threshold = find_baseline_reactivation(r_phase1[0])
                        av_threshold[0] = r_phase1[0][idx_av_threshold + 1]

                        # Data is appended to the results list
                        results_list[0].append(
                            [r_phase1.copy(), r_phase2.copy(), r_phase3.copy(), max_E.copy(), av_threshold.copy()])
                        results_list[1].append([J_EE_phase1.copy(), J_phase2.copy()])

                # Results are saved, the name of the weights that are spanned are added to the name of the file
                with open(dir_data + 'wEPwES_' + name + '.pkl', 'wb') as file:
                    pickle.dump(results_list, file)

                print('Initial conditions space is spanned.')
                print('\n')
            print('\n')

        # Plotting the memory specificity results for every initial condition pair of wEP and wES if plot_results is True
        if plot_results:
            # Reading the results
            with open(dir_data + 'wEPwES_' + name + '.pkl', 'rb') as file:
                results_list = pickle.load(file)

            # Reading the operating region and holding the binary results in an array
            with open(dir_data + r'wEPwES_Op_reg.pkl', 'rb') as file:
                results_op_region = pickle.load(file)
            op_region_binary = np.zeros((n ** 2, 1))
            for idx, i in enumerate(results_op_region[1]):
                w_EE_within, w_EE_cross = i[0][0][0], i[0][1][0]
                op_region_binary[idx] = (i[0][:, -1] > [w_EE_within, w_EE_cross, w_EE_cross, w_EE_within]).all()
            print('Data is read')
            print('\n')

            # Arrays created to hold data
            l_res_rates, l_res_weights = results_list[0], results_list[1]
            rE1_phase1 = np.zeros((n ** 2, l_res_rates[0][0].shape[1]))
            rE1_phase2 = np.zeros((n ** 2, l_res_rates[0][1].shape[1]))
            rE2_phase3 = np.zeros((n ** 2, l_res_rates[0][2].shape[1]))
            max_E = np.zeros((n ** 2, 1))
            av_threshold = np.zeros((n ** 2, 1))
            min_r = np.zeros((n ** 2, 1))

            # Data is registered to the correct arrays
            for idx, i in enumerate(l_res_rates):
                rE1_phase1[idx], rE1_phase2[idx], rE2_phase3[idx] = i[0][0], i[1][0], i[2][1]
                max_E[idx], av_threshold[idx] = i[3], i[4]
                min_r[idx] = min(np.min(i[0]), np.min(i[2]))  # min(min(np.min(i[0]), np.min(i[1])), np.min(i[2]))

            # The span of the initial conditions to determine the memory specificity is started
            dot = 0  # every initial condition pair is called as one dot on the initial condition space
            res_mem_spec = []  # array to hold memory specificity results at each initial condition pair of wEP and wES
            for w_ES_within in wES_withins:
                for w_EP_within in wEP_withins:
                    # If the dot is out of the operating regime (either LTD is induced during conditioning and/or the
                    # firing rates goes to zero)
                    exclude = (not op_region_binary[dot])  # or (min_r[dot] < 0.005)

                    if np.round(w_EP_within, 3) == 0.5 and np.round(w_ES_within, 3) == 0.75:
                        print(max_E[dot])
                        print(np.max(rE1_phase1))

                    # If the rates explode assign np.nan
                    if max_E[dot] > 50:
                        res_mem_spec.append(np.nan)

                    # If the rates do not explode and if there is no degeneracy, determine the memory specificity
                    elif max_E[dot] <= 50 and not exclude:
                        specific_memory = (rE2_phase3[dot][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                                           int(stim_times[0][1] * (
                                                                       1 / (delta_t * sampling_rate_stim)))] <
                                           av_threshold[dot]).all()
                        if specific_memory:
                            res_mem_spec.append(1)
                        else:
                            res_mem_spec.append(0)

                    # If the dot is out of the operating regime
                    elif exclude:
                        res_mem_spec.append(2)

                    dot = dot + 1

            # Titles for plotting
            x_title = '$W_{E_{1}_{P_{1}}$'
            y_title = '$W_{E_{1}_{S_{1}}$'
            plot_span_init_conds(res_mem_spec, wEP_withins, wES_withins, x_title, y_title,
                                 dir_plot, name, n, plot_bars=0, plot_legends=0, format='.png', title=title)
            plot_span_init_conds(res_mem_spec, wEP_withins, wES_withins, x_title, y_title,
                                 dir_plot, name, n, plot_bars=0, plot_legends=0, format='.svg', title=title)

    # Spanning the initial conditions for wEE and wEP
    if search_wEE_wEP == 1:
        # Loop for two simulations (operating region and initial conditions search)
        for flags, name, hour_sim in zip(flags_list, names_list, l_hour_sim):
            # The initial conditions of the weights
            wEE_withins = np.linspace(0, 1, n)
            wEP_withins = np.linspace(0, 1, n)

            # Empty lists are created to hold data
            l_res_rates = ()
            l_res_weights = ()

            # Simulation will be run if the run_search flag is set to True
            if run_search:
                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 15 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 15 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 15 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Total number of timepoints in stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1

                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each,
                # 2 extra seconds to reach steady state initially
                sim_duration = int(hour_sim * 60 * 60 + (stim_duration + 10) * 2 + 2)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((10, n_time_points_phase2),
                                    dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.zeros((12, n_time_points_phase2),
                                    dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)
                av_threshold = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)
                results_list = [[], []]

                # Initial conditions are spanned in the loops
                print('Span of initial conditions started.')
                print('...')
                for w_EP_within in wEP_withins:
                    for w_EE_within in wEE_withins:
                        # Weights
                        w_ES_within = 0.7;
                        w_ES_cross = 0.21
                        w_PE_within = 0.3;
                        w_PE_cross = 0.1
                        w_PP_within = 0.2;
                        w_PP_cross = 0.1
                        w_PS_within = 0.3;
                        w_PS_cross = 0.1
                        w_SE_within = 0.4;
                        w_SE_cross = 0.1

                        # Cross-connections for the weights that are spanned
                        cross_within_ratio = 0.3  # Cross connection to within connection ratio
                        w_EP_cross = np.round(w_EP_within * cross_within_ratio, 3)
                        w_EE_cross = np.round(w_EE_within * cross_within_ratio, 3)

                        weights = (
                            w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                            w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                        # The model is run for the given parameters
                        model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)),
                              weights,
                              back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                        # The aversion threshold is calculated, see the find_baseline_reactivation() function for details
                        idx_av_threshold = find_baseline_reactivation(r_phase1[0])
                        av_threshold = r_phase1[0][idx_av_threshold + 1]

                        # Data is appended to the results list
                        results_list[0].append(
                            [r_phase1.copy(), r_phase2.copy(), r_phase3.copy(), max_E.copy(), av_threshold.copy()])
                        results_list[1].append([J_EE_phase1.copy(), J_phase2.copy()])

                # Results are saved, the name of the weights that are spanned are added to the name of the file
                with open(dir_data + 'wEEwEP_' + name + '.pkl', 'wb') as file:
                    pickle.dump(results_list, file)

                print('Initial conditions space is spanned.')
                print('\n')
            print('\n')

        # Plotting the memory specificity results for every initial condition pair of wEE and wEP if plot_results is True
        if plot_results:
            # Reading the results
            with open(dir_data + 'wEEwEP_' + name + '.pkl', 'rb') as file:
                results_list = pickle.load(file)

            # Reading the operating region and holding the binary results in an array
            with open(dir_data + r'wEEwEP_Op_reg.pkl', 'rb') as file:
                results_op_region = pickle.load(file)
            op_region_binary = np.zeros((n ** 2, 1))
            for idx, i in enumerate(results_op_region[1]):
                w_EE_within, w_EE_cross = i[0][0][0], i[0][1][0]
                op_region_binary[idx] = (i[0][:, -1] > [w_EE_within, w_EE_cross, w_EE_cross, w_EE_within]).all()
            print('Data is read')
            print('\n')

            # Arrays created to hold data
            l_res_rates, l_res_weights = results_list[0], results_list[1]
            rE1_phase1 = np.zeros((n ** 2, l_res_rates[0][0].shape[1]))
            rE1_phase2 = np.zeros((n ** 2, l_res_rates[0][1].shape[1]))
            rE2_phase3 = np.zeros((n ** 2, l_res_rates[0][2].shape[1]))
            max_E = np.zeros((n ** 2, 1))
            av_threshold = np.zeros((n ** 2, 1))
            min_r = np.zeros((n ** 2, 1))

            # Data is registered to the correct arrays
            for idx, i in enumerate(l_res_rates):
                rE1_phase1[idx], rE1_phase2[idx], rE2_phase3[idx] = i[0][0], i[1][0], i[2][1]
                max_E[idx], av_threshold[idx] = i[3], i[4]
                min_r[idx] = min(np.min(i[0]), np.min(i[2]))  # min(min(np.min(i[0]), np.min(i[1])), np.min(i[2]))

            # The span of the initial conditions to determine the memory specificity is started
            dot = 0  # every initial condition pair is called as one dot on the initial condition space
            res_mem_spec = []  # array to hold memory specificity results at each initial condition pair of wEE and wEP
            for w_EP_within in wEP_withins:
                for w_EE_within in wEE_withins:
                    # If the dot is out of the operating regime (either LTD is induced during conditioning and/or the
                    # firing rates goes to zero)
                    exclude = (not op_region_binary[dot])  # or (min_r[dot] < 0.005)

                    # If the rates explode assign np.nan
                    if max_E[dot] > 50:
                        res_mem_spec.append(np.nan)

                    # If the rates do not explode and if there is no degeneracy, determine the memory specificity
                    elif max_E[dot] <= 50 and not exclude:
                        specific_memory = (rE2_phase3[dot][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                                           int(stim_times[0][1] * (
                                                                       1 / (delta_t * sampling_rate_stim)))] <
                                           av_threshold[dot]).all()

                        if specific_memory:
                            res_mem_spec.append(1)
                        else:
                            res_mem_spec.append(0)

                    # If the dot is out of the operating regime
                    elif exclude:
                        res_mem_spec.append(2)

                    dot = dot + 1

            # Titles for plotting
            x_title = '$W_{E_{1}_{E_{1}}$'
            y_title = '$W_{E_{1}_{P_{1}}$'
            plot_span_init_conds(res_mem_spec, wEP_withins, wEE_withins, x_title, y_title,
                                 dir_plot, name, n, plot_bars=0, plot_legends=0, format='.png', title=title)
            plot_span_init_conds(res_mem_spec, wEP_withins, wEE_withins, x_title, y_title,
                                 dir_plot, name, n, plot_bars=0, plot_legends=0, format='.svg', title=title)

    # Spanning the initial conditions for wEE and wES
    if search_wEE_wES == 1:
        # loop for two simulations (operating region and initial conditions search)
        for flags, name, hour_sim in zip(flags_list, names_list, l_hour_sim):
            # The initial conditions of the weights
            wEE_withins = np.linspace(0, 1, n)
            wES_withins = np.linspace(0, 1, n)

            # Empty lists are created to hold data
            l_res_rates = ()
            l_res_weights = ()

            # Simulation will be run if the run_search flag is set to True
            if run_search:
                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(1, 0), (0, 1)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 24 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 28 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 15 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 15 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 15 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_E = 4.5
                g_P = 3.2
                g_S = 3
                back_inputs = (g_E, g_P, g_S, g_top_down_to_S)

                # Total number of timepoints in stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1

                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each,
                # 2 extra seconds to reach steady state initially
                sim_duration = int(hour_sim * 60 * 60 + (stim_duration + 10) * 2 + 2)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                J_EE_phase1 = np.zeros((4, n_time_points_stim), dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((10, n_time_points_phase2),
                                    dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2,theta1,theta2,beta1,beta2
                J_phase2 = np.zeros((12, n_time_points_phase2),
                                    dtype=np.float32)  # WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)
                av_threshold = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, r_phase2, r_phase3, max_E)
                l_res_weights = (J_EE_phase1, J_phase2)
                results_list = [[], []]

                print('Span of initial conditions started.')
                print('...')
                for w_ES_within in wES_withins:
                    for w_EE_within in wEE_withins:
                        # Weights
                        w_EP_within = 0.7;
                        w_EP_cross = 0.21
                        w_PE_within = 0.3;
                        w_PE_cross = 0.1
                        w_PP_within = 0.2;
                        w_PP_cross = 0.1
                        w_PS_within = 0.3;
                        w_PS_cross = 0.1
                        w_SE_within = 0.4;
                        w_SE_cross = 0.1

                        # Cross-connections for the weights that are spanned
                        cross_within_ratio = 0.3  # Cross connection to within connection ratio
                        w_EE_cross = np.round(w_EE_within * cross_within_ratio, 3)
                        w_ES_cross = np.round(w_ES_within * cross_within_ratio, 3)

                        weights = (
                        w_EE_within, w_EP_within, w_ES_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                        w_EE_cross, w_EP_cross, w_ES_cross, w_PE_cross, w_PP_cross, w_PS_cross, w_SE_cross)

                        # The model is run for the given parameters
                        model(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration * (1 / delta_t)),
                              weights,
                              back_inputs, g_stim, stim_times, taus, K, rheobases, flags=flags, flags_theta=flags_theta)

                        # The aversion threshold is calculated, see the find_baseline_reactivation() function for details
                        idx_av_threshold = find_baseline_reactivation(r_phase1[0])
                        av_threshold = r_phase1[0][idx_av_threshold + 1]

                        # Data is appended to the results list
                        results_list[0].append(
                            [r_phase1.copy(), r_phase2.copy(), r_phase3.copy(), max_E.copy(), av_threshold.copy()])
                        results_list[1].append([J_EE_phase1.copy(), J_phase2.copy()])

                # Results are saved, the name of the weights that are spanned are added to the name of the file
                with open(dir_data + 'wEEwES_' + name + '.pkl', 'wb') as file:
                    pickle.dump(results_list, file)

                print('Initial conditions space is spanned.')
                print('\n')
            print('\n')

        # Plotting the memory specificity results for every initial condition pair of wEE and wES if plot_results is True
        if plot_results:
            # Reading the results
            with open(dir_data + 'wEEwES_' + name + '.pkl', 'rb') as file:
                results_list = pickle.load(file)

            # Reading the operating region and holding the binary results in an array
            with open(dir_data + r'wEEwES_Op_reg.pkl', 'rb') as file:
                results_op_region = pickle.load(file)
            op_region_binary = np.zeros((n ** 2, 1))
            for idx, i in enumerate(results_op_region[1]):
                w_EE_within, w_EE_cross = i[0][0][0], i[0][1][0]
                op_region_binary[idx] = (i[0][:, -1] > [w_EE_within, w_EE_cross, w_EE_cross, w_EE_within]).all()
            print('Data is read')
            print('\n')

            # Arrays created to hold data
            l_res_rates, l_res_weights = results_list[0], results_list[1]
            rE1_phase1 = np.zeros((n ** 2, l_res_rates[0][0].shape[1]))
            rE1_phase2 = np.zeros((n ** 2, l_res_rates[0][1].shape[1]))
            rE2_phase3 = np.zeros((n ** 2, l_res_rates[0][2].shape[1]))
            max_E = np.zeros((n ** 2, 1))
            av_threshold = np.zeros((n ** 2, 1))
            min_r = np.zeros((n ** 2, 1))

            # Data is registered to the correct arrays
            for idx, i in enumerate(l_res_rates):
                rE1_phase1[idx], rE1_phase2[idx], rE2_phase3[idx] = i[0][0], i[1][0], i[2][1]
                max_E[idx], av_threshold[idx] = i[3], i[4]
                min_r[idx] = min(np.min(i[0]), np.min(i[2]))  # min(min(np.min(i[0]), np.min(i[1])), np.min(i[2]))

            # The span of the initial conditions to determine the memory specificity is started
            dot = 0  # every initial condition pair is called as one dot on the initial condition space
            res_mem_spec = []  # array to hold memory specificity results at each initial condition pair of wEE and wES
            for w_ES_within in wES_withins:
                for w_EE_within in wEE_withins:
                    # If the dot is out of the operating regime (either LTD is induced during conditioning and/or the
                    # firing rates goes to zero)
                    exclude = (not op_region_binary[dot])  # or (min_r[dot] < 0.005)

                    # If the rates explode assign np.nan
                    if max_E[dot] > 50:
                        res_mem_spec.append(np.nan)

                    # If the rates do not explode and if there is no degeneracy, determine the memory specificity
                    elif max_E[dot] <= 50 and not exclude:
                        specific_memory = (rE2_phase3[dot][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                                           int(stim_times[0][1] * (
                                                                       1 / (delta_t * sampling_rate_stim)))] <
                                           av_threshold[dot]).all()
                        if specific_memory:
                            res_mem_spec.append(1)
                        else:
                            res_mem_spec.append(0)

                    # If the dot is out of the operating regime
                    elif exclude:
                        res_mem_spec.append(2)

                    dot = dot + 1

            # Titles for plotting
            x_title = '$W_{E_{1}_{E_{1}}$'
            y_title = '$W_{E_{1}_{S_{1}}$'
            plot_span_init_conds(res_mem_spec, wEE_withins, wES_withins, x_title, y_title,
                                 dir_plot, 'wEEwES_' + name, n, plot_bars=0, plot_legends=0, format='.png', title=title)
            plot_span_init_conds(res_mem_spec, wEE_withins, wES_withins, x_title, y_title,
                                 dir_plot, 'wEEwES_' + name, n, plot_bars=0, plot_legends=0, format='.svg', title=title)




def analyze_model_3_compartmental(hour_sim, run_simulation=True, save_results=False, plot_results=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    directory = os.getcwd()

    dir_data = directory + r'\data\new_model\3_compartmental\\'
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\new_model\3_compartmental\stronger_exc_coupling\a\y\c\\'

    delta_t = 0.0001  # time step in seconds (0.1 ms)
    stim_duration = 15  # stimulation duration in seconds
    # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
    sim_duration = int(((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)*(1/delta_t))
    sampling_rate_stim = 20  # register data at every 20 step during phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = 200000  # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Total number of timepoints for stimulation and simulation
    n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
    n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1  # total no the rest

    l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
    l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta = 1.5 * (60 * 60)  # time constant of target activity in seconds(24h)
    tau_beta = 1.6 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
    tau_scaling_E = 1.2 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 1.2 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 1.2 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
    taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    g_stim, rheobases, g, K, lambdas = get_model_parameters(coupling='strong', excitation='high')

    # Initial conditions for plastic weights
    w_EP_within = 0.6; w_EP_cross = 0.18
    w_DS_within = 0.6; w_DS_cross = 0.18
    w_DE_within = 0.5; w_DE_cross = 0.3
    w_EE_within = 0.5; w_EE_cross = 0.3

    # Weights
    w_PE_within = 0.35; w_PE_cross = 0.10
    w_PP_within = 0.20; w_PP_cross = 0.10
    w_PS_within = 0.30; w_PS_cross = 0.10
    w_SE_within = 0.15; w_SE_cross = 0.10

    weights = (w_DE_within, w_EE_within, w_EP_within, w_DS_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
               w_DE_cross,  w_EE_cross,  w_EP_cross,  w_DS_cross,  w_PE_cross,  w_PP_cross,  w_PS_cross,  w_SE_cross)

    # Arrays created to hold data
    r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    I_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_exc_phase1 = np.zeros((8, n_time_points_stim), dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22
    r_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    set_phase2 = np.zeros((12, n_time_points_phase2), dtype=np.float32) # thetaDD1,thetaDD2,thetaBD1,thetaBD2,thetaE1,thetaE2,betaAD1,betaAD2,betaBD1,betaBD2,betaE1,betaE2
    I_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32) # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_phase2 = np.zeros((16, n_time_points_phase2),
                        dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
    r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    max_E = np.zeros(1, dtype=np.float32)

    # Lists to hold data arrays
    l_res_rates = (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E)
    l_res_weights = (J_exc_phase1, J_phase2)

    # The flags for activating the following plasticity mechanisms in the given order: Hebbian learning, three-factor Hebbian learning,
    # adaptive set point, E-to-E scaling, P-to-E scaling, S-to-E scaling



    flags_list = [(1, 1, 1, 1, 1, 1)]
    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1), (1, 1, 1, 1, 1, 0),
                  (1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 0), (1, 1, 1, 0, 0, 1)]
    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1)]
    flags_theta = (1, 1)

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_' + str(hour_sim) + 'h' # + '_k' + str(K) + '_td' + str(g_top_down_to_S)
        print('*****', title, '*****')

        if run_simulation:

            print('Simulation started.')
            print('\n')
            model_3_compartmental(delta_t, sampling_rate, l_res_rates, l_res_weights, sim_duration, weights, g, g_stim,
                                  stim_times, taus, K, rheobases, lambdas, flags=flags,flags_theta=flags_theta)

            idx_av_threshold = find_baseline_reactivation(r_phase1[0])
            av_threshold = r_phase1[0][idx_av_threshold + 1]

            if save_results:
                l_results = [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates,
                             l_res_weights, av_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates, l_res_weights,
             av_threshold, stim_times, stim_duration, sim_duration] = l_results

        if plot_results:
            print('Plotting the results.')
            plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2*10], l_res_rates, l_res_weights,
                                      av_threshold, stim_times, dir_plot + name, hour_sim, format='.png', scale_y=False)
            """plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights,
                                      av_threshold, stim_times, dir_plot + name, hour_sim, format='.svg')"""





def analyze_model_3_compartmental2(hour_sim, run_simulation=True, save_results=False, plot_results=False):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param run_simulation: True to run the numerical simulation, False to read the already saved data
    :param save_results: True to save the results
    :param plot_results: True to plot the results

    Multi-purpose function to analyze the model. Here we run (if run_simulation is True) our computational model to
    investigate the role of cell-type dependent synaptic scaling mechanisms in associative learning. We replicate the
    experimental procedure in [1] in model() in model.py. The model has two subnetworks, each consisted of a canonical
    circuit of excitatory pyramidal neurons (E), parvalbumin-positive interneurons (P), somatostatin-positive
    interneurons (S). The simulation procedure is divided into three phases:
        Phase 1 - Conditioning: The first subnetwork receives extra input representing the conditioned stimulus in [1].
        The parameters describing the stimulation (when and how much stimulation) is described in analyze_model()
        function. The onset response of the excitatory firing rate of the first subnetwork is defined as the aversion
        threshold of this network. Three-factor Hebbian learning is active during this period. Also, synaptic scaling
        mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 2: In the experiment [1], the novel stimulus is presented to the mice at 4h/24h/48h after conditioning.
        This phase corresponds to the waiting time after conditioning and before testing. During this phase, synaptic
        scaling mechanisms, adaptive target activity (theta) and target activity regulator (beta) are active.

        Phase 3 - Testing: In this phase, the second subnetwork receives extra input corresponds to the novel stimulus
        in [1]. The memory specificity/overgeneralization is determined whether the excitatory rate in the second
        subnetwork is below/above the aversion threshold, respectively.


    During simulation model() writes data to the data arrays. This data can be saved (if save_results is set to True)
    and the results can be plotted (if plot_results is set to True).

    [1] Wu, C. H., Ramos, R., Katz, D. B., & Turrigiano, G. G. (2021). Homeostatic synaptic scaling establishes the
    specificity of an associative memory. Current biology, 31(11), 2274-2285.
    """

    directory = os.getcwd()

    dir_data = directory + r'\data\new_model\3_compartmental\\'
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\new_model\3_compartmental\without gE\coupling\\'

    delta_t = 0.0001  # time step in seconds (0.1 ms)
    stim_duration = 15  # stimulation duration in seconds
    # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each, 2 extra seconds to reach steady state initially
    sim_duration = int(((hour_sim) * 60 * 60 + (stim_duration + 10) * 2 + 2)*(1/delta_t))
    sampling_rate_stim = 20  # register data at every 20 step during phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = 200000  # register data at every 2e5 time step (20 seconds) during phase 2 (in between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Total number of timepoints for stimulation and simulation
    n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
    n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1  # total no the rest

    l_time_points_stim = np.linspace(0, stim_duration + 10, n_time_points_stim)
    l_time_points_phase2 = np.linspace(0, hour_sim, n_time_points_phase2)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim * 60 * 60) + 5, int(hour_sim * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    # The stimuli are given as inputs to the populations.
    g_stim_E = np.array([(2.25, 0), (0, 2.25)])
    g_stim_P = np.array([(0.5, 0), (0, 0.5)])
    g_stim_S = np.array([(0, 0), (0, 0)])
    g_stim = (g_stim_E, g_stim_P, g_stim_S)

    # Time constants
    tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
    tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
    tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
    tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
    tau_theta = 2 * (60 * 60)  # time constant of target activity in seconds(24h)
    tau_beta = 2.6 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
    tau_scaling_E = 1.8 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
    tau_scaling_P = 1.8 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
    tau_scaling_S = 1.8 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
    taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

    # Rheobases (minimum input needed for firing rates to be above zero)
    rheobase_E, rheobase_P, rheobase_S = 1, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # Background inputs
    g_AD = 4
    g_BD = 6
    g_E = 0
    g_P = 4
    g_S = 3.2
    g_top_down_to_S = 0
    g = (g_AD, g_BD, g_E, g_P, g_S, g_top_down_to_S)

    # K parameter in target regulator equation, it tunes the steady state value of target activity and its regulator
    K = 0.2
    print('line 1638 in model.py !!!!!!!!!W term in ss formula is commented out!!!!!!!')

    # Constant that define the contribution of each current
    lambda_AD = 0.2 # in stronger lambda config it is 0.5
    lambda_BD = 0.4 # in stronger lambda config it is 0.3
    lambdas = (lambda_AD, lambda_BD)

    # Initial conditions for plastic weights
    w_EP_within = 0.6; w_EP_cross = 0.18
    w_DS_within = 0.6; w_DS_cross = 0.18
    w_DE_within = 0.5; w_DE_cross = 0.4
    w_EE_within = 0.5; w_EE_cross = 0.4

    # Weights
    w_PE_within = 0.35; w_PE_cross = 0.10
    w_PP_within = 0.20; w_PP_cross = 0.10
    w_PS_within = 0.30; w_PS_cross = 0.10
    w_SE_within = 0.15; w_SE_cross = 0.10

    weights = (w_DE_within, w_EE_within, w_EP_within, w_DS_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
               w_DE_cross,  w_EE_cross,  w_EP_cross,  w_DS_cross,  w_PE_cross,  w_PP_cross,  w_PS_cross,  w_SE_cross)

    # Arrays created to hold data
    r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    I_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_exc_phase1 = np.zeros((8, n_time_points_stim), dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22
    r_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    set_phase2 = np.zeros((12, n_time_points_phase2), dtype=np.float32) # thetaDD1,thetaDD2,thetaBD1,thetaBD2,thetaE1,thetaE2,betaAD1,betaAD2,betaBD1,betaBD2,betaE1,betaE2
    I_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32) # IAD1, IAD2, IBD1, IBD2, IE1, IE2
    J_phase2 = np.zeros((16, n_time_points_phase2),
                        dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
    r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
    max_E = np.zeros(1, dtype=np.float32)

    # Lists to hold data arrays
    l_res_rates = (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E)
    l_res_weights = (J_exc_phase1, J_phase2)

    # The flags for activating the following plasticity mechanisms in the given order: Hebbian learning, three-factor Hebbian learning,
    # adaptive set point, E-to-E scaling, P-to-E scaling, S-to-E scaling

    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1), (1, 1, 1, 1, 1, 0),
                  (1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 0), (1, 1, 1, 0, 0, 1)]
    flags_list = [(1, 1, 1, 1, 0, 0)]
    flags_list = [(1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 0)]
    flags_theta = (1, 1)

    for flags in flags_list:
        id, title = determine_name(flags)
        name = 'Case' + id + '_' + str(hour_sim) + 'h' # + '_k' + str(K) + '_td' + str(g_top_down_to_S)
        print('*****', title, '*****')

        if run_simulation:

            print('Simulation started.')
            print('\n')
            model_3_compartmental(delta_t, sampling_rate, l_res_rates, l_res_weights, sim_duration, weights, g, g_stim,
                                  stim_times, taus, K, rheobases, lambdas, flags=flags,flags_theta=flags_theta)

            idx_av_threshold = find_baseline_reactivation(r_phase1[0])
            av_threshold = r_phase1[0][idx_av_threshold + 1]

            if save_results:
                l_results = [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates,
                             l_res_weights, av_threshold, stim_times, stim_duration, sim_duration]

                # Open a file and save
                with open(dir_data + name + '.pkl', 'wb') as file:
                    # A new file will be created
                    pickle.dump(l_results, file)
                print('Data is saved.')

        else:
            # Open the file and read
            with open(dir_data + name + '.pkl', 'rb') as file:
                l_results = pickle.load(file)
            print('Data is read.')

            [l_time_points_stim, l_time_points_phase2, delta_t, sampling_rate, l_res_rates, l_res_weights,
             av_threshold, stim_times, stim_duration, sim_duration] = l_results

        if plot_results:
            print('Plotting the results.')
            plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2*10], l_res_rates, l_res_weights,
                                      av_threshold, stim_times, dir_plot + name, hour_sim, format='.png', scale_y=False)
            """plot_all_3_compartmental([l_time_points_stim, l_time_points_phase2], l_res_rates, l_res_weights,
                                      av_threshold, stim_times, dir_plot + name, hour_sim, format='.svg')"""





def span_initial_conditions_3_compartmental(flag_case, hour_sim_case, g_top_down_to_S = 0, K = 0.5,
                            search_wEP_wAS=0, search_wEE_wEP=0, search_wEE_wES=0,
                            run_search=0, plot_results=1):
    """
    :param hour_sim: Defines how many hours does the simulation lasts
    :param run_search: True to run the numerical simulation
    :param plot_results: True to read the saved data and plot

    This function spans the initial conditions in pairs of W_ES, W_EP and W_EE weights if the corresponding flag for
    each combination is true (search_wEP_wES, search_wEE_wEP, search_wEE_wES). It saves the results (if run_search is
    True) or reads the saved data (if run_search is False). The memory specificity/overgeneralization result for every
    dot in the initial condition space can be plotted (if plot_results is True).
    """

    # Data and plotting directories are defined
    directory = r'C:\Users\AYCA\PycharmProjects\SynapticScaling' #os.getcwd()
    dir_data = directory + r'\data\new_model\3_compartmental\span\\'
    dir_plot = directory + r'\png\solving_it\10_corrected_eqs\new_model\3_compartmental\span\\'

    n = 21 # the number of different values for each weight
    stim_duration = 15 # stimulation duration in seconds
    delta_t = np.float32(0.0001) # time step in seconds (0.1 ms)
    sampling_rate_stim = 20 # register data at every 20 step in phase 1 and 3 (conditioning and testing)
    sampling_rate_sim = (1/delta_t)*10*60*60 # register data at every 10 min in phase 2 (between conditioning and testing)
    sampling_rate = (sampling_rate_stim, sampling_rate_sim)

    # Timepoints of the onset (first column) and offset (second column) of the first (first row) and second (second) stimuli.
    stim_times = np.array([[5, 5 + stim_duration],
                           [int(hour_sim_case * 60 * 60) + 5, int(hour_sim_case * 60 * 60) + 5 + stim_duration]]).reshape(2, 2)

    flags_list = [(1, 1, 1, 1, 1, 1), flag_case]
    flags_theta = (1, 1)
    id, title = determine_name(flag_case)

    # the name to identify the current case under investigation
    name = 'Case' + id + '_' + str(hour_sim_case) + 'h' + '_td' + str(g_top_down_to_S) + '_k' + str(K)

    # Lists of the names and simulation times for two simulations. The first simulation determines the operating region
    # whereas the second simulation runs the search
    names_list = [r'Op_reg', name]
    l_hour_sim = [int((stim_duration + 10 + 3)/(60 * 60)), # for determining only LTP cases, only conditioning phase needed
                  hour_sim_case] # case under investigation

    ### Following 3 if conditions provide the search for the 3 pairs of wEE, wEP and wES depending on the flags.

    # Spanning the initial conditions for wEP and wES
    if search_wEP_wAS == 1:
        # loop for two simulations (operating region and initial conditions search)
        for flags, name, hour_sim in zip(flags_list, names_list, l_hour_sim):
            # The initial conditions of the weights
            wAS_withins = np.linspace(0, 1, n)
            wEP_withins = np.linspace(0, 1, n)

            # Empty lists are created to hold data
            l_res_rates = ()
            l_res_weights = ()

            # Simulation will be run if the run_search flag is set to True
            if run_search:
                # The stimuli are given as inputs to the populations.
                g_stim_E = np.array([(2, 0), (0, 2)])
                g_stim_P = np.array([(0.5, 0), (0, 0.5)])
                g_stim_S = np.array([(0, 0), (0, 0)])
                g_stim = (g_stim_E, g_stim_P, g_stim_S)

                # Time constants
                tau_E = 0.02  # time constant of E population firing rate in seconds(20ms)
                tau_P = 0.005  # time constant of P population firing rate in seconds(5ms)
                tau_S = 0.01  # time constant of S population firing rate in seconds(10ms)
                tau_hebb = 120  # time constant of three-factor Hebbian learning in seconds(2min)
                tau_theta = 1.5 * (60 * 60)  # time constant of target activity in seconds(24h)
                tau_beta = 1.6 * (60 * 60)  # time constant of target activity regulator in seconds(28h)
                tau_scaling_E = 1.2 * (60 * 60)  # time constant of E-to-E scaling in seconds (15h)
                tau_scaling_P = 1.2 * (60 * 60)  # time constant of P-to-E scaling in seconds (15h)
                tau_scaling_S = 1.2 * (60 * 60)  # time constant of S-to-E scaling in seconds (15h)
                taus = (tau_E, tau_P, tau_S, tau_hebb, tau_scaling_E, tau_scaling_P, tau_scaling_S, tau_theta, tau_beta)

                # Rheobases (minimum input needed for firing rates to be above zero)
                rheobase_E, rheobase_P, rheobase_S = 1.5, 1.5, 1.5
                rheobases = (rheobase_E, rheobase_P, rheobase_S)

                # Background inputs
                g_AD = 2
                g_BD = 2
                g_E = 2
                g_P = 4
                g_S = 3.2
                g_top_down_to_S = 0
                g = (g_AD, g_BD, g_E, g_P, g_S, g_top_down_to_S)

                # Constant that define the contribution of each current
                lambda_AD = 0.2  # in stronger lambda config it is 0.5
                lambda_BD = 0.4  # in stronger lambda config it is 0.3
                lambdas = (lambda_AD, lambda_BD)

                # Total number of timepoints in stimulation and simulation
                n_time_points_stim = int((stim_duration + 10) * (1 / delta_t) * (1 / sampling_rate_stim))
                n_time_points_phase2 = int((hour_sim * 60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim)) + 1

                # Simulation duration in seconds, 5 extra seconds for pre- and post-stimulation each,
                # 2 extra seconds to reach steady state initially
                sim_duration = int(hour_sim * 60 * 60 + (stim_duration + 10) * 2 + 2)

                # Arrays created to hold data
                r_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                I_phase1 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
                J_exc_phase1 = np.zeros((8, n_time_points_stim),
                                        dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22
                r_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                set_phase2 = np.zeros((12, n_time_points_phase2),
                                      dtype=np.float32)  # thetaDD1,thetaDD2,thetaBD1,thetaBD2,thetaE1,thetaE2,betaAD1,betaAD2,betaBD1,betaBD2,betaE1,betaE2
                I_phase2 = np.zeros((6, n_time_points_phase2), dtype=np.float32)  # IAD1, IAD2, IBD1, IBD2, IE1, IE2
                J_phase2 = np.zeros((16, n_time_points_phase2),
                                    dtype=np.float32)  # WDE11,WDE12,WDE21,WDE22,WEE11,WEE12,WEE21,WEE22,WEP11,WEP12,WEP21,WEP22,WES11,WES12,WES21,WES22
                r_phase3 = np.zeros((6, n_time_points_stim), dtype=np.float32)  # rE1,rE2,rP1,rP2,rS1,rS2
                max_E = np.zeros(1, dtype=np.float32)

                # Lists to hold data arrays
                l_res_rates = (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E)
                l_res_weights = (J_exc_phase1, J_phase2)
                av_threshold = np.zeros(1, dtype=np.float32)

                results_list = [[], []]

                # Initial conditions are spanned in the loops
                print('Span of initial conditions started.')
                print('...')
                for w_AS_within in wAS_withins:
                    for w_EP_within in wEP_withins:

                        # Initial conditions for plastic weights
                        w_AB_within = 0.5; w_AB_cross = 0.4
                        w_BB_within = 0.5; w_BB_cross = 0.4

                        # Weights
                        w_PE_within = 0.35; w_PE_cross = 0.10
                        w_PP_within = 0.20; w_PP_cross = 0.10
                        w_PS_within = 0.30; w_PS_cross = 0.10
                        w_SE_within = 0.15; w_SE_cross = 0.10


                        # Cross-connections for the weights that are spanned
                        cross_within_ratio = 0.3  # Cross connection to within connection ratio
                        w_EP_cross = np.round(w_EP_within * cross_within_ratio, 3)
                        w_AS_cross = np.round(w_AS_within * cross_within_ratio, 3)

                        weights = (w_AB_within, w_BB_within, w_EP_within, w_AS_within, w_PE_within, w_PP_within, w_PS_within, w_SE_within,
                                   w_AB_cross,  w_BB_cross,  w_EP_cross,  w_AS_cross,  w_PE_cross,  w_PP_cross,  w_PS_cross,  w_SE_cross)

                        # The model is run for the given parameters
                        model_3_compartmental(delta_t, sampling_rate, l_res_rates, l_res_weights, int(sim_duration*(1/delta_t)), weights,
                              g, g_stim, stim_times, taus, K, rheobases, lambdas, flags=flags, flags_theta=flags_theta)

                        # The aversion threshold is calculated, see the find_baseline_reactivation() function for details
                        idx_av_threshold = find_baseline_reactivation(r_phase1[0])
                        av_threshold[0] = r_phase1[0][idx_av_threshold + 1]

                        # Data is appended to the results list
                        results_list[0].append([r_phase1.copy(), r_phase2.copy(), r_phase3.copy(), max_E.copy(), av_threshold.copy()])
                        results_list[1].append([J_exc_phase1.copy(), J_phase2.copy()])

                # Results are saved, the name of the weights that are spanned are added to the name of the file
                with open(dir_data + 'wEPwAS_' + name + '.pkl', 'wb') as file:
                    pickle.dump(results_list, file)

                print('Initial conditions space is spanned.')
                print('\n')
            print('\n')

        # Plotting the memory specificity results for every initial condition pair of wEP and wES if plot_results is True
        if plot_results:
            # Reading the results
            with open(dir_data + 'wEPwAS_' + name + '.pkl', 'rb') as file:
                results_list = pickle.load(file)

            # Reading the operating region and holding the binary results in an array
            with open(dir_data + r'wEPwAS_Op_reg.pkl', 'rb') as file:
                results_op_region = pickle.load(file)
            op_region_binary = np.zeros((n ** 2, 1))
            for idx, i in enumerate(results_op_region[1]):
                w_AB_within, w_AB_cross = i[0][0][0], i[0][1][0]
                w_BB_within, w_BB_cross = i[0][4][0], i[0][5][0]
                op_region_binary[idx] = (np.max(i[0], axis=1) > [w_AB_within, w_AB_cross, w_AB_cross, w_AB_within,
                                                        w_BB_within, w_BB_cross, w_BB_cross, w_BB_within]).all()
            print('Data is read')
            print('\n')

            # Arrays created to hold data
            l_res_rates, l_res_weights = results_list[0], results_list[1]
            rE1_phase1 = np.zeros((n ** 2, l_res_rates[0][0].shape[1]))
            rE2_phase3 = np.zeros((n ** 2, l_res_rates[0][2].shape[1]))
            max_E = np.zeros((n ** 2, 1))
            av_threshold = np.zeros((n ** 2, 1))
            min_r = np.zeros((n ** 2, 1))

            # Data is registered to the correct arrays
            for idx, i in enumerate(l_res_rates):
                rE1_phase1[idx], rE2_phase3[idx] = i[0][0], i[2][1]
                max_E[idx] , av_threshold[idx] = i[3], i[4]
                min_r[idx] = min(np.min(i[0]), np.min(i[2])) #min(min(np.min(i[0]), np.min(i[1])), np.min(i[2]))

            # The span of the initial conditions to determine the memory specificity is started
            dot = 0 # every initial condition pair is called as one dot on the initial condition space
            res_mem_spec = [] # array to hold memory specificity results at each initial condition pair of wEP and wES
            for w_AS_within in wAS_withins:
                for w_EP_within in wEP_withins:
                    # If the dot is out of the operating regime (either LTD is induced during conditioning and/or the
                    # firing rates goes to zero)
                    exclude = (not op_region_binary[dot]) #or (min_r[dot] < 0.005)

                    if np.round(w_EP_within,3)==0.5 and np.round(w_AS_within,3)==0.75:
                        print(max_E[dot])
                        print(np.max(rE1_phase1))

                    # If the rates explode assign np.nan
                    if max_E[dot] > 50:
                        res_mem_spec.append(np.nan)

                    # If the rates do not explode and if there is no degeneracy, determine the memory specificity
                    elif max_E[dot] <= 50 and not exclude:
                        specific_memory = (rE2_phase3[dot][int(stim_times[0][0] * (1 / (delta_t * sampling_rate_stim))):
                                           int(stim_times[0][1] * (1 / (delta_t * sampling_rate_stim)))] < av_threshold[dot]).all()
                        if specific_memory:
                            res_mem_spec.append(1)
                        else:
                            res_mem_spec.append(0)

                    # If the dot is out of the operating regime
                    elif exclude:
                        res_mem_spec.append(2)

                    dot = dot + 1

            # Titles for plotting
            x_title = '$W_{E_{1}_{P_{1}}$'
            y_title = '$W_{A_{1}_{S_{1}}$'
            plot_span_init_conds(res_mem_spec, wEP_withins, wAS_withins, x_title, y_title,
                              dir_plot, name, n, plot_bars=0, plot_legends=0, format='.png', title=title)
            plot_span_init_conds(res_mem_spec, wEP_withins, wAS_withins, x_title, y_title,
                              dir_plot, name, n, plot_bars=0, plot_legends=0, format='.svg', title=title)

