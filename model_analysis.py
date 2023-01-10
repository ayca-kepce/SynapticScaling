import numpy as np
import matplotlib.pyplot as plt
from util import *
import sys
from model import *
import os

def analyze_model():
    directory = os.getcwd()
    sim_duration = 250  # s
    delta_t = 0.0001  # s
    timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, timepoints)
    stim_start = 5
    stim_stop = 10
    stim_strength_E = 1
    stim_strength_P = .5
    stim_strengths = (stim_strength_E,stim_strength_P)
    upper_bound_E = 5
    upper_bound_P = 1
    upper_bound_S = 1
    upper_bounds = (upper_bound_E,upper_bound_P,upper_bound_S)

    rE1 = np.zeros(timepoints); rE2 = np.zeros(timepoints)
    rP1 = np.zeros(timepoints); rP2 = np.zeros(timepoints)
    rS1 = np.zeros(timepoints); rS2 = np.zeros(timepoints)
    av_I1 = np.zeros(timepoints); av_I2 =np.zeros(timepoints)

    J_EE11 = np.zeros(timepoints); J_EE12 = np.zeros(timepoints)
    J_EE21 = np.zeros(timepoints); J_EE22 = np.zeros(timepoints)

    J_EP11 = np.zeros(timepoints); J_EP12 = np.zeros(timepoints)
    J_EP21 = np.zeros(timepoints); J_EP22 = np.zeros(timepoints)

    J_ES11 = np.zeros(timepoints); J_ES12 = np.zeros(timepoints)
    J_ES21 = np.zeros(timepoints); J_ES22 = np.zeros(timepoints)

    vars = (rE1, rE2, rP1, rP2, rS1, rS2,
            av_I1,av_I2,
            J_EE11, J_EE12, J_EE21, J_EE22,
            J_EP11, J_EP12, J_EP21, J_EP22,
            J_ES11, J_ES12, J_ES21, J_ES22)

    hebEE11,hebEE12,hebEE21,hebEE22 = np.zeros(timepoints),np.zeros(timepoints),np.zeros(timepoints),np.zeros(timepoints)
    ss1_list, ss2_list = np.zeros(timepoints),np.zeros(timepoints)
    av_theta_I1,av_theta_I2 = np.zeros(timepoints),np.zeros(timepoints)
    LR_EE11, LR_EE12, LR_EE21, LR_EE22 = np.zeros(timepoints),np.zeros(timepoints),np.zeros(timepoints),np.zeros(timepoints)
    plas_terms = (hebEE11,hebEE12,hebEE21,hebEE22,
                  ss1_list,ss2_list,
                  av_theta_I1,av_theta_I2,
                  LR_EE11, LR_EE12, LR_EE21, LR_EE22)

    tau_E = 0.02  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    tau_plas = 100 # ms
    tau_scaling = 1000 # ms
    tau_theta = 20 # ms
    tau_LR = 750 # ms
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta, tau_LR)

    lambda_D = 0.27; lambda_E = 0.31
    lambdas = (lambda_D, lambda_E)

    rheobase_E, rheobase_P, rheobase_S =  1.5, 1.5, 1.5
    rheobases = (rheobase_E, rheobase_P, rheobase_S)

    # background inputs
    x_E = 4.5
    x_P = 3
    x_S = 3
    back_inputs = (x_E, x_P, x_S)

    w_DE11, w_DE12, w_DE21, w_DE22 = .4, .3, .3, .4 #not active
    w_EE11, w_EE12, w_EE21, w_EE22 = .4, .3, .3, .4
    w_DS11, w_DS12, w_DS21, w_DS22 = .5, .2, .2, .5
    w_EP11, w_EP12, w_EP21, w_EP22 = .5, .2, .2, .5
    w_PE11, w_PE12, w_PE21, w_PE22 = .3, .1, .1, .3
    w_SE11, w_SE12, w_SE21, w_SE22 = .4, .1, .1, .4
    w_PS11, w_PS12, w_PS21, w_PS22 = .3, .1, .1, .3
    w_PP11, w_PP12, w_PP21, w_PP22 = .2, .1, .1, .2

    """w_DE11, w_DE12, w_DE21, w_DE22 = .4, .3, .3, .4
    w_EE11, w_EE12, w_EE21, w_EE22 = .4, .3, .3, .4
    w_DS11, w_DS12, w_DS21, w_DS22 = .5, .2, .2, .5
    w_EP11, w_EP12, w_EP21, w_EP22 = .5, .2, .2, .5
    w_PE11, w_PE12, w_PE21, w_PE22 = .3, .1, .1, .3
    w_SE11, w_SE12, w_SE21, w_SE22 = .4, .1, .1, .4
    w_PS11, w_PS12, w_PS21, w_PS22 = .3, .1, .1, .3
    w_PP11, w_PP12, w_PP21, w_PP22 = .2, .1, .1, .2"""

    weights = (
        w_EE11, w_EE12, w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11,
        w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
        w_EE22, w_EE21, w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21,
        w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

    LR_E01,LR_E02 = 1,1
    learning_rates = (LR_E01, LR_E02)
    adaptive_LR_method = "3-factor"
    synaptic_scaling_method = "multiplicative"
    synaptic_scaling_update_method = "every_timestep"
    synaptic_scaling_compare_method = "individual"

    BCM_p = 1
    threshold_ss = .07
    ss_exponential = 1

    hebbian_plasticity_flag = 1
    exc_scaling_flag = 0
    inh_scaling_flag = 0
    adaptive_threshold_flag = 1
    adaptive_LR_flag = 0

    flags = (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag,
             adaptive_threshold_flag, adaptive_LR_flag)

    flags_list = [(1,1,1,1,1),(1,0,0,1,1),(1,1,1,1,0),(1,1,1,0,1)]
    flags_list = [(1,0,0,0,1),(1,1,1,1,1),(1,0,0,1,1),(1,1,1,0,1)]
    flags_list = [(1,0,0,1,0)]

    """orig_stdout = sys.stdout
    f = open(directory + r"\png\explore2\back_to_the_future\\" + '0values.txt', 'w')
    sys.stdout = f"""

    for flags in flags_list:
        name, title = determine_name(flags)
        print("*****", name)
        model(delta_t, vars, plas_terms, t, weights,
                                                   back_inputs, stim_strengths, stim_start, stim_stop, taus, lambdas,
                                                   rheobases, upper_bounds, learning_rates, adaptive_LR_method,
                                                   synaptic_scaling_method, synaptic_scaling_update_method, synaptic_scaling_compare_method,
                                                   BCM_p, threshold_ss, ss_exponential, flags=flags)

        """plot_all_mass(t, vars, plas_terms, stim_start, stim_stop, stim_strength_E, delta_t,
                      r"\explore2\back_to_the_future\\" + name, title)"""

        # plotting configuration
        ratio = 1.5
        figure_len, figure_width = 15 * ratio, 12 * ratio
        font_size_1, font_size_2 = 36 * ratio, 36 * ratio
        legend_size = 18 * ratio
        line_width, tick_len = 3 * ratio, 12 * ratio
        marker_size = 15 * ratio
        marker_edge_width = 3 * ratio
        plot_line_width = 5 * ratio
        hfont = {'fontname': 'Arial'}

        sns.set(style='ticks')
        pal = sns.color_palette()
        color_list = pal.as_hex()
        b_plot_entire = True
        stim_time = int(stim_start * (1 / delta_t))
        xmin = 0.1
        xmax = 250

        # plot here
        ymin = 0
        ymax = 5
        stim_label_y = (ymax - ymin) * .95
        plt.figure(figsize=(figure_len, figure_width))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_start, stim_stop, color='gray', alpha=0.1)
        plt.text(7.5, stim_label_y, 'stimulus', fontsize=font_size_1,
                 horizontalalignment='center')

        plt.plot(t[0:len(t):200], plas_terms[6][0:len(t):200], 'r')
        plt.plot(t[0:len(t):200], rE1[0:len(t):200], 'b')
        plt.plot(t[0:len(t):200], plas_terms[7][0:len(t):200], 'r', linestyle='dashed')
        plt.plot(t[0:len(t):200], rE2[0:len(t):200], 'b', linestyle='dashed', )

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_1, **hfont)
        plt.title(title, fontsize=font_size_1, **hfont)
        ax.legend(fontsize=font_size_1, loc='upper right')

        plt.savefig('png/' + name + '_thetaANDactivity.png')
        plt.close()


        # show_plot_currents_mass(av_I1,av_I2,t,stim_start)
    """save_params([weights, back_inputs, stim_strengths, taus, lambdas, rheobases, upper_bounds,
                 learning_rates, adaptive_LR_method, synaptic_scaling_method,
                 synaptic_scaling_update_method, synaptic_scaling_compare_method,
                 BCM_p, threshold_ss, ss_exponential],
                directory + r"\png\explore2\back_to_the_future\\" + name)

    sys.stdout = orig_stdout
    f.close()"""
analyze_model()
