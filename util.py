import numpy as np
from matplotlib import pyplot as plt
import pickle
from numba import cuda, jit


def connection_probability(weights, probability, strength):
    """
    Function to randomly create the connections between two populations
    :param weights: A matrix filled with 1s in the size of number of post- and pre-synaptic neurons
    :param probability: The connectivity probability
    :param strength: The strength of the connections
    :return: 
    """
    np.random.seed(1)
    num_elements = weights.shape[1] * weights.shape[0]
    indices = np.random.choice(num_elements, replace=False, size=int(num_elements * (1-probability)))
    weights[np.unravel_index(indices, weights.shape)] = 0
    """print("probability is ", probability)
    print("calculated as ", np.count_nonzero(weights)/num_elements)"""
    return weights*strength

def save_params(path, config_id, params):
    pickle.dump(params, open(path + config_id + '.pkl', 'wb'))

def save_plot(name, rE, rP, rS, t):
    plt.plot(t[:], rE[:], 'r', label='PC, saturates at '+ str(np.round(np.mean(rE[-80:]),3)) + 'Hz')
    plt.plot(t[:], rP[:], 'g', label='PV, saturates at '+ str(np.round(np.mean(rP[-80:]),3)) + 'Hz')
    plt.plot(t[:], rS[:], 'b', label='SST, saturates at '+ str(np.round(np.mean(rS[-80:]),3)) + 'Hz')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('Firing rates [Hz]')
    plt.grid()
    plt.savefig(name)
    plt.clf()

def show_plot(rE, rP, rS, t):
    ratio = 1.5
    figure_len, figure_width = 15 * ratio, 12 * ratio
    font_size_1, font_size_2 = 36 * ratio, 36 * ratio
    legend_size = 18 * ratio
    line_width, tick_len = 3 * ratio, 10 * ratio
    marker_size = 15 * ratio
    plot_line_width = 5 * ratio
    hfont = {'fontname': 'Arial'}

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    #plt.yscale('symlog', linthreshy=0.1) # linthreshy throw an error, ignored since plot does not diverge to inf

    plt.plot(t[:], rE[:], 'xkcd:green',       linewidth=plot_line_width, label='PC, saturates at ' + str(np.round(np.mean(rE[-80:]),3)) + 'Hz')
    plt.plot(t[:], rP[:], 'xkcd:dodger blue', linewidth=plot_line_width, label='PV, saturates at ' + str(np.round(np.mean(rP[-80:]),3)) + 'Hz')
    plt.plot(t[:], rS[:], 'xkcd:coral',       linewidth=plot_line_width, label='SST, saturates at '+ str(np.round(np.mean(rS[-80:]),3)) + 'Hz')
    plt.legend(prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rates [Hz]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.show()

def show_plot_2PC(rE1, rE2, rP, rS, t):
    ratio = 1.5
    figure_len, figure_width = 15 * ratio, 12 * ratio
    font_size_1, font_size_2 = 36 * ratio, 36 * ratio
    legend_size = 18 * ratio
    line_width, tick_len = 3 * ratio, 10 * ratio
    marker_size = 15 * ratio
    plot_line_width = 5 * ratio
    hfont = {'fontname': 'Arial'}

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    #plt.yscale('symlog', linthreshy=0.1) # linthreshy throw an error, ignored since plot does not diverge to inf

    plt.plot(t[:], rE1[:], 'xkcd:green',       linewidth=plot_line_width, label='PC1, base activity at '+ str(np.round(np.mean(rE1[-80:]),3)), linestyle='dashed')
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', linewidth=plot_line_width, label='PC2, base activity at '+ str(np.round(np.mean(rE2[-80:]),3)), linestyle='dashed')
    plt.plot(t[:], rP[:],  'xkcd:dodger blue', linewidth=plot_line_width, label='PV,  base activity at :'+ str(np.round(np.mean(rP[-80:]),3)))
    plt.plot(t[:], rS[:],  'xkcd:coral',       linewidth=plot_line_width, label='SST, base activity at :'+ str(np.round(np.mean(rS[-80:]),3)))
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rates [Hz]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.show()


def show_plot_2PC_stim(rE1, rE2, rP, rS, t, stim_time):
    ratio = 1.5
    figure_len, figure_width = 15 * ratio, 12 * ratio
    font_size_1, font_size_2 = 36 * ratio, 36 * ratio
    legend_size = 18 * ratio
    line_width, tick_len = 3 * ratio, 10 * ratio
    marker_size = 15 * ratio
    plot_line_width = 5 * ratio
    hfont = {'fontname': 'Arial'}

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    # plt.yscale('symlog', linthreshy=0.1) # linthreshy throw an error, ignored since plot does not diverge to inf

    plt.plot(t[:], rE1[:], 'xkcd:green', label='PC1, base:'
        + str(np.round(np.mean(rE1[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE1[-80:]), 3)), linestyle='dashed')
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', label='PC2, base:'
        + str(np.round(np.mean(rE2[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE2[-80:]), 3)), linestyle='dashed')
    plt.plot(t[:], rP[:], 'xkcd:dodger blue', label='PV, base:'
        + str(np.round(np.mean(rP[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rP[-80:]), 3)))
    plt.plot(t[:], rS[:], 'xkcd:coral', label='SST, base:'
        + str(np.round(np.mean(rS[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rS[-80:]), 3)))


    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rates [Hz]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.show()



def show_plot_2PC_old(rE1, rE2, rP, rS, t, stim_time):
    plt.plot(t[:], rE1[:], 'xkcd:green', label='PC1, base:'
                                               + str(
        np.round(np.mean(rE1[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
                                               + str(np.round(np.mean(rE1[-80:]), 3)), linestyle='dashed')
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', label='PC2, base:'
                                                     + str(
        np.round(np.mean(rE2[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
                                                     + str(np.round(np.mean(rE2[-80:]), 3)), linestyle='dashed')
    plt.plot(t[:], rP[:], 'xkcd:dodger blue', label='PV, base:'
                                                    + str(
        np.round(np.mean(rP[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
                                                    + str(np.round(np.mean(rP[-80:]), 3)))
    plt.plot(t[:], rS[:], 'xkcd:coral', label='SST, base:'
                                              + str(
        np.round(np.mean(rS[stim_time - 80:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
                                              + str(np.round(np.mean(rS[-80:]), 3)))
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('Firing rates [Hz]')
    plt.grid()
    plt.show()


def create_save_noise(N_PC,N_PV,N_SOM,t,noise_strength,path):
    np.random.seed(1)
    noise_E1 = np.random.rand(N_PC,len(t)) * noise_strength

    np.random.seed(2)
    noise_E2 = np.random.rand(N_PC,len(t)) * noise_strength

    np.random.seed(3)
    noise_P = np.random.rand(N_PV,len(t)) * noise_strength

    np.random.seed(4)
    noise_S = np.random.rand(N_SOM,len(t)) * noise_strength

    noises = (noise_E1,noise_E2,noise_P,noise_S)

    pickle.dump(noises, open(path, 'wb'))

def create_save_weights(N_PC,N_PV,N_SOM,weight_strengths,path):
    s_DE1, s_DE12, s_DS1, s_EP1, s_PE1, s_SE1, s_DE2, s_DE21, s_DS2, s_EP2, s_PE2, s_SE2, s_PS, s_PP = weight_strengths
    np.random.seed(124)
    w_DE1 = connection_probability(1 * np.random.rand(N_PC, N_PC) + .5 * np.ones((N_PC, N_PC)), 0.10, s_DE1)  # self-excitation of PC1
    w_DE12 = connection_probability(1 * np.random.rand(N_PC, N_PC) + .5 * np.ones((N_PC, N_PC)), 0.10, s_DE12)  # lateral-excitation of PC1-PC2
    w_DS1 = connection_probability(1 * np.random.rand(N_PC, N_SOM) + .5 * np.ones((N_PC, N_SOM)), 0.55, s_DS1)  # inhibition of D1
    w_EP1 = connection_probability(1 * np.random.rand(N_PC, N_PV) + .5 * np.ones((N_PC, N_PV)), 0.60, s_EP1)  # inhibition of E1
    # excitation of inhibitory neurons
    w_PE1 = connection_probability(1 * np.random.rand(N_PV, N_PC) + .5 * np.ones((N_PV, N_PC)), 0.45, s_PE1)
    w_SE1 = connection_probability(1 * np.random.rand(N_SOM, N_PC) + .5 * np.ones((N_SOM, N_PC)), 0.35, s_SE1)

    # PC neuron population 2
    # excitation and inhibition on PC neurons
    w_DE2 = connection_probability(1 * np.random.rand(N_PC, N_PC) + .5 * np.ones((N_PC, N_PC)), 0.10, s_DE2)  # self-excitation of PC2
    w_DE21 = connection_probability(1 * np.random.rand(N_PC, N_PC) + .5 * np.ones((N_PC, N_PC)), 0.10, s_DE21)  # lateral-excitation of PC1-PC2
    w_DS2 = connection_probability(1 * np.random.rand(N_PC, N_SOM) + .5 * np.ones((N_PC, N_SOM)), 0.55, s_DS2)  # inhibition of D2
    w_EP2 = connection_probability(1 * np.random.rand(N_PC, N_PV) + .5 * np.ones((N_PC, N_PV)), 0.60, s_EP2)  # inhibition of E2
    # excitation of inhibitory neurons
    w_PE2 = connection_probability(1 * np.random.rand(N_PV, N_PC) + .5 * np.ones((N_PV, N_PC)), 0.45, s_PE2)
    w_SE2 = connection_probability(1 * np.random.rand(N_SOM, N_PC) + .5 * np.ones((N_SOM, N_PC)), 0.35, s_SE2)

    # excitation and inhibition on inhibitory neurons
    w_PS = connection_probability(1 * np.random.rand(N_PV, N_SOM) + .5 * np.ones((N_PV, N_SOM)), 0.60, s_PS)  # inhibition of P, disinhibiting E
    w_PP = connection_probability(1 * np.random.rand(N_PV, N_PV) + .5 * np.ones((N_PV, N_PV)), 0.50, s_PP)  # self-inhibition of P, disinhibiting E

    pickle.dump((w_DE1,w_DE12,w_DS1,w_EP1,w_PE1,w_SE1,w_DE2,w_DE21,w_DS2,w_EP2,w_PE2,w_SE2,w_PS,w_PP), open(path, 'wb'))
