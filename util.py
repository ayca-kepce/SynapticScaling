import numpy as np
from matplotlib import pyplot as plt
import pickle
from numba import cuda, jit
from numba.types import float64, int64

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
    indices = np.random.choice(num_elements, replace=False, size=int(np.round(num_elements * (1-probability),1)))
    weights[np.unravel_index(indices, weights.shape)] = 0
    """print("probability is ", probability)
    print("calculated as ", np.count_nonzero(weights)/num_elements)"""
    return weights*strength

def show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_time, delta_t, title = None, save_figure=None, name=None):
    ratio = .6
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
    plt.axvline(x=stim_time, color='r', label='Stimulus', linestyle='dashed')
    stim_time = int(stim_time*(1/delta_t))
    plt.plot(t[:], rE1[:], 'xkcd:green', label='PC1, base:'
        + str(np.round(np.mean(rE1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE1[-200:]), 3)), linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', label='PC2, base:'
        + str(np.round(np.mean(rE2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE2[-200:]), 3)),linewidth=plot_line_width)
    plt.plot(t[:], rP1[:], 'xkcd:dodger blue', label='PV, base:'
        + str(np.round(np.mean(rP1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rP1[-200:]), 3)), linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rP2[:], 'xkcd:duck egg blue', label='PV, base:'
        + str(np.round(np.mean(rP2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rP2[-200:]), 3)),linewidth=plot_line_width)
    plt.plot(t[:], rS1[:], 'xkcd:coral', label='SST, base:'
        + str(np.round(np.mean(rS1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rS1[-200:]), 3)), linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rS2[:], 'xkcd:pale pink', label='SST, base:'
        + str(np.round(np.mean(rS2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rS2[-200:]), 3)),linewidth=plot_line_width)

    if title is not None:
        plt.title(str(title))

    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rates [Hz]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    #plt.ylim(0,1e-14)

    if save_figure:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def show_plot_weights(w1,w2,w3,w4, t, stim_time, title = None, save_figure=None, name=None):
    ratio = .6
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
    plt.axvline(x=stim_time, color='r', label='Stimulus', linestyle='dashed')
    plt.plot(t[:], w1[:], 'xkcd:green',         label='Average J_EE11 after stimulus:'+ str(np.round(w1[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], w2[:], 'xkcd:yellowgreen',   label='Average J_EE12 after stimulus:'+ str(np.round(w2[-1], 3)), linewidth=plot_line_width)
    plt.plot(t[:], w3[:], 'xkcd:dodger blue',   label='Average J_EE21 after stimulus:'+ str(np.round(w3[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], w4[:], 'xkcd:duck egg blue', label='Average J_EE22 after stimulus:'+ str(np.round(w4[-1], 3)), linewidth=plot_line_width)
    if title is not None:
        plt.title(str(title))

    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Weigths [a.u.]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if save_figure:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def show_plot_plasticity_terms(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_time, title = None, save_figure=None, name=None):
    ratio = .6
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
    plt.axvline(x=stim_time, color='r', label='Stimulus', linestyle='dashed')
    plt.plot(t[:], rE1[:], 'xkcd:green', label='Hebbian term EE22', linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', label='Syn. scaling term EE22',linewidth=plot_line_width)
    plt.plot(t[:], rP1[:], 'xkcd:dodger blue', label='Hebbian term EE12', linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rP2[:], 'xkcd:duck egg blue', label='Syn. scaling term EE22',linewidth=plot_line_width)
    plt.plot(t[:], rS1[:], 'xkcd:coral', label='Syn. scaling term EP21', linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rS2[:], 'xkcd:pale pink', label='Syn. scaling term ES21',linewidth=plot_line_width)

    if title is not None:
        plt.title(str(title))

    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Value of the term [a.u.]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    #plt.ylim(-1e-15,1e-15)
    #plt.xlim(1.395,t[-1])

    if save_figure:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def show_plot_indv_neurons(neurons, t, stim_time):
    ratio = .7
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
    for i in np.arange(neurons.shape[0]):
        plt.plot(t[:], neurons[i,:])

    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rates [Hz]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.show()

def create_save_noise_2PC(N_PC,N_PV,N_SOM,t,noise_strength,path):
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

def create_save_weights_2PC(N_PC,N_PV,N_SOM,weight_strengths,path):
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

def create_save_noise_2D(N_PC,N_PV,N_SOM,t,noise_strength,path):
    np.random.seed(1)
    noise_E1 = np.float32(np.random.rand(N_PC,len(t)) * noise_strength)

    np.random.seed(2)
    noise_E2 = np.float32(np.random.rand(N_PC,len(t)) * noise_strength)

    np.random.seed(3)
    noise_P1 = np.float32(np.random.rand(N_PV,len(t)) * noise_strength)

    np.random.seed(5)
    noise_P2 = np.float32(np.random.rand(N_PV,len(t)) * noise_strength)

    np.random.seed(4)
    noise_S1 = np.float32(np.random.rand(N_SOM,len(t)) * noise_strength)

    np.random.seed(6)
    noise_S2 = np.float32(np.random.rand(N_SOM,len(t)) * noise_strength)

    noises = (noise_E1,noise_E2,noise_P1,noise_P2,noise_S1,noise_S2)

    pickle.dump(noises, open(path, 'wb'))

def create_save_weights_2D(N_PC,N_PV,N_SOM,weight_strengths,weight_probabilities,uniform_distribution=0,path=None,save_weigths=0):
    s_DE11, s_DE12, s_DS11, s_EP11, s_PE11, s_SE11, s_PS11, s_PP11, s_DS12, s_EP12, s_PE12, s_SE12, s_PS12, s_PP12, \
    s_DE22, s_DE21, s_DS21, s_EP21, s_PE21, s_SE21, s_PS21, s_PP21, s_DS22, s_EP22, s_PE22, s_SE22, s_PS22, s_PP22 = weight_strengths

    p_DE11, p_DE12, p_DS11, p_EP11, p_PE11, p_SE11, p_PS11, p_PP11, p_DS12, p_EP12, p_PE12, p_SE12, p_PS12, p_PP12, \
    p_DE22, p_DE21, p_DS21, p_EP21, p_PE21, p_SE21, p_PS21, p_PP21, p_DS22, p_EP22, p_PE22, p_SE22, p_PS22, p_PP22 = weight_probabilities

    if uniform_distribution:
        a = 1
        b = .5
    else:
        a = 0
        b = 1
    np.random.seed(124)
    w_DE11 = connection_probability(a * np.random.rand(N_PC, N_PC) + b * np.ones((N_PC, N_PC)), p_DE11, s_DE11)  # self-excitation of PC1
    w_DE12 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_DE12, s_DE12)  # lateral-excitation of PC1-PC2
    w_DS11 = connection_probability(a* np.random.rand(N_PC, N_SOM) + b * np.ones((N_PC, N_SOM)), p_DS11, s_DS11)  # inhibition of D1
    w_DS12 = connection_probability(a* np.random.rand(N_PC, N_SOM) + b * np.ones((N_PC, N_SOM)), p_DS12, s_DS12)  # inhibition of D1
    w_EP11 = connection_probability(a* np.random.rand(N_PC, N_PV)  + b * np.ones((N_PC, N_PV)), p_EP11, s_EP11)  # inhibition of E1
    w_EP12 = connection_probability(a* np.random.rand(N_PC, N_PV)  + b * np.ones((N_PC, N_PV)), p_EP12, s_EP12)  # inhibition of E1
    # excitation of inhibitory neurons
    w_PE11 = connection_probability(a* np.random.rand(N_PV, N_PC)  + b * np.ones((N_PV, N_PC)), p_PE11, s_PE11)
    w_PE12 = connection_probability(a* np.random.rand(N_PV, N_PC)  + b * np.ones((N_PV, N_PC)), p_PE12, s_PE12)
    w_SE11 = connection_probability(a* np.random.rand(N_SOM, N_PC) + b * np.ones((N_SOM, N_PC)), p_SE11, s_SE11)
    w_SE12 = connection_probability(a* np.random.rand(N_SOM, N_PC) + b * np.ones((N_SOM, N_PC)), p_SE12, s_SE12)

    # PC neuron population 2
    # excitation and inhibition on PC neurons
    w_DE21 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_DE21, s_DE21)  # self-excitation of PC2
    w_DE22 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_DE22, s_DE22)  # lateral-excitation of PC1-PC2
    w_DS21 = connection_probability(a* np.random.rand(N_PC, N_SOM) + b * np.ones((N_PC, N_SOM)), p_DS21, s_DS21)  # inhibition of D2
    w_DS22 = connection_probability(a* np.random.rand(N_PC, N_SOM) + b * np.ones((N_PC, N_SOM)), p_DS22, s_DS22)  # inhibition of D2
    w_EP21 = connection_probability(a* np.random.rand(N_PC, N_PV)  + b * np.ones((N_PC, N_PV)), p_EP21, s_EP21)  # inhibition of E2
    w_EP22 = connection_probability(a* np.random.rand(N_PC, N_PV)  + b * np.ones((N_PC, N_PV)), p_EP22, s_EP22)  # inhibition of E2
    # excitation of inhibitory neurons
    w_PE21 = connection_probability(a* np.random.rand(N_PV, N_PC)  + b * np.ones((N_PV, N_PC)), p_PE21, s_PE21)
    w_PE22 = connection_probability(a* np.random.rand(N_PV, N_PC)  + b * np.ones((N_PV, N_PC)), p_PE22, s_PE22)
    w_SE21 = connection_probability(a* np.random.rand(N_SOM, N_PC) + b * np.ones((N_SOM, N_PC)), p_SE21, s_SE21)
    w_SE22 = connection_probability(a* np.random.rand(N_SOM, N_PC) + b * np.ones((N_SOM, N_PC)), p_SE22, s_SE22)

    # excitation and inhibition on inhibitory neurons
    w_PS11 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS11, s_PS11)  # inhibition of P1, disinhibiting E1
    w_PS12 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS12, s_PS12)  # inhibition of P2, disinhibiting E1
    w_PS21 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS21, s_PS21)  # inhibition of P1, disinhibiting E2
    w_PS22 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS22, s_PS22)  # inhibition of P2, disinhibiting E2
    w_PP11 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP11, s_PP11)  # self-inhibition of P1, disinhibiting E1
    w_PP12 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP12, s_PP12)  # self-inhibition of P2, disinhibiting E1
    w_PP21 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP21, s_PP21)  # self-inhibition of P1, disinhibiting E2
    w_PP22 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP22, s_PP22)  # self-inhibition of P2, disinhibiting E2

    if save_weigths:
        pickle.dump((w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
                 w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22), open(path, 'wb'))
    else:
        return (w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
                 w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)
def show_weights(weights):
    names = ["w_DE11", "w_DE12", "w_DS11", "w_EP11", "w_PE11", "w_SE11", "w_PS11", "w_PP11", "w_DS12", "w_EP12", "w_PE12", "w_SE12", "w_PS12", "w_PP12",
             "w_DE22", "w_DE21", "w_DS21", "w_EP21", "w_PE21", "w_SE21", "w_PS21", "w_PP21", "w_DS22", "w_EP22", "w_PE22", "w_SE22", "w_PS22", "w_PP22"]

    for i in range(len(weights)):
        #i = i + 15
        plt.figure()
        plt.title(names[i])
        plt.imshow(weights[i])
        plt.savefig(r"C:\Users\AYCA\Desktop\Internship\figures\weights\\" + str(names[i]) + ".png")

@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def round_array(x, decimals, out):
    return np.round_(x, decimals, out)

@jit(nopython=True)
def apply_hard_bound(array, lower_bound, upper_bound):
    for row in array:
        row[row < lower_bound] = lower_bound
        row[row > upper_bound] = upper_bound
    return array



"""
def save_params(path, config_id, params):
    pickle.dump(params, open(path + config_id + '.pkl', 'wb'))

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
"""