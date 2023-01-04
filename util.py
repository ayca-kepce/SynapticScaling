import numpy as np
from matplotlib import pyplot as plt
import pickle
"""from numba import cuda, jit, njit
import numba as nb
from numba.types import float64, int64, ListType, List"""
import seaborn as sns

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


def show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_start, stim_stop,
                 delta_t, title = None, save_figure=None, name=None):
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
    # plt.axvline(x=stim_time, color='r', label='Stimulus', linestyle='dashed')

    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus")
    stim_time = int(stim_start*(1/delta_t))
    plt.plot(t[:], rE1[:], 'xkcd:green', label='PC1, base:'
        + str(np.round(np.mean(rE1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE1[-200:]), 3)), linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rE2[:], 'xkcd:yellowgreen', label='PC2, base:'
        + str(np.round(np.mean(rE2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rE2[-200:]), 3)),linewidth=plot_line_width)
    plt.plot(t[:], rP1[:], 'xkcd:dodger blue', label='PV1, base:'
        + str(np.round(np.mean(rP1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'
        + str(np.round(np.mean(rP1[-200:]), 3)), linestyle='dashed',linewidth=plot_line_width)
    plt.plot(t[:], rP2[:], 'xkcd:duck egg blue', label='PV2, base:'
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

    plt.legend(prop={"family": "Arial", 'size': legend_size})
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
    plt.plot(t[:], w1[:], 'xkcd:green',
             label='Average J_EE11 after stimulus:'+ str(np.round(w1[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], w2[:], 'xkcd:yellowgreen',
             label='Average J_EE12 after stimulus:'+ str(np.round(w2[-1], 3)), linewidth=plot_line_width)
    plt.plot(t[:], w3[:], 'xkcd:dodger blue',
             label='Average J_EE21 after stimulus:'+ str(np.round(w3[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], w4[:], 'xkcd:duck egg blue',
             label='Average J_EE22 after stimulus:'+ str(np.round(w4[-1], 3)), linewidth=plot_line_width)
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


def show_plot_currents(IE1,IE2,ID1,ID2, t, stim_time, title = None, save_figure=None, name=None):
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
    plt.plot(t[:], IE1[:], 'xkcd:green',
             label='Average I_E1 after stimulus:'+ str(np.round(IE1[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], IE2[:], 'xkcd:yellowgreen',
             label='Average I_E2 after stimulus:'+ str(np.round(IE2[-1], 3)), linewidth=plot_line_width)
    plt.plot(t[:], ID1[:], 'xkcd:dodger blue',
             label='Average I_D1 after stimulus:'+ str(np.round(ID1[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], ID2[:], 'xkcd:duck egg blue',
             label='Average I_D2 after stimulus:'+ str(np.round(ID2[-1], 3)), linewidth=plot_line_width)
    if title is not None:
        plt.title(str(title))

    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Currents [a.u.]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if save_figure:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def show_plot_currents_mass(IE1,IE2, t, stim_time, title = None, save_figure=None, name=None):
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
    plt.plot(t[:], IE1[:], 'xkcd:green',         label='Average I_E1 after stimulus:'+ str(np.round(IE1[-1], 3)), linewidth=plot_line_width, linestyle='dashed')
    plt.plot(t[:], IE2[:], 'xkcd:yellowgreen',   label='Average I_E2 after stimulus:'+ str(np.round(IE2[-1], 3)), linewidth=plot_line_width)
    if title is not None:
        plt.title(str(title))

    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont)
    plt.ylabel('Currents [a.u.]', fontsize=font_size_1, **hfont)
    plt.grid()
    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if save_figure:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def show_plot_plasticity_terms_2compartmental(plas_terms, t, stim_start, stim_stop,
                                              stim_strength_E, delta_t, title = None, save_figure=None, name=None):
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
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))
    
    labels = ["hebEE11", "hebEE12", "hebEE21", "hebEE22", "hebDE11", "hebDE12", "hebDE21", "hebDE22",
              "ssEE11", "ssEE12", "ssEE21", "ssEE22", "ssDE11", "ssDE12", "ssDE21", "ssDE22",
              "ssEP11", "ssEP12", "ssEP21", "ssEP22", "ssES11", "ssES12", "ssES21", "ssES22",
              "av_theta_E1", "av_theta_E2", "av_theta_D1", "av_theta_D2"]

    plt.subplot(421)
    plt.plot(t,plas_terms[0], label=labels[0])# + "last value is " + str(plas_terms[0][-1]))
    plt.plot(t,plas_terms[1], label=labels[1])# + "last value is " + str(plas_terms[1][-1]))
    plt.plot(t,plas_terms[2], label=labels[2])# + "last value is " + str(plas_terms[2][-1]))
    plt.plot(t,plas_terms[3], label=labels[3])# + "last value is " + str(plas_terms[3][-1]))
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')

    plt.subplot(422)
    plt.plot(t,plas_terms[4], label=labels[4])# + "last value is " + str(plas_terms[4][-1]))
    plt.plot(t,plas_terms[5], label=labels[5])# + "last value is " + str(plas_terms[5][-1]))
    plt.plot(t,plas_terms[6], label=labels[6])# + "last value is " + str(plas_terms[6][-1]))
    plt.plot(t,plas_terms[7], label=labels[7])# + "last value is " + str(plas_terms[7][-1]))
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')

    plt.subplot(423)
    plt.plot(t,plas_terms[8], label=labels[8])
    plt.plot(t,plas_terms[9], label=labels[9])
    plt.plot(t,plas_terms[10], label=labels[10])
    plt.plot(t,plas_terms[11], label=labels[11])
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')
    plt.ylabel('Value of the term [a.u.]', fontsize=font_size_1, **hfont)

    plt.subplot(424)
    plt.plot(t,plas_terms[12], label=labels[12])
    plt.plot(t,plas_terms[13], label=labels[13])
    plt.plot(t,plas_terms[14], label=labels[14])
    plt.plot(t,plas_terms[15], label=labels[15])
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')

    plt.subplot(425)
    plt.plot(t,plas_terms[16], label=labels[16])
    plt.plot(t,plas_terms[17], label=labels[17])
    plt.plot(t,plas_terms[18], label=labels[18])
    plt.plot(t,plas_terms[19], label=labels[19])
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')

    plt.subplot(426)
    plt.plot(t,plas_terms[20], label=labels[20])
    plt.plot(t,plas_terms[21], label=labels[11])
    plt.plot(t,plas_terms[22], label=labels[22])
    plt.plot(t,plas_terms[23], label=labels[23])
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')

    plt.subplot(427)
    plt.plot(t,plas_terms[24], label=labels[24] + " final value " + str(plas_terms[24,-1]))
    plt.plot(t,plas_terms[25], label=labels[25] + " final value " + str(plas_terms[25,-1]))
    plt.plot(t,plas_terms[26], label=labels[26] + " final value " + str(plas_terms[26,-1]))
    plt.plot(t,plas_terms[27], label=labels[27] + " final value " + str(plas_terms[27,-1]))
    plt.legend(prop={"family": "Arial", 'size': legend_size*.5}, loc='upper right')
    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont, loc='right')

    if title is not None:
        plt.title(title, fontsize=font_size_1, **hfont)

    plt.grid()

    if save_figure:
        plt.savefig('png/' + name + '_plasticity_terms.png')
        plt.close()
    else:
        plt.show()


def plot_all_2compartmental(t, vars, plas_terms, stim_start, stim_stop, stim_strength_E, delta_t, s_name, title):
    # extracting variables to plot
    (rE1, rE2, rP1, rP2, rS1, rS2, av_I_E1, av_I_E2, av_I_D1, av_I_D2,
     J_EE11, J_EE12, J_EE21, J_EE22, J_DE11, J_DE12, J_DE21, J_DE22,
     J_EP11, J_EP12, J_EP21, J_EP22, J_DS11, J_DS12, J_DS21, J_DS22) = vars
    stim_time = int(stim_start*(1/delta_t))

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


    # plotting
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))

    plt.plot(t,rE1, color=color_list[0], linewidth=plot_line_width, label = r'$r_{E1}$')
    """ +
    str(np.round(np.mean(rE1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
    str(np.round(np.mean(rE1[-200:]), 3)))"""
    plt.plot(t,rE2, color=color_list[0], linestyle='dashed', linewidth=plot_line_width, label = r'$r_{E2}$')
    """+
             str(np.round(np.mean(rE2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
             str(np.round(np.mean(rE2[-200:]), 3)))"""
    plt.plot(t,rP1, color=color_list[1], linewidth=plot_line_width, label = r'$r_{P1}$')
    """+
             str(np.round(np.mean(rP1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
             str(np.round(np.mean(rP1[-200:]), 3)))"""
    plt.plot(t,rP2, color=color_list[1], linestyle='dashed', linewidth=plot_line_width, label = r'$r_{P2}$')
    """+
             str(np.round(np.mean(rP2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
             str(np.round(np.mean(rP2[-200:]), 3)))"""
    plt.plot(t,rS1, color=color_list[2], linewidth=plot_line_width, label = r'$r_{S1}$')
    """+
             str(np.round(np.mean(rS1[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
             str(np.round(np.mean(rS1[-200:]), 3)))"""
    plt.plot(t,rS2, color=color_list[2], linestyle='dashed', linewidth=plot_line_width, label = r'$r_{S2}$')
    """+
             str(np.round(np.mean(rS2[stim_time - 200:stim_time]), 3)) + 'Hz, s.s. after stimulus:'+
             str(np.round(np.mean(rS2[-200:]), 3)))"""

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, 8])
    plt.yticks(np.arange(0, 9, 2), [0, 2, 4, 6, 8], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1,loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_activity.png')
    plt.close()

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))
    
    plt.plot(t,J_DE11, linewidth=plot_line_width, label = r'$J_{DE}^{11}$')
    plt.plot(t,J_DE21, linewidth=plot_line_width, label = r'$J_{DE}^{21}$')
    plt.plot(t,J_DE12, linewidth=plot_line_width, label = r'$J_{DE}^{12}$')
    plt.plot(t,J_DE22, linewidth=plot_line_width, label = r'$J_{DE}^{22}$')

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, 0.6])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{DE}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_weight_DE.png')
    plt.close()


    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))
    
    plt.plot(t,J_EE11, linewidth=plot_line_width, label = r'$J_{EE}^{11}$')
    plt.plot(t,J_EE21, linewidth=plot_line_width, label = r'$J_{EE}^{21}$')
    plt.plot(t,J_EE12, linewidth=plot_line_width, label = r'$J_{EE}^{12}$')
    plt.plot(t,J_EE22, linewidth=plot_line_width, label = r'$J_{EE}^{22}$')

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, 0.6])
    plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_weight_EE.png')
    plt.close()

    #### W_EP plots
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))
    
    plt.plot(t,J_EP11, linewidth=plot_line_width, label = r'$J_{EP}^{11}$')
    plt.plot(t,J_EP21, linewidth=plot_line_width, label = r'$J_{EP}^{21}$')
    plt.plot(t,J_EP12, linewidth=plot_line_width, label = r'$J_{EP}^{12}$')
    plt.plot(t,J_EP22, linewidth=plot_line_width, label = r'$J_{EP}^{22}$')

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, 0.6])
    plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{EP}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_weight_EP.png')
    plt.close()


    #### W_DS plots
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))
    
    plt.plot(t,J_DS11, linewidth=plot_line_width, label = r'$J_{DS}^{11}$')
    plt.plot(t,J_DS21, linewidth=plot_line_width, label = r'$J_{DS}^{21}$')
    plt.plot(t,J_DS12, linewidth=plot_line_width, label = r'$J_{DS}^{12}$')
    plt.plot(t,J_DS22, linewidth=plot_line_width, label = r'$J_{DS}^{22}$')

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, 0.6])
    plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{DS}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_weight_DS.png')
    plt.close()

    # learning rates of E
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))

    plt.plot(t, plas_terms[28], color=color_list[0], linewidth=plot_line_width, label=r'$LR_{EE11}$')
    plt.plot(t, plas_terms[30], color=color_list[1], linewidth=plot_line_width, label=r'$LR_{EE21}$')
    plt.plot(t, plas_terms[29], color=color_list[2], linewidth=plot_line_width, label=r'$LR_{EE12}$')
    plt.plot(t, plas_terms[31], color=color_list[3], linewidth=plot_line_width, label=r'$LR_{EE22}$')

    plt.plot(t, plas_terms[32], color=color_list[0], linestyle='dashed', linewidth=plot_line_width, label=r'$LR_{DE11}$')
    plt.plot(t, plas_terms[34], color=color_list[1], linestyle='dashed', linewidth=plot_line_width, label=r'$LR_{DE21}$')
    plt.plot(t, plas_terms[33], color=color_list[2], linestyle='dashed', linewidth=plot_line_width, label=r'$LR_{DE12}$')
    plt.plot(t, plas_terms[35], color=color_list[3], linestyle='dashed', linewidth=plot_line_width, label=r'$LR_{DE22}$')

    plt.xticks(fontsize=font_size_1, **hfont)

    plt.ylim([0, .2])
    plt.yticks(np.arange(0, .21, .05), [0, .05, .1, .15, .2], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Learning rate', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(fontsize=font_size_1, loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig('png/' + s_name + '_learning_rates.png')
    plt.close()

    # bottom-up E/I ratio
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    ax = sns.barplot(data=[[(J_EE11[int(stim_start*(1/delta_t))-10] + J_EE12[int(stim_start*(1/delta_t))-10])/
                            (J_EP11[int(stim_start*(1/delta_t))-10] + J_EP12[int(stim_start*(1/delta_t))-10])],
                           [(J_EE11[-10] + J_EE12[-10])/(J_EP11[-10] + J_EP12[-10])],
                           [(J_EE21[int(stim_start*(1/delta_t))-10] + J_EE22[int(stim_start*(1/delta_t))-10])/
                            (J_EP21[int(stim_start*(1/delta_t))-10] + J_EP22[int(stim_start*(1/delta_t))-10])],
                           [(J_EE21[-10] + J_EE22[-10])/(J_EP21[-10] + J_EP22[-10])]],
                     linewidth=line_width, errwidth=line_width, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")

    widthbars = [0.3, 0.3, 0.3, 0.3]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    plt.xticks(range(4), ['E1 Before', 'E1 After', 'E2 Before', 'E2 After'], fontsize=font_size_1)
    plt.yticks([0, 0.5, 1.0, 1.5], fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1.5])
    plt.ylabel('Bottom up E/I ratio', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    plt.savefig('png/'+ s_name + '_bottom_up_EI_ratio.png')
    plt.close()

    # top-down E/I ratio
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    ax = sns.barplot(data=[[(J_DE11[int(stim_start*(1/delta_t))-10] + J_DE12[int(stim_start*(1/delta_t))-10])/
                            (J_DS11[int(stim_start*(1/delta_t))-10] + J_DS12[int(stim_start*(1/delta_t))-10])],
                           [(J_DE11[-10] + J_DE12[-10])/(J_DS11[-10] + J_DS12[-10])],
                           [(J_DE21[int(stim_start*(1/delta_t))-10] + J_DE22[int(stim_start*(1/delta_t))-10])/
                            (J_DS21[int(stim_start*(1/delta_t))-10] + J_DS22[int(stim_start*(1/delta_t))-10])],
                           [(J_DE21[-10] + J_DE22[-10])/(J_DS21[-10] + J_DS22[-10])]],
                     linewidth=line_width, errwidth=line_width, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")
    widthbars = [0.3, 0.3, 0.3, 0.3]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    plt.xticks(range(4), ['E1 Before', 'E1 After', 'E2 Before', 'E2 After'], fontsize=font_size_1)
    plt.yticks([0, 0.5, 1.0, 1.5], fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1.5])
    plt.ylabel('Bottom up E/I ratio', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    plt.savefig('png/' + s_name + '_top_down_EI_ratio.png')
    plt.close()

    show_plot_plasticity_terms_2compartmental(plas_terms, t, stim_start, stim_stop,
                                              stim_strength_E, delta_t, title, save_figure=1, name=s_name)


def show_plot_plasticity_terms_mass(plas_terms, rE1, rE2, t, stim_start, stim_stop,
                                    stim_strength_E, delta_t, title = None, save_figure=None, name=None):
    ratio = .8
    figure_len, figure_width = 15 * ratio, 12 * ratio
    font_size_1, font_size_2 = 36 * ratio, 36 * ratio
    legend_size = 18 * ratio
    line_width, tick_len = 3 * ratio, 10 * ratio
    marker_size = 15 * ratio
    plot_line_width = 5 * ratio
    hfont = {'fontname': 'Arial'}

    plt.figure(figsize=(figure_len, figure_width))
    if title is not None:
        plt.title(title, fontsize=font_size_1, **hfont)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " +str(stim_strength_E))

    labels = ["hebEE11", "hebEE12", "hebEE21", "hebEE22",
              "Synaptic scaling of E1 and S1", "Synaptic scaling of E2 and S2",
              "Synaptic scaling of P1", "Synaptic scaling of P2",
              "theta_I1", "theta_I2", "LR_EE11", "LR_EE12", "LR_EE21", "LR_EE22"]

    plt.subplot(421)
    plt.plot(t[0:len(t):200],plas_terms[0][0:len(t):200], label=labels[0])# + "last value is " + str(plas_terms[0][-1]))
    plt.plot(t[0:len(t):200],plas_terms[1][0:len(t):200], label=labels[1])# + "last value is " + str(plas_terms[1][-1]))
    plt.plot(t[0:len(t):200],plas_terms[2][0:len(t):200], label=labels[2])# + "last value is " + str(plas_terms[2][-1]))
    plt.plot(t[0:len(t):200],plas_terms[3][0:len(t):200], label=labels[3])# + "last value is " + str(plas_terms[3][-1]))
    plt.xlim(0, 20)
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')

    plt.subplot(422)
    plt.plot(t[0:len(t):200], plas_terms[8][0:len(t):200], label=labels[10])
    plt.plot(t[0:len(t):200], plas_terms[9][0:len(t):200], label=labels[11])
    plt.plot(t[0:len(t):200], plas_terms[10][0:len(t):200], label=labels[12])
    plt.plot(t[0:len(t):200], plas_terms[11][0:len(t):200], label=labels[13])
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    # plt.xlim([1, 200])

    plt.subplot(423)
    plt.plot(t[0:len(t):200],plas_terms[4][0:len(t):200], label=labels[4])# + str(np.sum(plas_terms[4])))
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')

    plt.subplot(424)
    plt.plot(t[0:len(t):200], plas_terms[5][0:len(t):200],  label=labels[5])# + str(np.sum(plas_terms[5])))
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')

    plt.subplot(425)
    plt.plot(t[0:len(t):200], -plas_terms[4][0:len(t):200],  label=labels[6])
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.ylabel('Value of the term [a.u.]', fontsize=font_size_1, **hfont)

    plt.subplot(426)
    plt.plot(t[0:len(t):200], -plas_terms[5][0:len(t):200],  label=labels[7])
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')

    plt.subplot(427)
    plt.plot(t[0:len(t):200],plas_terms[6][0:len(t):200], 'b', )#label=labels[8] + " final value " + str(np.round(plas_terms[6][-1],4)))
    plt.plot(t[0:len(t):200],rE1[0:len(t):200],  'b', linestyle= 'dashed', )#label="rE1 final value " + str(np.round(rE1[-1],4)))
    plt.plot(t[0:len(t):200],plas_terms[7][0:len(t):200], 'r', )#label=labels[9] + " final value " + str(np.round(plas_terms[7][-1],4)))
    plt.plot(t[0:len(t):200],rE2[0:len(t):200],  'r', linestyle= 'dashed', )#label="rE2 final value " + str(np.round(rE2[-1],4)))
    #plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.ylim([.9, max(plas_terms[6])+0.3])

    plt.subplot(428)
    plt.plot(t[0:len(t):200],plas_terms[6][0:len(t):200], 'b', )#label=labels[8] + " final value " + str(np.round(plas_terms[6][-1],4)))
    plt.plot(t[0:len(t):200],rE1[0:len(t):200],  'b', linestyle= 'dashed', )#label="rE1 final value " + str(np.round(rE1[-1],4)))
    plt.plot(t[0:len(t):200],plas_terms[7][0:len(t):200], 'r', )#label=labels[9] + " final value " + str(np.round(plas_terms[7][-1],4)))
    plt.plot(t[0:len(t):200],rE2[0:len(t):200],  'r', linestyle= 'dashed', )#label="rE2 final value " + str(np.round(rE2[-1],4)))
    #plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    plt.ylim([.9, max(plas_terms[6])+0.3])
    plt.xlim(0, 100)

    """plt.subplot(428)
    plt.plot(t[0:len(t):200],plas_terms[8][0:len(t):200],  label=labels[10])
    plt.plot(t[0:len(t):200],plas_terms[9][0:len(t):200],  label=labels[11])
    plt.plot(t[0:len(t):200],plas_terms[10][0:len(t):200], label=labels[12])
    plt.plot(t[0:len(t):200],plas_terms[11][0:len(t):200], label=labels[13])
    plt.legend(prop={"family": "Arial", 'size': legend_size}, loc='upper right')
    #plt.xlim([1, 200])"""

    plt.xlabel('Time [s]', fontsize=font_size_1, **hfont, loc='right')
    plt.grid()

    if save_figure:
        plt.savefig('png/' + name + '_plasticity_terms.png')
        plt.close()
    else:
        plt.show()


def plot_all_mass(t, vars, plas_terms, stim_start, stim_stop, stim_strength_E, delta_t, s_name, title):
    # extracting variables to plot
    (rE1, rE2, rP1, rP2, rS1, rS2, av_I1, av_I2,
     J_EE11, J_EE12, J_EE21, J_EE22, J_EP11, J_EP12, J_EP21, J_EP22, J_DS11, J_DS12, J_DS21, J_DS22) = vars
    stim_time = int(stim_start * (1 / delta_t))

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

    # rates
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " + str(stim_strength_E))

    plt.plot(t[0:len(t):100], rE1[0:len(t):100], color=color_list[0], linewidth=plot_line_width,
             label=r'difference in $r_{E1}$ is '+ str(np.round(-rE1[stim_time-2] + rE1[-2], 4)))
    plt.plot(t[0:len(t):100], rE2[0:len(t):100], color=color_list[0], linestyle='dashed', linewidth=plot_line_width,
             label=r'difference in $r_{E2}$ is '+ str(np.round(-rE2[stim_time-2] + rE2[-2], 4)))
    plt.plot(t[0:len(t):100], rP1[0:len(t):100], color=color_list[1], linewidth=plot_line_width,
             label=r'$r_{P1}$')
    plt.plot(t[0:len(t):100], rP2[0:len(t):100], color=color_list[1], linestyle='dashed', linewidth=plot_line_width,
             label=r'$r_{P2}$')
    plt.plot(t[0:len(t):100], rS1[0:len(t):100], color=color_list[2], linewidth=plot_line_width,
             label=r'$r_{S1}$')
    plt.plot(t[0:len(t):100], rS2[0:len(t):100], color=color_list[2], linestyle='dashed', linewidth=plot_line_width,
             label=r'$r_{S2}$')


    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    #plt.ylim([0, 4])
    #plt.yticks(np.arange(0, 4.1, 1), [0, 1, 2, 3, 4], fontsize=font_size_1, **hfont)
    #plt.xlim([1, 500])
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    ax.legend(fontsize=legend_size, loc='center right')

    plt.savefig('png/' + s_name + '_activity.png')
    plt.close()

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " + str(stim_strength_E))

    plt.plot(t[0:len(t):100], J_EE11[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EE}^{11}$: ' + str(np.round(J_EE11[-1], 3)))
    plt.plot(t[0:len(t):100], J_EE12[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EE}^{12}$: ' + str(np.round(J_EE12[-1], 3)))
    plt.plot(t[0:len(t):100], J_EE21[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EE}^{21}$: ' + str(np.round(J_EE21[-1], 3)))
    plt.plot(t[0:len(t):100], J_EE22[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EE}^{22}$: ' + str(np.round(J_EE22[-1], 3)))

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    #plt.ylim([0, 0.6])
    #plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    #plt.xlim([1, 500])
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    plt.legend(fontsize=font_size_1*.5, loc='center right')
    plt.savefig('png/' + s_name + '_weight_EE.png')
    plt.close()

    #### W_EP plots
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " + str(stim_strength_E))

    plt.plot(t[0:len(t):100], J_EP11[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EP}^{11}$: ' + str(np.round(J_EP11[-1], 3)))
    plt.plot(t[0:len(t):100], J_EP12[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EP}^{12}$: ' + str(np.round(J_EP12[-1], 3)))
    plt.plot(t[0:len(t):100], J_EP21[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EP}^{21}$: ' + str(np.round(J_EP21[-1], 3)))
    plt.plot(t[0:len(t):100], J_EP22[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{EP}^{22}$: ' + str(np.round(J_EP22[-1], 3)))

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    #plt.ylim([0, 0.6])
    #plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    #plt.xlim([1, 500])
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{EP}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    ax.legend(fontsize=font_size_1*.5, loc='center right')
    plt.savefig('png/' + s_name + '_weight_EP.png')
    plt.close()

    #### W_DS plots
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3, label="Stimulus of " + str(stim_strength_E))

    plt.plot(t[0:len(t):100], J_DS11[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{DS}^{11}$: ' + str(np.round(J_DS11[-1], 3)))
    plt.plot(t[0:len(t):100], J_DS12[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{DS}^{12}$: ' + str(np.round(J_DS12[-1], 3)))
    plt.plot(t[0:len(t):100], J_DS21[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{DS}^{21}$: ' + str(np.round(J_DS21[-1], 3)))
    plt.plot(t[0:len(t):100], J_DS22[0:len(t):100], linewidth=plot_line_width,
             label=r'final $J_{DS}^{22}$: ' + str(np.round(J_DS22[-1], 3)))

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    
    #plt.ylim([0, 0.6])
    #plt.yticks([0, 0.2, 0.4, 0.6], fontsize=font_size_1, **hfont)
    #plt.xlim([1, 500])
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel(r'$J_{DS}$', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    ax.legend(fontsize=font_size_1*.5, loc='center right')
    plt.savefig('png/' + s_name + '_weight_DS.png')
    plt.close()

    # learning rates of E
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_start, stim_stop, color='red', alpha=0.3,
                label="Stimulus between " + str(stim_start) + str(-stim_stop))

    plt.plot(t[0:len(t):100], plas_terms[-4][0:len(t):100], color=color_list[0],
             linewidth=plot_line_width,
             label=r'$LR_{EE11}$')
    plt.plot(t[0:len(t):100], plas_terms[-3][0:len(t):100], color=color_list[2], linewidth=plot_line_width,
             label=r'$LR_{EE12}$')
    plt.plot(t[0:len(t):100], plas_terms[-2][0:len(t):100], color=color_list[1], linewidth=plot_line_width,
             label=r'$LR_{EE21}$')
    plt.plot(t[0:len(t):100], plas_terms[-1][0:len(t):100], color=color_list[3], linewidth=plot_line_width,
             label=r'$LR_{EE22}$')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    #plt.ylim([0, 1.2])
    #plt.yticks(np.arange(0, 1.21, .20), [0, .20, .40, .60, .80, 1, 1.2], fontsize=font_size_1, **hfont)
    plt.xlim([1,30])
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Learning rate', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    ax.legend(fontsize=font_size_1*.5, loc='center right')
    plt.savefig('png/' + s_name + '_learning_rates.png')
    plt.close()

    # bottom-up E/I ratio
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    ax = sns.barplot(data=[[(J_EE11[int(stim_start * (1 / delta_t)) - 10] + J_EE12[
        int(stim_start * (1 / delta_t)) - 10]) / (J_EP11[int(stim_start * (1 / delta_t)) - 10] + J_EP12[
        int(stim_start * (1 / delta_t)) - 10])],
                           [(J_EE11[-10] + J_EE12[-10]) / (J_EP11[-10] + J_EP12[-10])],
                           [(J_EE21[int(stim_start * (1 / delta_t)) - 10] + J_EE22[
                               int(stim_start * (1 / delta_t)) - 10]) / (
                                        J_EP21[int(stim_start * (1 / delta_t)) - 10] + J_EP22[
                                    int(stim_start * (1 / delta_t)) - 10])],
                           [(J_EE21[-10] + J_EE22[-10]) / (J_EP21[-10] + J_EP22[-10])]],
                     linewidth=line_width, errwidth=line_width, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")

    widthbars = [0.3, 0.3, 0.3, 0.3]
    for bar, newwidth in zip(ax.patches, widthbars):
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - newwidth / 2.)
        bar.set_width(newwidth)

    plt.xticks(range(4), ['E1 Before', 'E1 After', 'E2 Before', 'E2 After'], fontsize=font_size_1)
    plt.yticks([0, 0.5, 1.0, 1.5], fontsize=font_size_1, **hfont)
    plt.xlim([-0.5, 3.5])
    plt.ylim([0, 1.51])
    plt.ylabel('Bottom up E/I ratio', fontsize=font_size_1, **hfont)
    plt.title(title, fontsize=font_size_1, **hfont)
    plt.savefig('png/' + s_name + '_bottom_up_EI_ratio.png')
    plt.close()

    show_plot_plasticity_terms_mass(plas_terms, rE1, rE2, t, stim_start, stim_stop, stim_strength_E,
                                    delta_t, title, save_figure=1,name=s_name)


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

def create_save_weights_2D(N_PC,N_PV,N_SOM,weight_strengths,
                           weight_probabilities,uniform_distribution=0,
                           path=None,save_weigths=0):
    s_EE11, s_EE12, s_DE11, s_DE12, s_DS11, s_EP11, s_PE11, s_SE11, s_PS11, s_PP11, \
    s_DS12, s_EP12, s_PE12, s_SE12, s_PS12, s_PP12, \
    s_EE21, s_EE22, s_DE22, s_DE21, s_DS21, s_EP21, s_PE21, s_SE21, s_PS21, s_PP21, \
    s_DS22, s_EP22, s_PE22, s_SE22, s_PS22, s_PP22 = weight_strengths

    p_EE11, p_EE12, p_DE11, p_DE12, p_DS11, p_EP11, p_PE11, p_SE11, \
    p_PS11, p_PP11, p_DS12, p_EP12, p_PE12, p_SE12, p_PS12, p_PP12, \
    p_EE21, p_EE22, p_DE22, p_DE21, p_DS21, p_EP21, p_PE21, p_SE21, \
    p_PS21, p_PP21, p_DS22, p_EP22, p_PE22, p_SE22, p_PS22, p_PP22 = weight_probabilities

    if uniform_distribution:
        a = 1
        b = .5
    else:
        a = 0
        b = 1
    np.random.seed(124)
    w_EE11 = connection_probability(a * np.random.rand(N_PC, N_PC) + b * np.ones((N_PC, N_PC)), p_EE11, s_EE11)  # self-excitation of PC1
    w_EE12 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_EE12, s_EE12)  # lateral-excitation of PC1-PC2
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
    w_EE21 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_EE21, s_EE21)  # self-excitation of PC2
    w_EE22 = connection_probability(a* np.random.rand(N_PC, N_PC)  + b * np.ones((N_PC, N_PC)), p_EE22, s_EE22)  # lateral-excitation of PC1-PC2
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
    w_PS11 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS11, s_PS11)
    w_PS12 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS12, s_PS12)
    w_PS21 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS21, s_PS21)
    w_PS22 = connection_probability(a* np.random.rand(N_PV, N_SOM) + b * np.ones((N_PV, N_SOM)), p_PS22, s_PS22)
    w_PP11 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP11, s_PP11)
    w_PP12 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP12, s_PP12)
    w_PP21 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP21, s_PP21)
    w_PP22 = connection_probability(a* np.random.rand(N_PV, N_PV)  + b * np.ones((N_PV, N_PV)), p_PP22, s_PP22)

    if save_weigths:
        pickle.dump((
        w_EE11, w_EE12, w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11,
        w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
        w_EE22, w_EE21, w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21,
        w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22), open(path, 'wb'))
    else:
        return (
        w_EE11, w_EE12, w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11,
        w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
        w_EE22, w_EE21, w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21,
        w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)


def determine_name(flags):
    (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag,
     adaptive_threshold_flag, adaptive_LR_flag) = flags

    if flags == (0,0,0,0,0):
        return "0", "No plasticity"
    elif flags == (1,0,0,0,0):
        return "1heb", "Only Hebbian learning"
    elif flags == (1,0,0,0,1):
        return "2heb_adapLR", "Hebbian and adaptive learning rate"
    elif flags == (1,0,0,1,0):
        return "3heb_adapThr", "Hebbian and adaptive threshold"
    elif flags == (1,0,0,1,1):
        return "4heb_adapLR_adapThr", "Hebbian, adaptive learning rate, and adaptive threshold"
    elif flags == (1,1,1,0,1):
        return "5heb_adapLR_excSS_inhSS", "Hebbian, adaptive LR, and both SS"
    elif flags == (1,1,1,1,1):
        return "6heb_adapLR_excSS_inhSS_adapThr",  "Hebbian, adaptive LR, adaptive threshold, and both SS"
    elif flags == (1,1,1,1,0):
        return "7heb_adapThr_excSS_inhSS_adapThr",  "Hebbian, both SS, and adaptive threshold"
    else:
        print("Please enter a valid flag configuration. Check the determine_name function.")
        quit()

def save_params(params, name):
    with open(name + ".txt", "w") as f:
        f.writelines("Parameters")
        f.write('\n')
        for i in params:
            f.writelines(str(i))
            f.write('\n')

def save_values(vars, plas_terms, name):
    with open(name + ".txt", "w") as f:
        f.writelines("Variables")
        f.write('n')
        f.write('n')

        f.writelines("rE1 final " + str(vars[0][-1]))
        f.write('n')
        f.writelines("rE2 final " + str(vars[1][-1]))
        f.write('n')
        f.writelines("rE1-rE2 " + str(vars[0][-1]-vars[1][-1]))
        f.write('n')
        f.write('n')

        f.writelines(str(np.max(plas_terms[4])),
                     str(np.min(plas_terms[5])),
                     str(np.max(plas_terms[6])),
                     str(np.min(plas_terms[7])))


