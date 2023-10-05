import numpy as np
import pylab as pl
from numpy.linalg import eig
from matplotlib import pyplot as plt
from matplotlib import colormaps as colmaps
from matplotlib.legend_handler import HandlerTuple
import pickle
import cmath
"""from numba import cuda, jit, njit
import numba as nb
from numba.types import float64, int64, ListType, List"""
import seaborn as sns


def determine_name(flags):
    (hebbian_plasticity_flag, exc_scaling_flag,
     inh_scaling_flag_p, inh_scaling_flag_s,
     adaptive_threshold_flag, adaptive_LR_flag) = flags

    if flags == (0,0,0,0,0,0):
        return "0", "0", "No plasticity"
    elif flags == (1,0,0,0,0,0):
        return "1", "1heb", "Only Hebbian learning"
    elif flags == (1,0,0,0,0,1):
        return "2", "2heb_adapLR", "Hebbian and 3-factor learning rate"
    elif flags == (1,0,0,0,1,0):
        return "3", "3heb_adapThr", "Hebbian and adaptive threshold"
    elif flags == (1,0,0,0,1,1):
        return "4", "4heb_adapLR_adapThr", "Hebbian, 3-factor learning rate, and adaptive threshold"
    elif flags == (1,1,1,1,0,1):
        return "5", "5heb_adapLR_excSS_inhSS", "Hebbian, 3-factor learning rate, and all SS"
    elif flags == (1,1,1,1,1,1):
        return "6", "6heb_adapLR_excSS_inhSS_adapThr",  "Full model"
    elif flags == (1,1,1,1,0,0):
        return "7", "7heb_adapThr_excSS_inhSS", "Hebbian and both SS"
    elif flags == (1,1,1,1,1,0):
        return "8", "8heb_adapThr_excSS_inhSS_adapLR", "Hebbian, adaptive threshold, and both SS"
    elif flags == (0,1,1,1,1,0):
        return "9", "9heb_adapThr_excSS_inhSS_adapLR", "Adaptive threshold, and both SS"
    elif flags == (1,0,1,1,1,1):
        return "10", "10without_excitatory_scaling", "E off"
    elif flags == (1,1,0,0,1,1):
        return "11", "11without_inhibitory_scaling", "P and S off"
    elif flags == (1,1,0,1,1,1):
        return "12", "12without_P_scaling", "P off (E+S)"
    elif flags == (1,1,1,0,1,1):
        return "13", "13without_S_scaling", "S off (E+P)"
    elif flags == (1,0,0,1,1,1):
        return "14", "14without_P_and_E_scaling", "only S on"
    elif flags == (1,0,1,0,1,1):
        return "15", "15without_S_and_E_scaling", "only P on"
    else:
        print("Please enter a valid flag configuration. Check the determine_name function.")
        quit()




def plot_all(t, res_rates, res_weights, p_threshold, stim_times, name, dk, format='.svg'):

    (t_stimulation, t_simulation) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1
    figure_len, figure_width = 13 * ratio, 15 * ratio
    font_size_1, font_size_2 = 80 * ratio, 36 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 45 * ratio
    line_width, tick_len = 5 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 7 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    pal1 = sns.color_palette()
    color_list = pal1.as_hex()
    palette_tab20 = sns.color_palette("tab20", 20)
    palette_tab20b = sns.color_palette("tab20b", 20)

    color_list = ['#00338E', '#8FBBD9',
                  '#C10000', '#EFABAB',
                  '#007100', '#87CB87']


    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
            ymax = 4
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]
            ymax = 3

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5
        ymin = 0

        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.1)


        if stim_applied == 1:
            transparency = 1
        else:
            transparency = 1


        """p1, = ax.plot(t_stimulation, rP1,
                      color=color_list[3], linewidth=plot_line_width, label=r'$r_{P1}$', alpha=transparency)
        p2, = ax.plot(t_stimulation, rP2, '-o', markersize=30, markevery=0.1,
                      color=color_list[3], linewidth=plot_line_width, label=r'$r_{P2}$', alpha=transparency)
        s1, = ax.plot(t_stimulation, rS1,
                      color=color_list[2], linewidth=plot_line_width, label=r'$r_{S1}$', alpha=transparency)
        s2, = ax.plot(t_stimulation, rS2, '-o', markersize=30, markevery=0.1,
                      color=color_list[2], linewidth=plot_line_width, label=r'$r_{S2}$', alpha=transparency)"""
        e1, = ax.plot(t_stimulation, rE1, color=color_list[0],
                      linewidth=plot_line_width, label=r'$r_{E1}$', alpha=transparency)
        e2, = ax.plot(t_stimulation, rE2, color=color_list[1],
                      linewidth=plot_line_width, label=r'$r_{E2}$')

        # add here the first and the last 5 seconds of the stimulations
        r_th, = plt.plot(t_stimulation, p_threshold * np.ones_like(t_stimulation),
                         linestyle=(0, (3, 5, 1, 5)), color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(t_stimulation, r_phase1[0][0] * np.ones_like(t_stimulation),
                       linestyle=(0, (1, 3)), color=color_list[0], linewidth=plot_line_width)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks(np.arange(ymax+1), fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        """ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_th], [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$',
                   '$r_{S1}$, $r_{S2}$', 'Baseline of $r_E$', r'Perception threshold $\theta_{th}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',
                  handlelength=3)"""
        """ax.legend([(e1, e2), rb, r_th], [r'$r_{E1}$, $r_{E2}$', '$r_b$',
                  r'$\theta_{pt}$'],handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)"""
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1

    # plot the long term behaviour only at 48 hours
    if dk > 48 * 60 -1:

        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        # rates
        xmin = 0
        xmax = t_simulation[-1]
        ymin = 0
        ymax = 3
        plt.figure(figsize=(figure_width*1.5, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)

        p1, = ax.plot(t_simulation, rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(t_simulation, rP2, color=color_list[3], linewidth=plot_line_width)
        s1, = ax.plot(t_simulation, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(t_simulation, rS2, color=color_list[5], linewidth=plot_line_width)
        e1, = ax.plot(t_simulation, rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(t_simulation, rE2, color=color_list[1], linewidth=plot_line_width)
        r_th, = plt.plot(t_simulation, p_threshold * np.ones_like(t_simulation), linestyle=(0, (3, 5, 1, 5)),
                         color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(t_simulation, r_phase1[0][0] * np.ones_like(t_simulation),  linestyle=(0, (1, 3)),
                         color=color_list[0], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )
        plt.vlines(24, ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )
        plt.vlines(48, ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_th],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$',
                   'Baseline $r_{E}$', r'$\theta_{pt}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        """ax.legend([(e1, e2), rb, r_th, ], [r'$r_{E1}$, $r_{E2}$', 'Baseline $r_{b}$',
                   'Perception threshold $r_{th}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                    fontsize=legend_size, loc='upper right', handlelength=3)"""
        plt.tight_layout()

        plt.savefig(name + '_long_activity' + format)
        plt.close()


        ######### All plastic weights during scaling ##########
        ymin = 0
        ymax = 2
        plt.figure(figsize=(figure_width*1.5, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1 * .7)

        wEE1, = ax.plot(t_simulation, J_EE11, linewidth=plot_line_width,
                        linestyle = 'solid', color=palette_tab20b[-2])

        wEE2, = ax.plot(t_simulation, J_EE22, linewidth=plot_line_width,
                        linestyle = 'solid', color=palette_tab20[-6])

        wEP1, = ax.plot(t_simulation, J_EP11, linewidth=plot_line_width,
                        linestyle = (0, (5, 7)), color=palette_tab20b[-2])

        wEP2, = ax.plot(t_simulation, J_EP22, linewidth=plot_line_width,
                        linestyle = (0, (5, 7)), color=palette_tab20[-6])

        wES1, = ax.plot(t_simulation, J_DS11, linewidth=plot_line_width*1.2,
                        linestyle = (0, (1, 4)), color=palette_tab20b[-2])

        wES2, = ax.plot(t_simulation, J_DS22, linewidth=plot_line_width*1.2,
                        linestyle = (0, (1, 4)), color=palette_tab20[-6])

        ax.legend([(wEE1, wEE2), (wEP1, wEP2), (wES1, wES2)],
                  [r'$w_{E_{1}E_{1}}$, $w_{E_{2}E_{2}}$', r'$w_{E_{1}P_{1}}$, $w_{E_{2}P_{2}}$',
                   r'$w_{E_{1}S_{1}}$, $w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=6)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)

        plt.tight_layout()

        plt.savefig(name + '__long' + '_ALL_WEIGHTS' + format)
        plt.close()



def plot_all_VIP(t, res_rates, res_weights, p_threshold, stim_times, name, dk, format='.svg'):

    (t_stimulation, t_simulation) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1
    figure_len, figure_width = 13 * ratio, 15 * ratio
    font_size_1, font_size_2 = 80 * ratio, 36 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 45 * ratio
    line_width, tick_len = 5 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 7 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    pal1 = sns.color_palette()
    color_list = pal1.as_hex()
    palette_tab20 = sns.color_palette("tab20", 20)
    palette_tab20b = sns.color_palette("tab20b", 20)

    color_list = ['#00338E', '#8FBBD9',
                  '#C10000', '#EFABAB',
                  '#007100', '#87CB87',
                  '#ff7f0e', '#ffcb9e']


    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
            rV1 = r_phase1[6]; rV2 = r_phase1[7]

            print(rE1[3000])
            print(rP1[3000])
            print(rS1[3000])
            print(rV1[3000])
            ymax = 4
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]
            rV1 = r_phase3[6]; rV2 = r_phase3[7]

            ymax = 3

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5
        ymin = 0

        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.1)


        if stim_applied == 1:
            transparency = 1
        else:
            transparency = 1


        p1, = ax.plot(t_stimulation, rP1,
                      color=color_list[2], linewidth=plot_line_width, label=r'$r_{P1}$', alpha=transparency)
        p2, = ax.plot(t_stimulation, rP2, '-o', markersize=30, markevery=0.1,
                      color=color_list[3], linewidth=plot_line_width, label=r'$r_{P2}$', alpha=transparency)
        s1, = ax.plot(t_stimulation, rS1,
                      color=color_list[4], linewidth=plot_line_width, label=r'$r_{S1}$', alpha=transparency)
        s2, = ax.plot(t_stimulation, rS2, '-o', markersize=30, markevery=0.1,
                      color=color_list[5], linewidth=plot_line_width, label=r'$r_{S2}$', alpha=transparency)

        v1, = ax.plot(t_stimulation, rV1, color=color_list[6], linewidth=plot_line_width)
        v2, = ax.plot(t_stimulation, rV2, color=color_list[7], linewidth=plot_line_width)
        e1, = ax.plot(t_stimulation, rE1, color=color_list[0],
                      linewidth=plot_line_width, label=r'$r_{E1}$', alpha=transparency)
        e2, = ax.plot(t_stimulation, rE2, color=color_list[1],
                      linewidth=plot_line_width, label=r'$r_{E2}$')

        # add here the first and the last 5 seconds of the stimulations
        r_th, = plt.plot(t_stimulation, p_threshold * np.ones_like(t_stimulation),
                         linestyle=(0, (3, 5, 1, 5)), color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(t_stimulation, r_phase1[0][0] * np.ones_like(t_stimulation),
                       linestyle=(0, (1, 3)), color=color_list[0], linewidth=plot_line_width)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        """plt.ylim([ymin, ymax])
        plt.yticks(np.arange(ymax+1), fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)"""
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        """ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_th], [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$',
                   '$r_{S1}$, $r_{S2}$', 'Baseline of $r_E$', r'Perception threshold $\theta_{th}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',
                  handlelength=3)"""
        """ax.legend([(e1, e2), rb, r_th], [r'$r_{E1}$, $r_{E2}$', '$r_b$',
                  r'$\theta_{pt}$'],handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)"""
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1

    # plot the long term behaviour only at 48 hours
    if dk > .8 * 60 -1:

        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]
        rV1 = r_phase2[6]; rV2 = r_phase2[7]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        # rates
        xmin = 0
        xmax = t_simulation[-1]
        ymin = 0
        ymax = 3
        plt.figure(figsize=(figure_width*1.5, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)

        p1, = ax.plot(rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(rP2, color=color_list[3], linewidth=plot_line_width)
        s1, = ax.plot(rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(rS2, color=color_list[5], linewidth=plot_line_width)
        v1, = ax.plot(rV1, color=color_list[6], linewidth=plot_line_width)
        v2, = ax.plot(rV2, color=color_list[7], linewidth=plot_line_width)
        e1, = ax.plot(rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(rE2, color=color_list[1], linewidth=plot_line_width)
        r_th, = plt.plot(t_simulation, p_threshold * np.ones_like(t_simulation), linestyle=(0, (3, 5, 1, 5)),
                         color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(t_simulation, r_phase1[0][0] * np.ones_like(t_simulation),  linestyle=(0, (1, 3)),
                         color=color_list[0], linewidth=plot_line_width)

        """plt.vlines(4,  ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )
        plt.vlines(24, ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )
        plt.vlines(48, ymin, ymax, 'black', linestyles=(0, (5, 8)), linewidth=plot_line_width, )"""

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        """plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)"""
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_th],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$',
                   'Baseline $r_{E}$', r'$\theta_{pt}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        """ax.legend([(e1, e2), rb, r_th, ], [r'$r_{E1}$, $r_{E2}$', 'Baseline $r_{b}$',
                   'Perception threshold $r_{th}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                    fontsize=legend_size, loc='upper right', handlelength=3)"""
        plt.tight_layout()

        plt.savefig(name + '_long_activity' + format)
        plt.close()


        ######### All plastic weights during scaling ##########
        ymin = 0
        ymax = 2
        plt.figure(figsize=(figure_width*1.5, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width * 1.3)
        plt.tick_params(width=line_width * 1.3, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1 * .7)

        wEE1, = ax.plot(J_EE11, linewidth=plot_line_width,
                        linestyle = 'solid', color=palette_tab20b[-2])

        wEE2, = ax.plot(J_EE22, linewidth=plot_line_width,
                        linestyle = 'solid', color=palette_tab20[-6])

        wEP1, = ax.plot(J_EP11, linewidth=plot_line_width,
                        linestyle = (0, (5, 7)), color=palette_tab20b[-2])

        wEP2, = ax.plot(J_EP22, linewidth=plot_line_width,
                        linestyle = (0, (5, 7)), color=palette_tab20[-6])

        wES1, = ax.plot(J_DS11, linewidth=plot_line_width*1.2,
                        linestyle = (0, (1, 4)), color=palette_tab20b[-2])

        wES2, = ax.plot(J_DS22, linewidth=plot_line_width*1.2,
                        linestyle = (0, (1, 4)), color=palette_tab20[-6])

        ax.legend([(wEE1, wEE2), (wEP1, wEP2), (wES1, wES2)],
                  [r'$w_{E_{1}E_{1}}$, $w_{E_{2}E_{2}}$', r'$w_{E_{1}P_{1}}$, $w_{E_{2}P_{2}}$',
                   r'$w_{E_{1}S_{1}}$, $w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=6)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        """plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)"""
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)

        plt.tight_layout()

        plt.savefig(name + '__long' + '_ALL_WEIGHTS' + format)
        plt.close()




def plot_span_init_conds(results_list, wEP_iis, wES_iis, directory, name, n,
                              plot_bars=0, plot_legends=0, format='.png', title=''):

    result = np.array(results_list).reshape(n,n)

   # plotting configuration
    ratio = 1
    figure_len, figure_width = 13 * ratio, 15 * ratio
    font_size_1, font_size_2 = 65 * ratio, 36 * ratio
    font_size_label = 65 * ratio
    legend_size = 50 * ratio
    line_width, tick_len = 5 * ratio, 20 * ratio
    marker_size = 360 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 7 * ratio
    hfont = {'fontname': 'Arial'}

    cmap_name = 'PiYG'
    cmap =colmaps[cmap_name]

    ones_matrix = np.ones((n,n))
    xmin = wEP_iis[0]
    xmax = wEP_iis[-1]
    ymin = wES_iis[0]
    ymax = wES_iis[-1]

    # Memory specificity
    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width * 1.3, length=tick_len)

    # checks if all values are NaN
    if np.isnan(result).all():
        for idx, i in enumerate(ones_matrix):
            plt.scatter(idx * wES_iis, i * wEP_iis[idx], c='black',
                        cmap=cmap, s=marker_size, marker=r"$\infty$")
    elif np.all(result == np.inf):
        for idx in np.arange(len(wES_iis)):
            for idy in np.arange(len(wEP_iis)):
                plt.scatter(wES_iis[idx], wEP_iis[idy], c='black', s=marker_size * 1.1, marker='X',
                            edgecolors='black', linewidths=marker_edge_width, label='Degenerate case')
    else:

        plt.scatter(-1, -1, c='black', s=marker_size, marker=r"$\infty$", label='Unstable case')
        plt.scatter(-1, -1, c='white', s=marker_size * 1.1, edgecolors='black',
                    linewidths=marker_edge_width, marker='*', label='Specialized memory')
        plt.scatter(-1, -1, c='white', s=marker_size, edgecolors='black',
                    linewidths=marker_edge_width, marker='o', label='Generalized memory')
        plt.scatter(-1, -1, c='white', s=marker_size, edgecolors='black',
                    linewidths=marker_edge_width, marker='X', label='Degenerate case')

        for idx in np.arange(len(wES_iis)):
            for idy in np.arange(len(wEP_iis)):
                if np.isnan(result[idy][idx]):
                    plt.scatter(wES_iis[idx], wEP_iis[idy], c='black',
                                s=marker_size, marker='s',
                                linewidths=marker_edge_width)
                else:
                    if result[idy][idx] == 1:
                        plt.scatter(wES_iis[idx], wEP_iis[idy], c='green',
                                        s=marker_size, marker='s',
                                        linewidths=marker_edge_width)
                    elif result[idy][idx] == 2:
                        plt.scatter(wES_iis[idx], wEP_iis[idy], c='gray',
                                        s=marker_size, marker='s',
                                        linewidths=marker_edge_width)
                    elif result[idy][idx] == 0:
                        plt.scatter(wES_iis[idx], wEP_iis[idy], c='red',
                                    s=marker_size, marker='s',
                                    linewidths=marker_edge_width)


    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    d_tick = (xmax - xmin)/4
    plt.xlim([xmin-d_tick*(3/50), xmax+d_tick*(3/50)])
    plt.xticks([xmin, xmin+d_tick, xmin+2*d_tick, xmin+3*d_tick, xmax], fontsize=font_size_1, **hfont)
    plt.ylim([ymin-d_tick*(4/50), ymax+d_tick*(4/50)])
    plt.yticks([ymin, ymin+d_tick, ymin+2*d_tick, ymin+3*d_tick, ymax], fontsize=font_size_1, **hfont)

    plt.xlabel('$W_{EP}$', fontsize=font_size_label, **hfont)
    plt.ylabel('$W_{ES}$', fontsize=font_size_label, **hfont)
    #plt.title(title, fontsize=font_size_1, **hfont)
    plt.tight_layout()

    if plot_bars:
        # Plot colorbar
        cb = plt.colorbar(shrink=0.9)
        cb.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_2)
        cb.ax.set_ylabel('Memory specificity', rotation=270, fontsize=font_size_2, labelpad=50)

    if plot_legends:
        # Shrink by 20%,
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15,
                         box.width, box.height * 0.8])

        # Put a legend to the right of the current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.55, -0.17),
                  fancybox=True,
                  scatterpoints=1, ncol=2, fontsize=legend_size)

    plt.savefig(directory + 'mem_spec_' + name + format)
    #plt.show()
    plt.close()
