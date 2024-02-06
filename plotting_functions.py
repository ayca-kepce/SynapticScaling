import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as colmaps
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple

import seaborn as sns



def time_plots(t, res_rates, res_weights, av_threshold, stim_times, name, hour_sim, flag_only_S_on=0, format='.svg'):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # Plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 13.7 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    # Line styles for dashed and dotted lines
    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    
    # The colors for given variables
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                  '#6600cc'] # timepoints in long simulation

    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rE_y_labels = [1, 1.5, 2, 2.5, 3]  # , 3.5] #[0,5,10,15]
            rE_ymin = 1
            rE_ymax = 3
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]

            # Only S on case has different margins for excitatory firing rates
            if flag_only_S_on:
                if t[-1][-1] < 5:
                    rE_ymin = 3
                    rE_ymax = 5
                    rE_y_labels = [1, 2, 3, 4, 5]  # , 3.5] #[0,5,10,15]
                else:
                    rE_ymin = 20
                    rE_ymax = 40
                    rE_y_labels = [25, 30, 35, 40]  # , 3.5] #[0,5,10,15]
            else:
                rE_ymin = 1
                rE_ymax = 2.5
                rE_y_labels = [1, 1.5, 2, 2.5]  # , 3.5] #[0,5,10,15]


        ### Excitatory firing rates during stimulation ###
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        e1, = ax.plot(l_time_points_stim, rE1, color=color_list[0], linewidth=plot_line_width, label=r'$r_{E1}$')
        e2, = ax.plot(l_time_points_stim, rE2, color=color_list[1], linewidth=plot_line_width, label=r'$r_{E2}$')
        r_at, = plt.plot(l_time_points_stim, av_threshold * np.ones_like(l_time_points_stim), dash_capstyle='round',
                         linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_stim, r_phase1[0][0] * np.ones_like(l_time_points_stim), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        ax.legend([(e1, e2), rb, r_at], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$', r'$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1



    ### Excitatory weights ###
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.2
    ymax = 0.8
    plt.figure(figsize=(figure_width, figure_len))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    wee11, = ax.plot(l_time_points_stim, J_EE_phase1[0], color=color_list[0], linewidth=plot_line_width)
    wee12, = ax.plot(l_time_points_stim, J_EE_phase1[1], '--', color=color_list[0], linewidth=plot_line_width)
    wee21, = ax.plot(l_time_points_stim, J_EE_phase1[2], color=color_list[1], linewidth=plot_line_width)
    wee22, = ax.plot(l_time_points_stim, J_EE_phase1[3], '--', color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([0.2, 0.5, 0.8], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    ax.legend([(wee11, wee12), (wee21, wee22)], [r'$w_{E_{1}E_{1}}$, $w_{E_{1}E_{2}}$', '$w_{E_{2}E_{1}}$, $w_{E_{2}E_{2}}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()

    plt.savefig(name + 'conditioning_wEE' + format)
    plt.close()



    # Plot the long term behaviour only at 48 hours
    if hour_sim >= 48:
        # Assign data to variables
        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        ### Excitatory rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        e1, = ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(l_time_points_phase2, rE2, color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), rb, r_at], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_E_rates' + format)
        plt.close()



        ### PV rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 1
        ymax = 1.2
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        p1, = ax.plot(l_time_points_phase2, rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(l_time_points_phase2, rP2, color=color_list[3], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([1, 1.1, 1.2], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(p1), (p2), ],
                  ['$r_{P1}$', '$r_{P2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_P_rates' + format)
        plt.close()



        ### SST rates in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 2
        ymax = 2.6
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        s1, = ax.plot(l_time_points_phase2, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_phase2, rS2, color=color_list[5], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([2, 2.2, 2.4, 2.6], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(s1), (s2)], ['$r_{S1}$', '$r_{S2}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper right', handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_S_rates' + format)
        plt.close()



        ### Set-point in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        theta1, = ax.plot(l_time_points_phase2, r_phase2[6], color=color_list[0], linewidth=plot_line_width)
        theta2, = ax.plot(l_time_points_phase2, r_phase2[7], color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(theta1, theta2), rb, r_at], [r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()



        ### Set-point regulator in the long term ###
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        beta1, = ax.plot(l_time_points_phase2, r_phase2[8], color=color_list[0], linewidth=plot_line_width)
        beta2, = ax.plot(l_time_points_phase2, r_phase2[9], color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color='black', linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(beta1, beta2), rb, r_at], [r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ### All plastic within-population weights in the long term ###
        ymin = 0
        ymax = 1.8
        plt.figure(figsize=(figure_width, figure_len))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        mag_of_y = ax.yaxis.get_offset_text()
        mag_of_y.set_size(font_size_1)

        plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        wEE1, = ax.plot(l_time_points_phase2, J_EE11, linewidth=plot_line_width, color=color_list[0])
        wEE2, = ax.plot(l_time_points_phase2, J_EE22, linewidth=plot_line_width, color=color_list[1])
        wEP1, = ax.plot(l_time_points_phase2, J_EP11, linewidth=plot_line_width, color=color_list[2])
        wEP2, = ax.plot(l_time_points_phase2, J_EP22, linewidth=plot_line_width, color=color_list[3])
        wES1, = ax.plot(l_time_points_phase2, J_DS11, linewidth=plot_line_width, color=color_list[4])
        wES2, = ax.plot(l_time_points_phase2, J_DS22, linewidth=plot_line_width, color=color_list[5])
        ax.legend([(wEE1, wEE2), (wEP1, wEP2), (wES1, wES2)],
                  [r'$w_{E_{1}E_{1}}$, $w_{E_{2}E_{2}}$', r'$w_{E_{1}P_{1}}$, $w_{E_{2}P_{2}}$',
                   r'$w_{E_{1}S_{1}}$, $w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 0.9, 1.8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (h)', fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weights' + format)
        plt.close()






def change_in_reactivation_every_h(r_phase1, l_time_points_phase2, r_phase2, hour_sims, l_delta_rE1, av_threshold,
                                    delta_t, sampling_rate_sim, name, flag_only_S_on=0, format='.svg'):

    # plotting configuration
    ratio = 1.5
    figure_len2, figure_width2 = 14.5 * ratio, 13.7 * ratio
    font_size_1, font_size_2 = 80 * ratio, 65 * ratio
    font_size_label = 80 * ratio
    legend_size = 50 * ratio
    legend_size2 = 65 * ratio
    line_width, tick_len = 9 * ratio, 20 * ratio
    marker_size = 15 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 9 * ratio
    hfont = {'fontname': 'Arial'}
    sns.set(style='ticks')

    x_label_text = 'Time (h)'

    line_style_rb = (0, (0.05, 2.5))
    line_style_r_at = (0, (5, 5))
    # defining the colors for
    color_list = ['#3276b3', '#91bce0', # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB', # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87', # rS1 and WES11, rS2 and WES22
                  '#6600cc'] # timepoints in long simulation

    ### % Change in reactivation of E1
    xmin = 0
    xmax = l_time_points_phase2[-1]

    # Only SST-to-E scaling on case has different margins
    if flag_only_S_on:
        ymin = 1
        ymax = 10000
    else:
        ymin = -10
        ymax = 100

    plt.figure(figsize=(figure_width2, figure_len2))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    baseline_reactivation = av_threshold / 1.15
    change_in_reactivation = 100*(np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=15, vmax=100)

    # Normalize the change in reactivation
    values_range = norm(change_in_reactivation)  # Normalize using TwoSlopeNorm

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        color = cmap(values_range[i])
        ax.fill_between(hour_sims[i:i + 2], 0, change_in_reactivation[i:i + 2], color=color,
                        edgecolor=(0, 0, 0, 0))

    plt.plot(hour_sims, change_in_reactivation, color=color_list[0], linewidth=plot_line_width, label='% change')
    plt.plot(hour_sims, 15 * np.ones_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='Tolerance')
    plt.plot(hour_sims, np.zeros_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_rb, color='black', linewidth=plot_line_width, label='Baseline reactivation')

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([-10, 15, 100], fontsize=font_size_1, **hfont)
    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness

    """plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)"""

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)

    if flag_only_S_on:
        plt.ylabel('% change in $r_{E1,re}$ (log)', fontsize=font_size_label, **hfont)
        ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='lower center',
                  bbox_to_anchor=(0.5, 1.05), handlelength=3)
    else:
        plt.yticks([-20, 0, 15, 50, 100], fontsize=font_size_1, **hfont)
        plt.ylabel('% change in $r_{E1,re}$', fontsize=font_size_label, **hfont)
        ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,loc='lower center', bbox_to_anchor=(0.5, 1.05), handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation_color' + format)
    plt.close()
