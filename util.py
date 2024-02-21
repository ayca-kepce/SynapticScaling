import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as colmaps
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns


def determine_name(flags):
    (hebbian_flag, three_factor_flag, adaptive_threshold_flag,
     E_scaling_flag, P_scaling_flag, S_scaling_flag) = flags

    if flags == (0,0,0,0,0,0):
        return "0", "No plasticity"

    elif flags == (1,1,1,1,1,1):
        return "1", "Full model"

    elif flags == (1,1,1,0,1,1):
        return "2", "E off (P+S)"

    elif flags == (1,1,1,1,0,1):
        return "3", "P off (E+S)"

    elif flags == (1,1,1,1,1,0):
        return "4", "S off (E+P)"

    elif flags == (1,1,1,1,0,0):
        return "5", "only E on"

    elif flags == (1,1,1,0,1,0):
        return "6", "only P on"

    elif flags == (1,1,1,0,0,1):
        return "7", "only S on"

    elif flags == (1,1,1,0,0,0):
        return "8", "No scaling"



def find_baseline_reactivation(rE1_conditioning):
    """
    :param rE1_conditioning: np.array that holds firing rate of the first excitatory population during conditioning
    :return: perception threshold of the network to the stimuli received during conditioning

    This function calculates the onset response of the first excitatory population in response to the stimuli, which is
    defined as the perception threshold of the network. The first excitatory firing rate (rE1) increase during
    conditioning both due to the stimuli and the Three-factor Hebbian learning.


    Since the Hebbian learning is way slower than the rate dynamics, the firing rate increase due to Hebbian learning
    should be observed later. To simplify, we can assume the change due to Hebbian learning is zero for a couple of
    miliseconds following the stimuli onset. The change is sudden when the stimuli is presented, then the rate dynamics
    needs a couple of miliseconds to reach the steady state. The following change in rE1 is due to the Hebbian learning.

    For this purpose, the change in rE1 is calculated at every time point. This change is greater at the beginning due
    to the stimuli onset. Later, the change of the change is calculated. The sign of every element of this array
    indicates whether the increase in rE1 accelerates (plus) or decelerates (minus). The change in rE1 due to stimuli
    initially accelerates, then it decelerates and becomes constant.
    """

    # Finding change of the change in rE1 every time point
    change_rE1 = rE1_conditioning - np.roll(rE1_conditioning, 1)
    change_of_change_rE1 = change_rE1 - np.roll(change_rE1, 1)

    # Finding at which index the sign of the second derivative changes. First two elements are ignored since np.roll
    # carries out a circular shift which assigns the last element of the input to the first element of the output.
    # The indexing should be preserved, thus two is added after calculating the sign change
    l_idx_sign_change = np.where(np.diff(np.sign(change_of_change_rE1[2:])) != 0)[0] + 2

    # When the firing rates explodes due to lack of inhibition, the change in rE1 only accelerates and the perception
    # threshold becomes irrelevant because the test cannot be conducted due to exploded rates. In this case, the
    # perception threshold is assigned to the baseline activity (pre-conditioning rate). When the firing rates stabilize
    # with the present inhibition, the perception threshold can be assigned at the index where the change in rE1 becomes
    # stable after the sudden acceleration followed by deceleration, which corresponds to the 3rd sign change.
    if l_idx_sign_change.shape[0] < 2:
        idx = 0
    else:
        idx = l_idx_sign_change[2] + 1

    return idx



def plot_all(t, res_rates, res_weights, av_threshold, stim_times, name, hour_sim, format='.svg'):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
    figure_len2, figure_width2 = 13 * ratio, 14.35 * ratio
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

    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
            rE_y_labels = [1, 1.5, 2, 2.5, 3]  # , 3.5] #[0,5,10,15]
            rE_ymin = 1
            rE_ymax = 3
            fig_size_stimulation = (figure_width1, figure_len1)
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]
            rE_ymin = 1
            rE_ymax = 2.5
            rE_y_labels = [1, 1.5, 2, 2.5]  # , 3.5] #[0,5,10,15]
            fig_size_stimulation = (figure_width1, figure_len1)

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        #p1, = ax.plot(l_time_points_stim, rP1, color=color_list[2], linewidth=plot_line_width)
        #p2, = ax.plot(l_time_points_stim, rP2, color=color_list[3], linewidth=plot_line_width)
        #s1, = ax.plot(l_time_points_stim, rS1, color=color_list[4], linewidth=plot_line_width)
        #s2, = ax.plot(l_time_points_stim, rS2, color=color_list[5], linewidth=plot_line_width)
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


    ######### excitatory weights ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.2
    ymax = 0.8
    plt.figure(figsize=(figure_width1, figure_len1))
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



    """######### inputs pre-test ###########
    ymin = 0
    ymax = 3
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    # text and line to identify the bars of neurons
    plt.text(.4, stim_label_y - 0.1, 'inputs to E1', fontsize=legend_size2, horizontalalignment='center')
    plt.axhline(stim_label_y - 0.2, 0.1/1.7, 0.7/1.7, color=color_list[0], linewidth=15, )

    plt.text(1.3, stim_label_y - 0.1, 'inputs to E2', fontsize=legend_size2, horizontalalignment='center')
    plt.axhline(stim_label_y - 0.2, 1/1.7, 1.6/1.7, color=color_list[1], linewidth=15, )


    E1_input = r_phase2[0][-1] * J_phase2[0][-1] + r_phase2[1][-1] * J_phase2[1][-1]
    E2_input = r_phase2[0][-1] * J_phase2[2][-1] + r_phase2[1][-1] * J_phase2[3][-1]
    P1_input = r_phase2[2][-1] * J_phase2[4][-1] + r_phase2[3][-1] * J_phase2[5][-1]
    P2_input = r_phase2[2][-1] * J_phase2[6][-1] + r_phase2[3][-1] * J_phase2[7][-1]
    S1_input = r_phase2[4][-1] * J_phase2[8][-1] + r_phase2[5][-1] * J_phase2[9][-1]
    S2_input = r_phase2[4][-1] * J_phase2[10][-1] + r_phase2[5][-1] * J_phase2[11][-1]

    data=[[E1_input], [E2_input], [P1_input], [P2_input], [S1_input], [S2_input]]

    X = np.arange(2) * 3
    ax.bar(X + 0.2, E1_input, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, P1_input, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, S1_input, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, E2_input, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, P2_input, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, S2_input, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$E$', r'$P$', r'$S$', r'$E$', r'$P$', r'$S$'], fontsize=font_size_1)
    plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute inputs', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_inputs' + format)
    plt.close()



    ######### absolute weight change pre-test ###########
    ymin = 0
    ymax = 1.2
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len1, figure_width1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    w_EE11_change = np.abs(J_phase2[0][-1] - J_phase2[0][1])
    w_EE22_change = np.abs(J_phase2[3][-1] - J_phase2[3][1])
    w_EP11_change = np.abs(J_phase2[4][-1] - J_phase2[4][1])
    w_EP22_change = np.abs(J_phase2[7][-1] - J_phase2[7][1])
    w_ES11_change = np.abs(J_phase2[8][-1] - J_phase2[8][1])
    w_ES22_change = np.abs(J_phase2[11][-1] - J_phase2[11][1])

    X = np.arange(2) * 3
    ax.bar(X + 0.2, w_EE11_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, w_EP11_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, w_ES11_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, w_EE22_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, w_EP22_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, w_ES22_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$|\Delta W_{E_{1}E_{1}}|$', r'$|\Delta W_{E_{1}P_{1}}|$', r'$|\Delta W_{E_{1}S_{1}}|$',
                                                r'$|\Delta W_{E_{2}E_{2}}|$', r'$|\Delta W_{E_{2}P_{2}}|$', r'$|\Delta W_{E_{2}S_{2}}|$', ],
               fontsize=font_size_2, rotation=90, ha='right', **hfont)
    plt.yticks([0, 0.6, 1.2], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute weight change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_weight_change' + format)
    plt.close()



    ######### absolute rate change pre-test ###########
    ymin = 0
    ymax = 2
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_change = np.abs(r_phase2[0][-1] - r_phase2[0][1])
    rE2_change = np.abs(r_phase2[1][-1] - r_phase2[1][1])
    rP1_change = np.abs(r_phase2[2][-1] - r_phase2[2][1])
    rP2_change = np.abs(r_phase2[3][-1] - r_phase2[3][1])
    rS1_change = np.abs(r_phase2[4][-1] - r_phase2[4][1])
    rS2_change = np.abs(r_phase2[5][-1] - r_phase2[5][1])

    X = np.arange(2) * 3
    ax.bar(X + 0.2, rE1_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, rP1_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, rS1_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, rE2_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, rP2_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, rS2_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$|\Delta r_{E_{1}}|$', r'$|\Delta r_{P_{1}}|$', r'$|\Delta r_{S_{1}}|$',
                                                r'$|\Delta r_{E_{2}}|$', r'$|\Delta r_{P_{2}}|$', r'$|\Delta r_{S_{2}}|$', ],
               fontsize=font_size_1, rotation=90, ha='right', **hfont)
    plt.yticks([0, 1, 2], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute rate change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_abs_rate_change' + format)
    plt.close()



    ######### rate change pre-test ###########
    ymin = -1.5
    ymax = 1.5
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_change = r_phase2[0][-1] - r_phase2[0][1]
    rE2_change = r_phase2[1][-1] - r_phase2[1][1]
    rP1_change = r_phase2[2][-1] - r_phase2[2][1]
    rP2_change = r_phase2[3][-1] - r_phase2[3][1]
    rS1_change = r_phase2[4][-1] - r_phase2[4][1]
    rS2_change = r_phase2[5][-1] - r_phase2[5][1]

    X = np.arange(2) * 3
    ax.bar(X + 0.2, rE1_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, rP1_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, rS1_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, rE2_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, rP2_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, rS2_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$\Delta r_{E_{1}}$', r'$\Delta r_{P_{1}}$', r'$\Delta r_{S_{1}}$',
                                                r'$\Delta r_{E_{2}}$', r'$\Delta r_{P_{2}}$', r'$\Delta r_{S_{2}}$', ],
               fontsize=font_size_2, rotation=90, ha='right', **hfont)
    plt.yticks([-1.5, 0, 1.5], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Rate change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_rate_change' + format)
    plt.close()"""

    # plot the long term behaviour only at 48 hours
    if hour_sim > 24:
        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        # rates ALL
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 1
        ymax = 2.5
        plt.figure(figsize=(figure_width1, figure_len1))
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
        s1, = ax.plot(l_time_points_phase2, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_phase2, rS2, color=color_list[5], linewidth=plot_line_width)
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

        plt.ylim([ymin, ymax])
        plt.yticks([1, 1.5, 2, 2.5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_at],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$', '$r_{b}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)

        plt.tight_layout()
        plt.savefig(name + '_long_ALL_rates' + format)
        plt.close()



        # rates E
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), rb, r_at], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_E_rates' + format)
        plt.close()



        # rates P
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 1
        ymax = 1.2
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(p1), (p2), ],
                  ['$r_{P1}$', '$r_{P2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_P_rates' + format)
        plt.close()



        # rates S
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 2
        ymax = 2.6
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(s1), (s2)], ['$r_{S1}$', '$r_{S2}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper right', handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_S_rates' + format)
        plt.close()



        # thetas
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(theta1, theta2), rb, r_at], [r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()



        # betas
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(beta1, beta2), rb, r_at], [r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ######### All plastic weights during scaling ##########
        ymin = 0
        ymax = 1.8
        plt.figure(figsize=(figure_width1, figure_len1))
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
        #plt.ylim([0, 3])
        #plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weights' + format)
        plt.close()




        ######### Weight ratio during scaling ##########
        ymin = 0
        ymax = 1.5
        plt.figure(figsize=(figure_width1, figure_len1))
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

        wEP1_wES1, = ax.plot(l_time_points_phase2, np.abs(J_EP11 - 0.7)/np.abs(J_DS11 - 0.7), linewidth=plot_line_width, color='black')
        wEP2_wES2, = ax.plot(l_time_points_phase2, np.abs(J_EP22 - 0.7)/np.abs(J_DS22 - 0.7), linewidth=plot_line_width, color='gray')

        plt.plot(l_time_points_phase2, np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)


        ax.legend([wEP1_wES1, wEP2_wES2],
                  [r'$w_{E_{1}P_{1}}/w_{E_{1}S_{1}}$, $w_{E_{2}P_{2}}/w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weight ratio P-to-S', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weight_ratio_P_S' + format)
        plt.close()



def plot_all_only_S(t, res_rates, res_weights, av_threshold, stim_times, name, hour_sim, format='.svg'):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, r_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
    figure_len2, figure_width2 = 13 * ratio, 14.35 * ratio
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

    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
            rE_y_labels = [1, 1.5, 2, 2.5, 3]  # , 3.5] #[0,5,10,15]
            rE_ymin = 1
            rE_ymax = 3
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]

            if t[-1][-1] < 5:
                rE_ymin = 3
                rE_ymax = 5
                rE_y_labels = [1, 2, 3, 4, 5]  # , 3.5] #[0,5,10,15]
            else:
                rE_ymin = 20
                rE_ymax = 40
                rE_y_labels = [25, 30, 35, 40]  # , 3.5] #[0,5,10,15]

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        #p1, = ax.plot(l_time_points_stim, rP1, color=color_list[2], linewidth=plot_line_width)
        #p2, = ax.plot(l_time_points_stim, rP2, color=color_list[3], linewidth=plot_line_width)
        #s1, = ax.plot(l_time_points_stim, rS1, color=color_list[4], linewidth=plot_line_width)
        #s2, = ax.plot(l_time_points_stim, rS2, color=color_list[5], linewidth=plot_line_width)
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


    ######### excitatory weights ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.2
    ymax = 0.8
    plt.figure(figsize=(figure_width1, figure_len1))
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



    """######### inputs pre-test ###########
    ymin = 0
    ymax = 3
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    # text and line to identify the bars of neurons
    plt.text(.4, stim_label_y - 0.1, 'inputs to E1', fontsize=legend_size2, horizontalalignment='center')
    plt.axhline(stim_label_y - 0.2, 0.1/1.7, 0.7/1.7, color=color_list[0], linewidth=15, )

    plt.text(1.3, stim_label_y - 0.1, 'inputs to E2', fontsize=legend_size2, horizontalalignment='center')
    plt.axhline(stim_label_y - 0.2, 1/1.7, 1.6/1.7, color=color_list[1], linewidth=15, )


    E1_input = r_phase2[0][-1] * J_phase2[0][-1] + r_phase2[1][-1] * J_phase2[1][-1]
    E2_input = r_phase2[0][-1] * J_phase2[2][-1] + r_phase2[1][-1] * J_phase2[3][-1]
    P1_input = r_phase2[2][-1] * J_phase2[4][-1] + r_phase2[3][-1] * J_phase2[5][-1]
    P2_input = r_phase2[2][-1] * J_phase2[6][-1] + r_phase2[3][-1] * J_phase2[7][-1]
    S1_input = r_phase2[4][-1] * J_phase2[8][-1] + r_phase2[5][-1] * J_phase2[9][-1]
    S2_input = r_phase2[4][-1] * J_phase2[10][-1] + r_phase2[5][-1] * J_phase2[11][-1]

    data=[[E1_input], [E2_input], [P1_input], [P2_input], [S1_input], [S2_input]]

    X = np.arange(2) * 3
    ax.bar(X + 0.2, E1_input, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, P1_input, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, S1_input, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, E2_input, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, P2_input, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, S2_input, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$E$', r'$P$', r'$S$', r'$E$', r'$P$', r'$S$'], fontsize=font_size_1)
    plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute inputs', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_inputs' + format)
    plt.close()



    ######### absolute weight change pre-test ###########
    ymin = 0
    ymax = 1.2
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len1, figure_width1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    w_EE11_change = np.abs(J_phase2[0][-1] - J_phase2[0][1])
    w_EE22_change = np.abs(J_phase2[3][-1] - J_phase2[3][1])
    w_EP11_change = np.abs(J_phase2[4][-1] - J_phase2[4][1])
    w_EP22_change = np.abs(J_phase2[7][-1] - J_phase2[7][1])
    w_ES11_change = np.abs(J_phase2[8][-1] - J_phase2[8][1])
    w_ES22_change = np.abs(J_phase2[11][-1] - J_phase2[11][1])

    X = np.arange(2) * 3
    ax.bar(X + 0.2, w_EE11_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, w_EP11_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, w_ES11_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, w_EE22_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, w_EP22_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, w_ES22_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$|\Delta W_{E_{1}E_{1}}|$', r'$|\Delta W_{E_{1}P_{1}}|$', r'$|\Delta W_{E_{1}S_{1}}|$',
                                                r'$|\Delta W_{E_{2}E_{2}}|$', r'$|\Delta W_{E_{2}P_{2}}|$', r'$|\Delta W_{E_{2}S_{2}}|$', ],
               fontsize=font_size_2, rotation=90, ha='right', **hfont)
    plt.yticks([0, 0.6, 1.2], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute weight change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_weight_change' + format)
    plt.close()



    ######### absolute rate change pre-test ###########
    ymin = 0
    ymax = 2
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_change = np.abs(r_phase2[0][-1] - r_phase2[0][1])
    rE2_change = np.abs(r_phase2[1][-1] - r_phase2[1][1])
    rP1_change = np.abs(r_phase2[2][-1] - r_phase2[2][1])
    rP2_change = np.abs(r_phase2[3][-1] - r_phase2[3][1])
    rS1_change = np.abs(r_phase2[4][-1] - r_phase2[4][1])
    rS2_change = np.abs(r_phase2[5][-1] - r_phase2[5][1])

    X = np.arange(2) * 3
    ax.bar(X + 0.2, rE1_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, rP1_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, rS1_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, rE2_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, rP2_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, rS2_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$|\Delta r_{E_{1}}|$', r'$|\Delta r_{P_{1}}|$', r'$|\Delta r_{S_{1}}|$',
                                                r'$|\Delta r_{E_{2}}|$', r'$|\Delta r_{P_{2}}|$', r'$|\Delta r_{S_{2}}|$', ],
               fontsize=font_size_1, rotation=90, ha='right', **hfont)
    plt.yticks([0, 1, 2], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Absolute rate change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_abs_rate_change' + format)
    plt.close()



    ######### rate change pre-test ###########
    ymin = -1.5
    ymax = 1.5
    stim_label_y = (ymax - ymin) * .95
    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    mag_of_y = ax.yaxis.get_offset_text()
    mag_of_y.set_size(font_size_1 * .7)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_change = r_phase2[0][-1] - r_phase2[0][1]
    rE2_change = r_phase2[1][-1] - r_phase2[1][1]
    rP1_change = r_phase2[2][-1] - r_phase2[2][1]
    rP2_change = r_phase2[3][-1] - r_phase2[3][1]
    rS1_change = r_phase2[4][-1] - r_phase2[4][1]
    rS2_change = r_phase2[5][-1] - r_phase2[5][1]

    X = np.arange(2) * 3
    ax.bar(X + 0.2, rE1_change, width=0.2, color=color_list[0], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.4, rP1_change, width=0.2, color=color_list[2], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.6, rS1_change, width=0.2, color=color_list[4], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.2, rE2_change, width=0.2, color=color_list[1], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.4, rP2_change, width=0.2, color=color_list[3], edgecolor='black', linewidth=line_width)  # , hatch='/')
    ax.bar(X + 0.9 + 0.6, rS2_change, width=0.2, color=color_list[5], edgecolor='black', linewidth=line_width)  # , hatch='/')

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)
    plt.xticks([0.2, 0.4, 0.6, 1.1, 1.3, 1.5], [r'$\Delta r_{E_{1}}$', r'$\Delta r_{P_{1}}$', r'$\Delta r_{S_{1}}$',
                                                r'$\Delta r_{E_{2}}$', r'$\Delta r_{P_{2}}$', r'$\Delta r_{S_{2}}$', ],
               fontsize=font_size_2, rotation=90, ha='right', **hfont)
    plt.yticks([-1.5, 0, 1.5], fontsize=font_size_1, **hfont)
    plt.xlim([0, 1.7])
    plt.ylim([ymin, ymax])
    plt.ylabel('Rate change', fontsize=font_size_1, **hfont)
    plt.tight_layout()

    plt.savefig(name + '_rate_change' + format)
    plt.close()"""

    # plot the long term behaviour only at 48 hours
    if hour_sim > 24:
        rE_ymin = 0
        rE_y_labels = [0, 20, 40]  # , 3.5] #[0,5,10,15]

        rE1 = r_phase2[0]; rE2 = r_phase2[1]
        rP1 = r_phase2[2]; rP2 = r_phase2[3]
        rS1 = r_phase2[4]; rS2 = r_phase2[5]

        J_EE11 = J_phase2[0]; J_EE22 = J_phase2[3]
        J_EP11 = J_phase2[4]; J_EP22 = J_phase2[7]
        J_DS11 = J_phase2[8]; J_DS22 = J_phase2[11]

        # rates ALL
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        s1, = ax.plot(l_time_points_phase2, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_phase2, rS2, color=color_list[5], linewidth=plot_line_width)
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

        plt.ylim([ymin, ymax])
        plt.ylim([rE_ymin, rE_ymax])
        plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_at],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$', '$r_{b}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)

        plt.tight_layout()
        plt.savefig(name + '_long_ALL_rates' + format)
        plt.close()



        # rates E
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), rb, r_at], [r'$r_{E1}$, $r_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_E_rates' + format)
        plt.close()



        # rates P
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 1
        ymax = 1.2
        plt.figure(figsize=(figure_width1, figure_len1))
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

        #plt.ylim([ymin, ymax])
        #plt.yticks([1, 1.1, 1.2], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(p1), (p2), ],
                  ['$r_{P1}$', '$r_{P2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_P_rates' + format)
        plt.close()



        # rates S
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 2
        ymax = 2.6
        plt.figure(figsize=(figure_width1, figure_len1))
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

        #plt.ylim([ymin, ymax])
        #plt.yticks([2, 2.2, 2.4, 2.6], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(s1), (s2)], ['$r_{S1}$', '$r_{S2}$'], handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper right', handlelength=3)
        plt.tight_layout()

        plt.savefig(name + '_long_S_rates' + format)
        plt.close()



        # thetas
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(theta1, theta2), rb, r_at], [r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()



        # betas
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(beta1, beta2), rb, r_at], [r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ######### All plastic weights during scaling ##########
        ymin = 0
        ymax = 1.8
        plt.figure(figsize=(figure_width1, figure_len1))
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
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weights', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weights' + format)
        plt.close()




        ######### Weight ratio during scaling ##########
        ymin = 0
        ymax = 1.5
        plt.figure(figsize=(figure_width1, figure_len1))
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

        wEP1_wES1, = ax.plot(l_time_points_phase2, np.abs(J_EP11 - 0.7)/np.abs(J_DS11 - 0.7), linewidth=plot_line_width, color='black')
        wEP2_wES2, = ax.plot(l_time_points_phase2, np.abs(J_EP22 - 0.7)/np.abs(J_DS22 - 0.7), linewidth=plot_line_width, color='gray')

        plt.plot(l_time_points_phase2, np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3)


        ax.legend([wEP1_wES1, wEP2_wES2],
                  [r'$w_{E_{1}P_{1}}/w_{E_{1}S_{1}}$, $w_{E_{2}P_{2}}/w_{E_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        plt.ylim([ymin, ymax])
        plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Weight ratio P-to-S', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_weight_ratio_P_S' + format)
        plt.close()




def plot_all_3_compartmental(t, res_rates, res_weights, av_threshold, stim_times, name, hour_sim, format='.svg', scale_y=True):

    (l_time_points_stim, l_time_points_phase2) = t
    (r_phase1, I_phase1, r_phase2, I_phase2, set_phase2, r_phase3, max_E) = res_rates
    (J_EE_phase1, J_phase2) = res_weights

    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
    figure_len2, figure_width2 = 13 * ratio, 14.35 * ratio
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
                  '#6600cc', # timepoints in long simulation
                  'indigo', 'mediumpurple', # I_AD1 and WDE11, I_AD2 and WDE22
                  'olive', 'darkkhaki']     # I_BD1, I_BD2

    rE_y_labels = [0.5, 1, 1.5, 2, 2.5] #, 3.5] #[0,5,10,15]
    rE_ymax = 2.5
    rE_ymin = 0.5

    stim_applied = 1

    for i in stim_times:
        (stim_start, stim_stop) = i

        if stim_applied == 1:
            rE1 = r_phase1[0]; rE2 = r_phase1[1]
            rP1 = r_phase1[2]; rP2 = r_phase1[3]
            rS1 = r_phase1[4]; rS2 = r_phase1[5]
        elif stim_applied == 2:
            rE1 = r_phase3[0]; rE2 = r_phase3[1]
            rP1 = r_phase3[2]; rP2 = r_phase3[3]
            rS1 = r_phase3[4]; rS2 = r_phase3[5]

        ######### rates ###########
        xmin = 0
        xmax = stim_times[0][1] + 5

        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)
        plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

        p1, = ax.plot(l_time_points_stim, rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(l_time_points_stim, rP2, color=color_list[3], linewidth=plot_line_width)
        s1, = ax.plot(l_time_points_stim, rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_stim, rS2, color=color_list[5], linewidth=plot_line_width)
        e1, = ax.plot(l_time_points_stim, rE1, color=color_list[0], linewidth=plot_line_width, label=r'$r_{E1}$')
        e2, = ax.plot(l_time_points_stim, rE2, color=color_list[1], linewidth=plot_line_width, label=r'$r_{E2}$')
        r_at, = plt.plot(l_time_points_stim, av_threshold * np.ones_like(l_time_points_stim), dash_capstyle='round',
                         linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_stim, r_phase1[0][0] * np.ones_like(l_time_points_stim), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([rE_ymin, rE_ymax])
            plt.yticks(rE_y_labels, fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
        plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)

        ax.legend([(e1, e2), (rb, r_at)], ['$r_{E1}$, $r_{E2}$', '$r_b$, $r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()

        plt.savefig(name + "_stim" + str(stim_applied) + '_activity' + format)
        plt.close()

        stim_applied = stim_applied + 1

    ######### wAB during conditioning ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.3
    ymax = 0.8
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    wde11, = ax.plot(l_time_points_stim, J_EE_phase1[0], color=color_list[0], linewidth=plot_line_width)
    wde12, = ax.plot(l_time_points_stim, J_EE_phase1[1], '--', color=color_list[0], linewidth=plot_line_width)
    wde21, = ax.plot(l_time_points_stim, J_EE_phase1[2], color=color_list[1], linewidth=plot_line_width)
    wde22, = ax.plot(l_time_points_stim, J_EE_phase1[3], '--', color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if scale_y:
        plt.ylim([ymin, ymax])
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, ], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    ax.legend([(wde11, wde12), (wde21, wde22)],
              [r'$w_{A_{1}B_{1}}$, $w_{A_{1}B_{2}}$', '$w_{A_{2}B_{1}}$', r'$w_{A_{2}B_{2}}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()

    plt.savefig(name + 'conditioning_wAB' + format)
    plt.close()

    #########  wBB during conditioning  ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 0.3
    ymax = 0.8
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    plt.axvspan(stim_times[0][0], stim_times[0][1], color='gray', alpha=0.15)

    wee11, = ax.plot(l_time_points_stim, J_EE_phase1[4], color=color_list[0], linewidth=plot_line_width)
    wee12, = ax.plot(l_time_points_stim, J_EE_phase1[5], '--', color=color_list[0], linewidth=plot_line_width)
    wee21, = ax.plot(l_time_points_stim, J_EE_phase1[6], color=color_list[1], linewidth=plot_line_width)
    wee22, = ax.plot(l_time_points_stim, J_EE_phase1[7], '--', color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if scale_y:
        plt.ylim([ymin, ymax])
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, ], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Weights', fontsize=font_size_label, **hfont)

    ax.legend([(wee11, wee12), (wee21, wee22)],
              [r'$w_{B_{1}B_{1}}$, $w_{B_{1}B_{2}}$', '$w_{B_{2}B_{1}}$', r'$w_{B_{2}B_{2}}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()

    plt.savefig(name + 'conditioning_wBB' + format)
    plt.close()

    ######### conditioning currents ###########
    xmin = 0
    xmax = stim_times[0][1] + 5
    ymin = 1
    ymax = 5
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

    I_A1, = ax.plot(l_time_points_stim, I_phase1[0], color=color_list[7], linewidth=plot_line_width)
    I_A2, = ax.plot(l_time_points_stim, I_phase1[1], color=color_list[8], linewidth=plot_line_width)
    I_B1, = ax.plot(l_time_points_stim, I_phase1[2], color=color_list[9], linewidth=plot_line_width)
    I_B2, = ax.plot(l_time_points_stim, I_phase1[3], color=color_list[10],linewidth=plot_line_width)
    I_E1, = ax.plot(l_time_points_stim, I_phase1[4], color=color_list[0], linewidth=plot_line_width)
    I_E2, = ax.plot(l_time_points_stim, I_phase1[5], color=color_list[1], linewidth=plot_line_width)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    if scale_y:
        plt.ylim([ymin, ymax])
        plt.yticks([1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([5, 10, 15, 20, 25], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_label, **hfont)
    plt.ylabel('Currents', fontsize=font_size_label, **hfont)
    ax.legend([(I_A1, I_A2), (I_B1, I_B2), (I_E1, I_E2), rb, r_at],
              [r'$I_{A1}$, $I_{A2}$', r'$I_{B1}$, $I_{BD2}$', r'$I_{E1}$, $I_{E2}$'],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=legend_size, loc='upper left', handlelength=5)
    plt.tight_layout()
    plt.savefig(name + 'conditioning_currents' + format)
    plt.close()

    # plot the long term behaviour only at 48 hours
    if hour_sim >= 4.8:
        l_rE1 = r_phase2[0]; l_rE2 = r_phase2[1]
        l_rP1 = r_phase2[2]; l_rP2 = r_phase2[3]
        l_rS1 = r_phase2[4]; l_rS2 = r_phase2[5]

        l_thetaA1, l_thetaA2, l_thetaB1, l_thetaB2, l_thetaE1, l_thetaE2 = set_phase2[0], set_phase2[1], set_phase2[2], set_phase2[3], set_phase2[4], set_phase2[5]
        l_betaA1, l_betaA2, l_betaB1, l_betaB2, l_betaE1, l_betaE2 = set_phase2[0], set_phase2[1], set_phase2[2], set_phase2[3], set_phase2[4], set_phase2[5]
        l_IA1, l_IA2, l_IB1, l_IB2, l_IE1, l_IE2 = I_phase2[0], I_phase2[1], I_phase2[2], I_phase2[3], I_phase2[4], I_phase2[5]


        l_J_AB11, l_J_AB12, l_J_AB21, l_J_AB22 = J_phase2[0], J_phase2[1], J_phase2[2], J_phase2[3]
        l_J_BB11, l_J_BB12, l_J_BB21, l_J_BB22 = J_phase2[4], J_phase2[5], J_phase2[6], J_phase2[7]
        l_J_EP11, l_J_EP12, l_J_EP21, l_J_EP22 = J_phase2[8], J_phase2[9], J_phase2[10],J_phase2[11]
        l_J_AS11, l_J_AS12, l_J_AS21, l_J_AS22 = J_phase2[12],J_phase2[13],J_phase2[14],J_phase2[15]

        # rates ALL
        xmin = 0
        xmax = l_time_points_phase2[-1]
        ymin = 0.5
        ymax = 2.5
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        p1, = ax.plot(l_time_points_phase2, l_rP1, color=color_list[2], linewidth=plot_line_width)
        p2, = ax.plot(l_time_points_phase2, l_rP2, color=color_list[3], linewidth=plot_line_width)
        s1, = ax.plot(l_time_points_phase2, l_rS1, color=color_list[4], linewidth=plot_line_width)
        s2, = ax.plot(l_time_points_phase2, l_rS2, color=color_list[5], linewidth=plot_line_width)
        e1, = ax.plot(l_time_points_phase2, l_rE1, color=color_list[0], linewidth=plot_line_width)
        e2, = ax.plot(l_time_points_phase2, l_rE2, color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.5, 1, 1.5, 2, 2.5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
        ax.legend([(e1, e2), (p1, p2), (s1, s2), rb, r_at],
                  [r'$r_{E1}$, $r_{E2}$', '$r_{P1}$, $r_{P2}$', '$r_{S1}$, $r_{S2}$', '$r_{b}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right',
                  handlelength=3)

        plt.tight_layout()
        plt.savefig(name + '_long_ALL_rates' + format)
        plt.close()



        # currents
        ymin = 1
        ymax = 5
        xmin = 0
        xmax = l_time_points_phase2[-1]
        plt.figure(figsize=(figure_width1, figure_len1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(line_width)
        plt.tick_params(width=line_width, length=tick_len)

        I_A1, = ax.plot(l_time_points_phase2, l_IA1, color=color_list[7], linewidth=plot_line_width)
        I_A2, = ax.plot(l_time_points_phase2, l_IA2, color=color_list[8], linewidth=plot_line_width)
        I_B1, = ax.plot(l_time_points_phase2, l_IB1, color=color_list[9], linewidth=plot_line_width)
        I_B2, = ax.plot(l_time_points_phase2, l_IB2, color=color_list[10],linewidth=plot_line_width)
        I_E1, = ax.plot(l_time_points_phase2, l_IE1, color=color_list[0], linewidth=plot_line_width)
        I_E2, = ax.plot(l_time_points_phase2, l_IE2, color=color_list[1], linewidth=plot_line_width)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel('Currents', fontsize=font_size_label, **hfont)
        ax.legend([(I_A1, I_A2), (I_B1, I_B2), (I_E1, I_E2), rb, r_at],
                  [r'$I_{A1}$, $I_{A2}$', r'$I_{B1}$, $I_{B2}$', r'$I_{E1}$, $I_{E2}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)
        plt.tight_layout()
        plt.savefig(name + '_long_currents' + format)
        plt.close()


        # thetas
        ymin = 1
        ymax = 5
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

        thetaA1, = ax.plot(l_time_points_phase2, l_thetaA1, color=color_list[7], linewidth=plot_line_width)
        thetaA2, = ax.plot(l_time_points_phase2, l_thetaA2, color=color_list[8], linewidth=plot_line_width)
        thetaB1, = ax.plot(l_time_points_phase2, l_thetaB1, color=color_list[9], linewidth=plot_line_width)
        thetaB2, = ax.plot(l_time_points_phase2, l_thetaB2, color=color_list[10],linewidth=plot_line_width)
        thetaE1, = ax.plot(l_time_points_phase2, l_thetaE1, color=color_list[0], linewidth=plot_line_width)
        thetaE2, = ax.plot(l_time_points_phase2, l_thetaE2, color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point $\theta$', fontsize=font_size_label, **hfont)
        ax.legend([(thetaA1, thetaA2), (thetaB1, thetaB2), (thetaE1, thetaE2), rb, r_at],
                  [r'$\theta_{A1}$, $\theta_{A2}$', r'$\theta_{B1}$, $\theta_{B2}$', r'$\theta_{E1}$, $\theta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  fontsize=legend_size, loc='upper left', handlelength=5)

        plt.tight_layout()

        plt.savefig(name + '_long_thetas' + format)
        plt.close()


        # betas
        ymin = 1
        ymax = 5
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

        betaA1, = ax.plot(l_time_points_phase2, l_betaA1, color=color_list[7], linewidth=plot_line_width)
        betaA2, = ax.plot(l_time_points_phase2, l_betaA2, color=color_list[8], linewidth=plot_line_width)
        betaB1, = ax.plot(l_time_points_phase2, l_betaB1, color=color_list[9], linewidth=plot_line_width)
        betaB2, = ax.plot(l_time_points_phase2, l_betaB2, color=color_list[10],linewidth=plot_line_width)
        betaE1, = ax.plot(l_time_points_phase2, l_betaE1, color=color_list[0], linewidth=plot_line_width)
        betaE2, = ax.plot(l_time_points_phase2, l_betaE2, color=color_list[1], linewidth=plot_line_width)
        r_at, = plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                         linestyle=line_style_r_at, color=color_list[0], linewidth=plot_line_width)
        rb, = plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                       linestyle=line_style_rb, color=color_list[0], linewidth=plot_line_width*1.3)

        plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
        plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([1, 2, 3, 4, 5], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'Set-point regulator $\beta$', fontsize=font_size_label, **hfont)
        ax.legend([(betaA1, betaA2), (betaB1, betaB2), (betaE1, betaE2), rb, r_at],
                  [r'$\beta_{A1}$, $\beta_{A2}$', r'$\beta_{B1}$, $\beta_{B2}$', r'$\beta_{E1}$, $\beta_{E2}$', '$r_{bs}$', '$r_{at}$'],
                  handler_map={tuple:HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper left',handlelength=5)
        plt.tight_layout()

        plt.savefig(name + '_long_betas' + format)
        plt.close()



        ######### All plastic weights during scaling ##########

        # wAB
        ymin = 0.3
        ymax = 0.8
        plt.figure(figsize=(figure_width1, figure_len1))
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

        wAB11, = ax.plot(l_time_points_phase2, l_J_AB11, linewidth=plot_line_width, color=color_list[7])
        wAB12, = ax.plot(l_time_points_phase2, l_J_AB12, '--', linewidth=plot_line_width, color=color_list[7])
        wAB21, = ax.plot(l_time_points_phase2, l_J_AB21, linewidth=plot_line_width, color=color_list[8])
        wAB22, = ax.plot(l_time_points_phase2, l_J_AB22, '--', linewidth=plot_line_width, color=color_list[8])
        ax.legend([(wAB11, wAB12), (wAB21, wAB22)],
                  [r'$w_{A_{1}B_{1}}$, $w_{A_{1}B_{2}}$', r'$w_{A_{2}B_{1}}$, $w_{A_{2}B_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)
        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{AB}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_AB' + format)
        plt.close()

        # wBB
        ymin = 0.3
        ymax = 0.8
        plt.figure(figsize=(figure_width1, figure_len1))
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

        wBB11, = ax.plot(l_time_points_phase2, l_J_BB11, linewidth=plot_line_width, color=color_list[0])
        wBB12, = ax.plot(l_time_points_phase2, l_J_BB12, '--', linewidth=plot_line_width, color=color_list[0])
        wBB21, = ax.plot(l_time_points_phase2, l_J_BB21, linewidth=plot_line_width, color=color_list[1])
        wBB22, = ax.plot(l_time_points_phase2, l_J_BB22, '--', linewidth=plot_line_width, color=color_list[1])
        ax.legend([(wBB11, wBB12), (wBB21, wBB22)],
                  [r'$w_{B_{1}B_{1}}$, $w_{B_{1}B_{2}}$', r'$w_{B_{2}B_{1}}$, $w_{B_{2}B_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{BB}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_BB' + format)
        plt.close()

        # wEP
        ymin = 0
        ymax = 0.75
        plt.figure(figsize=(figure_width2, figure_len2))
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

        wEP11, = ax.plot(l_time_points_phase2, l_J_EP11, linewidth=plot_line_width, color=color_list[2])
        wEP12, = ax.plot(l_time_points_phase2, l_J_EP12, '--', linewidth=plot_line_width, color=color_list[2])
        wEP21, = ax.plot(l_time_points_phase2, l_J_EP21, linewidth=plot_line_width, color=color_list[3])
        wEP22, = ax.plot(l_time_points_phase2, l_J_EP22, '--', linewidth=plot_line_width, color=color_list[3])
        ax.legend([(wEP11, wEP12), (wEP21, wEP22)],
                  [r'$w_{E_{1}P_{1}}$, $w_{E_{1}P_{2}}$', r'$w_{E_{2}P_{1}}$, $w_{E_{2}P_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0, 0.25, 0.5, 0.75], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{EP}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_EP' + format)
        plt.close()

        # wAS
        ymin = 0
        ymax = 0.75
        plt.figure(figsize=(figure_width2, figure_len2))
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

        wAS11, = ax.plot(l_time_points_phase2, l_J_AS11, linewidth=plot_line_width, color=color_list[4])
        wAS12, = ax.plot(l_time_points_phase2, l_J_AS12, '--', linewidth=plot_line_width, color=color_list[4])
        wAS21, = ax.plot(l_time_points_phase2, l_J_AS21, linewidth=plot_line_width, color=color_list[5])
        wAS22, = ax.plot(l_time_points_phase2, l_J_AS22, '--', linewidth=plot_line_width, color=color_list[5])
        ax.legend([(wAS11, wAS12), (wAS21, wAS22)],
                  [r'$w_{A_{1}S_{1}}$, $w_{A_{1}S_{2}}$', r'$w_{A_{2}S_{1}}$, $w_{A_{2}S_{2}}$'],
                  handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,
                  loc='upper right', handlelength=3)

        plt.xticks(fontsize=font_size_1, **hfont)
        plt.yticks(fontsize=font_size_1, **hfont)

        if scale_y:
            plt.ylim([ymin, ymax])
            plt.yticks([0, 0.25, 0.5, 0.75], fontsize=font_size_1, **hfont)
        plt.xlim([xmin, xmax])
        plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
        plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
        plt.ylabel(r'$W_{AS}$ connections', fontsize=font_size_label, **hfont)
        plt.tight_layout()

        plt.savefig(name + '_long_W_AS' + format)
        plt.close()




def plot_rates_at_regular_intervals(r_phase1, l_time_points_phase2, r_phase2, hour_sims, l_delta_rE1, av_threshold,
                                    delta_t, sampling_rate_sim, name, format='.svg'):


    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
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



    ### plot the long term behaviour for 48 hours
    # the phase2 arrays hold the 48h simulation results
    rE1 = r_phase2[0]; rE2 = r_phase2[1]

    baseline_reactivation = av_threshold / 1.15
    change_in_reactivation = 100*(np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    """# Reactivation E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 1
    ymax = 3
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_at_every_hour = rE1[int((60*60-20)* (1 / delta_t) * (1 / sampling_rate_sim))::int(60*60* (1 / delta_t) * (1 / sampling_rate_sim))]

    ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width, label='$r_{E1}$')
    ax.fill_between(hour_sims, rE1_at_every_hour, np.array(l_delta_rE1), color=color_list[0], alpha=0.2, label='$\Delta r_{E_{1}}$')
    plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='$r_{at}$')
    plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                   linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3, label='$r_{b}$')
    plt.plot(hour_sims[np.where(np.array(l_delta_rE1)-av_threshold*np.ones_like(hour_sims) < 0.01)][0],
             np.array(l_delta_rE1)[np.where(np.array(l_delta_rE1)-av_threshold < 0.01)][0], 'r*', markersize=plot_line_width*4, label='Onset specificity')
    
    plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([1, 2, 3], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3, ncol=1)

    plt.tight_layout()
    plt.savefig(name + '_long_E1_rate' + format)
    plt.close()


    # delta reactivation
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = -20
    ymax = 100
    plt.figure(figsize=(figure_width1, figure_len1))
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

    plt.plot(hour_sims, change_in_reactivation, color=color_list[0], linewidth=plot_line_width, label='% change')
    plt.plot(hour_sims, 15 * np.ones_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='Tolerance')
    plt.plot(hour_sims, np.zeros_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_rb, color='black', linewidth=plot_line_width, label='Baseline reactivation')


    plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([-20, 0, 15, 50, 100], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('% change in $r^{re}_{E1}$', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation' + format)
    plt.close()"""


    ### Gradient colors

    # Reactivation E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 1
    ymax = 3
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    # Plot rE1 for 48h
    ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width, label='$r_{E1}$')

    ### Fill the region with gradient color
    rE1_at_every_hour = rE1[int((60*60-20)* (1 / delta_t) * (1 / sampling_rate_sim))::int(60*60* (1 / delta_t) * (1 / sampling_rate_sim))]

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap_name = 'green_white_red'

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=15, vmax=100)

    # Normalize the change in reactivation
    values_range = norm(change_in_reactivation)  # Normalize using TwoSlopeNorm


    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        color = cmap(values_range[i])
        ax.fill_between(hour_sims[i:i + 2], rE1_at_every_hour[i:i + 2], l_delta_rE1[i:i + 2], color=color,
                        edgecolor=(0, 0, 0, 0))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([-10, 15, 100], fontsize=font_size_1, **hfont)

    plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='$r_{at}$')
    plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                   linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3, label='$r_{b}$')
    """plt.plot(hour_sims[np.where(np.array(l_delta_rE1)-av_threshold*np.ones_like(hour_sims) < 0.01)][0],
             np.array(l_delta_rE1)[np.where(np.array(l_delta_rE1)-av_threshold < 0.01)][0], 'r*', markersize=plot_line_width*4, label='Onset specificity')
    """
    plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([1, 2, 3], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3, ncol=1)

    plt.tight_layout()
    plt.savefig(name + '_long_E1_rate_colorful' + format)
    plt.close()




    # delta reactivation
    xmin = 0
    xmax = l_time_points_phase2[-1]
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
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap_name = 'green_white_red'

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
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
    plt.yticks([-20, 0, 15, 50, 100], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('% change in $r_{E1,re}$', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size,loc='lower center', bbox_to_anchor=(0.5, 1.05), handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation_color' + format)
    plt.close()




def plot_rates_at_regular_intervals_only_S(r_phase1, l_time_points_phase2, r_phase2, hour_sims, l_delta_rE1, av_threshold,
                                    delta_t, sampling_rate_sim, name, format='.svg'):
    # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 12.75 * ratio
    figure_len1, figure_width1 = 13 * ratio, 13.7 * ratio
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
    color_list = ['#3276b3', '#91bce0',  # rE1 and WEE11, rE2 and WEE22
                  '#C10000', '#EFABAB',  # rP1 and WEP11, rP2 and WEP22
                  '#007100', '#87CB87',  # rS1 and WES11, rS2 and WES22
                  '#6600cc']  # timepoints in long simulation

    ### plot the long term behaviour for 48 hours
    # the phase2 arrays hold the 48h simulation results
    rE1 = r_phase2[0];
    rE2 = r_phase2[1]

    baseline_reactivation = av_threshold / 1.15
    change_in_reactivation = 100 * (np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    """# Reactivation E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 1
    ymax = 3
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    rE1_at_every_hour = rE1[int((60*60-20)* (1 / delta_t) * (1 / sampling_rate_sim))::int(60*60* (1 / delta_t) * (1 / sampling_rate_sim))]

    ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width, label='$r_{E1}$')
    ax.fill_between(hour_sims, rE1_at_every_hour, np.array(l_delta_rE1), color=color_list[0], alpha=0.2, label='$\Delta r_{E_{1}}$')
    plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='$r_{at}$')
    plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
                   linestyle=line_style_rb, color='black', linewidth=plot_line_width*1.3, label='$r_{b}$')
    plt.plot(hour_sims[np.where(np.array(l_delta_rE1)-av_threshold*np.ones_like(hour_sims) < 0.01)][0],
             np.array(l_delta_rE1)[np.where(np.array(l_delta_rE1)-av_threshold < 0.01)][0], 'r*', markersize=plot_line_width*4, label='Onset specificity')

    plt.vlines(4,  ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([1, 2, 3], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3, ncol=1)

    plt.tight_layout()
    plt.savefig(name + '_long_E1_rate' + format)
    plt.close()


    # delta reactivation
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = -20
    ymax = 100
    plt.figure(figsize=(figure_width1, figure_len1))
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

    plt.plot(hour_sims, change_in_reactivation, color=color_list[0], linewidth=plot_line_width, label='% change')
    plt.plot(hour_sims, 15 * np.ones_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='Tolerance')
    plt.plot(hour_sims, np.zeros_like(hour_sims), dash_capstyle='round',
                     linestyle=line_style_rb, color='black', linewidth=plot_line_width, label='Baseline reactivation')


    plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([-20, 0, 15, 50, 100], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('% change in $r^{re}_{E1}$', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation' + format)
    plt.close()"""

    ### Gradient colors

    # Reactivation E1
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 1
    ymax = 40
    plt.figure(figsize=(figure_width1, figure_len1))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)

    # Plot rE1 for 48h
    ax.plot(l_time_points_phase2, rE1, color=color_list[0], linewidth=plot_line_width, label='$r_{E1}$')

    ### Fill the region with gradient color
    rE1_at_every_hour = rE1[int((60 * 60 - 20) * (1 / delta_t) * (1 / sampling_rate_sim))::int(
        60 * 60 * (1 / delta_t) * (1 / sampling_rate_sim))]

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap_name = 'green_white_red'

    # Define the colors for the custom colormap
    cmap_colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=15, vmax=100)

    # Normalize the change in reactivation
    values_range = norm(change_in_reactivation)  # Normalize using TwoSlopeNorm

    # Plotting and using the values to pick colors
    for i in range(len(hour_sims) - 1):
        color = cmap(values_range[i])
        ax.fill_between(hour_sims[i:i + 2], rE1_at_every_hour[i:i + 2], l_delta_rE1[i:i + 2], color=color,
                        edgecolor=(0, 0, 0, 0))

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([-10, 15, 100], fontsize=font_size_1, **hfont)

    plt.plot(l_time_points_phase2, av_threshold * np.ones_like(l_time_points_phase2), dash_capstyle='round',
             linestyle=line_style_r_at, color='black', linewidth=plot_line_width, label='$r_{at}$')
    plt.plot(l_time_points_phase2, r_phase1[0][0] * np.ones_like(l_time_points_phase2), dash_capstyle='round',
             linestyle=line_style_rb, color='black', linewidth=plot_line_width * 1.3, label='$r_{b}$')
    """plt.plot(hour_sims[np.where(np.array(l_delta_rE1)-av_threshold*np.ones_like(hour_sims) < 0.01)][0],
             np.array(l_delta_rE1)[np.where(np.array(l_delta_rE1)-av_threshold < 0.01)][0], 'r*', markersize=plot_line_width*4, label='Onset specificity')
    """
    plt.vlines(4, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(24, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)
    plt.vlines(48, ymin, ymax, color_list[6], linewidth=plot_line_width, alpha=0.15)

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    plt.yticks([1, 20, 40], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('Firing rate', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='upper right', handlelength=3,
              ncol=1)

    plt.tight_layout()
    plt.savefig(name + '_long_E1_rate_colorful' + format)
    plt.close()

    # delta reactivation
    xmin = 0
    xmax = l_time_points_phase2[-1]
    ymin = 1
    ymax = 10000
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
    change_in_reactivation = 100 * (np.array(l_delta_rE1) - baseline_reactivation) / baseline_reactivation

    # Define the colors for the custom colormap
    cmap_colors = [(1, 1, 1), (1, 0, 0)]  # Green, White, Red
    cmap_name = 'green_white_red'

    # Define the colors for the custom colormap
    cmap_colors = [(1, 1, 1), (1, 0, 0)]  # White, Red
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

    # Define the TwoSlopeNorm with vcenter at 15, vmin at -10, and vmax at 100
    norm = mcolors.TwoSlopeNorm(vmin=15, vcenter=115/2, vmax=100)

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
    plt.yscale('log')  # Set the y-axis to logarithmic scale

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(width=line_width, length=tick_len, labelsize=font_size_1)
    cbar.set_ticks([15, 100], fontsize=font_size_1, **hfont)
    cbar.outline.set_linewidth(line_width)  # Set colorbar border thickness

    plt.xticks(fontsize=font_size_1, **hfont)
    plt.yticks(fontsize=font_size_1, **hfont)

    plt.ylim([ymin, ymax])
    #plt.yticks([0, 100, 1000, 10000], fontsize=font_size_1, **hfont)
    plt.xlim([xmin, xmax])
    plt.xticks([4, 24, 48], fontsize=font_size_1, **hfont)
    plt.xlabel(x_label_text, fontsize=font_size_label, **hfont)
    plt.ylabel('% change in $r_{E1,re}$ (log)', fontsize=font_size_label, **hfont)
    ax.legend(handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=legend_size, loc='lower center',
              bbox_to_anchor=(0.5, 1.05), handlelength=3)

    plt.tight_layout()
    plt.savefig(name + '_delta_reactivation_color' + format)
    plt.close()




def plot_span_init_conds(results_list, w_x_axis, w_y_axis, title_x_axis, title_y_axis,
                         directory, name, n, plot_bars=0, plot_legends=0, format='.png', title=''):

    result = np.array(results_list).reshape(n,n)

   # plotting configuration
    ratio = 1.5
    figure_len, figure_width = 13 * ratio, 14.4 * ratio
    font_size_1, font_size_2 = 65 * ratio, 36 * ratio
    font_size_label = 65 * ratio
    legend_size = 50 * ratio
    line_width, tick_len = 5 * ratio, 20 * ratio
    marker_size = 550 * ratio
    marker_edge_width = 3 * ratio
    plot_line_width = 7 * ratio
    hfont = {'fontname': 'Arial'}

    cmap_name = 'PiYG'
    cmap =colmaps[cmap_name]

    ones_matrix = np.ones((n,n))
    xmin = w_x_axis[0]
    xmax = w_x_axis[-1]
    ymin = w_y_axis[0]
    ymax = w_y_axis[-1]

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

    for idx in np.arange(len(w_y_axis)):
        for idy in np.arange(len(w_x_axis)):
            if np.isnan(result[idy][idx]):
                plt.scatter(w_y_axis[idx], w_x_axis[idy], c='black', s=marker_size, marker='s',
                            linewidths=marker_edge_width)
            else:
                if result[idy][idx] == 1:
                    plt.scatter(w_y_axis[idx], w_x_axis[idy], c='green', s=marker_size, marker='s',
                                linewidths=marker_edge_width)
                elif result[idy][idx] == 2:
                    plt.scatter(w_y_axis[idx], w_x_axis[idy], c='silver', s=marker_size, marker='s',
                                linewidths=marker_edge_width)
                elif result[idy][idx] == 0:
                    plt.scatter(w_y_axis[idx], w_x_axis[idy], c='red', s=marker_size, marker='s',
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




