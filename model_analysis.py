import numpy as np
from numba import cuda, jit
import matplotlib.pyplot as plt
from util import *
from model import *

"""
def analyze_model_2PC():
    sim_duration = 1  # s
    delta_t = 0.0005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))

    # flags are set
    plasticity = 0
    synaptic_scaling = 0
    flags = (plasticity, synaptic_scaling)

    rE1 = np.zeros(sim_timepoints, dtype=np.float64)
    rE2 = np.zeros(sim_timepoints, dtype=np.float64)
    rP = np.zeros(sim_timepoints, dtype=np.float64)
    rS = np.zeros(sim_timepoints, dtype=np.float64)
    vars = (rE1,rE2,rP,rS)

    t = np.linspace(0, sim_duration, sim_timepoints)

    # number of neurons
    N_PC = 80
    N_PV = 20
    N_SOM = 20
    num_neurons = (N_PC,N_PV,N_SOM)

    stim_start = 1.5
    stim_strength = 2
    stimulus = np.zeros(len(t))
    stimulus[int(stim_start*(1/delta_t)):] = stim_strength

    tau_E = 0.06 # s
    tau_I = 0.002 # s
    taus = (tau_E,tau_I)

    theta = np.ones(N_PC)*5 # 1/s
    lambda_D = 0.35
    lambda_E = 0.35
    exc_params = (theta,lambda_D,lambda_E)

    # background inputs
    x_E = 20 * np.ones(N_PC)
    x_D = 0 * np.ones(N_PC)
    x_P = 10 * np.ones(N_PV)
    x_S = 10 * np.ones(N_SOM)
    back_inputs = (x_E,x_D,x_P,x_S)

    w_DE1,w_DE2 = 2, 2
    w_DE12,w_DE21 = 2, 2
    w_DS1,w_DS2 = .35, .35
    w_EP1,w_EP2 = .6, .6
    w_PE1,w_PE2 = .7, .7
    w_SE1,w_SE2 = .1, .1
    w_PS = .5
    w_PP = .5

    weight_strengths = (w_DE1,w_DE12,w_DS1,w_EP1,w_PE1,w_SE1,w_DE2,w_DE21,w_DS2,w_EP2,w_PE2,w_SE2,w_PS,w_PP)
    path = r"params/weights_model_2PC.pkl"
    create_save_weights_2PC(N_PC,N_PV,N_SOM,weight_strengths,path)
    with open(path, 'rb') as f:
        weights = pickle.load(f)

    # read the noise created before
    path = r"params/noise_model_2PC.pkl"
    with open(path, 'rb') as f:
        noises = pickle.load(f)


    model_2PC(delta_t,vars,t,num_neurons,weights,noises,back_inputs,stimulus,taus,exc_params,flags)
    show_plot_2PC_old(rE1,rE2, rP, rS, t, int((1/delta_t)*stim_start))
#show_plot_2PC_stim(rE1_sol,rE2_sol, rP_sol, rS_sol, t, int((1/delta_t)*stim_start))
"""

def grid_search_model_2D():
    sim_duration = 3  # s
    delta_t = 0.00005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, sim_timepoints)

    # flags are set
    plasticity = 0
    synaptic_scaling = 0
    flags = (plasticity, synaptic_scaling)

    rE1 = np.zeros(sim_timepoints, dtype=np.float64)
    rE2 = np.zeros(sim_timepoints, dtype=np.float64)
    rP1 = np.zeros(sim_timepoints, dtype=np.float64)
    rP2 = np.zeros(sim_timepoints, dtype=np.float64)
    rS1 = np.zeros(sim_timepoints, dtype=np.float64)
    rS2 = np.zeros(sim_timepoints, dtype=np.float64)
    vars = (rE1, rE2, rP1, rP2, rS1, rS2)


    # number of neurons
    N_PC = 80
    N_PV = 11
    N_SOM = 9
    num_neurons = (N_PC, N_PV, N_SOM)

    tau_E = 0.02  # s
    tau_P = 0.005  # s
    tau_S = 0.01
    taus = (tau_E, tau_P, tau_S)

    lambda_D = 0.27
    lambda_E = 0.31

    x_E_strengths = [15]
    x_D_strengths = [1]
    x_P_strengths = [7]
    x_S_strengths = [1]
    stim_strengths = [10]
    theta_strengths = [3]
    w_DEs = [1.5,2.5,3]
    w_PEs = [.7,.9,1.2]
    w_SEs = [.1,.3]
    w_PSs = [.1,.3,.7]
    # read the noise created before
    path = r"params/noise_model_2D.pkl"
    # create_save_noise_2D(N_PC, N_PV, N_SOM, t, 1, path)
    with open(path, 'rb') as f:
        noises = pickle.load(f)
    grid_no = 0
    for x_E_strength in x_E_strengths:
        for x_P_strength in x_P_strengths:
            for x_S_strength in x_S_strengths:
                for x_D_strength in x_D_strengths:
                    for stim_strength in stim_strengths:
                        for theta_strength in theta_strengths:
                            for w_DE in w_DEs:
                                for w_PE in w_PEs:
                                    for w_SE in w_SEs:
                                        for w_PS in w_PSs:
                                            theta = np.ones(N_PC) * theta_strength  # 1/s
                                            exc_params = (theta, lambda_D, lambda_E)
                                            stim_start = 1.5
                                            stimulus = np.zeros(len(t))
                                            stimulus[int(stim_start * (1 / delta_t)):] = stim_strength
                                            # background inputs
                                            x_E = x_E_strength * np.ones(N_PC)
                                            x_D = x_D_strength * np.ones(N_PC)
                                            x_P = x_P_strength * np.ones(N_PV)
                                            x_S = x_S_strength * np.ones(N_SOM)
                                            back_inputs = (x_E, x_D, x_P, x_S)

                                            w_DE11, w_DE12, w_DE21, w_DE22 = w_DE,w_DE,w_DE,w_DE
                                            w_DS11, w_DS12, w_DS21, w_DS22 = .55,.55,.55,.55
                                            w_EP11, w_EP12, w_EP21, w_EP22 = .75,.75,.75,.75
                                            w_PE11, w_PE12, w_PE21, w_PE22 = w_PE,w_PE,w_PE,w_PE
                                            w_SE11, w_SE12, w_SE21, w_SE22 = w_SE,w_SE,w_SE,w_SE
                                            w_PS11, w_PS12, w_PS21, w_PS22 = w_PS,w_PS,w_PS,w_PS
                                            w_PP11, w_PP12, w_PP21, w_PP22 = .9,.9,.9,.9

                                            weight_strengths = (
                                            w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
                                            w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

                                            path = r"params/weights_model_2D.pkl"
                                            create_save_weights_2D(N_PC, N_PV, N_SOM, weight_strengths, path)
                                            with open(path, 'rb') as f:
                                                weights = pickle.load(f)

                                            #indv_neurons = np.zeros((40,len(t)))
                                            model_2D(delta_t, vars, t, num_neurons, weights, noises, back_inputs, stimulus, taus, exc_params, flags)

                                            if ((np.abs(rS1[int((1 / delta_t) * stim_start)-2]-rS1[int((1 / delta_t) * stim_start)-302]) < 0.1) and
                                                (np.abs(rS1[int((1 / delta_t) * stim_start)-2]-rS1[int((1 / delta_t) * stim_start)-2002]) < 0.1) and
                                                (rE1[-1]-rE1[int((1 / delta_t) * stim_start)-2]>0.2) and (rE2[-1]-rE2[int((1 / delta_t) * stim_start) - 2]>0.2) and
                                                (rP1[-1]>0.5 and rP2[8888]>0.5)):
                                                show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, int((1 / delta_t) * stim_start),
                                                                       title=str(weight_strengths[:8])+"  "+str(back_inputs[0][0])+"  "+str(back_inputs[1][0])+"  "+str(back_inputs[2][0])+"  "+str(back_inputs[3][0])+" "+str(stim_strength),
                                                                       save_figure=1, name= r"C:\Users\AYCA\PycharmProjects\SynapticScaling\figs2\\" + str(grid_no) + ".png")
                                                grid_no = grid_no + 1


def grid_search_model_2D_plastic():
    sim_duration = 15 # s
    delta_t = 0.00005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, sim_timepoints)
    stim_start = 2.5

    # number of neurons
    N_PC = 2
    N_PV = 2
    N_SOM = 2
    num_neurons = (N_PC, N_PV, N_SOM)

    rE1 = np.zeros(sim_timepoints); rE2 = np.zeros(sim_timepoints)
    rP1 = np.zeros(sim_timepoints); rP2 = np.zeros(sim_timepoints)
    rS1 = np.zeros(sim_timepoints); rS2 = np.zeros(sim_timepoints)

    J_EE11 = np.zeros(sim_timepoints); J_EE12 = np.zeros(sim_timepoints)
    J_EE21 = np.zeros(sim_timepoints); J_EE22 = np.zeros(sim_timepoints)

    J_EP11 = np.zeros(sim_timepoints)
    J_EP12 = np.zeros(sim_timepoints)
    J_EP21 = np.zeros(sim_timepoints)
    J_EP22 = np.zeros(sim_timepoints)

    J_ES11 = np.zeros(sim_timepoints)
    J_ES12 = np.zeros(sim_timepoints)
    J_ES21 = np.zeros(sim_timepoints)
    J_ES22 = np.zeros(sim_timepoints)

    av_theta_E1 = np.zeros(sim_timepoints)
    av_theta_E2 = np.zeros(sim_timepoints)

    hebb11, ss_EE11, hebb21, ss_EE21, ss_EP21, ss_ES21  = np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints)
    vars = (rE1, rE2, rP1, rP2, rS1, rS2,
     J_EE11, J_EE12, J_EE21, J_EE22,
     J_EP11, J_EP12, J_EP21, J_EP22,
     J_ES11, J_ES12, J_ES21, J_ES22,
     av_theta_E1, av_theta_E2,
    hebb11,ss_EE11,hebb21,ss_EE21,ss_EP21,ss_ES21)


    tau_E = 0.06  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    tau_plas = 1
    tau_scaling = 10
    tau_theta = 5
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta)

    lambda_D = 0.27; lambda_E = 0.31
    lambdas = (lambda_D, lambda_E)

    x_E_strength = 15
    x_D_strength = 0
    x_P_strength = 7
    x_S_strength = 1
    stim_strengths = [5,10]
    theta_strengths = [.5,1,3]
    w_DEs = [.1,.2,.4]
    w_DSs = [.2,.4]
    w_EPs = [.2,.4]
    w_PEs = [.2,.4]
    w_SEs = [.1,.2]
    w_PSs = [.1,.2]
    w_PPs = [.3,.6]
    upper_bounds = [1,1.5]
    stim_stops = [stim_start+.1, stim_start+.5]

    # read the noise created before
    path = r"params/noise_model_2D.pkl"
    # create_save_noise_2D(N_PC, N_PV, N_SOM, t, 1, path)
    with open(path, 'rb') as f:
        noises = pickle.load(f)

    grid_no = 0
    for stim_strength in stim_strengths:
        for theta_strength in theta_strengths:
            for w_DE in w_DEs:
                for w_DS in w_DSs:
                    for w_EP in w_EPs:
                        for w_PE in w_PEs:
                            for w_SE in w_SEs:
                                for w_PS in w_PSs:
                                    for w_PP in w_PPs:
                                        for upper_bound in upper_bounds:
                                            for stim_stop in stim_stops:

                                                theta_default = np.ones(N_PC) * theta_strength  # 1/s

                                                # background inputs
                                                x_E = x_E_strength * np.ones(N_PC)
                                                x_D = x_D_strength * np.ones(N_PC)
                                                x_P = x_P_strength * np.ones(N_PV)
                                                x_S = x_S_strength * np.ones(N_SOM)
                                                back_inputs = (x_E, x_D, x_P, x_S)

                                                w_DE11, w_DE12, w_DE21, w_DE22 = w_DE, w_DE, w_DE, w_DE
                                                w_DS11, w_DS12, w_DS21, w_DS22 = w_DS, w_DS, w_DS, w_DS
                                                w_EP11, w_EP12, w_EP21, w_EP22 = w_EP, w_EP, w_EP, w_EP
                                                w_PE11, w_PE12, w_PE21, w_PE22 = w_PE, w_PE, w_PE, w_PE
                                                w_SE11, w_SE12, w_SE21, w_SE22 = w_SE, w_SE, w_SE, w_SE
                                                w_PS11, w_PS12, w_PS21, w_PS22 = w_PS, w_PS, w_PS, w_PS
                                                w_PP11, w_PP12, w_PP21, w_PP22 = w_PP, w_PP, w_PP, w_PP
                                                weight_strengths = (
                                                    w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
                                                    w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

                                                p_DE11, p_DE12, p_DE21, p_DE22 = 1, 1, 1, 1
                                                p_DS11, p_DS12, p_DS21, p_DS22 = 1, 1, 1, 1
                                                p_EP11, p_EP12, p_EP21, p_EP22 = 1, 1, 1, 1
                                                p_PE11, p_PE12, p_PE21, p_PE22 = 1, 1, 1, 1
                                                p_SE11, p_SE12, p_SE21, p_SE22 = 1, 1, 1, 1
                                                p_PS11, p_PS12, p_PS21, p_PS22 = 1, 1, 1, 1
                                                p_PP11, p_PP12, p_PP21, p_PP22 = 1, 1, 1, 1
                                                weight_probabilities = (
                                                    p_DE11, p_DE12, p_DS11, p_EP11, p_PE11, p_SE11, p_PS11, p_PP11, p_DS12, p_EP12, p_PE12, p_SE12, p_PS12, p_PP12,
                                                    p_DE22, p_DE21, p_DS21, p_EP21, p_PE21, p_SE21, p_PS21, p_PP21, p_DS22, p_EP22, p_PE22, p_SE22, p_PS22, p_PP22)

                                                path = r"params/weights_model_2D.pkl"
                                                weights = create_save_weights_2D(N_PC, N_PV, N_SOM, weight_strengths, weight_probabilities)

                                                E01 = np.random.rand(N_PC); E02 = np.random.rand(N_PC)
                                                P01 = np.random.rand(N_PV); P02 = np.random.rand(N_PV)
                                                S01 = np.random.rand(N_SOM); S02 = np.random.rand(N_SOM)
                                                EE110, EE120, EE210, EE220 = weights[0], weights[1], weights[15], weights[14]
                                                EP110, EP120, EP210, EP220 = weights[3], weights[9], weights[17], weights[23]
                                                ES110, ES120, ES210, ES220 = weights[2], weights[8], weights[16], weights[22]

                                                initial_values = (E01, E02, P01, P02, S01, S02,
                                                                  EE110, EE120, EE210, EE220,
                                                                  EP110, EP120, EP210, EP220,
                                                                  ES110, ES120, ES210, ES220)

                                                #indv_neurons = np.zeros((40,len(t)))
                                                hebbian_plasticity_flag = 1
                                                exc_scaling_flag = 1
                                                inh_scaling_flag = 1
                                                flags = (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag)

                                                model_2D_plasticity_scaling(delta_t, vars, initial_values, t, num_neurons, weights, noises, back_inputs,
                                                                            stim_strength, stim_start, stim_stop, taus,lambdas, theta_default, upper_bound, flags=flags)

                                                if (((J_EE12[-1] < 0.9) and (J_EE21[-1] < 0.9) and (J_EE22[-1] < 0.9)) or
                                                    ((J_EE12[-100]-J_EE12[-1] > .1) or (J_EE21[-100]-J_EE21[-1] > .1) or (J_EE22[-100]-J_EE22[-1] > .1))):
                                                    show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_start, delta_t,
                                                                 title=str(weight_strengths[:8]) + "  " + str(back_inputs[0][0]) + "  " + str(back_inputs[1][0]) + "  " +
                                                                       str(back_inputs[2][0]) + "  " + str(back_inputs[3][0]) + " " + str(stim_strength),
                                                                 save_figure=1, name=r"C:\Users\AYCA\PycharmProjects\SynapticScaling\figs2\\" + str(grid_no) + ".png")
                                                grid_no = grid_no + 1


def analyze_model_2D():
    sim_duration = 2  # s
    delta_t = 0.00005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, sim_timepoints)

    rE1 = np.zeros(sim_timepoints, dtype=np.float64)
    rE2 = np.zeros(sim_timepoints, dtype=np.float64)
    rP1 = np.zeros(sim_timepoints, dtype=np.float64)
    rP2 = np.zeros(sim_timepoints, dtype=np.float64)
    rS1 = np.zeros(sim_timepoints, dtype=np.float64)
    rS2 = np.zeros(sim_timepoints, dtype=np.float64)
    vars = (rE1, rE2, rP1, rP2, rS1, rS2)


    # number of neurons
    N_PC = 80
    N_PV = 11
    N_SOM = 9
    num_neurons = (N_PC, N_PV, N_SOM)

    stim_start = 1.5
    stim_strength = 1
    stimulus = np.zeros(len(t))
    stimulus[int(stim_start * (1 / delta_t)):] = stim_strength

    tau_E = 0.02  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    taus = (tau_E, tau_P, tau_S)

    theta = np.ones(N_PC) * 3  # 1/s
    lambda_D = 0.27
    lambda_E = 0.31
    exc_params = (theta, lambda_D, lambda_E)

    # background inputs
    x_E = 15* np.ones(N_PC)
    x_D = 10 * np.ones(N_PC)
    x_P = 1 * np.ones(N_PV)
    x_S = 1* np.ones(N_SOM)
    back_inputs = (x_E, x_D, x_P, x_S)

    w_DE11, w_DE12, w_DE21, w_DE22 = 2,2,2,2
    w_DS11, w_DS12, w_DS21, w_DS22 = .55,.55,.55,.55
    w_EP11, w_EP12, w_EP21, w_EP22 = .75,.75,.75,.75
    w_PE11, w_PE12, w_PE21, w_PE22 = .9,.9,.9,.9
    w_SE11, w_SE12, w_SE21, w_SE22 = .1,.1,.1,.1
    w_PS11, w_PS12, w_PS21, w_PS22 = .3,.3,.3,.3
    w_PP11, w_PP12, w_PP21, w_PP22 = .9,.9,.9,.9

    weight_strengths = (
    w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
    w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

    # read the noise created before
    path = r"params/noise_model_2D.pkl"
    #create_save_noise_2D(N_PC, N_PV, N_SOM, t, 0, path)
    with open(path, 'rb') as f:
        noises = pickle.load(f)

    path = r"params/weights_model_2D.pkl"
    create_save_weights_2D(N_PC, N_PV, N_SOM, weight_strengths, path)
    with open(path, 'rb') as f:
        weights = pickle.load(f)

    flag_save_rates = 0
    #indv_neurons = np.zeros((40,len(t)))
    model_2D(delta_t, vars, t, num_neurons, weights, noises, back_inputs, stimulus, stim_start,  taus, exc_params, flag_save_rates=flag_save_rates)
    show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_start, delta_t,
                 title=str(weight_strengths[:8])+"  "+str(back_inputs[0][0])+"  "+str(back_inputs[1][0])+"  "+str(back_inputs[2][0])+"  "+str(back_inputs[3][0])+" "+str(stim_strength))
    #show_plot_indv_neurons(indv_neurons,t,int((1 / delta_t) * stim_start))


def analyze_model_2D_plasticity_scaling():
    sim_duration = 10# s
    delta_t = 0.00005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, sim_timepoints)
    stim_start = 3
    stim_stop = 3.5
    stim_strength = 1
    upper_bound = 1.5

    # number of neurons
    N_PC = 80
    N_PV = 11
    N_SOM = 9
    num_neurons = (N_PC, N_PV, N_SOM)

    rE1 = np.zeros(sim_timepoints, dtype=np.float32)
    rE2 = np.zeros(sim_timepoints, dtype=np.float32)
    rP1 = np.zeros(sim_timepoints, dtype=np.float32)
    rP2 = np.zeros(sim_timepoints, dtype=np.float32)
    rS1 = np.zeros(sim_timepoints, dtype=np.float32)
    rS2 = np.zeros(sim_timepoints, dtype=np.float32)

    J_EE11 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EE12 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EE21 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EE22 = np.zeros(sim_timepoints, dtype=np.float32)

    J_EP11 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EP12 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EP21 = np.zeros(sim_timepoints, dtype=np.float32)
    J_EP22 = np.zeros(sim_timepoints, dtype=np.float32)

    J_ES11 = np.zeros(sim_timepoints, dtype=np.float32)
    J_ES12 = np.zeros(sim_timepoints, dtype=np.float32)
    J_ES21 = np.zeros(sim_timepoints, dtype=np.float32)
    J_ES22 = np.zeros(sim_timepoints, dtype=np.float32)

    av_theta_E1 = np.zeros(sim_timepoints, dtype=np.float32)
    av_theta_E2 = np.zeros(sim_timepoints, dtype=np.float32)

    hebb11, ss_EE11, hebb21, ss_EE21, ss_EP21, ss_ES21  = np.zeros(sim_timepoints, dtype=np.float32),np.zeros(sim_timepoints, dtype=np.float32),np.zeros(sim_timepoints, dtype=np.float32),np.zeros(sim_timepoints, dtype=np.float32),np.zeros(sim_timepoints, dtype=np.float32),np.zeros(sim_timepoints, dtype=np.float32),
    vars = (rE1, rE2, rP1, rP2, rS1, rS2,
     J_EE11, J_EE12, J_EE21, J_EE22,
     J_EP11, J_EP12, J_EP21, J_EP22,
     J_ES11, J_ES12, J_ES21, J_ES22,
     av_theta_E1, av_theta_E2,
    hebb11,ss_EE11,hebb21,ss_EE21,ss_EP21,ss_ES21)


    tau_E = 0.06  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    tau_plas = 1
    tau_scaling = 10
    tau_theta = 5
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta)

    lambda_D = 0.27; lambda_E = 0.31
    lambdas = (lambda_D, lambda_E)

    # background inputs
    x_E = 15* np.ones(N_PC)
    x_D = 10 * np.ones(N_PC)
    x_P = 1 * np.ones(N_PV)
    x_S = 1* np.ones(N_SOM)
    back_inputs = (x_E, x_D, x_P, x_S)

    w_DE11, w_DE12, w_DE21, w_DE22 = 1,1,1,1
    w_DS11, w_DS12, w_DS21, w_DS22 = .55,.55,.55,.55
    w_EP11, w_EP12, w_EP21, w_EP22 = .75,.75,.75,.75
    w_PE11, w_PE12, w_PE21, w_PE22 = .9,.9,.9,.9
    w_SE11, w_SE12, w_SE21, w_SE22 = .1,.1,.1,.1
    w_PS11, w_PS12, w_PS21, w_PS22 = .3,.3,.3,.3
    w_PP11, w_PP12, w_PP21, w_PP22 = .9,.9,.9,.9
    weight_strengths = (
        w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
        w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

    p_DE11, p_DE12, p_DE21, p_DE22 = .1,.1,.1,.1
    p_DS11, p_DS12, p_DS21, p_DS22 = .55,.55,.55,.55
    p_EP11, p_EP12, p_EP21, p_EP22 = .6,.6,.6,.6
    p_PE11, p_PE12, p_PE21, p_PE22 = .45,.45,.45,.45
    p_SE11, p_SE12, p_SE21, p_SE22 = .35,.35,.35,.35
    p_PS11, p_PS12, p_PS21, p_PS22 = .6,.6,.6,.6
    p_PP11, p_PP12, p_PP21, p_PP22 = .5,.5,.5,.5
    weight_probabilities = (
    p_DE11, p_DE12, p_DS11, p_EP11, p_PE11, p_SE11, p_PS11, p_PP11, p_DS12, p_EP12, p_PE12, p_SE12, p_PS12, p_PP12,
    p_DE22, p_DE21, p_DS21, p_EP21, p_PE21, p_SE21, p_PS21, p_PP21, p_DS22, p_EP22, p_PE22, p_SE22, p_PS22, p_PP22)

    path = r"params/weights_model_2D.pkl"
    create_save_weights_2D(N_PC, N_PV, N_SOM, weight_strengths, weight_probabilities,path)
    with open(path, 'rb') as f:
        weights = pickle.load(f)

    E01 = np.float32(np.random.rand(N_PC)); E02 = np.float32(np.random.rand(N_PC))
    P01 = np.float32(np.random.rand(N_PV)); P02 = np.float32(np.random.rand(N_PV))
    S01 = np.float32(np.random.rand(N_SOM)); S02 = np.float32(np.random.rand(N_SOM))
    EE110, EE120, EE210, EE220 = weights[0],weights[1],weights[15],weights[14]
    EP110, EP120, EP210, EP220 = weights[3],weights[9],weights[17],weights[23]
    ES110, ES120, ES210, ES220 = weights[2],weights[8],weights[16],weights[22]

    initial_values = (E01, E02, P01, P02, S01, S02,
                      EE110, EE120, EE210, EE220,
                      EP110, EP120, EP210, EP220,
                      ES110, ES120, ES210, ES220)

    # read the noise created before
    path = r"params/noise_model_2D.pkl"
    #create_save_noise_2D(N_PC, N_PV, N_SOM, t, 0, path)
    with open(path, 'rb') as f:
        noises = pickle.load(f)

    theta_default = np.float32(np.ones(N_PC) * 3)  # 1/s

    hebbian_plasticity_flag = 1
    exc_scaling_flag = 1
    inh_scaling_flag = 1
    flags = (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag)

    model_2D_plasticity_scaling(delta_t, vars, initial_values,t, num_neurons, weights, noises, back_inputs,
                                stim_strength, stim_start, stim_stop, taus, lambdas, theta_default, upper_bound, flags=flags)
    show_plot_plasticity_terms(hebb11, ss_EE11, hebb21, ss_EE21, ss_EP21, ss_ES21, t, stim_start)
    show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_start, delta_t,
                 title=str(weight_strengths[:8])+"  "+str(back_inputs[0][0])+"  "+str(back_inputs[1][0])+"  "+
                       str(back_inputs[2][0])+"  "+str(back_inputs[3][0])+" "+str(stim_strength))
    show_plot_weights(J_EE11,J_EE12,J_EE21,J_EE22,t,stim_start)
    #indv_neurons = np.zeros((40,len(t)))
    #show_plot_indv_neurons(indv_neurons,t,int((1 / delta_t) * stim_start))


def analyze_model_2D_mass_plasticity_scaling():
    sim_duration = 10 # s
    delta_t = 0.00005  # s
    sim_timepoints = int(sim_duration * (1 / delta_t))
    t = np.linspace(0, sim_duration, sim_timepoints)
    stim_start = 2
    stim_stop = 2.5
    stim_strength = 1
    upper_bound = 2.2

    # number of neurons
    N_PC = 1
    N_PV = 1
    N_SOM = 1
    num_neurons = (N_PC, N_PV, N_SOM)

    rE1 = np.zeros(sim_timepoints); rE2 = np.zeros(sim_timepoints)
    rP1 = np.zeros(sim_timepoints); rP2 = np.zeros(sim_timepoints)
    rS1 = np.zeros(sim_timepoints); rS2 = np.zeros(sim_timepoints)

    J_EE11 = np.zeros(sim_timepoints); J_EE12 = np.zeros(sim_timepoints)
    J_EE21 = np.zeros(sim_timepoints); J_EE22 = np.zeros(sim_timepoints)

    J_EP11 = np.zeros(sim_timepoints)
    J_EP12 = np.zeros(sim_timepoints)
    J_EP21 = np.zeros(sim_timepoints)
    J_EP22 = np.zeros(sim_timepoints)

    J_ES11 = np.zeros(sim_timepoints)
    J_ES12 = np.zeros(sim_timepoints)
    J_ES21 = np.zeros(sim_timepoints)
    J_ES22 = np.zeros(sim_timepoints)

    av_theta_E1 = np.zeros(sim_timepoints)
    av_theta_E2 = np.zeros(sim_timepoints)

    hebb11, ss_EE11, hebb21, ss_EE21, ss_EP21, ss_ES21  = np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints),np.zeros(sim_timepoints)
    vars = (rE1, rE2, rP1, rP2, rS1, rS2,
     J_EE11, J_EE12, J_EE21, J_EE22,
     J_EP11, J_EP12, J_EP21, J_EP22,
     J_ES11, J_ES12, J_ES21, J_ES22,
     av_theta_E1, av_theta_E2,
    hebb11,ss_EE11,hebb21,ss_EE21,ss_EP21,ss_ES21)


    tau_E = 0.02  # s
    tau_P = 0.005  # s
    tau_S = 0.01  # s
    tau_plas = 1
    tau_scaling = 10
    tau_theta = 8
    taus = (tau_E, tau_P, tau_S, tau_plas, tau_scaling, tau_theta)

    lambda_D = 0.27; lambda_E = 0.31
    lambdas = (lambda_D, lambda_E)

    rheobase_E, rheobase_P, rheobase_S = 5,1,1
    rheobases = (rheobase_E, rheobase_P, rheobase_S)
    # background inputs
    x_E = 15 * np.ones(N_PC)
    x_D = 10 * np.ones(N_PC)
    x_P = 1 * np.ones(N_PV)
    x_S = 1 * np.ones(N_SOM)
    back_inputs = (x_E, x_D, x_P, x_S)

    w_DE11, w_DE12, w_DE21, w_DE22 = 1.75,1.5,1.5,1.75 # fixed
    w_DS11, w_DS12, w_DS21, w_DS22 = .6,.3,.3,.6
    w_EP11, w_EP12, w_EP21, w_EP22 = .4,.2,.2,.4
    w_PE11, w_PE12, w_PE21, w_PE22 = .8,.6,.6,.8
    w_SE11, w_SE12, w_SE21, w_SE22 = .3,.15,.15,.3
    w_PS11, w_PS12, w_PS21, w_PS22 = .4,.2,.2,.4
    w_PP11, w_PP12, w_PP21, w_PP22 = .3,.15,.15,.3
    weight_strengths = (
        w_DE11, w_DE12, w_DS11, w_EP11, w_PE11, w_SE11, w_PS11, w_PP11, w_DS12, w_EP12, w_PE12, w_SE12, w_PS12, w_PP12,
        w_DE22, w_DE21, w_DS21, w_EP21, w_PE21, w_SE21, w_PS21, w_PP21, w_DS22, w_EP22, w_PE22, w_SE22, w_PS22, w_PP22)

    p_DE11, p_DE12, p_DE21, p_DE22 = 1,1,1,1
    p_DS11, p_DS12, p_DS21, p_DS22 = 1,1,1,1
    p_EP11, p_EP12, p_EP21, p_EP22 = 1,1,1,1
    p_PE11, p_PE12, p_PE21, p_PE22 = 1,1,1,1
    p_SE11, p_SE12, p_SE21, p_SE22 = 1,1,1,1
    p_PS11, p_PS12, p_PS21, p_PS22 = 1,1,1,1
    p_PP11, p_PP12, p_PP21, p_PP22 = 1,1,1,1
    weight_probabilities = (
        p_DE11, p_DE12, p_DS11, p_EP11, p_PE11, p_SE11, p_PS11, p_PP11, p_DS12, p_EP12, p_PE12, p_SE12, p_PS12, p_PP12,
        p_DE22, p_DE21, p_DS21, p_EP21, p_PE21, p_SE21, p_PS21, p_PP21, p_DS22, p_EP22, p_PE22, p_SE22, p_PS22, p_PP22)

    path = r"params/weights_model_2D.pkl"
    weights = create_save_weights_2D(N_PC, N_PV, N_SOM, weight_strengths, weight_probabilities)


    E01 = np.random.rand(N_PC); E02 = np.random.rand(N_PC)
    P01 = np.random.rand(N_PV); P02 = np.random.rand(N_PV)
    S01 = np.random.rand(N_SOM); S02 = np.random.rand(N_SOM)
    EE110, EE120, EE210, EE220 = weights[0],weights[1],weights[15],weights[14]
    EP110, EP120, EP210, EP220 = weights[3],weights[9],weights[17],weights[23]
    ES110, ES120, ES210, ES220 = weights[2],weights[8],weights[16],weights[22]

    initial_values = (E01, E02, P01, P02, S01, S02,
                      EE110, EE120, EE210, EE220,
                      EP110, EP120, EP210, EP220,
                      ES110, ES120, ES210, ES220)

    BCM_p = 1

    hebbian_plasticity_flag = 1
    exc_scaling_flag = 1
    inh_scaling_flag = 1
    BCM_flag = 1
    flags = (hebbian_plasticity_flag, exc_scaling_flag, inh_scaling_flag, BCM_flag)

    model_2D_plasticity_scaling(delta_t, vars, initial_values,t, num_neurons, weights, back_inputs,
                                stim_strength, stim_start, stim_stop, taus, lambdas, rheobases, upper_bound,
                                BCM_p, flags=flags)
    show_plot_plasticity_terms(hebb11, ss_EE11, hebb21, ss_EE21, ss_EP21, ss_ES21, t, stim_start)
    show_plot_2D(rE1, rE2, rP1, rP2, rS1, rS2, t, stim_start, stim_stop, delta_t,
                 title=str(weight_strengths[:8])+"  "+str(back_inputs[0][0])+"  "+str(back_inputs[1][0])+"  "+
                       str(back_inputs[2][0])+"  "+str(back_inputs[3][0])+" "+str(stim_strength))
    show_plot_weights(J_EE11,J_EE12,J_EE21,J_EE22,t,stim_start)
    #indv_neurons = np.zeros((40,len(t)))
    #show_plot_indv_neurons(indv_neurons,t,int((1 / delta_t) * stim_start))

#analyze_model_2D()
#analyze_moszdel_2D_plasticity_scaling()
analyze_model_2D_mass_plasticity_scaling()
#grid_search_model_2D()
#grid_search_model_2D_plastic()

"""
E>PV>SST
    w_DE11, w_DE12, w_DE21, w_DE22 = 1.75,1.5,1.5,1.75 # fixed
    w_DS11, w_DS12, w_DS21, w_DS22 = .6,.3,.3,.6
    w_EP11, w_EP12, w_EP21, w_EP22 = .4,.2,.2,.4
    w_PE11, w_PE12, w_PE21, w_PE22 = .8,.6,.6,.8
    w_SE11, w_SE12, w_SE21, w_SE22 = .3,.15,.15,.3
    w_PS11, w_PS12, w_PS21, w_PS22 = .4,.2,.2,.4
    w_PP11, w_PP12, w_PP21, w_PP22 = .3,.15,.15,.3

"""