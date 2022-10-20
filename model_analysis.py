import numpy as np
from numba import cuda, jit
import matplotlib.pyplot as plt
from util import *
from model import *

sim_duration = 4   # s
delta_t = 0.00005  # s
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
path = r"./params/weights.pkl"
create_save_weights(N_PC,N_PV,N_SOM,weight_strengths,path)
with open(path, 'rb') as f:
    weights = pickle.load(f)

# read the noise created before
path = r"./params/noise.pkl"
with open(path, 'rb') as f:
    noises = pickle.load(f)


euler_loop(delta_t,vars,t,num_neurons,weights,noises,back_inputs,stimulus,taus,exc_params,flags)
show_plot_2PC_old(rE1,rE2, rP, rS, t, int((1/delta_t)*stim_start))
#show_plot_2PC_stim(rE1_sol,rE2_sol, rP_sol, rS_sol, t, int((1/delta_t)*stim_start))