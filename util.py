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
    plt.plot(t[:], rE[:], 'r', label='PC, saturates at '+ str(np.round(np.mean(rE[-80:]),3)) + 'Hz')
    plt.plot(t[:], rP[:], 'g', label='PV, saturates at '+ str(np.round(np.mean(rP[-80:]),3)) + 'Hz')
    plt.plot(t[:], rS[:], 'b', label='SST, saturates at '+ str(np.round(np.mean(rS[-80:]),3)) + 'Hz')
    plt.legend(loc='best')
    plt.xlabel('time [s]')
    plt.ylabel('Firing rates [Hz]')
    plt.grid()
    plt.show()