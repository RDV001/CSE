# IMPORTS

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# REWARD FUNCTIONS

def reward_func_1(observation, action):
    return (1 - np.abs(observation[0]))*(np.abs(observation[1]))*(1 - np.abs(observation[2]))*(1 - np.abs(observation[3]))*(1 - np.abs(action[1]))

def reward_func_2(observation, action):
    return (1 - np.abs(observation[0]))*(np.abs(observation[1]))*(1 - np.abs(observation[2]))*(1 - np.abs(observation[3]))

def reward_func_3(observation, action):
    return (1 - np.abs(observation[0]))*(np.abs(observation[1]))

# UTILS

def evaluate(X, U, dynamics):

    reward_1 = []
    reward_2 = []
    reward_3 = []

    for i in range(len(U)):
        state = dynamics.normalize_state(X[i])

        action = U[i]
        if(dynamics.torque_limit[0] > 0): action[0] = action[0]/dynamics.torque_limit[0]
        else: action[0] = 0
        if(dynamics.torque_limit[1] > 0): action[1] = action[1]/dynamics.torque_limit[1]
        else: action[1] = 0

        reward_1.append(reward_func_1(state, action))
        reward_2.append(reward_func_2(state, action))
        reward_3.append(reward_func_3(state, action))

    s_1 = np.sum(reward_1)
    s_2 = np.sum(reward_2)
    s_3 = np.sum(reward_3)

    print(str(s_1) + '\t' + str(s_2) + '\t' + str(s_3))

    return (s_1, s_2, s_3)

def save(X, T, dynamics, path, scale=1.0):

    SMALL_SIZE = 16 * scale
    MEDIUM_SIZE = 20 * scale
    BIGGER_SIZE = 24 * scale

    mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
    mpl.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    mpl.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.subplots(
        1, 1, figsize=(9 * scale, 1 * 6 * scale), sharex="all"
    )

    '''
    plt.subplots(
        1, 1, figsize=(18 * scale, 1 * 3 * scale), sharex="all"
    )
    '''
    
    reward_1 = []
    reward_2 = []
    reward_3 = []

    for x in X:
        obs = dynamics.normalize_state(x)
        reward_1.append(reward_func_1(obs, None))
        reward_2.append(reward_func_2(obs, None))
        reward_3.append(reward_func_3(obs, None))

    plt.plot(T, reward_1)
    plt.plot(T, reward_2)
    plt.plot(T, reward_3)

    plt.ylabel('rewards')
    plt.xlabel('time [s]')
    plt.legend(['1st reward function', '2nd reward function', '3rd reward function'])

    plt.savefig(path)

    