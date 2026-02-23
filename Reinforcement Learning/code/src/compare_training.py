# IMPORT

import os
import csv
import numpy as np

import matplotlib.pyplot as plt

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum

from double_pendulum.simulation.simulation import Simulator
import double_pendulum.simulation.gym_env as gym_env

import reward_functions as rf

# SAVING DIRECTORY

ddpg_1_directory = os.path.join('data', 'ddpg_1', 'training')
ddpg_2_directory = os.path.join('data', 'ddpg_2', 'training')
ddpg_3_directory = os.path.join('data', 'ddpg_3', 'training')

ilqr_directory = os.path.join('data', 'ilqr', 'trajectory')

saving_directory = os.path.join('data', 'ddpg', 'training')
os.makedirs(saving_directory, exist_ok=True)

# LOAD

data_ddpg_1 = np.load(os.path.join(ddpg_1_directory, 'evaluations.npz'))
data_ddpg_2 = np.load(os.path.join(ddpg_2_directory, 'evaluations.npz'))
data_ddpg_3 = np.load(os.path.join(ddpg_3_directory, 'evaluations.npz'))

X_ilqr = []
U_ilqr = []
with open(os.path.join(ilqr_directory, 'trajectory.csv'), mode ='r') as file:
    reader = csv.reader(file)
    for line in reader:
        try:
            buffer = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
            X_ilqr.append(buffer)
            buffer = [float(line[5]), float(line[6])]
            U_ilqr.append(buffer)
        except:
            continue

# ILQR VALUES

parameters_path = 'acrobot_parameters.yml'
model_parameters = model_parameters(filepath=parameters_path)

plant = SymbolicDoublePendulum(model_pars=model_parameters)
simulator = Simulator(plant=plant)
dt = 0.005

acrobot_dynamics = gym_env.double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    robot='acrobot',
    torque_limit=plant.torque_limit,
    scaling=True,
    max_velocity=20.0
)

rf_ilqr = rf.evaluate(X_ilqr, U_ilqr, acrobot_dynamics)

data_ilqr = np.load(os.path.join(ddpg_3_directory, 'evaluations.npz'))

T =  data_ddpg_1['timesteps']
X_1 = data_ddpg_1['results']
X_2 = data_ddpg_2['results']
X_3 = data_ddpg_3['results']

# PRINT

plt.figure()

plt.plot(T, rf_ilqr[0]*np.ones(len(T)), color='tab:blue', linewidth=1, label='_nolegend_', linestyle='dashed')
plt.plot(T, rf_ilqr[1]*np.ones(len(T)), color='tab:orange', linewidth=1, label='_nolegend_', linestyle='dashed')
plt.plot(T, rf_ilqr[2]*np.ones(len(T)), color='tab:green', linewidth=1, label='_nolegend_', linestyle='dashed')
plt.plot(T, X_1, color='tab:blue', linewidth=1)
plt.plot(T, X_2, color='tab:orange', linewidth=1)
plt.plot(T, X_3, color='tab:green', linewidth=1)

plt.ylabel('evaluations')
plt.xlabel('timesteps')
plt.legend(['1st reward function', '2nd reward function', '3rd reward function'])

plt.savefig(os.path.join(saving_directory, 'training.pdf'))

