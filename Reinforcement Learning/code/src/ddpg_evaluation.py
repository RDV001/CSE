# IMPORT

import os
import numpy as np

from stable_baselines3 import DDPG

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters

from double_pendulum.simulation.simulation import Simulator
import double_pendulum.simulation.gym_env as gym_env

from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries

import reward_functions as rf 

# SAVING DIRECTORY

saving_directory = os.path.join('data', 'ddpg')
saving_directory_1 = os.path.join('data', 'ddpg_1')
saving_directory_2 = os.path.join('data', 'ddpg_2')
saving_directory_3 = os.path.join('data', 'ddpg_3')

os.makedirs(saving_directory_1, exist_ok=True)
os.makedirs(os.path.join(saving_directory_1, 'trajectory'), exist_ok=True)
os.makedirs(os.path.join(saving_directory_1, 'timeseries'), exist_ok=True)

os.makedirs(saving_directory_2, exist_ok=True)
os.makedirs(os.path.join(saving_directory_2, 'trajectory'), exist_ok=True)
os.makedirs(os.path.join(saving_directory_2, 'timeseries'), exist_ok=True)

os.makedirs(saving_directory_3, exist_ok=True)
os.makedirs(os.path.join(saving_directory_3, 'trajectory'), exist_ok=True)
os.makedirs(os.path.join(saving_directory_3, 'timeseries'), exist_ok=True)

# MODEL PARAMETERS

parameters_path = "acrobot_parameters.yml"
model_parameters = model_parameters(filepath=parameters_path)

# SIMULATION PARAMETERS

t0 = 0.0
dt = 0.005
t_final = 5.0
integrator = "runge_kutta"
start_position = [0.0, 0.0, 0.0, 0.0]
goal_position = [np.pi, 0.0, 0.0, 0.0]

# PLANT

plant = SymbolicDoublePendulum(model_pars=model_parameters)
simulator = Simulator(plant=plant)
acrobot_dynamics = gym_env.double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    robot='acrobot',
    torque_limit=plant.torque_limit,
    scaling=True,
    max_velocity=20.0
)

# SIMULATION

ddpg_1 = DDPG.load(os.path.join(saving_directory_1, 'controller', 'best_model'))
ddpg_2 = DDPG.load(os.path.join(saving_directory_2, 'controller', 'best_model'))
ddpg_3 = DDPG.load(os.path.join(saving_directory_3, 'controller', 'best_model'))

# EVALUATION ENVIRONMENT

def terminated_func(observation):
    return False

def reset_func():
    return [-1.0, -1.0, 0.0, 0.0]

eval_env_1 = gym_env.CustomEnv(
    dynamics_func=acrobot_dynamics,
    reward_func=rf.reward_func_1,
    terminated_func=terminated_func,
    reset_func=reset_func,
    max_episode_steps=1000,
)
eval_env_2 = gym_env.CustomEnv(
    dynamics_func=acrobot_dynamics,
    reward_func=rf.reward_func_2,
    terminated_func=terminated_func,
    reset_func=reset_func,
    max_episode_steps=1000,
)
eval_env_3 = gym_env.CustomEnv(
    dynamics_func=acrobot_dynamics,
    reward_func=rf.reward_func_3,
    terminated_func=terminated_func,
    reset_func=reset_func,
    max_episode_steps=1000,
)

models_envs_id = [
    (ddpg_1, eval_env_1, '1'), 
    (ddpg_2, eval_env_2, '2'), 
    (ddpg_3, eval_env_3, '3')
]

# EVALUATION

for (model, env, id) in models_envs_id:

    state, _ = env.reset()

    rewards_list = []
    X = []
    U = []

    X.append(acrobot_dynamics.unscale_state(state))
    U.append(acrobot_dynamics.unscale_action([0.0, 0.0]))
    for timestep in range(1000):
        action, _ = model.predict(state)
        state, reward, _, _, _ = env.step(action)
        rewards_list.append(reward)
        X.append(acrobot_dynamics.unscale_state(state))
        U.append(acrobot_dynamics.unscale_action(action))
    X.pop()
    U.pop()

    T = np.arange(t0, t_final, dt)

    for x in X:
        if x[1] > np.pi: x[1] = x[1] - 2*np.pi

    save_trajectory(os.path.join(saving_directory + '_' + str(id), 'trajectory', 'trajectory.csv'), T, X, U)
    plot_timeseries(
        T, X, U, None,
        plot_energy=False,
        pos_y_lines=[0.0, np.pi],
        tau_y_lines=[-model_parameters.tl[1], model_parameters.tl[1]],
        save_to=os.path.join(saving_directory + '_' + str(id), 'timeseries', 'timeseries.pdf')
    )






