# IMPORT

import numpy as np
import os

# import gymnasium as gym
from stable_baselines3 import DDPG

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters

from double_pendulum.simulation.simulation import Simulator
import double_pendulum.simulation.gym_env as gym_env

# Choose one among the three - the second is the best one
# from reward_functions import reward_func_1 as rf
from reward_functions import reward_func_2 as rf
# from reward_functions import reward_func_3 as rf

# SAVING DIRECTORY

# Choose one among the three - the second is the best one
# saving_directory = os.path.join('data', 'ddpg_1') 
saving_directory = os.path.join('data', 'ddpg_2')
# saving_directory = os.path.join('data', 'ddpg_3')

os.makedirs(os.path.join(saving_directory, 'controller'), exist_ok=True)
os.makedirs(os.path.join(saving_directory, 'training'), exist_ok=True)

# MODEL PAAMETERS

parameters_path = 'acrobot_parameters.yml'
model_parameters = model_parameters(filepath=parameters_path)

# SIMULATION PARAMETERS

dt = 0.005
integrator = 'runge_kutta'
robot = 'acrobot'
max_velocity = 20.0

# PLANT

plant = SymbolicDoublePendulum(model_pars=model_parameters)
simulator = Simulator(plant=plant)

# ENVIRONMENT PARAMETERS

max_steps = 1000
termination = False

n_envs = 100
episode_number = 20000
training_steps = max_steps*episode_number
verbose = 2

reward_threshold = 1000.0
eval_freq=2000
n_eval_episodes=1

acrobot_dynamics = gym_env.double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    torque_limit=plant.torque_limit,
    scaling=True,
    max_velocity=max_velocity
)

def terminated_func(observation):
    return False

def reset_func():
    return [-1.0, -1.0, 0.0, 0.0]

env = gym_env.CustomEnv(
    dynamics_func=acrobot_dynamics,
    reward_func=rf,
    terminated_func=terminated_func,
    reset_func=reset_func,
    max_episode_steps=max_steps,
)

# training env
envs = make_vec_env(
    env_id=gym_env.CustomEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": acrobot_dynamics,
        "reward_func": rf,
        "terminated_func": terminated_func,
        "reset_func": reset_func,
        "max_episode_steps": max_steps,
    },
)

# evaluation env
eval_env = gym_env.CustomEnv(
    dynamics_func=acrobot_dynamics,
    reward_func=rf,
    terminated_func=terminated_func,
    reset_func=reset_func,
    max_episode_steps=max_steps,
)

# training callbacks
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(saving_directory, 'controller'),
    log_path=os.path.join(saving_directory, 'training'),
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# AGENT

policy = 'MlpPolicy'

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

model = DDPG(
    policy=policy,
    env=envs,
    verbose=verbose,
    action_noise=action_noise,
    learning_starts=max_steps*100,
    learning_rate = 0.001,
    seed=0,
    train_freq=1
)

# TRAINING OF THE AGENT

model.learn(total_timesteps=training_steps, log_interval=1, callback=eval_callback)

