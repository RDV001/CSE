# IMPORT

import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters

from double_pendulum.simulation.simulation import Simulator
import double_pendulum.simulation.gym_env as gym_env

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController

from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries

import reward_functions as rf

# SAVING DIRECTORY

saving_directory = os.path.join('data', 'ilqr')
os.makedirs(saving_directory, exist_ok=True)
os.makedirs(os.path.join(saving_directory, 'controller'), exist_ok=True)
os.makedirs(os.path.join(saving_directory, 'trajectory'), exist_ok=True)
os.makedirs(os.path.join(saving_directory, 'timeseries'), exist_ok=True)

# MODEL PARAMETERS

parameters_path = 'acrobot_parameters.yml'
model_parameters = model_parameters(filepath=parameters_path)

# SIMULATION PARAMETERS

t0 = 0.0
dt = 0.005
t_final = 5.0
integrator = 'runge_kutta'
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

# CONTROLLER

controller = ILQRMPCCPPController(model_pars=model_parameters)

controller.set_start(start_position)
controller.set_goal(goal_position)

controller.set_parameters(
    N=100,
    dt=dt,
    max_iter=100, 
    regu_init=1.0,
    max_regu=10000.0,
    min_regu=1e-6,
    break_cost_redu=1e-6,
    integrator=integrator,
    trajectory_stabilization=True,
    shifting=1
)
controller.set_cost_parameters(
    sCu=[0.0, 0.0],
    sCp=[0.1, 0.1],
    sCv=[0.01, 0.1],
    sCen=0.0,
    fCp=[100.0, 10.0],
    fCv=[10.0, 10.0],
    fCen=0.0
)
controller.set_final_cost_parameters(
    sCu=[0.0, 0.0],
    sCp=[0.1, 0.1],
    sCv=[0.01, 0.1],
    sCen=0.0,
    fCp=[10.0, 10.0],
    fCv=[10.0, 10.0],
    fCen=0.0
)

controller.compute_init_traj(
    N=100,
    dt=dt,
    max_iter=100, 
    regu_init=1.0,
    max_regu=10000.0,
    min_regu=1e-6,
    break_cost_redu=1e-6, 
    sCu=[0.0, 0.0],
    sCp=[0.1, 0.1],
    sCv=[0.01, 0.1],
    sCen=0.0,
    fCp=[100.0, 10.0],
    fCv=[10.0, 10.0],
    fCen=0.0,
    integrator=integrator
)

controller.init()

controller.save(os.path.join(saving_directory, 'controller'))

# EVALUATION

T, X, U = simulator.simulate(
    t0=t0,
    x0=start_position,
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator
)

save_trajectory(os.path.join(saving_directory, 'trajectory', 'trajectory.csv'), T, X, U)

plot_timeseries(
    T, X, U, None,
    plot_energy=False,
    pos_y_lines=[0.0, np.pi],
    tau_y_lines=[-model_parameters.tl[1], model_parameters.tl[1]],
    save_to=os.path.join(saving_directory, 'timeseries', 'timeseries.pdf')
)

rf.evaluate(X, U, acrobot_dynamics)






