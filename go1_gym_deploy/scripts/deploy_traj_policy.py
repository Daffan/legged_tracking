import glob
import pickle as pkl
import lcm
import sys
import torch
import traceback
sys.path.append("../..")

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_traj_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import RandomTrajectoryProfile, DummyFrontGoalProfile
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name):
    # load agent
    logdir = f"../../runs/{label}"

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg
        print(cfg.keys())


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = DummyFrontGoalProfile(dt=control_dt, state_estimator=se, command_xy_only=cfg["env"]["command_xy_only"])

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()
    print("Initialized the agent")

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)
    print("Initialized the policy")

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)
    print("Initialized the deployment runner")

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    label = "trajectory_tracking/run-20230904_112307-rhi1my71"

    experiment_name = "example_experiment"
    try: 
        load_and_run_policy(label, experiment_name=experiment_name)
    except Exception as e:
        print("########################## Error ##########################")
        print(traceback.format_exc())
        print("########################## Error ##########################")
