import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.trajectory_tracking import TrajectoryTrackingEnv

from tqdm import tqdm

LOAD_PATH = "wandb/run-20230903_221111-2tbiy4ay/files"

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(logdir, headless=False):

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg  # ["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    """ # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False """

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = args.num_env * args.n_repeat
    Cfg.env.look_from_back = True

    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    # Cfg.control.control_type = "P"

    if False:
        Cfg.terrain.mesh_type = 'plane'
    else:
        # By default random pyramid terrain
        Cfg.terrain.num_cols = int(args.num_env ** 0.5)
        Cfg.terrain.num_rows = int(args.num_env ** 0.5)
        Cfg.terrain.terrain_length = 5.0
        Cfg.terrain.terrain_width = 1.6
        Cfg.terrain.terrain_ratio_x = 0.5
        Cfg.terrain.terrain_ratio_y = 1.0
        Cfg.terrain.pyramid_num_x=3
        Cfg.terrain.pyramid_num_y=4
        Cfg.terrain.pyramid_var_x=0.3
        Cfg.terrain.pyramid_var_y=0.3

    if args.hybrid:
        Cfg.commands.sampling_based_planning = True
        Cfg.commands.plan_interval = 100
        Cfg.commands.switch_dist = 0.2

    Cfg.commands.traj_function = "fixed_target"
    Cfg.commands.base_x = 3.5
    Cfg.commands.base_y = 0.0
    Cfg.curriculum_thresholds.cl_fix_target = False
    # Cfg.env.rotate_camera = False
    # Cfg.terrain.measure_front_half = True
    # Cfg.env.camera_zero = False

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = TrajectoryTrackingEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    # from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(logdir, headless=True):
    # from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    env, policy = load_env(logdir, headless=headless)
    env.start_recording()

    num_eval_steps = 1005

    measured_x_vels = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    env.episode_length_buf[:] = 0 # torch.zeros_like(env.episode_length_buf)
    # import ipdb; ipdb.set_trace()
    reached = [None] * args.num_env * args.n_repeat

    for i in range(num_eval_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        step_reached = env.episode_sums["reach_goal"]
        for j, (d, r) in enumerate(zip(done, rew)):
            if d and reached[j] is None:
                reached[j] = (r > 2.0).item()

                if j == 0:
                    frames = env.get_complete_frames()
                    fps = 25
                    if args.hybrid:
                        path = os.path.join(logdir, "hybrid.mp4")
                    else:
                        path = os.path.join(logdir, "e2e.mp4")
                    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]), True)
                    for frame in frames:
                        out.write(frame[:, :, :3])
                    out.release()
        ei = env.episode_length_buf[0].item()
        print(f"Step: {i}; env step: {ei}", end='\r')

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    print("Reached: ", np.sum(reached))


if __name__ == '__main__':
    # to see the environment rendering, set headless=False

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--num_env", type=int, default=100)
    parser.add_argument("--n_repeat", type=int, default=10)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()

    play_go1(args.logdir, headless=False)
