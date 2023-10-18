import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2

import glob
import pickle as pkl
from params_proto import PrefixProto, ParamsProto

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.trajectory_tracking import TrajectoryTrackingEnv

from tqdm import tqdm

# LOAD_PATH = "wandb/run-20230914_164301-yuilazvd/files"
# LOAD_PATH = "wandb/run-20230914_173527-luqqe6t6/files"
# LOAD_PATH = "wandb/run-20230915_062051-30d5ikhi/files"
# LOAD_PATH = "wandb/run-20231008_084214-c26g0as0/files"
LOAD_PATH = "wandb/run-20231016_233057-wn7jngb6/files"

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        action += torch.randn(action.shape) * 1.0
        info['latent'] = latent
        return action

    return policy

from go1_gym_learn.ppo_cse_cnn import ActorCritic, AC_Args
import os

def load_policy(logdir, env, device='cuda:0'):
    AC_Args.use_cnn = True
    AC_Args.use_gru = False
    actor_critic = ActorCritic(env.num_obs,
                                env.num_privileged_obs,
                                env.num_obs_history,
                                env.num_actions,
                                ).to(device)

    weights = torch.load(os.path.join(logdir, "checkpoints", "ac_weights.pt"))
    actor_critic.load_state_dict(state_dict=weights)

    def policy(obs, info={}):
       return actor_critic.act(obs["obs_history"])
    
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
                    if key2 not in ["top", "bottom"]:
                        setattr(getattr(Cfg, key), key2, value2)

    Cfg.env.recording_width_px = 640
    Cfg.env.recording_height_px = 480
    Cfg.env.episode_length_s = 20
    Cfg.env.timestep_in_obs = True
    Cfg.env.num_history = 5
    Cfg.commands.switch_dist = 0.6

    # turn off DR for evaluation script
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
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.env.look_from_back = True
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.env.viewer_look_at_robot = True

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = TrajectoryTrackingEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)
    env.start_recording()
    env.reset()

    # load policy
    # from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir, env, device='cuda:0')

    return env, policy


def play_go1(headless=True):
    # from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    logdir = LOAD_PATH

    env, policy = load_env(logdir, headless=headless)
    env.start_recording()

    # num_eval_steps = 2005
    num_eval_steps = 1001

    measured_x_vels = np.zeros(num_eval_steps)
    measured_rolls = np.zeros(num_eval_steps)
    measured_pitchs = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    env.episode_length_buf[:] = 0

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        # print(env.episode_sums["reaching_local_goal"])
        # print(env.episode_sums["exploration"])

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_rolls[i] = env.base_rotation[0, 0]
        measured_pitchs[i] = env.base_rotation[0, 1]
        joint_positions[i] = env.dof_pos[0, :].cpu()

        if done.any():
            break

    frames = env.get_complete_frames()

    # """
    fps = 25
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]), True)
    for frame in frames:
        out.write(frame[:, :, :3])
    out.release()
    # """ 
    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, i), measured_x_vels[:i], color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, i), measured_rolls[:i], linestyle="-", label="Measured")
    axs[1].set_title("Rotation - Roll")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("(rad)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, i), measured_pitchs[:i], linestyle="-", label="Measured")
    axs[2].set_title("Rotation - Pitch")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("(rad)")

    plt.tight_layout()
    plt.show() 


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
