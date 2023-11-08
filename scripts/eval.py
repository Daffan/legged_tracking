import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2

import glob
import pickle as pkl
from params_proto import PrefixProto, ParamsProto
import os

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.trajectory_tracking import TrajectoryTrackingEnv

from tqdm import tqdm

LOAD_PATH = "/tmp/wandb/run-20231026_081701-3u41i1hk/files"

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        action += torch.randn(action.shape) * 1.0
        info['latent'] = latent
        return action

    return policy



def load_policy(logdir, env, device='cuda:0', old_ppo=False):
    if old_ppo:
        from go1_gym_learn.ppo_cse import ActorCritic
        actor_critic = ActorCritic(env.num_obs,
                                    env.num_privileged_obs,
                                    env.num_obs_history,
                                    env.num_actions,
                                    ).to(device)
        weights = torch.load(os.path.join(logdir, "checkpoints", "ac_weights.pt"), map_location=device)
        actor_critic.load_state_dict(state_dict=weights)
        def policy(obs, info={}):
            return actor_critic.act(obs["obs_history"])
    else:
        from go1_gym_learn.ppo_cse_cnn import ActorCritic, AC_Args
        AC_Args.use_cnn = True
        AC_Args.use_gru = False
        actor_critic = ActorCritic(env.num_obs,
                                    env.num_privileged_obs,
                                    env.num_obs_history,
                                    env.num_actions,
                                    ac_args=AC_Args
                                    ).to(device)

        weights = torch.load(os.path.join(logdir, "checkpoints", "ac_weights.pt"), map_location=device)
        actor_critic.load_state_dict(state_dict=weights)
        def policy(obs, info={}):
            return actor_critic.act(obs["obs_history"])
    
    return policy

def load_env(args, logdir, headless=False, device='cuda:0'):

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg  # ["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    # if key2 not in ["top", "bottom"]:
                    setattr(getattr(Cfg, key), key2, value2)

    Cfg.env.recording_width_px = 640
    Cfg.env.recording_height_px = 480
    Cfg.env.num_envs = 16
    Cfg.env.look_from_back = True
    Cfg.env.record_all_envs = True
    Cfg.terrain.num_rows = 4
    Cfg.terrain.num_cols = 4
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

    Cfg.env.viewer_look_at_robot = True

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = TrajectoryTrackingEnv(sim_device=device, headless=False, cfg=Cfg)
    env = HistoryWrapper(env)
    env.start_recording()
    env.reset()

    # load policy
    # from ml_logger import logger

    policy = load_policy(logdir, env, device=device, old_ppo=args.old_ppo)

    return env, policy


def play_go1(args):
    # from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    logdir = args.logdir

    env, policy = load_env(args, logdir, headless=args.headless, device=f'cuda:{args.device}')
    env.start_recording()

    num_eval_steps = 500

    measured_x_vels = np.zeros(num_eval_steps)
    measured_rolls = np.zeros(num_eval_steps)
    measured_pitchs = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    env.episode_length_buf[:] = 0

    video_frames = []

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)
        video_frames.append(env.render_all_envs())

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_rolls[i] = env.base_rotation[0, 0]
        measured_pitchs[i] = env.base_rotation[0, 1]
        joint_positions[i] = env.dof_pos[0, :].cpu()


    video_frames = np.stack(video_frames).astype(np.uint8)
    video_frames = video_frames.transpose(1, 0, 2, 3, 4)
    fps = 25
    """
    for i, frames in enumerate(video_frames):
        out = cv2.VideoWriter(f'media/videos/env_{i}.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()
    """ 
    import imageio
    for i, frames in enumerate(video_frames):
        output_video_path = f'media/videos/env_{i}.mp4'
        video_writer = imageio.get_writer(output_video_path, fps=fps)
        for frame in frames:
            video_writer.append_data(frame)
        video_writer.close()

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--logdir", type=str, default=LOAD_PATH)
    parser.add_argument("--old_ppo", action="store_true")
    args = parser.parse_args()

    play_go1(args)

    # some commands:
    # tar -zcvf media/videos.tar.gz media/videos
