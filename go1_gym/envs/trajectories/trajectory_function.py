import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

from go1_gym.envs.base.base_task import BaseTask


class TrajectoryFunctions:
    def __init__(self, env: BaseTask):
        self.env = env

    def _traj_fn_fixed_target(self, env_ids):
        # fixed delta x, y between waypoints as specified by the config
        # return tractories of shape (num_envs, traj_length, 6)
        tcfg = self.env.cfg.commands
        x = torch.arange(tcfg.traj_length, device=self.env.device).repeat(len(env_ids), 1) * tcfg.base_x
        x += self.env.robot_states[env_ids, 0:1]
        y = torch.arange(tcfg.traj_length, device=self.env.device).repeat(len(env_ids), 1) * tcfg.base_y
        y += self.env.robot_states[env_ids, 1:2]
        z = torch.zeros_like(x) + tcfg.base_z
        yaw = torch.ones_like(x) * tcfg.base_yaw
        pitch = torch.zeros_like(x) + tcfg.base_pitch
        roll = torch.zeros_like(x) + tcfg.base_roll
        return torch.stack([x, y, z, roll, pitch, yaw], dim=2)
    
    def _traj_fn_random_target(self, env_ids):
        # random delta x, y, z, raw, pitch, yaw between waypoints with interpolation
        # return tractories of shape (num_envs, traj_length, 6)
        tcfg = self.env.cfg.commands
        num_interp = tcfg.num_interpolation
        assert tcfg.traj_length % num_interp == 0  # for now...
        num_targets = tcfg.traj_length // num_interp + 1
        
        x = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.x_range - tcfg.x_range
        y = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.y_range - tcfg.y_range
        z = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.z_range - tcfg.z_range
        yaw = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.yaw_range - tcfg.yaw_range
        pitch = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.pitch_range - tcfg.pitch_range 
        roll = torch.rand((*env_ids.shape, num_targets), device=self.env.device) * 2 * tcfg.roll_range - tcfg.roll_range

        target_pose = torch.stack([x, y, z, roll, pitch, yaw], dim=2)  # (num_envs, num_targets, 6)
        target_pose[: 0, :] = 0  # set the first target to be the current pose

        delta = (target_pose[1:] - target_pose[:-1]) / num_interp
        target_pose_interp = torch.stack([target_pose[:-1] + (i+1) * delta for i in range(num_interp)], dim=1)
        target_pose_interp = target_pose_interp.reshape(*env_ids.shape, -1, 6)

        target_pose_interp[:, :3] += self.env.robot_states[env_ids, :3]
        return target_pose_interp[1:]  # (num_envs, traj_length, 6)