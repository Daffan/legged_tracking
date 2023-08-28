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
        x = torch.arange(1, tcfg.traj_length+1, device=self.env.device).repeat(len(env_ids), 1) * tcfg.base_x
        x += self.env.root_states[::self.env.num_actor][env_ids, 0:1]
        y = torch.arange(1, tcfg.traj_length+1, device=self.env.device).repeat(len(env_ids), 1) * tcfg.base_y
        y += self.env.root_states[::self.env.num_actor][env_ids, 1:2]
        z = torch.zeros_like(x) + tcfg.base_z
        yaw = torch.zeros_like(x) * tcfg.base_yaw
        pitch = torch.zeros_like(x) + tcfg.base_pitch
        roll = torch.zeros_like(x) + tcfg.base_roll
        return torch.stack([x, y, z, roll, pitch, yaw], dim=2)
    
    def _traj_fn_random_goal(self, env_ids):
        tcfg = self.env.cfg.commands
        x_mean, x_range = tcfg.x_mean, tcfg.x_range
        y_mean, y_range = tcfg.y_mean, tcfg.y_range
        x = (torch.rand((*env_ids.shape, 1), device=self.env.device) - 0.5) * x_range + x_mean
        x += self.env.root_states[::self.env.num_actor][env_ids, 0:1]
        y = (torch.rand((*env_ids.shape, 1), device=self.env.device) - 0.5) * y_range + y_mean
        y += self.env.root_states[::self.env.num_actor][env_ids, 1:2]
        z = torch.zeros_like(x) + self.env.cfg.commands.base_z
        yaw = torch.rand((*env_ids.shape, 1), device=self.env.device) * 2 * tcfg.yaw_range - tcfg.yaw_range
        pitch = torch.zeros_like(x)
        roll = torch.zeros_like(x)
        return torch.stack([x, y, z, roll, pitch, yaw], dim=2)
    
    def _traj_fn_valid_goal(self, env_ids):
        # sample valid goal from the terrain
        tcfg = self.env.cfg.commands
        x_mean, x_range = tcfg.x_mean, tcfg.x_range

        # global elevation maps
        env_height_sample = self.env.env_height_samples[env_ids]  # (num_envs, 2, num_pixels_x, num_pixels_y)
        openings = env_height_sample[:, 0, :, :] - env_height_sample[:, 1, :, :]  # (num_envs, num_pixels_x, num_pixels_y)
        
        x = (torch.rand((*env_ids.shape, 1), device=self.env.device) - 0.5) * x_range + x_mean
        x += self.env.root_states[::self.env.num_actor][env_ids, 0:1]
        x -= self.env.env_terrain_origin[env_ids, 0:1]
        x_pixel = (x / self.env.terrain.cfg.horizontal_scale).long().unsqueeze(-1).repeat(1, 1, env_height_sample.shape[-1])  # (num_envs, 1, 1)
        y_openings = openings.gather(1, x_pixel)[:, 0, :]\
             - torch.linspace(-0.01, 0.01, env_height_sample.shape[-1], device=self.env.device).clip(0, 1)\
             - torch.linspace(0.01, -0.01, env_height_sample.shape[-1], device=self.env.device).clip(0, 1)
        y_pixel = torch.argmax(y_openings, dim=1).unsqueeze(-1)  # find the y_pixel that has the largest opening
        y = y_pixel * self.env.terrain.cfg.horizontal_scale  # (num_envs, 1)

        x += self.env.env_terrain_origin[env_ids, 0:1]
        y += self.env.env_terrain_origin[env_ids, 1:2]
        z = torch.zeros_like(x) + self.env.cfg.commands.base_z
        yaw = torch.zeros_like(x)
        pitch = torch.zeros_like(x)
        roll = torch.zeros_like(x)
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
        target_pose[:, 0, :] = 0  # set the first target to be the current pose

        delta = (target_pose[:, 1:, :] - target_pose[:, :-1, :]) / num_interp
        target_pose_interp = torch.stack([target_pose[:, :-1] + (i+1) * delta for i in range(num_interp)], dim=2)
        target_pose_interp = target_pose_interp.reshape(*env_ids.shape, -1, 6)

        target_pose_interp[:, :, :3] += self.env.root_states[::self.env.num_actor][env_ids, :3][:, None, :]
        return target_pose_interp  # (num_envs, traj_length, 6)