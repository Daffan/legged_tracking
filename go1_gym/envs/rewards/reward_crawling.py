import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

EPSILON = 1e-6

class RewardsCrawling:
    # reward for crawling behavior
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ---------------- penalty ----------------
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)
    
    def _reward_base_height(self):
        return torch.square(self.env.relative_linear[:, 2] - self.env.cfg.rewards.base_height_target)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    # TODO: consider adding foot to body center

    # ---------------- task ----------------

    def _reward_e2e(self):
        # when the distance is smaller than 0.5m and T > T_reach, perform velocity tracking
        # this can help keep stable at goal position
        target_linear_vel = self.env.relative_linear[:, :2]
        magnitude = torch.linalg.norm(target_linear_vel, dim=1, keepdim=True)
        # target linear velocity (default 0.25 m/s)
        target_linear_vel = target_linear_vel / (magnitude + EPSILON) * self.env.cfg.rewards.target_lin_vel
        # if in a distance range of 0.05, set the target to be 0
        target_linear_vel *= (magnitude > self.env.cfg.rewards.lin_reaching_criterion)
        linear_vel_error = torch.sum(torch.square(target_linear_vel - self.env.base_lin_vel[:, :2]), dim=-1)

        in_dist = torch.norm(self.env.relative_linear[:, :2], dim=1) < self.env.cfg.rewards.large_dist_threshold
        after_t_reach = self.env.episode_length_buf > self.env.cfg.rewards.T_reach
        return torch.exp(-linear_vel_error/self.env.cfg.rewards.tracking_sigma_lin) * in_dist.float() * after_t_reach.float()
    
    def _reward_exploration_lin(self):
        """ rewarding the linear velocity to be close to
            the specified target velocity toward the target pose
        """
        # TODO: consider adding yaw as well
        target_linear_vel = self.env.relative_linear[:, :2]
        magnitude = torch.linalg.norm(target_linear_vel, dim=1, keepdim=True)
        # target linear velocity (default 0.25 m/s)
        target_linear_vel = target_linear_vel / (magnitude + EPSILON) * self.env.cfg.rewards.target_lin_vel
        # if in a distance range of 0.05, set the target to be 0
        target_linear_vel *= (magnitude > self.env.cfg.rewards.lin_reaching_criterion)
        if self.env.cfg.rewards.lin_vel_form == "exp":
            linear_vel_error = torch.sum(torch.square(target_linear_vel - self.env.base_lin_vel[:, :2]), dim=-1)
            return torch.exp(-linear_vel_error/self.env.cfg.rewards.tracking_sigma_lin)
        if self.env.cfg.rewards.lin_vel_form == "l1":
            return torch.sum(torch.abs(target_linear_vel - self.env.base_lin_vel[:, :2]), dim=-1)
        if self.env.cfg.rewards.lin_vel_form == "l2":
            return torch.sum(torch.square(target_linear_vel - self.env.base_lin_vel[:, :2]), dim=-1)
        
    def _reward_exploration_yaw(self):
        ''' rewarding the angular velocity to be close to
            the specified target angualr velocity towards the target pose yaw
        '''
        target_angular_vel = self.env.relative_rotation[:, 2]
        magnitude = target_angular_vel.abs()
        target_angular_vel = target_angular_vel / (magnitude + EPSILON) * self.env.cfg.rewards.target_ang_vel  # pi/2 angular velocity
        target_angular_vel *= (magnitude > self.env.cfg.rewards.ang_reaching_criterion)  # if angle error smaller than pi/10.
        ang_vel_error = torch.square(target_angular_vel - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma_ang)
        
    def _reward_reaching_z(self):
        return torch.square(self.env.relative_linear[:, 2]) 

    def _reward_reaching_roll(self):
        return torch.square(self.env.relative_rotation[:, 0])

    def _reward_reaching_pitch(self):
        return torch.square(self.env.relative_rotation[:, 1])
