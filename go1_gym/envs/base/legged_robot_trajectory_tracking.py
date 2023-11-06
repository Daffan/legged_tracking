# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict
from collections import defaultdict, deque

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch
import numpy as np

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift, quat_without_yaw, quaternion_to_roll_pitch_yaw, quat_apply_yaw_inverse
from go1_gym.utils.tunnel import Terrain
from .legged_robot_trajectory_tracking_config import Cfg
from go1_gym.envs.trajectories.trajectory_function import TrajectoryFunctions


class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)


        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        # self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        """ import time
        start_time = time.time() """

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        """ post_forward_sim_time = time.time()
        try:
            self.forward_sim_time.append(-start_time + post_forward_sim_time)
        except:
            self.forward_sim_time = []
            self.forward_sim_time.append(-start_time + post_forward_sim_time)
        print("forward_sim_time", np.mean(self.forward_sim_time)) """
        
        self.post_physics_step()
        """ try:
            self.post_phyics_time.append(time.time() - post_forward_sim_time)
        except:
            self.post_phyics_time = []
            self.post_phyics_time.append(time.time() - post_forward_sim_time)
        print("post_phyics_time", np.mean(self.post_phyics_time)) """

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1  # this is for each individual environments

        # prepare quantities
        self.base_pos[:] = self.root_states[::self.num_actor][:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[::self.num_actor][:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.num_actor][:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[::self.num_actor][:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices, 0:3] - self.base_pos[:, None, :]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        self.update_curriculum()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[::self.num_actor][:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if self.cfg.env.viewer_look_at_robot:
            # set the viewer always at the back of the robot in the first environment
            pos_target = torch.tensor([
                [-0.295-0.1, 0, 0.35],
                [0, 0, 0.25]
            ], device=self.device)
            # Right now, the camera does not rotate with the robot
            # pos_target = quat_apply_yaw(self.base_quat[[0]*2], pos_target)
            pos_target[:, :2] += self.root_states[0, :2]
            self.set_camera(pos_target[0], pos_target[1])

        self._render_headless()

    def update_curriculum(self):
        if "exploration" in self.reward_scales and self.reward_scales["exploration"] > 0:
            if self.common_step_counter > self.cfg.rewards.exploration_steps:
                self.reward_scales["exploration"] -= self.cfg.reward_scales.exploration / self.cfg.rewards.exploration_steps
                if self.reward_scales["exploration"] <= 0:
                    print("Exploration reward disabled")

        if self.cfg.curriculum_thresholds.cl_fix_target and \
            np.mean(self.extras["train/episode"]["reached"]) > \
            self.cfg.curriculum_thresholds.cl_switch_threshold and \
            len(self.extras["train/episode"]["reached"]) == 4000:
            
            self.current_target_dist += self.cfg.curriculum_thresholds.cl_switch_delta
            self.current_target_dist = min(self.current_target_dist, self.cfg.curriculum_thresholds.cl_goal_target_dist)
            self.cfg.commands.x_mean = self.current_target_dist

            self.extras["train/episode"]["reached"] = deque(maxlen=4000) # refresh deque 
            self.extras["eval/episode"]["reached"] = deque(maxlen=4000) # refresh deque 

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.cfg.rewards.use_terminal_body_height:
            # TODO: height to the floor should consider floor height
            self.body_height_buf = torch.mean(self.root_states[::self.num_actor][:, 2].unsqueeze(1), dim=1) \
                                   < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        if self.cfg.env.terminate_end_of_trajectory:
            reached_buf = torch.logical_and(self.reached_buf, self.episode_length_buf > self.cfg.rewards.T_reach)
            self.reset_buf = torch.logical_or(reached_buf, self.reset_buf)

        if self.cfg.env.use_terminal_body_rotation:
            self.reset_buf = torch.logical_or(self.reset_buf, self.projected_gravity[:, 2] > 0.0)  # any angle > 90 degree

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # reset robot states
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)

        # trajectory needs to be resampled after reset so that the origin matches the current position
        self._resample_trajectory(env_ids)

        episode_length_buf = self.episode_length_buf.float().clone().detach()

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.local_relative_linear[env_ids] = 0.
        self.local_relative_rotation[env_ids] = 0.
        self.plan_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            # self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key].extend(
                    self.episode_sums[key][train_env_ids].detach().cpu().numpy())
                self.episode_sums[key][train_env_ids] = 0.
            self.extras["train/episode"]['episode_length'].extend(
                episode_length_buf[train_env_ids].detach().cpu().numpy())
            self.extras["train/episode"]['reached'].extend(
                self.reached_buf[train_env_ids].detach().cpu().numpy())
            self.extras["train/episode"]['goal_distance'].extend(
                torch.norm(self.relative_linear[train_env_ids], dim=1).detach().cpu().numpy())
            
            self.reached_env_buf[train_env_ids] = self.reached_buf[train_env_ids]  # this tracks the last reached state
            self.collision_env_buf[train_env_ids] = self.collision_count[train_env_ids] * self.reached_buf[train_env_ids]  # this tracks the last reached state
            # print(self.episode_length_buf[0].item(), torch.mean(self.reached_env_buf.float()).item())
            self.collision_count[train_env_ids] = 0

            # log curriculum
            if self.cfg.curriculum_thresholds.cl_fix_target:
                self.extras["train/episode"]['goal_dist'] = self.current_target_dist
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.
            self.extras["eval/episode"]['episode_length'].extend(
                episode_length_buf[eval_env_ids].detach().cpu().numpy())

        if self.cfg.env.send_timeouts:
            self.extras["timeouts"].extend(self.time_out_buf[:self.num_train_envs])
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device) #  * self.num_actor

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # base position
        self.root_states[::self.num_actor][env_ids_int32] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32 * self.num_actor), len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sumss and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                             :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                                 :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)
            if self.cfg.env.timestep_in_obs:
                self.obs_buf = torch.cat((self.obs_buf,
                                          self.episode_length_buf.unsqueeze(1) / self.max_episode_length), dim=-1)
            
        if self.cfg.env.observe_heights:
            # take the second half as front
            if self.cfg.terrain.measure_front_half:
                x_start = int(self.measured_heights.shape[2] // 2 + 1)
            else:
                x_start = 0
            measured_heights_front = self.measured_heights[:, :, x_start:, :]
            if self.cfg.env.camera_zero:
                measured_heights_front -= self.root_states[::self.num_actor][..., 2:3, None, None]
                measured_heights_front = torch.clip(measured_heights_front, min=-0.3, max=0.2)
            else:
                measured_heights_front = torch.clip(measured_heights_front, min=0, max=self.cfg.terrain.ceiling_height)
                # to the range of [-0.5, 0.5]
                measured_heights_front /= self.cfg.terrain.ceiling_height
                measured_heights_front -= 0.5

            measured_heights_front = measured_heights_front.reshape(self.num_train_envs, -1) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((
                self.obs_buf, measured_heights_front), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        # if self.cfg.env.observe_desired_contact_states:
        #     self.obs_buf = torch.cat((self.obs_buf,
        #                               self.desired_contact_states), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[::self.num_actor][:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        assert self.obs_buf.shape[1] == self.cfg.env.num_observations, f"Observation shape {self.obs_buf.shape} does not match num_observations {self.cfg.env.num_observations}"

        # build privileged obs

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[::self.num_actor][:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[::self.num_actor][:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                # TODO: add eval terrain later
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)

        if mesh_type == "plane":
            self._create_ground_plane()
        if mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type not in [None, "plane"]:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target) \

    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        # TODO: check whether this is still needed
        # self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # resample commands
        # Update the realtive target poses at every timestep
        # measure terrain heights
        if self.cfg.env.observe_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device))

        env_ids = torch.arange(self.num_envs).to(self.device)
        self._plan_target_pose(env_ids)
        self._set_the_target_pose_visual(torch.arange(self.num_envs, device=self.device), self.local_target_poses)
        if self.cfg.env.command_type == "xy":
            self.commands = self.local_relative_linear[:, :2]
        elif self.cfg.env.command_type == "xy_norm":
            norm = torch.norm(self.local_relative_linear[:, :2])
            # if magnitue larger than 1 meters, normalize
            if norm > 1.0:
                self.commands = self.local_relative_linear[:, :2] / norm
                
        elif self.cfg.env.command_type == "6dof":
            self.commands = torch.cat([
                self.local_relative_linear[:, :2],  # relative x, y
                self.trajectories[env_ids, self.curr_pose_index[env_ids], 2:-1],  # z, roll, pitch: in real application, it is hard to access the relative z, roll, pitch
                self.local_relative_rotation[:, 2:]  # relative yaw
            ], axis=-1)
        else:
            raise ValueError("Command type not recognised. Allowed types are [xy, front, 6dof]")

        # push robots
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        # This setting switches the robot every fixed timestep intervals
        if self.cfg.commands.switch_upon_reach:
            self.switched_buf = torch.linalg.norm(self.relative_linear[:, :2], dim=1) < self.cfg.commands.switch_dist
        else:
            self.switched_buf = (self.episode_length_buf % self.cfg.commands.switch_interval == 0)
        env_ids = self.switched_buf.nonzero(as_tuple=False).flatten()
        self.curr_pose_index[env_ids] += 1
        # cap to traj_length
        self.curr_pose_index[env_ids] = torch.clip(self.curr_pose_index[env_ids], max=self.cfg.commands.traj_length-1)
        self.reached_buf = torch.logical_and(self.switched_buf, self.curr_pose_index==self.cfg.commands.traj_length-1)
        plan_buf_linear = torch.linalg.norm(self.local_relative_linear[:, :2], dim=1) < self.cfg.commands.switch_dist
        plan_buf_yaw = torch.abs(self.local_relative_rotation[:, 2]) < self.cfg.commands.switch_yaw
        self.plan_buf = torch.logical_and(plan_buf_linear, plan_buf_yaw)  # switch if both linear and yaw are reached
        self.collision_count += torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).long(), dim=-1)

    def _plan_target_pose(self, env_ids):
        # No planning, directly use the goal as the target pose
        # if not self.cfg.commands.sampling_based_planning:
        # this is in world frame
        self.target_poses = self.trajectories[env_ids, self.curr_pose_index[env_ids], :]
        self.relative_linear, self.relative_rotation = self._compute_relative_target_pose(self.target_poses)
        if self.cfg.commands.sampling_based_planning:
            # sampling-based planning
            self.plan_length_buf += 1
            close_to_goal = torch.norm(self.relative_linear[:, :2], dim=1) < 1.0
            if self.cfg.commands.plan_interval > 0:
                self.replan = (self.plan_length_buf % self.cfg.commands.plan_interval) == 0
                ep_start = self.episode_length_buf == 1
                # switch the plan whenenver the agent reaches and the plan interval is reached
                plan_buf = torch.logical_and(self.replan, self.plan_buf)
                plan_buf = torch.logical_or(ep_start, plan_buf)
            else:
                plan_buf = torch.logical_or(ep_start, self.plan_buf)
            plan_env_ids = plan_buf.nonzero(as_tuple=False).flatten()
            # plan_buf = torch.logical_and(self.plan_buf, ~close_to_goal)
            
            if len(plan_env_ids) > 0:
                target_poses = []
                goal_poses = self.trajectories[env_ids, self.curr_pose_index[env_ids]]
                self.plan_length_buf[plan_env_ids] = 0
                for env_id in plan_env_ids:
                    if close_to_goal[env_id]:
                        target_poses.append(self.target_poses[env_id])  
                    else:
                        candidate_target_poses_env = torch.from_numpy(self.cfg.commands.candidate_target_poses).to(self.device).float()
                        goal_pose = goal_poses[env_id]
                        goal_pose[:2] -= self.base_pos[env_id, :2].clone()
                        # sort by distance to goal pose
                        metrics = torch.linalg.norm(candidate_target_poses_env[:, :2] - goal_pose[:2], dim=1)
                        metrics += torch.linalg.norm(candidate_target_poses_env[:, 3:], dim=1) * 10.  # penality for rotation
                        metrics += torch.abs(candidate_target_poses_env[:, 3] - goal_pose[3]) * 100.  # penality for z from base
                        idxs = torch.argsort(
                            torch.linalg.norm(candidate_target_poses_env[:, :2] - goal_pose[:2], dim=1) + \
                            torch.linalg.norm(candidate_target_poses_env[:, 3:], dim=1) * 0.1  # penality for rotation
                        , dim=-1)
                        candidate_target_poses_env = candidate_target_poses_env[idxs, :]
                        candidate_target_poses_linear = candidate_target_poses_env[:, :3]
                        candidate_target_poses_quat = quat_from_euler_xyz(candidate_target_poses_env[:, 3], candidate_target_poses_env[:, 4], candidate_target_poses_env[:, 5])
                        height_points = self.height_points[env_id, None, ...].repeat(2, 1, 1, 1)  # (2, 21, 11, 3)
                        height_points[..., 2] = self.measured_heights[env_id]
                        height_points_cand = height_points.view(-1, 1, 3).repeat(1, len(candidate_target_poses_env), 1)
                        height_points_cand -= candidate_target_poses_linear[None, :, :]
                        height_points_cand = quat_apply_yaw_inverse(candidate_target_poses_quat.repeat(height_points_cand.shape[0], 1), height_points_cand.view(-1, 3)).view(*height_points_cand.shape)
                        height_points_cand = torch.norm(height_points_cand / self.robot_size[None, None, :], dim=-1) > 1.0
                        cands_idx = torch.all(height_points_cand, dim=0)
                        if torch.any(cands_idx):
                            target_pose_env = candidate_target_poses_env[cands_idx, :][0, :]
                            # bring the target pose to world frame 
                            # target_pose_env[:2] += self.base_pos[env_id, :2]
                            base_rotation = quaternion_to_roll_pitch_yaw(self.base_quat[env_id][None, :])[0]
                            target_pose_env[:2] = quat_apply_yaw(self.base_quat[[env_id]], target_pose_env[None, :3])[0, :2] + self.base_pos[env_id, :2]
                            target_pose_env[3:] = wrap_to_pi(target_pose_env[3:] + base_rotation)
                            target_poses.append(target_pose_env)
                        else:
                            print("env_%d has no valid local goal, using the global goal" %env_id)
                            target_poses.append(self.target_poses[env_id]) # candidate_target_poses_env[0, :]
                        
                target_poses = torch.stack(target_poses, dim=0)
                self.local_target_poses[plan_env_ids] = target_poses
            # update relative linear and rotation at every timestep
            self.local_relative_linear, self.local_relative_rotation = \
                self._compute_relative_target_pose(self.local_target_poses)
        else:
            self.local_target_poses = self.target_poses
            self.local_relative_linear, self.local_relative_rotation = \
                self._compute_relative_target_pose(self.local_target_poses)
        
    def _compute_relative_target_pose(self, target_poses):
        """ Computes the relative pose of the target with respect to the body
        Args:
            target_pose (torch.Tensor): Pose of the target (num_envs, 6)
        """
        relative_linear = target_poses[:, :3] - self.root_states[::self.num_actor][:, :3]
        relative_linear = quat_apply_yaw_inverse(self.base_quat, relative_linear)
        self.base_rotation = quaternion_to_roll_pitch_yaw(self.base_quat)
        relative_rotation = target_poses[:, 3:] - self.base_rotation
        relative_rotation = wrap_to_pi(relative_rotation)
        return relative_linear, relative_rotation

    def _set_the_target_pose_visual(self, env_ids, target_poses):
        """ This function sets the visualization of the target pose 
            with a green coordinate frame.
        """
        linear = target_poses[env_ids, :3]  # self.trajectories[env_ids, self.curr_pose_index[env_ids], :3]
        rotation = target_poses[env_ids, 3:]  #  self.trajectories[env_ids, self.curr_pose_index[env_ids], 3:]
        arrow_actor_ids = (env_ids + 1) * self.num_actor - 1
        self.root_states[arrow_actor_ids, :3] = linear
        target_quat = quat_from_euler_xyz(rotation[:, 0], rotation[:, 1], rotation[:, 2])
        self.root_states[arrow_actor_ids, 3:7] = target_quat
        arrow_actor_ids_int32 = to_torch(arrow_actor_ids, dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(arrow_actor_ids_int32), len(arrow_actor_ids_int32))

    def _resample_trajectory(self, env_ids):
        # if there is a curriculum, it should be updated here
        # move this index when moving to the next pose in a trajectory
        self.curr_pose_index[env_ids] = 0
        self.trajectories[env_ids] = self.trajectory_function(env_ids)
        # self.target_poses[env_ids] = self.trajectories[env_ids, self.curr_pose_index[env_ids], :]
        # TODO: add the target visual back

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32) * self.num_actor
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids, cfg: Cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[::self.num_actor][env_ids] = self.base_init_state
            self.root_states[::self.num_actor][env_ids, :3] += self.env_origins[env_ids]
            self.root_states[::self.num_actor][env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[::self.num_actor][env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[::self.num_actor][env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[::self.num_actor][env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[::self.num_actor][env_ids] = self.base_init_state
            self.root_states[::self.num_actor][env_ids, :3] += self.env_origins[env_ids]
        # import ipdb; ipdb.set_trace()

        # base yaws
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[::self.num_actor][env_ids, 3:7] = quat

        # base velocities
        self.root_states[::self.num_actor][env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32) #  * self.num_actor
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32 * self.num_actor), len(env_ids_int32))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
                self.complete_height_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
                self.complete_height_frames = self.height_frames[:]
            self.video_frames = []
            self.height_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
                self.complete_height_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
                self.complete_height_frames_eval = self.height_frames_eval[:]
            self.video_frames_eval = []
            self.height_frames_eval = []

    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]
            env_ids *= self.num_actor

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[::self.num_actor][env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        if self.cfg.env.observe_command:
            command_dim = 2 if self.cfg.env.command_xy_only else 6
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(command_dim),  # corresponding to the 6-DOF target pose
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        if self.cfg.env.timestep_in_obs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_heights:
            if self.cfg.terrain.measure_front_half:
                size = self.height_points.shape[1] // 2 * self.height_points.shape[2] * 2
                noise_vec = torch.cat((noise_vec, torch.zeros((size,))), dim=0)
            else:
                noise_vec = torch.cat((noise_vec,
                                    torch.zeros(2 * np.prod(self.height_points.shape[1:3]))
                                    ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.robot_states = self.root_states[::self.num_actor, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        # self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # [:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[::self.num_actor][:self.num_envs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[::self.num_actor][:self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices,
                              0:3] - self.base_pos[:, None, :]  # this might not be correct
        self.prev_base_pos = self.base_pos.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {
            "train/episode": defaultdict(lambda: deque([], 4000)),
            "eval/episode": defaultdict(lambda: deque([], 4000)),
            "timeouts": deque([], 4000),
        }
        if self.cfg.curriculum_thresholds.cl_fix_target:
            self.current_target_dist = self.cfg.curriculum_thresholds.cl_start_target_dist
            self.cfg.commands.x_mean = self.current_target_dist
        self.robot_size = torch.tensor([0.3762, 0.0935, 0.114]).to(self.device).float()
        self.reached_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.plan_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.cfg.env.observe_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[::self.num_actor][:, 7:13])

        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        
        self.reached_env_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.collision_env_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        # replace commands with trajectory waypoints
        self.trajectories = torch.zeros(
            (self.cfg.env.num_envs, self.cfg.commands.traj_length, 6),
            device=self.device
        )  # (num_envs, length of traj, poses dof=6)
        self.target_poses = torch.zeros_like(self.trajectories[:, 0, :])  # (num_envs, poses dof=6)
        self.curr_pose_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.local_relative_linear = torch.zeros_like(self.target_poses[:, :3])
        self.local_relative_rotation = torch.zeros_like(self.target_poses[:, 3:])
        self.local_target_poses = torch.zeros_like(self.target_poses)
        self.relative_linear = torch.zeros_like(self.target_poses[:, :3])
        self.relative_rotation = torch.zeros_like(self.target_poses[:, 3:])
        self.plan_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.plan_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.height_frame = torch.zeros((self.num_envs, 1, 64, 64), dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.env.command_xy_only:
            self.commands = torch.zeros_like(self.trajectories[:, 0, :2])  # (num_envs, poses dof=6)  
        else:  
            self.commands = torch.zeros_like(self.trajectories[:, 0, :])  # (num_envs, poses dof=6)

        self.commands_scale = torch.ones_like(self.commands)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[::self.num_actor][:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[::self.num_actor][:self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        from go1_gym.envs.rewards.trajectory_tracking_reward import TrajectoryTrackingRewards
        from go1_gym.envs.rewards.reward_crawling import RewardsCrawling
        reward_containers = {"CoRLRewards": CoRLRewards, "TrajectoryTrackingRewards": TrajectoryTrackingRewards, "RewardsCrawling": RewardsCrawling}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        for terrain_origin, vtb, ttb in zip((self.terrain.terrain_origins).reshape(-1, 3), self.terrain.vertices, self.terrain.triangles):
            for v, t in zip(vtb, ttb):
                tm_params.nb_vertices = v.shape[0]
                tm_params.nb_triangles = t.shape[0]
                tm_params.transform.p.x = terrain_origin[0] 
                tm_params.transform.p.y = terrain_origin[1]
                self.gym.add_triangle_mesh(self.sim, v.flatten(order='C'), t.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.height_field_raw).view(2, self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices_top.shape[0]
        tm_params.nb_triangles = self.terrain.triangles_top.shape[0]

        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices_top.flatten(order='C'),
                                   self.terrain.triangles_top.flatten(order='C'), tm_params)
        

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices_bottom.shape[0]
        tm_params.nb_triangles = self.terrain.triangles_bottom.shape[0]

        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices_bottom.flatten(order='C'),
                                   self.terrain.triangles_bottom.flatten(order='C'), tm_params)
                                   
        self.height_samples = torch.tensor(self.terrain.height_field_raw).view(2, self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # Arrow asset for visualizing the target pose

        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        arrow_asset_root = os.path.join(MINI_GYM_ROOT_DIR, "resources/robots", "arrow")
        arrow_asset_file = "arrow.urdf"
        arrow_asset = self.gym.load_asset(self.sim, arrow_asset_root, arrow_asset_file, asset_options)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.armature = 0.005
        # asset_options.density = 100000.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        terrain_length = self.cfg.terrain.terrain_length * self.cfg.terrain.terrain_ratio_x
        terrain_width = self.cfg.terrain.terrain_width * self.cfg.terrain.terrain_ratio_y
        wall_thickness = 10
        v_wall_asset = self.gym.create_box(self.sim, terrain_length, self.cfg.terrain.horizontal_scale * wall_thickness, 0.4, asset_options)
        h_wall_asset = self.gym.create_box(self.sim, terrain_width, self.cfg.terrain.horizontal_scale * wall_thickness, 0.4, asset_options)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.num_actor = 1

            """ if self.cfg.terrain.mesh_type != "plane":
                env_origin = np.array([
                    (self.grid_r[i].item() + 0.5) * terrain_length / self.cfg.terrain.terrain_ratio_x,
                    (self.grid_c[i].item() + 0.5) * terrain_width / self.cfg.terrain.terrain_ratio_y - self.cfg.terrain.horizontal_scale / 2.0,
                    0.0
                ])
                wall_pose = gymapi.Transform()

                pos = env_origin.copy()
                pass; pos[1] += (terrain_width/2. + self.cfg.terrain.horizontal_scale * wall_thickness / 2); pos[2] += 0.2
                wall_pose.p = gymapi.Vec3(*pos)
                v_wall_actor = self.gym.create_actor(env_handle, v_wall_asset, wall_pose, "wall_left", i, 2, 0)

                pos = env_origin.copy()
                pass; pos[1] -= (terrain_width/2. + self.cfg.terrain.horizontal_scale * wall_thickness / 2); pos[2] += 0.2
                wall_pose.p = gymapi.Vec3(*pos)
                v_wall_actor = self.gym.create_actor(env_handle, v_wall_asset, wall_pose, "wall_right", i, 2, 0)

                pos = env_origin.copy()
                pos[0] -= terrain_length/2.; pass; pos[2] += 0.2
                wall_pose.p = gymapi.Vec3(*pos)
                wall_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi * 0.5)
                self.gym.create_actor(env_handle, h_wall_asset, wall_pose, "wall_back", i, 0, 0)

                pos = env_origin.copy()
                pos[0] += terrain_length/2.; pass; pos[2] += 0.2
                wall_pose.p = gymapi.Vec3(*pos)
                wall_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi * 0.5)
                self.gym.create_actor(env_handle, h_wall_asset, wall_pose, "wall_front", i, 0, 0)
                
                self.num_actor += 4 """

            # adding arrow to visualize the goal position
            arrow = self.gym.create_actor(env_handle, arrow_asset, start_pose, "arrow_%d" %i, i, self.cfg.asset.self_collisions, 0)
            self.num_actor += 1
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

        self.height_frames = []
        self.height_frames_eval = []
        self.complete_height_frames = []
        self.complete_height_frames_eval = []

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])
    
    def get_height_frame(self, env_id):
        elevation_map = self.measured_heights[env_id].cpu().numpy()
        data = plot_elevation_map(elevation_map)
        a_channel = np.ones_like(data[:, :, 0]) * 255
        data = np.concatenate([data, a_channel[:, :, None]], axis=2)
        return data

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            env_id = 0  # np.random.randint(0, self.num_train_envs)
            bx = self.root_states[env_id * self.num_actor, 0], 
            by = self.root_states[env_id * self.num_actor, 1]
            bz = self.root_states[env_id * self.num_actor, 2]
            if self.cfg.env.look_from_back:
                pos_target = torch.tensor([
                [-0.295-0.2, 0, 0.35],
                [0, 0, 0.25]
                ], device=self.device)
                # Right now it does not rotate with the robot
                # pos_target = quat_apply_yaw(self.base_quat[[0]*2], pos_target)
                pos_target[:, :2] += self.root_states[env_id * self.num_actor, :2]
                self.gym.set_camera_location(
                    self.rendering_camera, self.envs[env_id],
                    gymapi.Vec3(*pos_target[0].detach().cpu().numpy()),
                    gymapi.Vec3(*pos_target[1].detach().cpu().numpy()))
            else:
                self.gym.set_camera_location(self.rendering_camera, self.envs[env_id], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                            gymapi.Vec3(bx, by, bz))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[env_id], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self.video_frames.append(self.video_frame)

            """ try:
                self.height_frame = self.get_height_frame(env_id)  # self.measured_heights.detach().cpu().numpy()
            except Exception as e:
                print(e)
                pass

            try:
                self.height_frames.append(self.height_frame)
            except:
                self.height_frames.append(np.ones_like(self.video_frame) * 255) """

        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            if self.eval_cfg is not None:
                bx, by, bz = self.root_states[::self.num_actor][self.num_train_envs, 0], self.root_states[::self.num_actor][self.num_train_envs, 1], \
                             self.root_states[::self.num_actor][self.num_train_envs, 2]
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                self.video_frames_eval.append(self.video_frame_eval)

                self.height_frame = self.plot_elevation_map(self.num_train_envs)  # self.measured_heights.detach().cpu().numpy()
                self.height_frames.append(self.height_frame)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.height_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.height_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        else:
            return self.complete_video_frames
        # return [np.concatenate([vf, hf], axis=1) for vf, hf in zip(self.complete_video_frames, self.complete_height_frames)]

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval
        # return [np.concatenate([vf, hf], axis=1) for vf, hf in zip(self.complete_video_frames_eval, self.complete_height_frames_eval)]

    def _get_env_origins(self, env_ids, cfg: Cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """

        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            # if tunnel is not empty
            # env origin depends on the tunnel
            r = torch.tensor(range(self.cfg.terrain.num_rows)).to(self.device).to(torch.long)
            c = torch.tensor(range(self.cfg.terrain.num_cols)).to(self.device).to(torch.long)
            grid_r, grid_c = torch.meshgrid(r, c)
            assert self.num_envs % (self.cfg.terrain.num_rows * self.cfg.terrain.num_cols) == 0, (self.num_envs, self.cfg.terrain.num_rows, self.cfg.terrain.num_cols)
            m = self.num_envs // (self.cfg.terrain.num_rows * self.cfg.terrain.num_cols)
            grid_r = grid_r.flatten() #.detach().cpu().numpy()
            grid_c = grid_c.flatten() #.detach().cpu().numpy()
            self.grid_r = grid_r.repeat(m); self.grid_c = grid_c.repeat(m)

            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_height_samples = torch.zeros(
                self.num_envs, 2,
                self.terrain.length_per_env_pixels,
                self.terrain.width_per_env_pixels, 
                device=self.device, requires_grad=False
            )  # (num_envs, top_bottom, length, width)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_terrain_origin = torch.from_numpy(self.terrain.all_terrain_origins).to(self.device).to(torch.float)

            self.env_origins[:] = self.terrain_origins[self.grid_r, self.grid_c]
            self.env_terrain_origin = self.env_terrain_origin[self.grid_r, self.grid_c]
            self.env_height_samples[:] = torch.from_numpy(
                self.terrain.height_samples_by_row_col * self.terrain.vertical_scale
            ).to(self.device).to(torch.float)[self.grid_r, self.grid_c]

            # global elevation_map
            start_x = int((0.5 - self.terrain.terrain_ratio_x/2.) * self.terrain.length_per_env_pixels)
            end_x = int((0.5 + self.terrain.terrain_ratio_x/2.) * self.terrain.length_per_env_pixels)
            start_y = int((0.5 - self.terrain.terrain_ratio_y/2.) * self.terrain.width_per_env_pixels)
            end_y = int((0.5 + self.terrain.terrain_ratio_y/2.) * self.terrain.width_per_env_pixels)
            self.global_elevation_map = self.env_height_samples[:, start_x: end_x, start_y: end_y]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            xx = xx.to(self.device); yy = yy.to(self.device)
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

        self.trajectory_function = getattr(TrajectoryFunctions(self), "_traj_fn_" + cfg.commands.traj_function)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[::self.num_actor][i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, pixel x, pixel y, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.elevation_map_shape = grid_x.shape
        points = torch.zeros(self.num_envs, *grid_x.shape, 3, device=self.device, requires_grad=False)
        points[:, :, :, 0] = grid_x
        points[:, :, :, 1] = grid_y
        return points  # shape (num_envs, pixel x, pixel y, 3)

    def _get_heights(self, env_ids):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Returns:
            heights: height point measurement of shape (num_envs, 2, num_pixel_x, num_pixel_y)
        """
        if self.cfg.terrain.mesh_type == "plane":
            # Dummy heights maps
            top_heights = torch.ones(len(env_ids), *self.elevation_map_shape, device=self.device, requires_grad=False)  # sufficient space at the top
            bottom_heights = torch.zeros(len(env_ids), *self.elevation_map_shape, device=self.device, requires_grad=False) # sufficient space at the top
            heights = torch.stack([top_heights, bottom_heights], dim=1)
        else:
            if self.cfg.env.rotate_camera:
                points = quat_apply(self.base_quat[env_ids, None, None, :].repeat(1, *self.elevation_map_shape, 1), self.height_points[env_ids])
            else:
                points = quat_apply_yaw(self.base_quat[env_ids, None, None, :].repeat(1, *self.elevation_map_shape, 1), self.height_points[env_ids])
            points[..., :2] += self.root_states[::self.num_actor][env_ids, None, None, :2]  # bring points to the world frame
            points -= self.env_terrain_origin[env_ids, None, None, :]  # bring points to the env frame
            # TODO: get the actual position of the camera
            points = (points / self.terrain.cfg.horizontal_scale).long()
            px = points[..., 0]
            py = points[..., 1]
            px = torch.clip(px, 0, self.env_height_samples.shape[2]-2)
            py = torch.clip(py, 0, self.env_height_samples.shape[3]-2)

            heights1 = torch.stack([self.env_height_samples[env_id, :, px[env_id], py[env_id]] for env_id in env_ids], dim=0)
            heights2 = torch.stack([self.env_height_samples[env_id, :, px[env_id]+1, py[env_id]] for env_id in env_ids], dim=0)
            heights3 = torch.stack([self.env_height_samples[env_id, :, px[env_id], py[env_id]+1] for env_id in env_ids], dim=0)

            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)
            # heights = heights * self.terrain.vertical_scale

        return heights
