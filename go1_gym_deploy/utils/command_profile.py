import torch
import numpy as np


class CommandProfile:
    def __init__(self, dt, max_time_s=10.):
        self.dt = dt
        self.max_timestep = int(max_time_s / self.dt)
        self.commands = torch.zeros((self.max_timestep, 9))
        self.start_time = 0

    def get_command(self, t):
        timestep = int((t - self.start_time) / self.dt)
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        return [0, 0, 0, 0]

    def reset(self, reset_time):
        self.start_time = reset_time

class DummyFrontGoalProfile(CommandProfile):
    def __init__(
        self, dt, state_estimator,
        max_time_s=10.,
        command_xy_only=True
    ):
        super().__init__(dt, max_time_s)
        self.dt = dt
        self.state_estimator = state_estimator
        self.currently_triggered = [0, 0, 0, 0]
        self.triggered_commands = {i: None for i in range(4)}
        self.button_states = [0, 0, 0, 0]
        self.command_xy_only = command_xy_only

    def get_command(self, t):
        # this is a dummy command with the goal 1 meter at the front of the robot with a height of 0.29 and no orientation.
        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.triggered_commands[button] is not None:
                if self.button_states[button] == 1 and prev_button_states[button] == 0:
                    if not self.currently_triggered[button]:
                        # reset the triggered action
                        self.triggered_commands[button].reset(t)
                        # reset the internal timing variable
                        reset_timer = True
                        self.currently_triggered[button] = True
                    else:
                        self.currently_triggered[button] = False
                # execute the triggered action
                if self.currently_triggered[button] and t < self.triggered_commands[button].max_timestep:
                    command = self.triggered_commands[button].get_command(t)
        if self.command_xy_only:
            return np.array([1.0, 0.0]), False
        else:
            return np.array([1.0, 0.0, 0.34, 0.0, 0.0, 0.0]), False

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()

class RandomTrajectoryProfile(CommandProfile):
    def __init__(
        self, dt, state_estimator,
        x_range = 0.5,
        y_range = 0.5,
        z_range = 0.1,  # 0.1
        roll_range = 30 * np.pi / 180,
        pitch_range = 30 * np.pi / 180,
        yaw_range = 180 * np.pi / 180,
        num_interpolation=10,
        traj_length=10,
        switch_upon_reach=False,
        max_time_s=10.
    ):
        super().__init__(dt, max_time_s)
        self.dt = dt
        self.se = state_estimator
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.roll_range = roll_range
        self.num_interpolation = num_interpolation
        self.traj_length = traj_length
        self.switch_upon_reach = switch_upon_reach
        self.max_timestep = int(max_time_s / self.dt)

        self.start_time = 0
        self.trajectory = self._traj_fn_random_target()

    '''def get_command(self, t):
        xy, yaw = self.se.get_xy_yaw()  # xy: (2,), yaw: scalar
        if not self.switch_upon_reach:
            curr_timestep = int((t - self.start_time) / self.dt)
            if curr_timestep >= self.max_timestep:
                self.start_time = t
                self.trajectory = self._traj_fn_random_target()
                curr_timestep = 0
                reset_timer = True
            else:
                reset_timer = False
            curr_pose = self.trajectory[int(curr_timestep / self.max_timestep * len(self.trajectory)), :]

            rel_xy = curr_pose[:2] - xy
            rel_xy = self._apply_yaw_inverse(yaw, rel_xy)
            rel_yaw = self._wrap_to_pi(curr_pose[5] - yaw)
            return np.concatenate([rel_xy, curr_pose[2:5], [rel_yaw]]), reset_timer
        else:
            raise NotImplementedError'''
    def get_command(self, t):
        return np.array([1.0, 0.0, 0.29, 0, 0, 0]), False
    
    def _apply_yaw_inverse(self, yaw, xy):
        return np.array([
            xy[0] * np.cos(yaw) + xy[1] * np.sin(yaw),
            -xy[0] * np.sin(yaw) + xy[1] * np.cos(yaw)
        ])
    
    def _wrap_to_pi(angles):
        angles %= 2 * np.pi
        angles -= 2 * np.pi * (angles > np.pi)
        return angles

    def _traj_fn_random_target(self):
        # random delta x, y, z, raw, pitch, yaw between waypoints with interpolation
        # return tractories of shape (num_envs, traj_length, 6)
        num_interp = self.num_interpolation
        assert self.traj_length % num_interp == 0  # for now...
        num_targets = self.traj_length // num_interp + 1
        
        base_z = 0.29
        x = np.random.uniform(num_targets) * 2 * self.x_range - self.x_range
        y = np.random.uniform(num_targets) * 2 * self.y_range - self.y_range
        z = np.random.uniform(num_targets) * 2 * self.z_range - self.z_range + base_z
        yaw = np.random.uniform(num_targets) * 2 * self.yaw_range - self.yaw_range
        pitch = np.random.uniform(num_targets) * 2 * self.pitch_range - self.pitch_range 
        roll = np.random.uniform(num_targets) * 2 * self.roll_range - self.roll_range

        target_pose = np.stack([x, y, z, roll, pitch, yaw], axis=1)  # (num_targets, 6)
        target_pose[0, :] = 0  # set the first target to be the current pose

        # iterpolation
        delta = (target_pose[1:, :] - target_pose[:-1, :]) / num_interp
        target_pose_interp = np.stack([target_pose[:, :-1] + (i+1) * delta for i in range(num_interp)], axis=1)
        target_pose_interp = target_pose_interp.reshape(-1, 6)

        # assuming starting from zero
        xy, _ = self.se.get_xy_yaw()
        target_pose_interp[:, :2] += xy.reshape(1, -1)
        return target_pose_interp  # (traj_length, 6)

    def get_buttons(self):
        return self.state_estimator.get_buttons()


class ConstantAccelerationProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, zero_buf_time=0):
        super().__init__(dt)
        zero_buf_timesteps = int(zero_buf_time / self.dt)
        accel_timesteps = int(accel_time / self.dt)
        self.commands[:zero_buf_timesteps] = 0
        self.commands[zero_buf_timesteps:zero_buf_timesteps + accel_timesteps, 0] = torch.arange(0, max_speed,
                                                                                                 step=max_speed / accel_timesteps)
        self.commands[zero_buf_timesteps + accel_timesteps:, 0] = max_speed


class ElegantForwardProfile(CommandProfile):
    def __init__(self, dt, max_speed, accel_time, duration, deaccel_time, zero_buf_time=0):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)


class ElegantYawProfile(CommandProfile):
    def __init__(self, dt, max_speed, zero_buf_time, accel_time, duration, deaccel_time, yaw_rate):
        import numpy as np

        zero_buf_timesteps = int(zero_buf_time / dt)
        accel_timesteps = int(accel_time / dt)
        duration_timesteps = int(duration / dt)
        deaccel_timesteps = int(deaccel_time / dt)

        total_time_s = zero_buf_time + accel_time + duration + deaccel_time

        super().__init__(dt, total_time_s)

        x_vel_cmds = [0] * zero_buf_timesteps + [*np.linspace(0, max_speed, accel_timesteps)] + \
                     [max_speed] * duration_timesteps + [*np.linspace(max_speed, 0, deaccel_timesteps)]

        yaw_vel_cmds = [0] * zero_buf_timesteps + [0] * accel_timesteps + \
                       [yaw_rate] * duration_timesteps + [0] * deaccel_timesteps

        self.commands[:len(x_vel_cmds), 0] = torch.Tensor(x_vel_cmds)
        self.commands[:len(yaw_vel_cmds), 2] = torch.Tensor(yaw_vel_cmds)


class ElegantGaitProfile(CommandProfile):
    def __init__(self, dt, filename):
        import numpy as np
        import json

        with open(f'../command_profiles/{filename}', 'r') as file:
                command_sequence = json.load(file)

        len_command_sequence = len(command_sequence["x_vel_cmd"])
        total_time_s = int(len_command_sequence / dt)

        super().__init__(dt, total_time_s)

        self.commands[:len_command_sequence, 0] = torch.Tensor(command_sequence["x_vel_cmd"])
        self.commands[:len_command_sequence, 2] = torch.Tensor(command_sequence["yaw_vel_cmd"])
        self.commands[:len_command_sequence, 3] = torch.Tensor(command_sequence["height_cmd"])
        self.commands[:len_command_sequence, 4] = torch.Tensor(command_sequence["frequency_cmd"])
        self.commands[:len_command_sequence, 5] = torch.Tensor(command_sequence["offset_cmd"])
        self.commands[:len_command_sequence, 6] = torch.Tensor(command_sequence["phase_cmd"])
        self.commands[:len_command_sequence, 7] = torch.Tensor(command_sequence["bound_cmd"])
        self.commands[:len_command_sequence, 8] = torch.Tensor(command_sequence["duration_cmd"])

class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
        super().__init__(dt)
        self.state_estimator = state_estimator
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale

        self.probe_vel_multiplier = probe_vel_multiplier

        self.triggered_commands = {i: None for i in range(4)}  # command profiles for each action button on the controller
        self.currently_triggered = [0, 0, 0, 0]
        self.button_states = [0, 0, 0, 0]

    def get_command(self, t, probe=False):

        command = self.state_estimator.get_command()
        command[0] = command[0] * self.x_scale
        command[1] = command[1] * self.y_scale
        command[2] = command[2] * self.yaw_scale

        reset_timer = False

        if probe:
            command[0] = command[0] * self.probe_vel_multiplier
            command[2] = command[2] * self.probe_vel_multiplier

        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.triggered_commands[button] is not None:
                if self.button_states[button] == 1 and prev_button_states[button] == 0:
                    if not self.currently_triggered[button]:
                        # reset the triggered action
                        self.triggered_commands[button].reset(t)
                        # reset the internal timing variable
                        reset_timer = True
                        self.currently_triggered[button] = True
                    else:
                        self.currently_triggered[button] = False
                # execute the triggered action
                if self.currently_triggered[button] and t < self.triggered_commands[button].max_timestep:
                    command = self.triggered_commands[button].get_command(t)


        return command, reset_timer

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()

class RCControllerProfileAccel(RCControllerProfile):
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt, state_estimator, x_scale=x_scale, y_scale=y_scale, yaw_scale=yaw_scale)
        self.x_scale, self.y_scale, self.yaw_scale = self.x_scale / 100., self.y_scale / 100., self.yaw_scale / 100.
        self.velocity_command = torch.zeros(3)

    def get_command(self, t):

        accel_command = self.state_estimator.get_command()
        self.velocity_command[0] = self.velocity_command[0]  + accel_command[0] * self.x_scale
        self.velocity_command[1] = self.velocity_command[1]  + accel_command[1] * self.y_scale
        self.velocity_command[2] = self.velocity_command[2]  + accel_command[2] * self.yaw_scale

        # check for action buttons
        prev_button_states = self.button_states[:]
        self.button_states = self.state_estimator.get_buttons()
        for button in range(4):
            if self.button_states[button] == 1 and self.triggered_commands[button] is not None:
                if prev_button_states[button] == 0:
                    # reset the triggered action
                    self.triggered_commands[button].reset(t)
                # execute the triggered action
                return self.triggered_commands[button].get_command(t)

        return self.velocity_command[:]

    def add_triggered_command(self, button_idx, command_profile):
        self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()





class KeyboardProfile(CommandProfile):
    # for control via keyboard inputs to isaac gym visualizer
    def __init__(self, dt, isaac_env, x_scale=1.0, y_scale=1.0, yaw_scale=1.0):
        super().__init__(dt)
        from isaacgym.gymapi import KeyboardInput
        self.gym = isaac_env.gym
        self.viewer = isaac_env.viewer
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_UP, "FORWARD")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_DOWN, "REVERSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_LEFT, "LEFT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, KeyboardInput.KEY_RIGHT, "RIGHT")

        self.keyb_command = [0, 0, 0]
        self.command = [0, 0, 0]

    def get_command(self, t):
        events = self.gym.query_viewer_action_events(self.viewer)
        events_dict = {event.action: event.value for event in events}
        print(events_dict)
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 1.0: self.keyb_command[0] = 1.0
        if "FORWARD" in events_dict and events_dict["FORWARD"] == 0.0: self.keyb_command[0] = 0.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 1.0: self.keyb_command[0] = -1.0
        if "REVERSE" in events_dict and events_dict["REVERSE"] == 0.0: self.keyb_command[0] = 0.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 1.0: self.keyb_command[1] = 1.0
        if "LEFT" in events_dict and events_dict["LEFT"] == 0.0: self.keyb_command[1] = 0.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 1.0: self.keyb_command[1] = -1.0
        if "RIGHT" in events_dict and events_dict["RIGHT"] == 0.0: self.keyb_command[1] = 0.0

        self.command[0] = self.keyb_command[0] * self.x_scale
        self.command[1] = self.keyb_command[2] * self.y_scale
        self.command[2] = self.keyb_command[1] * self.yaw_scale

        print(self.command)

        return self.command


if __name__ == "__main__":
    cmdprof = ConstantAccelerationProfile(dt=0.2, max_speed=4, accel_time=3)
    print(cmdprof.commands)
    print(cmdprof.get_command(2))
