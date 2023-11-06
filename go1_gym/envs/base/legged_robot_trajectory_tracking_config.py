# License: see [LICENSE, LICENSES/legged_gym/LICENSE]
import numpy as np
from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):
    class env(PrefixProto, cli=False):
        num_envs = 4096
        num_observations = 235
        num_scalar_observations = 42
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 6
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_heights = True
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = True
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = False
        observe_imu = False
        record_video = True
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1
        debug_viz = False
        all_agents_share = False
        look_from_back = False

        priv_observe_friction = True
        priv_observe_friction_indep = True
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = True
        priv_observe_com_displacement = True
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = True
        priv_observe_Kd_factor = True
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

        terminate_end_of_trajectory = False
        use_terminal_body_rotation = False
        rotate_camera = False
        camera_zero = True
        command_xy_only = True
        viewer_look_at_robot = False

    class terrain(PrefixProto, cli=False):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        terrain_type = 'random_pyramid'  # in ["random", "random_pyramid"]
        valid_tunnel_only = False
        ceiling_height = 0.5
        start_loc = 0.4  # 0.4 is 0.4 * env_length away from the center of the env

        # if all zero, it is deterministic starting position
        x_init_range = 0.
        y_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        yaw_init_range = 0.  # in rad

        # settings for ground
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        terrain_ratio_x = 0.5
        terrain_ratio_y = 0.5

        terrain_length = 8.0
        terrain_width = 3.6

        terrain_border_ratio_x = 0.9
        terrain_border_ratio_y = 0.5

        num_rows = 1
        num_cols = 1

        horizontal_scale = 0.05
        vertical_scale = 0.005

        measured_points_x = np.linspace(-1, 1, 21)
        measured_points_y = np.linspace(-0.5, 0.5, 11)
        measure_front_half = True

        terminate_end_of_trajectory = False

        # settings for random_pyramid
        class top(PrefixProto, cli=False):
            pyramid_num_x=3
            pyramid_num_y=5
            pyramid_var_x=0.5
            pyramid_var_y=0.3
            pyramid_length_min=0.2
            pyramid_length_max=0.4
            pyramid_height_min=0.2
            pyramid_height_max=0.4

        class bottom(PrefixProto, cli=False):
            pyramid_num_x=3
            pyramid_num_y=5
            pyramid_var_x=0.5
            pyramid_var_y=0.3
            pyramid_length_min=0.2
            pyramid_length_max=0.4
            pyramid_height_min=0.2
            pyramid_height_max=0.4


    class commands(PrefixProto, cli=False):
        switch_upon_reach = True  # switch waypoint when current waypoint is reached
        switch_interval = 0.5  # if switch_upon_reach is False, switch every plan_interval seconds
        traj_function = "fixed_target"  # in ["fixed_target", "random_target"]
        traj_length = 1
        num_interpolation = 1
        # fixed target parameqters
        base_x = 5.0
        base_y = 0.0
        base_z = 0.34
        base_roll = 0.0
        base_pitch = 0.0
        base_yaw = 0.0
        # random target parameters
        x_range = 0.5
        y_range = 0.5
        z_range = 0.1  # 0.1
        roll_range = 30 * np.pi / 180
        pitch_range = 30 * np.pi / 180
        yaw_range = 180 * np.pi / 180
        # random goal parameters
        x_mean = 3.6
        x_range = 0.4
        y_mean = 3.6
        y_eang = 0.4
        # for inference vel obs
        global_reference = False
        switch_dist = 0.05
        switch_yaw = 0.5

        sampling_based_planning = False
        plan_interval = 10  # replan every plan_interval steps
        candidate_target_poses = np.stack(np.meshgrid(
            np.linspace(0.5, 0.5, 1), # x
            # np.array([0, -0.15, 0.15, -0.3, 0.3]), # y
            np.array([0, -0.15, +0.15, -0.3, 0.3, -0.45, 0.45]), # y
            np.array([0.29, 0.27, 0.31, 0.25, 0.23]), # z
            np.array([0, -15, +15]) * np.pi / 180, # roll
            np.array([0, -15, +15]) * np.pi / 180, # pitch
            np.array([0, -22.5, +22.5, -45, +45]) * np.pi / 180, # yaw
        ), axis=-1).reshape(-1, 6)  # (num_cands=1125, xyz+rotation=6) in robot frame  

    class curriculum_thresholds(PrefixProto, cli=False):
        cl_fix_target = False
        cl_start_target_dist = 0.5
        cl_goal_target_dist = 3.6
        cl_switch_delta = 0.5
        cl_switch_threshold = 1.0

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(PrefixProto, cli=False):
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(PrefixProto, cli=False):
        rand_interval_s = 10
        randomize_motor_strength = True
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_rigids_after_start = True
        randomize_friction = True
        friction_range = [0.5, 1.25]  # increase range
        randomize_restitution = False
        restitution_range = [0, 1.0]
        randomize_base_mass = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1., 1.]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6

    class rewards(PrefixProto, cli=False):
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 5
        
        reward_container_name = "RewardsCrawling"
        # parameters for reward functions
        target_lin_vel = 0.5  # [m/s]
        lin_reaching_criterion = 0.1  # [m]
        tracking_sigma_lin = 0.10
        target_ang_vel = np.pi / 4.0  # [rad/s]
        ang_reaching_criterion = np.pi / 10.
        tracking_sigma_ang = 0.5
        use_terminal_body_height = True
        terminal_body_height = 0.1

    class reward_scales(ParamsProto, cli=False):
        torques = -0.00001  # -0.0002
        dof_acc = -2.5e-7
        collision = -1.
        action_rate = -0.01
        reaching_linear_vel = 0.0  # 0.6
        reaching_z = 0.0
        # reaching_roll = -0.5
        # reaching_pitch = -0.5
        reaching_yaw = 0.0  # 0.3
        # dof_pos_limits = -10.0

    class normalization(PrefixProto, cli=False):
        clip_observations = 100.
        clip_actions = 100.

        friction_range = [0.05, 4.5]
        ground_friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales(PrefixProto, cli=False):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 1.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise(PrefixProto, cli=False):
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales(PrefixProto, cli=False):
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(PrefixProto, cli=False):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        class physx(PrefixProto, cli=False):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
