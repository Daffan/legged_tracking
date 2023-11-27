def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch
    import wandb
    from params_proto import PrefixProto, ParamsProto

    from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
    from go1_gym.envs.go1.go1_crawling import config_go1
    from go1_gym.envs.go1.trajectory_tracking import TrajectoryTrackingEnv
  
    # from ml_logger import logger

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    if args.old_ppo:
        from go1_gym_learn.ppo_cse import Runner
        from go1_gym_learn.ppo_cse.actor_critic import AC_Args
        from go1_gym_learn.ppo_cse.ppo import PPO_Args
        from go1_gym_learn.ppo_cse import RunnerArgs
    else:
        from go1_gym_learn.ppo_cse_cnn import Runner
        from go1_gym_learn.ppo_cse_cnn.actor_critic import AC_Args
        from go1_gym_learn.ppo_cse_cnn.ppo import PPO_Args
        from go1_gym_learn.ppo_cse_cnn import RunnerArgs

    import random
    import numpy as np
    import torch
    import os

    seed = 11

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    AC_Args.use_gru = args.gru
    AC_Args.use_cnn = args.cnn
    AC_Args.normalize_obs = args.normalize_obs

    config_go1(Cfg)
    # observation space
    Cfg.env.observe_heights = True
    
    command_type = args.command_type  # ["xy, "6dof", "xy_norm"]
    if command_type in ["xy", "xy_norm"]:
        Cfg.env.command_type = command_type
        Cfg.env.num_observations = 261 if args.measure_front_half else 503
        Cfg.env.num_observations += int(args.timestep_in_obs)
        Cfg.env.num_scalar_observations = Cfg.env.num_observations
    else:
        Cfg.env.command_type = command_type
        Cfg.env.num_observations = 265 if args.measure_front_half else 507
        Cfg.env.num_observations += int(args.timestep_in_obs)
        Cfg.env.num_scalar_observations = Cfg.env.num_observations
    Cfg.env.num_privileged_obs = 2

    Cfg.terrain.measured_points_x = np.linspace(-1, 1, 21)
    Cfg.terrain.measured_points_y = np.linspace(-0.5, 0.5, 11)
    AC_Args.height_map_shape = (2, 21, 11)

    Cfg.env.num_observation_history = args.num_history
    Cfg.env.look_from_back = True
    Cfg.env.viewer_look_at_robot = False
    Cfg.env.terminate_end_of_trajectory = args.terminate_after_reach
    Cfg.env.record_all_envs = False
    Cfg.env.episode_length_s = 20
    Cfg.env.rotate_camera = args.rotate_camera
    Cfg.env.camera_zero = args.camera_zero
    Cfg.env.timestep_in_obs = args.timestep_in_obs
    Cfg.terrain.measure_front_half = args.measure_front_half

    # asset
    # change to not terminate on, but just penalize base contact, 
    Cfg.asset.penalize_contacts_on = ["thigh", "calf", "base"]
    Cfg.asset.terminate_after_contacts_on = []

    # rewards
    Cfg.rewards.small_vel_threshold = 0.1
    Cfg.rewards.lin_reaching_criterion = 0.3
    Cfg.rewards.ang_reaching_criterion = np.pi / 20.0
    Cfg.rewards.only_positive_rewards = args.only_positive
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = args.terminal_body_height
    Cfg.rewards.lin_vel_form = args.lin_vel_form
    Cfg.rewards.exploration_steps = +np.inf
    Cfg.rewards.tracking_sigma_lin = 0.05
    Cfg.rewards.base_height_target = 0.28
    Cfg.rewards.target_lin_vel = 0.25

    # penalty reward scales
    penalty_scaler = args.penalty_scaler
    Cfg.reward_scales.dof_acc = -2.5e-7 * penalty_scaler
    Cfg.reward_scales.torques = -1e-5 * penalty_scaler
    Cfg.reward_scales.action_rate = -1e-3 * penalty_scaler
    Cfg.reward_scales.dof_pos_limits = -10.0 * penalty_scaler
    Cfg.reward_scales.collision = -args.r_collision * penalty_scaler
    Cfg.reward_scales.base_height = -args.r_base_height * penalty_scaler
    Cfg.reward_scales.orientation = -args.r_orientation * penalty_scaler
    Cfg.reward_scales.ang_vel_xy = -args.r_ang_vel * penalty_scaler
    Cfg.reward_scales.large_vel = -args.r_large_vel * penalty_scaler
    # task reward scales
    Cfg.reward_scales.reaching_z = 0.0
    Cfg.reward_scales.reaching_roll = 0.0
    Cfg.reward_scales.reaching_pitch = 0.0
    if args.strategy == "vel":
        Cfg.reward_scales.e2e = 0
        Cfg.rewards.T_reach = args.t_reach
        Cfg.rewards.exploration_steps = 200000
    if args.strategy == "e2e":
        Cfg.reward_scales.e2e = args.r_task
        Cfg.rewards.T_reach = args.t_reach
        Cfg.rewards.exploration_steps = args.exploration_steps  # decay only applies for e2e
    elif args.strategy == "pms":  # parameterized motor skill
        # TODO: put a positive number here
        Cfg.reward_scales.reaching_z = 0.0
        Cfg.reward_scales.reaching_roll = 0.0
        Cfg.reward_scales.reaching_pitch = 0.0
    Cfg.reward_scales.exploration_lin = args.r_explore_lin
    Cfg.reward_scales.exploration_yaw = args.r_explore_yaw

    # terrain
    Cfg.env.num_envs = 1024
    Cfg.terrain.num_cols = 32
    Cfg.terrain.num_rows = 32
    if args.terrain == "plane":
        Cfg.terrain.mesh_type = 'plane'
    elif args.terrain == "single_path":
        Cfg.terrain.terrain_type = "single_path"
        Cfg.terrain.terrain_length = 4.0
        Cfg.terrain.terrain_width = 2.0
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.5
        Cfg.terrain.ceiling_height = 0.8
        Cfg.terrain.start_loc = 0.32
        Cfg.terrain.p_flat = 0.0 if args.empty_tunnel else 0.9
        Cfg.terrain.p_double = 0.6
        Cfg.env.episode_length_s = 10.0
        # single path do not need planning
        Cfg.commands.sampling_based_planning = False

    elif args.terrain == "multi_path":
        Cfg.terrain.terrain_type = "multi_path"
        Cfg.terrain.terrain_length = 3.0
        Cfg.terrain.terrain_width = args.tunnel_width
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8
        Cfg.env.episode_length_s = 8.0
        Cfg.terrain.start_loc = 0.4
        # multi-path needs planning
        Cfg.commands.sampling_based_planning = True
        Cfg.commands.plan_interval = 100
        
    if args.random_target:
        Cfg.commands.traj_function = "random_target"
        Cfg.commands.traj_length = 10
        Cfg.commands.num_interpolation = 1
        Cfg.commands.sampling_based_planning = False
    else:
        Cfg.commands.traj_function = "fixed_target"
        Cfg.commands.traj_length = 1
        Cfg.commands.num_interpolation = 1
        Cfg.commands.switch_dist = 0.3
        Cfg.commands.base_x = Cfg.terrain.terrain_length * Cfg.terrain.terrain_ratio_x - 1.0

    if args.blind:
        Cfg.env.observe_heights = False
        # Cfg.terrain.measured_points_x = np.linspace(-0.6, 0.6, 1)
        # Cfg.terrain.measured_points_y = np.linspace(-0.3, 0.3, 1)
        # AC_Args.height_map_shape = (2, 1, 1)
        Cfg.env.measure_front_half = False
        if command_type in ["xy", "xy_norm"]:   
            Cfg.env.command_type = command_type
            Cfg.env.num_observations = 45 + int(args.timestep_in_obs) - 4
            Cfg.env.num_scalar_observations = 45 + int(args.timestep_in_obs) - 4
        else:
            Cfg.env.command_type = command_type
            Cfg.env.num_observations = 45 + int(args.timestep_in_obs) + 2 + 4
            Cfg.env.num_scalar_observations = 45 + int(args.timestep_in_obs) + 2 + 4

    # domain randomization stuff
    enable_random = not args.no_domain_rand
    #    Cfg.domain_rand.push_robots = False

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = enable_random
    Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [0.1, 3.0]
    Cfg.domain_rand.randomize_restitution = enable_random
    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = enable_random
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = enable_random
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = enable_random
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = enable_random
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = enable_random
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.normalization.clip_actions = 10.0
    
    RunnerArgs.save_video_interval = 500
    RunnerArgs.resume = args.resume
    gpu_id = args.device
    env = TrajectoryTrackingEnv(sim_device=f"cuda:{gpu_id}", headless=args.headless, cfg=Cfg)
    env.reset()

    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.gamma = args.gamma

    # if logdir does not exist, create one
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if args.wandb:
        wandb.init(
            project="go1_gym",
            config=vars(Cfg),
            name=args.name,
            dir=args.logdir
        )

    env = HistoryWrapper(env)

    """ import time
    print("Is recording?", env.record_now)
    env.pause_recording()
    start = time.time()
    for i in range(100):
        print(i, end='\r')
        action = torch.rand(Cfg.env.num_envs, 12).to("cuda:0") / 5.0
        env.step(action)
    print(100 * Cfg.env.num_envs / (time.time() - start))
    import ipdb; ipdb.set_trace() """

    runner = Runner(env, device=f"cuda:{gpu_id}", runner_args=RunnerArgs, ac_args=AC_Args, log_wandb=args.wandb)
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True, eval_freq=100, update_model=not args.freeze_model)


if __name__ == '__main__':
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import argparse
    import os

    parser = argparse.ArgumentParser()
    # user setting
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default="velocity_tracking")
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--freeze_model", action="store_true")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--logdir", type=str, default=os.path.join(MINI_GYM_ROOT_DIR, "logs", "go1"))
    parser.add_argument("--strategy", default="vel", choices=["e2e", "pms", "vel"])

    # training setting
    parser.add_argument("--old_ppo", action="store_true")
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration_steps", type=int, default=2500)
    parser.add_argument("--normalize_obs", action="store_true")
    parser.add_argument("--num_steps_per_env", type=int, default=24)

    # env setting
    parser.add_argument("--command_type", default="xy", choices=["xy", "6dof", "xy_norm"])
    parser.add_argument("--timestep_in_obs", action="store_true")
    parser.add_argument("--num_history", type=int, default=1)
    parser.add_argument("--measure_front_half", action="store_true")
    parser.add_argument("--rotate_camera", action="store_true")
    parser.add_argument("--camera_zero", action="store_true")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--terminal_body_height", type=float, default=0.0)
    parser.add_argument("--terrain", default="single_path", choices=["single_path", "multi_path", "plane"])
    parser.add_argument("--no_domain_rand", action="store_true")
    parser.add_argument("--empty_tunnel", action="store_true")
    parser.add_argument("--random_target", action="store_true")
    # add boolen flag for terminate after reach target
    parser.add_argument("--terminate_after_reach", action="store_true")


    # reward setting
    parser.add_argument("--lin_vel_form", default="exp", choices=["l1", "l2", "exp", "prod"])
    parser.add_argument("--r_explore_lin", type=float, default=1.0)
    parser.add_argument("--r_explore_yaw", type=float, default=0.4)
    parser.add_argument("--penalty_scaler", type=float, default=1.0)
    parser.add_argument("--only_positive", action="store_true")
    parser.add_argument("--r_orientation", type=float, default=0.0)
    parser.add_argument("--r_base_height", type=float, default=20.0)
    parser.add_argument("--r_ang_vel", type=float, default=0.001)
    parser.add_argument("--t_reach", type=int, default=0, help="time step to assign the task reward")
    parser.add_argument("--r_task", type=float, default=1.0)
    parser.add_argument("--r_collision", type=float, default=5.0)
    parser.add_argument("--r_large_vel", type=float, default=0.0)


    args = parser.parse_args()

    stem = Path(__file__).stem
    # to see the environment rendering, set headless=False
    train_go1(args)