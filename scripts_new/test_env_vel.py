def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch
    import wandb
    from params_proto import PrefixProto, ParamsProto

    from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.trajectory_tracking import TrajectoryTrackingEnv
  
    # from ml_logger import logger

    from go1_gym_learn.ppo_cse_cnn import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
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

    config_go1(Cfg)
    # observation space
    Cfg.env.observe_heights = True
    Cfg.env.num_envs = 4000
    
    command_type = args.command_type  # ["xy, "6dof", "xy_norm"]
    if command_type in ["xy", "xy_norm"]:
        Cfg.env.command_type = command_type
        Cfg.env.num_observations = 261 + int(args.timestep_in_obs)
        Cfg.env.num_scalar_observations = 261 + int(args.timestep_in_obs)
    else:
        Cfg.env.command_type = command_type
        Cfg.env.num_observations = 265 + 1 # 507  (consider height meaurement only at front)
        Cfg.env.num_scalar_observations = 265 + 1  # 507
    Cfg.terrain.measured_points_x = np.linspace(-1, 1, 21)
    Cfg.terrain.measured_points_y = np.linspace(-0.5, 0.5, 11)
    Cfg.env.num_observation_history = args.num_history
    Cfg.env.look_from_back = True
    Cfg.env.terminate_end_of_trajectory = False
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
    Cfg.rewards.T_reach = 0
    Cfg.rewards.small_vel_threshold = 0.1
    Cfg.rewards.large_dist_threshold = 0.5
    Cfg.rewards.only_positive_rewards = args.only_positive
    Cfg.rewards.use_terminal_body_height = False

    # Cfg.reward_scales.stalling = args.r_stalling
    Cfg.reward_scales.reaching_yaw = 0
    Cfg.reward_scales.linear_vel = 0.0  # penalize large linear velocity > 0.7 m/s
    Cfg.reward_scales.reaching_yaw_abs = -0.0
    Cfg.reward_scales.reach_goal_t = 0.0  # 100
    Cfg.reward_scales.reach_goal = 0.0  # 100
    # Cfg.reward_scales.reaching_z = -5.0
    Cfg.reward_scales.exploration = args.r_explore
    Cfg.rewards.exploration_steps = 100000000  # always explore

    Cfg.reward_scales.reaching_linear_vel = args.r_linear_vel
    Cfg.rewards.target_lin_vel = 0.5
    Cfg.rewards.lin_vel_z = -2.0
    Cfg.rewards.ang_vel_xy = -0.05
    Cfg.rewards.lin_vel_form = args.lin_vel_form

    penalty_scaler = args.penalty_scaler
    Cfg.reward_scales.dof_acc = -2.5e-7 * penalty_scaler
    Cfg.reward_scales.torques = -1e-5 * penalty_scaler
    Cfg.reward_scales.dof_pos_limits = -10.0 * penalty_scaler
    Cfg.reward_scales.collision = -1.0 * penalty_scaler
    Cfg.reward_scales.action_rate = -0.01 * penalty_scaler
    # Cfg.reward_scales.feet_air_time = 1.0 * penalty_scaler
    Cfg.reward_scales.orientation = 0.0  # -5.0
    Cfg.reward_scales.reaching_z = 0.0
    Cfg.reward_scales.base_height = 0.0

    # terrain
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    if args.terrain == "plane":
        Cfg.terrain.mesh_type = 'plane'
    if args.terrain == "test_1":
        # By default random pyramid terrain
        Cfg.terrain.terrain_type = "test_env"
        Cfg.terrain.terrain_length = 4.0
        Cfg.terrain.terrain_width = 1.5
        Cfg.terrain.terrain_ratio_x = 0.5
        Cfg.terrain.terrain_ratio_y = 1.0
    elif args.terrain == "test_2":
        Cfg.terrain.terrain_type = "test_env_2"
        Cfg.terrain.horizontal_scale = 0.01
        Cfg.terrain.terrain_length = 4.0
        Cfg.terrain.terrain_width = 2.0
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8
    elif args.terrain == "test_3":
        Cfg.terrain.num_cols = 20
        Cfg.terrain.num_rows = 1
        Cfg.terrain.terrain_type = "test_env_3"
        Cfg.terrain.horizontal_scale = 0.01
        Cfg.terrain.terrain_length = 4.0 * 3
        Cfg.terrain.terrain_width = 2.0
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8

    elif args.terrain == "test_4":
        Cfg.terrain.num_cols = 20
        Cfg.terrain.num_rows = 1
        Cfg.terrain.terrain_type = "test_env_4"
        Cfg.terrain.horizontal_scale = 0.01
        Cfg.terrain.terrain_length = 4.0 * 3
        Cfg.terrain.terrain_width = 1.8
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8

    elif args.terrain == "test_5":
        Cfg.terrain.num_cols = 20
        Cfg.terrain.num_rows = 1
        Cfg.terrain.terrain_type = "test_env_5"
        Cfg.terrain.horizontal_scale = 0.01
        Cfg.terrain.terrain_length = 4.0 * 3
        Cfg.terrain.terrain_width = 1.8
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8

    elif args.terrain == "test_6":
        Cfg.terrain.num_cols = 20
        Cfg.terrain.num_rows = 20
        Cfg.terrain.terrain_type = "test_env_6"
        Cfg.terrain.horizontal_scale = 0.01
        Cfg.terrain.terrain_length = 3.0
        Cfg.terrain.terrain_width = 1.8
        Cfg.terrain.terrain_ratio_x = 0.9
        Cfg.terrain.terrain_ratio_y = 0.25
        Cfg.terrain.ceiling_height = 0.8
        Cfg.env.episode_length_s = 8.0

    if Cfg.terrain.horizontal_scale == 0.01:
        Cfg.terrain.measured_points_x = np.linspace(-0.6, 0.6, 61)
        Cfg.terrain.measured_points_y = np.linspace(-0.3, 0.3, 31)
        Cfg.env.measure_front_half = False
        if command_type in ["xy", "xy_norm"]:
            Cfg.env.command_type = command_type
            Cfg.env.num_observations = 45 + int(args.timestep_in_obs) + 61 * 31 * 2 - 4
            Cfg.env.num_scalar_observations = 45 + int(args.timestep_in_obs) + 61 * 31 * 2 - 4
        else:
            Cfg.env.command_type = command_type
            Cfg.env.num_observations = 45 + int(args.timestep_in_obs) + 61 * 31 * 2 + 4
            Cfg.env.num_scalar_observations = 45 + int(args.timestep_in_obs) + 61 * 31 * 2 + 4

    # goal
    Cfg.commands.base_z = 0.29
    if args.random_target:
        Cfg.commands.traj_function = "random_target"
        Cfg.commands.traj_length = 10
        Cfg.commands.num_interpolation = 1
        Cfg.commands.base_x = 4.0
        Cfg.commands.sampling_based_planning = False
        Cfg.commands.plan_interval = 10
    else:
        Cfg.commands.traj_function = "fixed_target"
        Cfg.commands.traj_length = 1
        Cfg.commands.num_interpolation = 1
        Cfg.commands.base_x = Cfg.terrain.terrain_length * Cfg.terrain.terrain_ratio_x - 1.0
        Cfg.commands.sampling_based_planning = False
        Cfg.commands.plan_interval = 100
    Cfg.commands.traj_length = 1
    Cfg.commands.num_interpolation = 1
    Cfg.commands.x_mean = Cfg.terrain.terrain_length * Cfg.terrain.terrain_ratio_x - 1.0
    Cfg.commands.y_mean = 0.0
    Cfg.commands.x_range = 0.2
    Cfg.commands.y_range = 0.0
    Cfg.commands.switch_dist = 0.6
    if args.start_target_dist > 0:
        Cfg.curriculum_thresholds.cl_fix_target = True
        Cfg.curriculum_thresholds.cl_start_target_dist = args.start_target_dist
        Cfg.curriculum_thresholds.cl_goal_target_dist = 3.2
        Cfg.curriculum_thresholds.cl_switch_delta = 0.2
        Cfg.curriculum_thresholds.cl_switch_threshold = 0.4
    else:
        Cfg.curriculum_thresholds.cl_fix_target = False

    
    RunnerArgs.save_video_interval = 500
    RunnerArgs.resume = args.resume
    gpu_id = args.device
    env = TrajectoryTrackingEnv(sim_device=f"cuda:{gpu_id}", headless=args.headless, cfg=Cfg)

    if args.wandb:
        wandb.init(project="go1_gym", config=vars(Cfg), name=args.name)

    env = HistoryWrapper(env)

    """ import time
    start = time.time()
    for i in range(100):
        print(i, end='\r')
        action = torch.rand(4000, 12).to("cuda:0")
        env.step(action)
    print(1000 * 4000 / (time.time() - start)) """

    runner = Runner(env, device=f"cuda:{gpu_id}", runner_args=RunnerArgs, ac_args=AC_Args, log_wandb=args.wandb)
    runner.learn(num_learning_iterations=5000000, init_at_random_ep_len=True, eval_freq=100, update_model=not args.freeze_model)


if __name__ == '__main__':
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--terrain", default="pyramid", choices=["pyramid", "plane", "test_1", "test_2", "test_3", "test_4", "test_5", "test_6"])
    parser.add_argument("--random_target", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default="e2e")
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--r_explore", type=float, default=1.0)
    parser.add_argument("--r_stalling", type=float, default=1.0)
    parser.add_argument("--freeze_model", action="store_true")
    parser.add_argument("--start_target_dist", type=float, default=0.0)
    parser.add_argument("--penalty_scaler", type=float, default=1.0)
    parser.add_argument("--only_positive", action="store_true")
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--timestep_in_obs", action="store_true")
    parser.add_argument("--num_history", type=int, default=5)
    parser.add_argument("--measure_front_half", action="store_true")
    parser.add_argument("--rotate_camera", action="store_true")
    parser.add_argument("--camera_zero", action="store_true")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--command_type", default="xy", choices=["xy", "6dof", "xy_norm"])
    parser.add_argument("--lin_vel_form", default="exp", choices=["l1", "l2", "exp"])
    parser.add_argument("--r_linear_vel", type=float, default=1.0)

    args = parser.parse_args()

    stem = Path(__file__).stem
    # to see the environment rendering, set headless=False
    train_go1(args)