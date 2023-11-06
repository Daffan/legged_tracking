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

    config_go1(Cfg)
    # observation space
    Cfg.terrain.measured_points_x = np.linspace(-1, 1, 21)
    Cfg.terrain.measured_points_y = np.linspace(-0.5, 0.5, 11)
    Cfg.env.observe_heights = True
    Cfg.env.num_envs = 4000
    
    command_xy_only = True
    if command_xy_only:
        Cfg.env.command_xy_only = True
        Cfg.env.num_observations = 261
        Cfg.env.num_scalar_observations = 261
    else:
        Cfg.env.command_xy_only = False
        Cfg.env.num_observations = 265  # 507  (consider height meaurement only at front)
        Cfg.env.num_scalar_observations = 265  # 507
    Cfg.env.num_observation_history = 5
    Cfg.env.look_from_back = True
    Cfg.env.terminate_end_of_trajectory = True
    Cfg.env.episode_length_s = 20
    Cfg.env.rotate_camera = False
    Cfg.env.camera_zero = True
    Cfg.terrain.measure_front_half = True

    # asset
    # change to not terminate on, but just penalize base contact, 
    Cfg.asset.penalize_contacts_on = ["thigh", "calf", "base"]
    Cfg.asset.terminate_after_contacts_on = []

    # rewards
    Cfg.rewards.T_reach = 200
    Cfg.rewards.small_vel_threshold = 0.1
    Cfg.rewards.large_dist_threshold = 0.5
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.use_terminal_body_height = False

    # Cfg.reward_scales.stalling = args.r_stalling
    Cfg.reward_scales.reaching_linear_vel = 0
    Cfg.reward_scales.reaching_yaw = 0
    Cfg.reward_scales.linear_vel = -1.0  # penalize large linear velocity > 0.7 m/s
    Cfg.reward_scales.reaching_yaw_abs = -0.1
    Cfg.reward_scales.reach_goal = 100
    # Cfg.reward_scales.reaching_z = -5.0
    Cfg.reward_scales.exploration = args.r_explore
    Cfg.rewards.exploration_steps = 100000000  # always explore

    Cfg.reward_scales.dof_acc = -2.5e-7 * 5
    Cfg.reward_scales.torques = -1e-5 * 2
    Cfg.reward_scales.dof_pos_limits = -10.0 * 2
    Cfg.reward_scales.collision = -1.0
    Cfg.reward_scales.action_rate = -0.05
    Cfg.reward_scales.orientation = 0.0  # -5.0
    Cfg.reward_scales.reaching_z = 0.0
    Cfg.reward_scales.base_height = 0.0

    # terrain
    if args.no_tunnel:
        Cfg.terrain.mesh_type = 'plane'
    else:
        # By default random pyramid terrain
        Cfg.terrain.terrain_type = "test_env"

        Cfg.terrain.num_cols = 20
        Cfg.terrain.num_rows = 20
        Cfg.terrain.terrain_length = 4.0
        Cfg.terrain.terrain_width = 3.0
        Cfg.terrain.terrain_ratio_x = 0.5
        Cfg.terrain.terrain_ratio_y = 1.0

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
        Cfg.commands.traj_function = "valid_goal"
        Cfg.commands.traj_length = 1
        Cfg.commands.num_interpolation = 1
        Cfg.commands.base_x = 5.0
        Cfg.commands.sampling_based_planning = False
        Cfg.commands.plan_interval = 100
    Cfg.commands.traj_length = 1
    Cfg.commands.num_interpolation = 1
    Cfg.commands.x_mean = 5.0
    Cfg.commands.y_mean = 0.0
    Cfg.commands.x_range = 0.4
    Cfg.commands.y_range = 0.0
    Cfg.commands.switch_dist = 0.3
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
    env = TrajectoryTrackingEnv(sim_device='cuda:0', headless=args.headless, cfg=Cfg)

    if args.wandb:
        wandb.init(project="go1_gym", config=vars(Cfg), name=args.name)

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}", runner_args=RunnerArgs, log_wandb=args.wandb)
    runner.learn(num_learning_iterations=1000000000, init_at_random_ep_len=True, eval_freq=100, update_model=not args.freeze_model)


if __name__ == '__main__':
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_tunnel", action="store_true")
    parser.add_argument("--random_target", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default="e2e")
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--r_explore", type=float, default=1.0)
    parser.add_argument("--r_stalling", type=float, default=1.0)
    parser.add_argument("--freeze_model", action="store_true")
    parser.add_argument("--start_target_dist", type=float, default=0.6)
    args = parser.parse_args()

    stem = Path(__file__).stem
    # to see the environment rendering, set headless=False
    train_go1(args)