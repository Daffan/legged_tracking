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

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs

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
    Cfg.env.num_observations = 507
    Cfg.env.num_scalar_observations = 507
    Cfg.env.num_observation_history = 1

    # control
    Cfg.control.control_type = 'P'

    # rewards
    Cfg.rewards.T_reach = 200
    Cfg.rewards.small_vel_threshold = 0.1
    Cfg.rewards.large_dist_threshold = 0.5

    Cfg.reward_scales.torques = -0.00001  # -0.0002
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.collision = -1.
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.reaching_linear_vel = 1.2  # 0.6
    Cfg.reward_scales.reaching_z = -10.0
    Cfg.reward_scales.reaching_yaw = 0.6  # 0.3


    # terrain
    # Cfg.env.num_envs = 4000
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_ratio_y = 1.0

    # goal
    Cfg.commands.traj_function = "fixed_target"
    Cfg.commands.traj_length = 1
    Cfg.commands.num_interpolation = 1
    Cfg.commands.base_x = 6.0
    Cfg.commands.sampling_based_planning = True
    Cfg.commands.plan_interval = 10

    env = TrajectoryTrackingEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    """ 
    import time
    start = time.time()
    for i in range(100):
        print(i, end='\r')
        action = torch.zeros(4000, 12).to(env.device)
        env.step(action)
    print(1000 * 4000 / (time.time() - start))
    import ipdb; ipdb.set_trace() 
    """

    # log the experiment parameters
    # logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
    #                   Cfg=vars(Cfg))
    log_wandb = True
    if log_wandb:
        wandb.init(project="go1_gym", config=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}", runner_args=RunnerArgs)
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100, log_wandb=log_wandb)


if __name__ == '__main__':
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    # logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
    #                  root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    '''
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)
    '''

    # to see the environment rendering, set headless=False
    train_go1(headless=False)
