import copy
import os
import pickle
import shutil
import time
from collections import deque

import numpy as np
import torch
# from ml_logger import logger
import wandb
from params_proto import PrefixProto

from .actor_critic import ActorCritic, AC_Args
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo.metrics_caches import DistCache, SlotCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100  # save video every this many iterations
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Runner:

    def __init__(self, env, device='cpu', runner_args=RunnerArgs, ac_args=AC_Args, log_wandb=False):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.runner_args = runner_args
        self.log_wandb = log_wandb

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)
        if log_wandb:
            save_path = os.path.join(wandb.run.dir, "parameters.pkl")
            pickle.dump(vars(self.env.cfg), open(save_path, "wb"))
            wandb.save(save_path)

        if runner_args.resume:
            # load pretrained weights from resume_path
            # from ml_logger import ML_Logger

            #loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
            #                   prefix=RunnerArgs.resume_path)
            # TODO: fix this later, not sure how to load from the WandB server
            # weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            weights = torch.load(runner_args.resume)
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and runner_args.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = runner_args.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -1000

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False, update_model=True):
        log_wandb = self.log_wandb
        # initialize writer
        # assert logger.prefix, "you will overwrite the entire instrument server"

        # logger.start('start', 'epoch', 'episode', 'run', 'step')

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = num_learning_iterations
        very_start = time.time()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        for it in range(tot_iter):
            start = time.time()
            # Rollout
            """ if torch.sum(self.env.reached_env_buf.float()) > 0:
                success_collision = torch.sum(self.env.collision_env_buf.float() * self.env.reached_env_buf.float()) / torch.sum(self.env.reached_env_buf.float())
                success_collision = success_collision.item()
            else:
                success_collision = "NaN" """
            # print(it * self.num_steps_per_env, "success: ", torch.mean(self.env.reached_env_buf.float()).item(), " collision", success_collision)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    # print(actions_train[0])
                    # import ipdb; ipdb.set_trace()
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    if update_model:
                        self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos and log_wandb:
                        # with logger.Prefix(metrics="train/episode"):
                        #     logger.store_metrics(**infos['train/episode'])
                        info = infos['train/episode']
                        metrics = {k: np.mean(v) for k, v in info.items()}
                        metrics["fps"] = (it+1) * self.env.cfg.env.num_envs * self.num_steps_per_env / (time.time() - very_start)
                        wandb.log({"train": metrics}, step=it * 24 + i)

                    if 'eval/episode' in infos and log_wandb:
                        # with logger.Prefix(metrics="eval/episode"):
                        #     logger.store_metrics(**infos['eval/episode'])
                        info = infos['eval/episode']
                        metrics = {k: np.mean(v) for k, v in info.items()}
                        wandb.log({"eval": metrics}, step=it * 24 + i)

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if update_model:
                    self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0 and log_wandb:
                    if not os.path.exists(os.path.join(wandb.run.dir, "curriculum")):
                        os.makedirs(os.path.join(wandb.run.dir, "curriculum"))
                    save_path = os.path.join(wandb.run.dir, "curriculum/info.pkl")
                    pickle.dump({"iteration": it,
                                    **caches.slot_cache.get_summary(),
                                    **caches.dist_cache.get_summary()},
                                    open(save_path, "wb"))
                    wandb.save(save_path)

                    if 'curriculum/distribution' in infos:
                        save_path = os.path.join(wandb.run.dir, "curriculum/distribution.pkl")
                        pickle.dump({"iteration": it,
                                        "distribution": distribution},
                                        open(save_path, "wb"))
                        wandb.save(save_path)

            if update_model:
                mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = 0, 0, 0, 0, 0, 0, 0, 0

            stop = time.time()
            learn_time = stop - start

            store_metrics = dict(
                # total_time=learn_time - collection_time,
                # time_elapsed=logger.since('start'),
                # time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )
            if log_wandb:
                wandb.log({"metrics": store_metrics}, step=it * 24 + i)

            if self.runner_args.save_video_interval and log_wandb:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            #if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
            #    logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
            #    logger.job_running()

            if it % self.runner_args.save_interval == 0:
                # with logger.Sync():
                if log_wandb:
                    save_path = os.path.join(wandb.run.dir, "checkpoints")
                else:
                    save_path = f"last_run/checkpoints"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.alg.actor_critic.state_dict(), os.path.join(save_path, "ac_weights.pt"))
                # shutil.copyfile(f"{wandb.run.dir}/checkpoints/ac_weights_{it:06d}.pt", f"{wandb.run.dir}/checkpoints/ac_weights_last.pt")

                adaptation_module_path = f'{save_path}/adaptation_module_latest.jit'
                adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                traced_script_adaptation_module = torch.jit.script(adaptation_module)
                traced_script_adaptation_module.save(adaptation_module_path)

                body_path = f'{save_path}/body_latest.jit'
                body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                traced_script_body_module = torch.jit.script(body_model)
                traced_script_body_module.save(body_path)

                if log_wandb:
                    wandb.save(os.path.join(save_path, "ac_weights.pt"))
                    wandb.save(adaptation_module_path)
                    wandb.save(body_path)

       
        torch.save(self.alg.actor_critic.state_dict(), os.path.join(save_path, "ac_weights.pt"))
        # shutil.copyfile(f"{wandb.run.dir}/checkpoints/ac_weights_{it:06d}.pt", f"{wandb.run.dir}/checkpoints/ac_weights_last.pt")

        os.makedirs(save_path, exist_ok=True)

        adaptation_module_path = f'{save_path}/adaptation_module_latest.jit'
        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
        traced_script_adaptation_module = torch.jit.script(adaptation_module)
        traced_script_adaptation_module.save(adaptation_module_path)

        body_path = f'{save_path}/body_latest.jit'
        body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(body_path)

        if log_wandb:
            wandb.save(os.path.join(save_path, "ac_weights.pt"))
            wandb.save(adaptation_module_path)
            wandb.save(body_path)


    def log_video(self, it):
        if it - self.last_recording_it >= self.runner_args.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            frames_np = np.stack(frames).transpose(0, 3, 1, 2)
            wandb.log({"train": {"video": wandb.Video(frames_np, fps=1 / self.env.dt, format="mp4")}}, step=it * 24)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                # logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)
                frames_np = np.stack(frames).transpose(0, 3, 1, 2)
                wandb.log({"eval": {"video": wandb.Video(frames_np, fps=1 / self.env.dt, format="mp4")}}, step=it * 24)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
