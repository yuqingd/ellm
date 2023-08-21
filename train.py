# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import time

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
import wandb

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # uncomment for exact reproducibility 
        # torch.set_default_dtype(torch.double)

        self.resume_run = False
        self.train_time = 0
        self.eval_time = 0
        self.agent_time = 0
        self.env_time = 0
        self.update_time = 0
        self.save_time = 0
        self.storage_time = 0
        self.log_time = 0
        self.env_spec = cfg.env_spec
        
        
        # create envs
        if 'Crafter' in self.cfg.env_spec.name:
            import crafter_env
            self.train_env = crafter_env.make(
                logdir=self.work_dir / 'train_episodes',
                env_spec=self.cfg.env_spec,
                save_video=False,
                frame_stack=cfg.frame_stack,
                seed=cfg.seed,
                env_reward=self.train_with_env_reward,
                use_wandb=cfg.use_wandb,
                debug=self.cfg.debug,
                device=cfg.device)

            self.eval_env = crafter_env.make(
                logdir=self.work_dir / 'eval_episodes',
                env_spec=self.cfg.env_spec,
                save_video=False,
                frame_stack=cfg.frame_stack,
                seed=cfg.seed + 1,
                env_reward=True,
                use_wandb=cfg.use_wandb,
                debug=self.cfg.debug,
                device=cfg.device)
        elif 'Housekeep' in self.cfg.env_spec.name: 
            import housekeep_env
            self.train_env = housekeep_env.make(
                logdir=self.work_dir / 'train_episodes',
                env_spec=self.cfg.env_spec,
                save_video=self.cfg.save_video,
                frame_stack=cfg.frame_stack,
                seed=cfg.seed,
                env_reward=self.train_with_env_reward,
                use_wandb=cfg.use_wandb,
                debug=self.cfg.debug,
                device=cfg.device)
            self.eval_env = self.train_env
        else:
            raise NotImplementedError

        # log action distributions for harder env 
        self.eval_actions_path = self.work_dir / 'eval_actions'
        self.eval_actions_path.mkdir(exist_ok=True, parents=True)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb, config=cfg, resume_id=cfg.resume_id)
        self._global_step = 0
        self._global_episode = 0
        if self.cfg.use_wandb:
            try:
                if hasattr(wandb.run, 'history_step'):
                    self._global_step = wandb.run.history_step
                elif hasattr(wandb.run, 'step'):
                    self._global_step = wandb.run.step
                else:
                    import pdb; pdb.set_trace()
            except:
                pass
        self.cumulative_train_reward = 0


        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      self.train_env.reward_spec(),
                      self.train_env.discount_spec())

        self.replay_storage = ReplayBufferStorage(self.train_env,
                                                  data_specs,
                                                  self.work_dir / 'buffer' if cfg.offline_buffer is None else Path.cwd().parents[1] / Path(cfg.offline_buffer + '/offline_buffer'),
                                                )

        if cfg.offline_buffer is None and cfg.use_offline_buffer:
            # for offline evals later, store episodes relabelled with environment reward
            self.offline_replay_storage = ReplayBufferStorage(self.train_env,
                                                    data_specs,
                                                    self.work_dir / 'offline_buffer',
                                                    )

        self.replay_loader = make_replay_loader(self.work_dir / 'buffer' if cfg.offline_buffer is None else Path.cwd().parents[1] / Path(cfg.offline_buffer + '/offline_buffer'),
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                cfg.save_buffer, cfg.nstep,
                                                cfg.discount, cfg.clip_reward, cfg.seed, cfg.frame_stack)
        self._replay_iter = None

        other_dim = np.sum(self.train_env.observation_spec().get('other', np.array(0)).shape)
        other_dim = int(other_dim)

        if self.train_env.action_spec().shape == ():
            ac_shape = self.train_env.action_spec().num_values
        else:
            ac_shape = self.train_env.action_spec().maximum
        
        self.agent = hydra.utils.instantiate(
            cfg.agent,
            obs_shape=self.train_env.observation_spec()['obs'].shape,
            num_actions=ac_shape,
            other_dim=other_dim)
        self.timer = utils.Timer()

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def pt_replay_iter(self):
        if self._pt_replay_iter is None:
            self._pt_replay_iter = iter(self.pt_replay_loader)
        return self._pt_replay_iter

    def eval_housekeep(self, save_video=False):
        # Eval, with the training configuration
        self.eval_env.set_env_reward(True)
        self.eval_env.update_dir(self.work_dir / 'eval_episodes', save_video)
        self.evaluate_env(self.eval_env, self.cfg.num_eval_episodes, 'eval', '', True)

        self.eval_env.reset()

    def eval_crafter(self, save_video=False, single_goal_only=False):

        #Single-goal eval
        if self.cfg.single_goal_eval:

            env_goals = list(self.train_env.goals_so_far.keys())
            video_goals = list(self.cfg.eval_video_goals)
            if self.cfg.env_spec.single_task:
                env_goals += [self.cfg.env_spec.single_task]
            all_goals = set(env_goals + video_goals)
            print("all goals", all_goals)
            for task in all_goals:
                if task in self.train_env.goal_compositions():
                    self.evaluate_goal_env(task)

        if single_goal_only:
            return

        # Eval with the training reward
        self.eval_env.set_env_reward(self.train_with_env_reward)
        self.eval_env.update_dir(self.work_dir / 'eval_episodes', False)
        self.eval_env.set_single_goal_hierarchical('reward_last')  # Set this so we only count achieving the true goal, not the intermediate goals
        self.evaluate_env(self.eval_env, self.cfg.num_eval_episodes, 'eval_extended', 'all_achievements', False)
        self.eval_env.set_single_goal_hierarchical(self.cfg.env_spec.single_goal_hierarchical)

        # Eval, with the environment reward. (This will be different than the training reward, for instance
        # if we're finetuning with a single goal or if we're using LM rewards)
        self.eval_env.set_env_reward(True)
        self.eval_env.update_dir(self.work_dir / 'eval_episodes', save_video)
        self.evaluate_env(self.eval_env, self.cfg.num_eval_episodes, 'eval', '', True)

    def evaluate_goal_env(self, task):
        env = self.eval_env
        env.set_env_reward(self.train_with_env_reward)
        env.set_end_on_success(True)
        env.set_step(self.global_frame)
        log_video = task in self.cfg.eval_video_goals or task == self.cfg.env_spec.single_task
        print('DOING EVAL FOR', task, log_video)
        env.update_dir(self.work_dir / 'eval_singlegoal_episodes', log_video, f'single_goal_eval_video/{task}')

        log_str = f"{task}_"
        episode, total_success, total_length = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        
        
        # Set env goal to only use single item
        env.set_single_task(task)

        while eval_until_episode(episode):
            time_step, _ = env.reset()
            while not time_step.last():
                # Decompose hierarchical goals into subtasks
                subtask = env.get_subtask()

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    env.set_rand_actions(self.agent.rand_action)
                final_subtask = env.on_final_subtask() or subtask == task
                time_step = env.step(action)
                total_length += 1
                if time_step.observation['goal_success'] == 1:
                    if final_subtask:  # Only count success if it's the last subtask
                        print(f'Final action: {env.get_action_name(action)}')
                        total_success += 1
                        break
            episode += 1
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval_extended') as log:
            log(f'{log_str}episode_reward', total_success / self.cfg.num_eval_episodes)
            log(f'{log_str}episode_length', total_length / self.cfg.num_eval_episodes)
        env.set_single_task(self.cfg.env_spec.single_task)  # Reset to the default
        env.set_end_on_success(False)


    def evaluate_env(self, env, num_episodes, log_str1, log_str2, log_full, log_actions=False, log_misplaced=False):
        if not log_str2 == '':
            log_str2 += '_' 
        step, episode, total_reward = 0, 0, 0
        if 'Housekeep' in self.cfg.env_spec.name:
            rearrange_success_total, initial_success_total, rearrange_misplaced_success_total = 0, 0, 0
        eval_until_episode = utils.Until(num_episodes)

        try:
            env.set_step(self.global_frame)
        except:
            pass
        while eval_until_episode(episode):
            time_step, _ = env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                    env.set_rand_actions(self.agent.rand_action)

                time_step = env.step(action)
                total_reward += time_step.reward
                if time_step.reward > .5 and log_actions:
                    print('action name', env.get_action_name(action), 'episode', episode, 'total reward', total_reward)
                step += 1
            episode += 1
            if 'Housekeep' in self.cfg.env_spec.name:
                rearrange_misplaced_success_total += env._rearrange_misplaced_success
                rearrange_success_total += env._rearrange_success
                initial_success_total += env._initial_success

        with self.logger.log_and_dump_ctx(self.global_frame, ty=log_str1) as log:
            log(f'{log_str2}episode_reward', total_reward / episode)
            if 'Housekeep' in self.cfg.env_spec.name:
                log('rearrange_success', rearrange_success_total / episode)
                log('rearrange_misplaced_success', rearrange_misplaced_success_total / episode)
                log('initial_success', initial_success_total / episode)
        if log_full:
            with self.logger.log_and_dump_ctx(self.global_frame, ty=log_str1) as log:
                log(f'{log_str2}episode_length', step / episode)
                log(f'{log_str2}episode', self.global_episode)
                log(f'{log_str2}step', self.global_step)

    def relabel_env_rew(self, time_step, env):
        if env.get_env_reward() is not None:
            return time_step._replace(reward=env.get_env_reward())
        return time_step

    @property
    def train_with_env_reward(self):
        return self.env_spec['name'] == 'CrafterReward-v1'

    def train(self):
        if self.cfg.finetune_snapshot:
            self.agent.new_opt(self.cfg.lr)
            self.agent.new_tau(self.cfg.agent.critic_target_tau)

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        seed_until_step = utils.Until(self.cfg.num_seed_frames)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)

        self.eval_every_step = eval_every_step
        video_every_step = utils.Every(self.cfg.eval_every_frames * 20)
        save_every_step = utils.Every(self.cfg.eval_every_frames)

        self.eval_env.set_step(self.global_frame)
        episode_step, episode_reward = 0, 0
        if 'Housekeep' in self.cfg.env_spec.name:
            max_rearrange_success = 0
        time_step, _ = self.train_env.reset()
        self.replay_storage.add(time_step)
        if self.cfg.use_offline_buffer:
            self.offline_replay_storage.add(self.relabel_env_rew(time_step, self.train_env))
        metrics = None
        next_log_itr = self.global_step
        log_time = False
        while train_until_step(self.global_step):
            train_start_time = time.time()
            log_time = False
            if time_step.last():
                start_time = time.time()
                self.replay_storage.store_episode()
                if self.cfg.use_offline_buffer:
                    self.offline_replay_storage.store_episode()
                self.storage_time += time.time() - start_time
                start_time = time.time()
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if self.global_step >= next_log_itr:
                    log_time = True
                    next_log_itr = self.global_step + self.cfg.train_log_every - 1
                if True:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step
                    if log_time:
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty='memory') as log:
                            import psutil, resource
                            log('MB', psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
                            current_process = psutil.Process(os.getpid())
                            mem = current_process.memory_info().rss
                            for child in current_process.children(recursive=True):
                                mem += child.memory_info().rss
                            log('MB_children', mem / (1024 ** 2))
                            log('MB_max', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                            mb_max_w_children = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                            log('MB_max_children', mb_max_w_children / 1024)
                            
                            

                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('spf', elapsed_time / episode_frame)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                            if 'Housekeep' in self.cfg.env_spec.name:
                                if self.train_env._rearrange_success is not None:
                                    log('rearrange_success', self.train_env._rearrange_success)
                                    log('max_rearrange_success', max_rearrange_success)
                                if self.train_env._rearrange_misplaced_success is not None:
                                    log('rearrange_misplaced_success', self.train_env._rearrange_misplaced_success)
                                if self.train_env._initial_success is not None:
                                    log('initial_success', self.train_env._initial_success)
                                if self.train_env._lm_rearrange_success is not None:
                                    log('lm_rearrange_success', self.train_env._lm_rearrange_success)
                                if self.train_env._oracle_lm_rearrange_success is not None:
                                    log('oracle_lm_rearrange_success', self.train_env._oracle_lm_rearrange_success)
                                if self.train_env._pick_success is not None:
                                    log('pick_success', self.train_env._pick_success)
                                    log('place_success', self.train_env._place_success)
                            log('train_eps_value', self.agent.compute_train_eps(self.global_step))
                            log('cumulative_train_reward', self.cumulative_train_reward)
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty='time') as log:
                            log('total_time', total_time)
                            
                            log('abs_train_time', (self.train_time - self.eval_time))  # Train time includes eval as well
                            log('abs_eval_time', self.eval_time)
                            log('abs_agent_time', self.agent_time)
                            log('abs_env_time', self.env_time)
                            log('abs_update_time', self.update_time)
                            log('abs_storage_time', self.storage_time)
                            log('abs_save_time', self.save_time)
                            log('abs_log_time', self.log_time)
                            
                            
                            log('train_time', (self.train_time - self.eval_time) / total_time)  # Train time includes eval as well
                            log('eval_time', self.eval_time / total_time)
                            log('agent_time', self.agent_time / total_time)
                            log('env_time', self.env_time / total_time)
                            log('update_time', self.update_time / total_time)
                            log('storage_time', self.storage_time / total_time)
                            log('save_time', self.save_time / total_time)
                            log('log_time', self.log_time / total_time)
                            accounted_time = self.agent_time + self.env_time + self.update_time + self.eval_time + self.storage_time + self.save_time + self.log_time
                            log('total_accounted_time', accounted_time / total_time)
                            log('total_unaccounted', (total_time - accounted_time) / total_time)


                self.log_time += time.time() - start_time
                # reset env
                start_time  = time.time()
                time_step, _ = self.train_env.reset()
                self.env_time += time.time() - start_time
                start_time = time.time()
                self.replay_storage.add(time_step)
                if self.cfg.use_offline_buffer:
                    self.offline_replay_storage.add(self.relabel_env_rew(time_step, self.train_env))
                self.storage_time += time.time() - start_time
                episode_step = 0
                episode_reward = 0
                if 'Housekeep' in self.cfg.env_spec.name:
                    max_rearrange_success = 0
            # try to save snapshot
            if self.cfg.save_snapshot and save_every_step(self.global_step):
                start_time = time.time()
                self.save_snapshot()
                self.save_time += time.time() - start_time
                    
            # Save cache
            if save_every_step(self.global_step):
                start_time = time.time()
                if self.train_env.use_sbert_sim:
                    self.train_env.lm.load_and_save_cache()
                    self.eval_env.lm.load_and_save_cache()
                if self.train_env.use_sbert_sim:
                    self.train_env.load_and_save_caches()
                    self.eval_env.load_and_save_caches()
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.log_time += time.time() - start_time

            # Evaluate every eval_every_step or on the last step
            if eval_every_step(self.global_step) or self.global_step == self.cfg.num_train_frames - 1:
                start_time = time.time()
                if 'Crafter' in self.cfg.env_spec.name:
                    self.eval_crafter(save_video=video_every_step(self.global_step))
                elif 'Housekeep' in self.cfg.env_spec.name:
                    self.eval_housekeep(save_video=video_every_step(self.global_step))
                self.eval_time += time.time() - start_time

            # sample action
            start_time  = time.time()
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False, other_model=self.other_model)
                self.train_env.set_rand_actions(self.agent.rand_action)
            self.agent_time += time.time() - start_time

            # try to update the agent
            start_time = time.time()
            if not seed_until_step(self.global_step) and (not self.cfg.train_after_reward or self.cumulative_train_reward >= 1):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.update_time += time.time() - start_time
                if log_time:
                    # Separate the dict into two dicts, one for the agent metrics and one for the grad norm metrics (every key which starts with 'grad')
                    grad_dict = {k: v for k, v in metrics.items() if k.startswith('grad')}
                    metrics = {k: v for k, v in metrics.items() if not k.startswith('grad')}
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    self.logger.log_metrics(grad_dict, self.global_frame, ty='gradients')
            real_lm = 'GPT' in self.env_spec.lm_spec['lm_class'] or ('SimpleOracle' in self.env_spec.lm_spec['lm_class'] and 'Housekeep' in self.cfg.env_spec.name)
            if log_time and self.cfg.use_wandb:
                if real_lm and 'Crafter' in self.cfg.env_spec.name:
                    self.train_env.log_real_lm(self.global_frame)
                else:
                    try:
                        self.train_env.log_lm(self.global_frame)
                    except:
                        pass

            # take env step
            start_time = time.time()
            self.train_env.set_step(self._global_step)
            time_step = self.train_env.step(action)
            self.cumulative_train_reward += time_step.reward
            if self.cfg.decay_after_reward:
                if self.cumulative_train_reward >= 1:
                    self.agent.decay_started = True
                    if self.agent.decay_started_step is None:
                        self.agent.decay_started_step = self.global_step
            else:
                self.agent.decay_started = True
                self.agent.decay_started_step = 0
            
            episode_reward += time_step.reward
            self.env_time += time.time() - start_time
            self.replay_storage.add(time_step)
            if self.cfg.use_offline_buffer:
                self.offline_replay_storage.add(self.relabel_env_rew(time_step, self.train_env))
            episode_step += 1
            self._global_step += 1
            self.train_time += time.time() - train_start_time
        print('SAVING COUNTS!!!')
        if hasattr(self.train_env, 'save_counts'):
            self.train_env.save_counts(self.cfg.exp_name)
  
    
    def train_offline(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)

        self.eval_env.set_step(self._global_step)
        episode_step, episode_reward = 0, 0
        time_step, _ = self.train_env.reset()
        metrics = None
        next_log_itr = self.global_step
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step
                    if self.global_step >= next_log_itr:
                        next_log_itr = self.global_step + self.cfg.train_log_every
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                
                if 'Crafter' in self.args.env_spec.name:
                    self.eval_crafter()
                elif 'Housekeep' in self.args.env_spec.name:
                    self.eval_housekeep()


            # try to update the agent
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty='train')

            episode_step += 1
            self._global_step += 1
            self.eval_env.set_step(self._global_step)

    def save_snapshot(self):
        temp_snapshot = self.work_dir / 'temp.pt'
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        # Remove the agent's cache from the snapshot
        agent = payload['agent']
        if hasattr(agent.encoder, 'lang_goal_encoder') and hasattr(agent.encoder.lang_goal_encoder, 'sbert_encoder'):
            agent.encoder.lang_goal_encoder.sbert_encoder.cache = {}
        if hasattr(agent.encoder, 'lang_state_encoder') and hasattr(agent.encoder.lang_state_encoder, 'sbert_encoder'):
            agent.encoder.lang_state_encoder.sbert_encoder.cache = {}
        payload['agent'] = agent
        
        with temp_snapshot.open('wb') as f:
            torch.save(payload, f)
            os.rename(temp_snapshot, snapshot)


    def load_snapshot(self, snapshot=None, agent_only=False):
        if snapshot is None:
            snapshot = self.work_dir / 'snapshot.pt'
        if not snapshot.exists():
            raise ValueError(f'Snapshot not found: {snapshot}')
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        if agent_only:
            new_agent = payload['agent']
            if 'reinit' in self.cfg.finetune_settings:
                if 'linear' in self.cfg.finetune_settings:
                    layers = [4]
                elif 'critic' in self.cfg.finetune_settings:
                    layers = [0, 2, 4]
                last_weights = []
                for layer in layers:
                    last_weights += [f'V.{layer}.weight', f'V.{layer}.bias', f'A.{layer}.weight', f'A.{layer}.bias']
                # Loop through loaded state dicts and re-initialize these weights
                for weight_name in last_weights:
                    # Reinitialize weights with Kaiming uniform
                    new_agent.critic.state_dict()[weight_name] = torch.nn.init.kaiming_uniform_(new_agent.critic.state_dict()[weight_name])
                    new_agent.critic_target.state_dict()[weight_name] = torch.nn.init.kaiming_uniform_(new_agent.critic_target.state_dict()[weight_name])                    
            self.__dict__['agent'].encoder.load_state_dict(new_agent.encoder.state_dict())
            self.__dict__['agent'].critic.load_state_dict(new_agent.critic.state_dict())
            self.__dict__['agent'].critic_target.load_state_dict(new_agent.critic_target.state_dict())
        else:
            for k, v in payload.items():
                self.__dict__[k] = v   

    def load_other_agent(self, snapshot_path):
        with snapshot_path.open('rb') as f:
            payload = torch.load(f)
        self.other_model = payload['agent']           
            
root_dir = Path.cwd()

@hydra.main(config_path='.', config_name='config')
def main(cfg):
    from train import Workspace as W   
    if not cfg.snapshot_path is None:
        if not cfg.finetune_snapshot:
            os.chdir(root_dir / Path(cfg.snapshot_path)) # Change to snapshot dir
        snapshot = root_dir / Path(cfg.snapshot_path + '/snapshot.pt')
        workspace = W(cfg)
    else:
        workspace = W(cfg)
        snapshot = Path.cwd() / 'snapshot.pt'
    
    if cfg.expl_agent_path is not None:
        workspace.load_other_agent(root_dir / Path(cfg.expl_agent_path + '/snapshot.pt'))
        workspace.agent.other_model_prob = cfg.other_model_prob
    else:
        workspace.other_model=None
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot, agent_only=cfg.finetune_snapshot)
        workspace.resume_run = not cfg.finetune_snapshot
    if cfg.offline_buffer is None:
        workspace.train()
    else:
        workspace.train_offline()

if __name__ == '__main__':
    main()
    print('Done!')
