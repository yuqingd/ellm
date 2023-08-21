import os

import text_crafter.text_crafter
import text_housekeep.recorder
import yaml
from pathlib import Path


import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

from text_housekeep.cos_eor.env.env import CosRearrangementRLEnv
from text_housekeep.cos_eor.env.text_env import BaseTextRearrangementEnv
from text_housekeep.cos_eor.policy.explore import ExploreModule
from text_housekeep.cos_eor.task.task import CosRearrangementTask

from text_housekeep.habitat_lab.habitat_baselines.common.environments import get_env_class
from text_housekeep.habitat_lab.habitat_baselines.config.default import get_config
from text_housekeep.cos_eor.task.measures import *
from text_housekeep.cos_eor.task.sensors import *
import framework

def update_paths(config):
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict):
                v = update_paths(v)
            elif isinstance(v, str) and ('data/' in v or 'cos_eor/' in v):
                v = os.getcwd() + '/text_housekeep/' + v
            config[k] = v
    return config


class RLTaskCRF:
    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.env_train = self.create_train_env()
        self.env_valid_deterministic = self.env_valid_stochastic = None
        if self.helper.args.env_version != 'housekeep':
            self.env_valid_deterministic = self.env_train #self.create_valid_env(env_prefix='valid_det', log_every_n_episodes=1)
            self.env_valid_stochastic = self.env_train #self.create_valid_env(env_prefix='valid_sto', log_every_n_episodes=1)
        self.model = self.create_model()

        if self.helper.args.ckpt.load_path is not None:
            self.model.set_parameters(self.helper.args.ckpt.load_path)
            self.helper.state.step = int(self.helper.args.ckpt.load_path.split('_')[-2])
            self.model.num_timesteps = self.helper.state.step

        framework.helpers.model_info.print_model_size(self.model.policy)
    
    def create_train_env(self):
        return self.create_single_env(env_prefix='train', el_vars=self.helper.args.el_vars, el_freq=self.helper.args.el_freq_train, el_app_freq=self.helper.args.el_app_freq_train)
    
    def create_valid_env(self, env_prefix, save_video=True, log_every_n_episodes=None):
        return self.create_single_env(env_prefix=env_prefix, seed=0, save_video=save_video, log_every_n_episodes=log_every_n_episodes, 
                                        el_vars=self.helper.args.el_vars, el_freq=self.helper.args.el_freq_valid, el_app_freq=self.helper.args.el_app_freq_valid, env_reward=True)

    def create_single_env(self, env_prefix='train', seed=None, save_video=False, log_every_n_episodes=None, el_vars='', el_freq='100,0,0,0', el_app_freq='sssss', env_reward=False):
        if self.helper.args.env_version == 'original':
            env = crafter.Env(
                size=(self.helper.args.crf.size, self.helper.args.crf.size), 
                render_scoreboard=self.helper.args.crf.render_scoreboard,
                seed=seed,
                length=self.helper.args.crf.max_ep_len,
                el_vars=el_vars,
                el_freq=el_freq,
                el_app_freq=el_app_freq,
            )
            env = crafter.Recorder(
                env,
                os.path.join(self.helper.save_dir, f'crafter-episodes-{env_prefix}'),
                self.helper,
                env_prefix=env_prefix,
                save_stats=True,
                save_video=save_video,
                save_episode=False,
                log_every_n_episodes=log_every_n_episodes,
            )
        elif self.helper.args.env_version == 'text':
            env = text_crafter.text_crafter.TextEnv(**self.helper.args.env_spec)
            self.helper.start_logging()
            env = text_housekeep.recorder.Recorder(
                env,
                os.path.join(self.helper.save_dir, f'crafter-episodes-{env_prefix}'),
                self.helper,
                env_prefix=env_prefix,
                save_stats=True,
                save_video=save_video,
                save_episode=False,
                log_every_n_episodes=log_every_n_episodes,
            )
        elif self.helper.args.env_version == 'housekeep':
            task_config_path = os.getcwd() + '/text_housekeep/logs/oracle/configs/' + self.helper.args.env_spec["housekeep_task"] + '.yaml'
            task_config = get_config(task_config_path, [])
            task_config.defrost()
            task_config = update_paths(task_config)
            task_config['TASK_CONFIG']['EPISODE_NUM'] = self.helper.args.env_spec.housekeep_ep_num
            task_config.freeze()
            env = get_env_class("BaseTextRearrangementRLEnv")(task_config, **self.helper.args.env_spec)
            self.helper.start_logging()
            
            env = text_housekeep.recorder.Recorder(
                env,
                os.path.join(self.helper.save_dir, f'crafter-episodes-{env_prefix}'),
                self.helper,
                env_prefix=env_prefix,
                save_stats=True,
                save_video=save_video,
                save_episode=False,
                log_every_n_episodes=log_every_n_episodes,
            )
        else:
            raise NotImplementedError

        env = BaseAlgorithm._wrap_env(env)
        # env = VecFrameStack(env, n_stack=1, n_skip=1)
        return env

    def create_model(self):
        if self.helper.args.ppo.recurrent:
            model = sb3.RecurrentPPO(
                self.helper, 
                self.env_train, 
                verbose=1
            )
        else:
            if self.helper.args.ppo.rnd:
                model = sb3.PPORND(
                    self.helper, 
                    self.env_train, 
                    verbose=1
                )
            elif self.helper.args.ppo.apt:
                model = sb3.PPOAPT(
                    self.helper, 
                    self.env_train, 
                    verbose=1
                )
            else:
                model = sb3.PPO(
                    self.helper, 
                    self.env_train, 
                    verbose=1
                )
        return model

    def train(self):
        try:
            print('-' * 89)
            print(f'Starting training for max {self.helper.args.max_train_steps} steps')
            print("At any point you can hit Ctrl + C to break out of the training loop early.")

            self.model.learn(
                total_timesteps=self.helper.args.max_train_steps,
                eval_env_det=self.env_valid_deterministic,
                eval_env_sto=self.env_valid_stochastic,
                eval_freq=self.helper.args.eval_n_steps,
                n_eval_episodes=self.helper.args.eval_n_episodes,
                reset_num_timesteps=self.helper.state.step == 0,
            )

        except KeyboardInterrupt:
            print('-' * 89)
            print('KeyboardInterrupt signal received. Exiting early from training.')
    
    def test(self):
        self.env_test_deterministic = self.create_valid_env(env_prefix='test_det', save_video=True, log_every_n_episodes=1)
        self.env_test_stochastic = self.create_valid_env(env_prefix='test_sto', save_video=True, log_every_n_episodes=1)

        # Evaluate deterministic policy
        reward_mean, reward_std, _, _ = evaluate_policy(self.model, self.env_test_deterministic, n_eval_episodes=20, deterministic=True)
        print(f"Mean deterministic reward = {reward_mean} +/- {reward_std}")
        helper_logs = {
            'eval_final/reward_det_mean': reward_mean,
            'eval_final/reward_det_std': reward_std,
        }
        self.helper.log(helper_logs, step=self.helper.state.step)

        # Evaluate stochastic policy
        reward_mean, reward_std, _, _ = evaluate_policy(self.model, self.env_test_stochastic, n_eval_episodes=20, deterministic=False)
        print(f"Mean stochastic reward = {reward_mean} +/- {reward_std}")
        helper_logs = {
            'eval_final/reward_stoch_mean': reward_mean,
            'eval_final/reward_stoch_std': reward_std,
        }
        self.helper.log(helper_logs, step=self.helper.state.step)
