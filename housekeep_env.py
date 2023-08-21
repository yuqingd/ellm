import copy
import numpy as np
from collections import deque

import os
import dm_env
import cv2
import text_crafter.text_crafter

from text_housekeep.cos_eor.task.measures import *
from text_housekeep.cos_eor.task.sensors import *
from dm_env import specs, StepType
from utils import ExtendedTimeStepWrapper

from text_housekeep.habitat_lab.habitat_baselines.common.environments import get_env_class
from text_housekeep.habitat_lab.habitat_baselines.config.default import get_config
from text_housekeep.cos_eor.env.text_env import BaseTextRearrangementEnv

def update_paths(config):
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict):
                v = update_paths(v)
            elif isinstance(v, str) and ('data/' in v or 'cos_eor/' in v):
                v = os.getcwd().split('exp_local')[0] + '/text_housekeep/' + v
            config[k] = v
    return config


class Housekeep(dm_env.Environment):
    def __init__(self,
                 logdir,
                 env_spec,
                 screen_size=84,
                 save_stats=True,
                 save_video=False,
                 save_episode=False,
                 seed=1,
                 env_reward=False,
                 use_wandb=False, 
                 debug=False,
                 device=None):
                 
        self.logdir = logdir
        self.debug = debug
        if debug:
            self.obs_strings = set()
        env_spec['env_reward'] = env_reward
        task_config_path = os.getcwd().split('exp_local')[0] + '/text_housekeep/logs/oracle/configs/' + env_spec["housekeep_task"] + '.yaml'
        task_config = get_config(task_config_path, [])
        task_config.defrost()
        task_config = update_paths(task_config)
        task_config['TASK_CONFIG']['EPISODE_NUM'] = env_spec['housekeep_ep_num']
        task_config.freeze()
        env = get_env_class("BaseTextRearrangementRLEnv")(task_config, **env_spec)
        self._env = text_crafter.text_crafter.Recorder(
            env,
            logdir,
            save_stats=save_stats,
            save_video=save_video,
            save_episode=save_episode,
            use_wandb=use_wandb)


        self._env.seed(seed)
        
        self._screen_size = screen_size
        self.max_seq_len = env_spec['max_seq_len']
        
        shape = (1, screen_size, screen_size)
        self._obs_spec = {'obs':specs.BoundedArray(shape=shape,
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation'),
                        'text_obs':specs.Array(shape=(env_spec['max_seq_len'],),
                                            dtype=np.uint8,
                                            name='text_obs'),
                        'goal':specs.Array(shape=(env_spec['max_seq_len'],),
                                            dtype=np.uint8,
                                            name='goal'),
                        'old_goals':specs.Array(shape=(env_spec['max_seq_len'],),
                                            dtype=np.uint8,
                                            name='old_goals'),
                        'success':specs.Array(shape=(),
                                            dtype=bool,
                                            name='success'),
                        'goal_success':specs.Array(shape=(),
                                            dtype=bool,
                                            name='goal_success'),
                                            
                         }
        self._action_spec = specs.DiscreteArray(
                                                num_values=self._env.action_space.n,
                                                dtype=np.int64,
                                                name='action')


    def _fetch_grayscale_observation(self, buffer):
        self._env.ale.getScreenGrayscale(buffer)
        return buffer

    def _pool_and_resize(self):
        # pool if there are enough screens to do so.
        if self._frame_skip > 1:
            np.maximum(self._screen_buffer[0],
                       self._screen_buffer[1],
                       out=self._screen_buffer[0])

        image = cv2.resize(self._screen_buffer[0],
                           (self._screen_size, self._screen_size),
                           interpolation=cv2.INTER_LINEAR)
        image = np.asarray(image, dtype=np.uint8)
        return np.expand_dims(image, axis=0)
     
    def _transform_observation(self, obs):
        obs['obs'] = self._transform_obs_array(obs['obs'])
        return obs

    def _transform_obs_array(self, obs):
        # gray scale
        obs = np.mean(obs, axis=-1)
        # resize
        image = cv2.resize(obs, (self._screen_size, self._screen_size), interpolation=cv2.INTER_LINEAR)
        obs = np.asarray(image, dtype=np.uint8)
        obs = np.expand_dims(obs, axis=0)
        return obs
    
    def get_env_reward(self):
        return self._env_reward

    def reset(self):
        obs, info = self._env.reset()
        
        if self.debug:
        ##To save in file
            if len(self.obs_strings) > 1:
                with open('obs_strings.txt','a+') as f:
                    obs_strings = 'NEW EPISODE: '
                    obs_strings += '\n '.join(self.obs_strings)
                    obs_strings += '\n \n'
                    print(obs_strings, file=f)
            self.obs_strings = set()
        self._env_reward = None
        self._rearrange_success = None
        self._rearrange_misplaced_success = None
        self._oracle_lm_rearrange_success = None
        self._lm_rearrange_success = None
        self._initial_success = None
        obs = self._transform_observation(obs)
        return dm_env.TimeStep(StepType.FIRST, 0.0, 1.0, obs), info  # StepType, reward, discount, obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._transform_observation(obs)
        self._env_reward = info['env_reward']
        self._rearrange_success = info['rearrange_success']
        self._rearrange_misplaced_success = info['rearrange_misplaced_success']
        self._initial_success = info['initial_success']
        self._pick_success = info['pick_success']
        self._place_success = info['place_success']
        if self.use_lm:
            self._oracle_lm_rearrange_success = info['oracle_lm_rearrange_success']
            self._lm_rearrange_success = info['lm_rearrange_success']
        
        # log strings
        if self.debug:
            self.obs_strings.add(info['string_obs_log'])
        if done:
            return dm_env.termination(reward, obs)
        return dm_env.transition(reward, obs)

    def relabel(self, time_step):
        return self._env.relabel(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return specs.Array(shape=(), name='reward', dtype=np.float16)

    def discount_spec(self):
        return specs.Array(shape=(), name='discount', dtype=np.uint8)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)
    


class FrameStack(dm_env.Environment):
    def __init__(self, env, k):
        self._env = env
        self._k = k
        self._frames = deque([], maxlen=k)

        env_obs_spec = env.observation_spec()
        obs_shape = env_obs_spec['obs'].shape
        env_obs_spec['obs'] = specs.BoundedArray(shape=np.concatenate(
            [[obs_shape[0] * k], obs_shape[1:]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._k
        frame_list = list(self._frames)
        if type(frame_list[0]) is dict:
            obs = copy.deepcopy(frame_list[-1])
            obs['obs'] = np.concatenate([o['obs'] for o in frame_list], axis=0)
        else:
            obs = np.concatenate(frame_list, axis=0)

        return time_step._replace(observation=obs)

    def reset(self):
        time_step, info = self._env.reset()
        pixels = time_step.observation
        for _ in range(self._k):
            self._frames.append(pixels)
        return self._transform_observation(time_step), info

    def step(self, action):
        time_step = self._env.step(action)
        pixels = time_step.observation
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def discount_spec(self):
        return self._env.discount_spec()
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    

def make(logdir, env_spec, save_video, frame_stack, seed=1, env_reward=False, use_wandb=False, debug=False, device=None):
    env = Housekeep(logdir, env_spec, save_video=save_video, seed=seed, env_reward=env_reward, use_wandb=use_wandb, debug=debug, device=device)
    env = FrameStack(env, k=frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
