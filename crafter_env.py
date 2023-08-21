""" Create crafter environment """

import copy
from collections import deque
import dm_env
import gym
import cv2
import numpy as np
from dm_env import specs, StepType

import text_crafter.text_crafter
from utils import ExtendedTimeStepWrapper
from text_crafter.text_crafter.logging_wrapper import CrafterLoggingWrapper
from text_crafter.text_crafter.goal_wrapper import CrafterGoalWrapper, CrafterLMGoalWrapper

class Crafter(dm_env.Environment):
    """A Crafter env, wrapped to do logging."""
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
        if env_spec['name'] == 'CrafterReward-v1':
            assert env_reward
        env_spec['env_reward'] = env_reward
        env_spec['device'] = device
        env_spec['seed'] = seed
        self.logdir = logdir
        env = gym.make(env_spec['name'], **env_spec)
        if 'CrafterTextEnv' in env_spec['name']:  # NOT baseline, so we have goals and should log
            env = CrafterLoggingWrapper(CrafterLMGoalWrapper(env,
                                                             env_spec['lm_spec'],
                                                             env_spec['env_reward'],
                                                             device=device,
                                                             threshold=env_spec['threshold'],
                                                             debug=debug,
                                                             single_task=env_spec['single_task'],
                                                             single_goal_hierarchical=env_spec['single_goal_hierarchical'],
                                                             use_state_captioner=env_spec['use_state_captioner'],
                                                             use_transition_captioner=env_spec['use_transition_captioner'],
                                                             check_ac_success=env_spec['check_ac_success']
                                                             ))
        else:
            env = CrafterGoalWrapper(env, env_spec['env_reward'], env_spec['single_task'], single_goal_hierarchical=env_spec['single_goal_hierarchical'])
        self._env = text_crafter.text_crafter.Recorder(
            env,
            logdir,
            save_stats=save_stats,
            save_video=save_video,
            save_episode=save_episode,
            use_wandb=use_wandb)
        self._env.seed(seed)
        self._screen_size = screen_size
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
        self._env_reward = None

    def _transform_observation(self, obs):
        obs['obs'] = self._transform_obs_array(obs['obs'])
        return obs

    def _transform_obs_array(self, obs):
        # gray scale
        obs = np.mean(obs, axis=-1)
        # resize
        image = cv2.resize(obs, (self._screen_size, self._screen_size),
                           interpolation=cv2.INTER_LINEAR)
        obs = np.asarray(image, dtype=np.uint8)
        obs = np.expand_dims(obs, axis=0)
        return obs

    def get_env_reward(self):
        """Return the most recent env reward."""
        return self._env_reward

    def reset(self):
        obs, info = self._env.reset()
        self._env_reward = None
        obs = self._transform_observation(obs)
        return dm_env.TimeStep(StepType.FIRST, 0.0, 1.0, obs), info  # StepType, reward, discount, obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._transform_observation(obs)
        self._env_reward = info['env_reward']

        if done:
            return dm_env.termination(reward, obs)
        return dm_env.transition(reward, obs)

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
    """A dm_env wrapper that stacks a list of the past k observations."""
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
        if isinstance(frame_list[0], dict):
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


def make(logdir, env_spec, save_video, frame_stack, seed=1, env_reward=False,
         use_wandb=False, debug=False, device=None):
    """Create and wrap crafter environment."""
    env = Crafter(logdir, env_spec, save_video=save_video, seed=seed, env_reward=env_reward,
                  use_wandb=use_wandb, debug=debug, device=device)
    env = FrameStack(env, k=frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
