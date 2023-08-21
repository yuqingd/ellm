import datetime
import json
import pathlib

import imageio
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2

class Recorder:

  def __init__(
      self, env, directory, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512), use_wandb=False):
    if directory and save_stats:
      env = StatsRecorder(env, directory)
    if directory:
      env = VideoRecorder(env, directory, video_size, use_wandb=use_wandb, save_video=save_video)
    self._env = env

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)


class StatsRecorder:

  def __init__(self, env, directory):
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._file = (self._directory / 'stats.jsonl').open('a')
    self._length = None
    self._reward = None
    self._unlocked = None
    self._stats = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs, info = self._env.reset()
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    return obs, info

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      if 'achievements' in info:
        for key, value in info['achievements'].items():
          self._stats[f'achievement_{key}'] = value
      if 'env_reward' in info:
        self._stats['env_reward'] = info['env_reward']
      if 'rearrange_success' in info:
        self._stats['rearrange_success'] = info['rearrange_success']
      if 'rearrange_misplaced_success' in info:
        self._stats['rearrange_misplaced_success'] = info['rearrange_misplaced_success']
      if 'initial_success' in info:
        self._stats['initial_success'] = info['initial_success']
      self._save()
    return obs, reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()

  def update_dir(self, directory, save_video, save_video_dir='eval_video'):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    try:
      self._env.update_dir(directory, save_video, save_video_dir=save_video_dir)
    except:
      pass


class VideoRecorder:

  def __init__(self, env, directory, size=(512, 512), use_wandb=False, save_video=False):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._size = size
    self._frames = None
    self.use_wandb = use_wandb
    self.global_step = 0
    self.save_video = save_video
    self.save_video_dir = 'eval_video'
    self._num_episodes_to_log = float('inf')
    self._num_episodes_so_far = 0
    self.rand_actions = []
    self.last_rand_action = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)
  
  def set_rand_actions(self, rand_action):
    self.rand_actions.append(rand_action)
    self.last_rand_action = rand_action

  def add_text_housekeep(self, img, text_goal, oracle_goal, gripped_obj, text_action):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)    
    draw.text((10, 0), 'Goal: ' + text_goal,(255,255,255))
    draw.text((10, 15),  'Receptacles: ' + oracle_goal,(255,255,255))
    draw.text((10, 30),  'Gripped Object: ' +  gripped_obj,(255,255,255))
    draw.text((10, 45), 'Took action: ' +  text_action,(255,255,255))
    img = np.asarray(img)
    return img    

  def add_text_crafter(self, img, text_goal, oracle_goal, text_action, rand_action, transition_caption=None, state_caption=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    if rand_action is None:
      rand_action = 'unknown'
    draw.text((0, 0), 'Current Goal: ' +  text_goal,(255,255,255))
    draw.text((0, 15), 'Oracle Goal: ' +  oracle_goal,(255,255,255))
    draw.text((0, 30), 'Took action: ' +  text_action + f' - {rand_action}',(255,255,255))

    if transition_caption is not None:
      draw.text((0, 90), 'Transition: ' +  transition_caption,(255,255,255))
    if state_caption is not None:
      draw.text((0, 105), 'State: ' +  state_caption,(255,255,255))

    draw.text((0, 45), 'Fraction random: ' +  str(np.sum([int(r == 'random') for r in self.rand_actions]) / max(1,len(self.rand_actions))),(255,255,255))
    draw.text((0, 60), 'Fraction policy: ' +  str(np.sum([int(r == 'policy') for r in self.rand_actions]) / max(1,len(self.rand_actions))),(255,255,255))
    draw.text((0, 75), 'Fraction other: ' +  str(np.sum([int(r == 'other') for r in self.rand_actions]) / max(1,len(self.rand_actions))),(255,255,255))
    img = np.asarray(img)
    return img

  def reset(self):
    obs, info = self._env.reset()
    obs['obs'] = cv2.resize(obs['obs'], (84, 84), interpolation=cv2.INTER_LINEAR)

    self.running_rew = 0
    if not self.save_video or self._num_episodes_so_far >= self._num_episodes_to_log:
      return obs, info
    img = self._env.render(self._size)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = np.asarray(img, dtype=np.uint8)
    if self._env.goal_str is not None:
      if hasattr(self._env, 'gripped_obj'):
        img = self.add_text_housekeep(img, self._env.goal_str, self._env.oracle_goal_str, self._env.gripped_obj,  'RESET')
      else:
        img = self.add_text_crafter(img, self._env.goal_str, self._env.oracle_goal_str,  'RESET', self.last_rand_action, self.env.transition_caption, self.env.state_caption)
    del self._frames
    self._frames = [img]
    self.rand_actions = []
    return obs, info

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    
    obs['obs'] = cv2.resize(obs['obs'], (84, 84), interpolation=cv2.INTER_LINEAR)
    self.running_rew += reward
    if not self.save_video or self._num_episodes_so_far >= self._num_episodes_to_log:
      return obs, reward, done, info
    img = self._env.render(self._size)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = np.asarray(img, dtype=np.uint8)
    if self._env.goal_str is not None:
      if hasattr(self._env, 'gripped_obj'):
        img = self.add_text_housekeep(img, self._env.goal_str, self._env.oracle_goal_str, self._env.gripped_obj, self._env.get_action_name(action))
      else:
        img = self.add_text_crafter(img, self._env.goal_str, self._env.oracle_goal_str, self._env.get_action_name(action), self.last_rand_action, self.env.transition_caption, self.env.state_caption)
    self._frames.append(img)

    if done:# and self.running_rew > 10:
      self._save()
      self._num_episodes_so_far += 1
    return obs, reward, done, info

  def set_step(self, step):
    self.global_step = step

  def _save(self):
    if not self.save_video or self._num_episodes_so_far >= self._num_episodes_to_log:
      return
    if self.use_wandb:
      import wandb
      frames = np.stack(self._frames) # THWC
      frames = np.moveaxis(frames, [3], [1])
      wandb.log({self.save_video_dir: wandb.Video(frames)}, step=self.global_step)
      self._frames = []
    # filename = str(self._directory / (self._env.episode_name + '.mp4'))
    # imageio.mimsave(filename, self._frames)

  def update_dir(self, directory, save_video, save_video_dir='eval_video', num_episodes=1):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self.save_video = save_video
    self.save_video_dir = save_video_dir
    self._num_episodes_so_far = 0
    self._num_episodes_to_log = num_episodes
    try:
      self._env.update_dir(directory, save_video)
    except:
      pass


class EpisodeRecorder:

  def __init__(self, env, directory):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._episode = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs = self._env.reset()
    self._episode = [{'image': obs}]
    return obs

  def step(self, action):
    # Transitions are defined from the environment perspective, meaning that a
    # transition contains the action and the resulting reward and next
    # observation produced by the environment in response to said action.
    obs, reward, done, info = self._env.step(action)
    transition = {
        'action': action, 'image': obs, 'reward': reward, 'done': done,
    }
    for key, value in info.items():
      if key in ('inventory', 'achievements'):
        continue
      transition[key] = value
    for key, value in info['achievements'].items():
      transition[f'achievement_{key}'] = value
    for key, value in info['inventory'].items():
      transition[f'ainventory_{key}'] = value
    self._episode.append(transition)
    if done:
      self._save()
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.npz'))
    # Fill in zeros for keys missing at the first time step.
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    episode = {
        k: np.array([step[k] for step in self._episode])
        for k in self._episode[0]}
    np.savez_compressed(filename, **episode)

  def update_dir(self, directory, save_video):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    try:
      self._env.update_dir(directory, save_video)
    except:
      pass


class EpisodeName:

  def __init__(self, env):
    self._env = env
    self._timestamp = None
    self._unlocked = None
    self._length = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self):
    obs, info = self._env.reset()
    self._timestamp = None
    self._unlocked = None
    self._length = 0
    return obs, info

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._length += 1
    if done:
      self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
      if 'achievements' in info:
        self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
    return obs, reward, done, info

  @property
  def episode_name(self):
    return 'debugvideo'
    # return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
