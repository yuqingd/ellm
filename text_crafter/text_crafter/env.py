import collections

import numpy as np

from text_crafter.text_crafter import constants
from text_crafter.text_crafter import engine
from text_crafter.text_crafter import objects
from text_crafter.text_crafter import worldgen

# Gym is an optional dependency.
try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object


class Env(BaseClass):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, seed=None, dying=True, **kwargs):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    self._area = area
    self._view = view
    self._size = size
    self._reward = reward
    self._length = length
    self._seed = seed
    self._episode = 0
    self.world = engine.World(area, constants.materials, (12, 12), seed=seed)
    self._textures = engine.Textures(constants.root / 'assets')
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self.world, self._textures, [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    self._sem_view = engine.SemanticView(self.world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.Skeleton, objects.Arrow, objects.Plant])
    self._local_token_view = engine.LocalSemanticView(self.world, [objects.Player, objects.Cow, objects.Zombie, objects.Skeleton, objects.Arrow, objects.Plant],
                                                      [view[0], view[1] - item_rows])
    self._step = None
    self.player = None
    self._last_health = None
    self._unlocked = None
    # Some libraries expect these attributes to be set.
    self.reward_range = None
    self.metadata = None
    self.dying = dying # If true, episode ends when health < 0 

  @property
  def observation_space(self):
    return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self):
    center = (self.world.area[0] // 2, self.world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self.world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._update_time()
    self.player = objects.Player(self.world, center)
    self._last_health = self.player.health
    self.world.add(self.player)
    self._unlocked = set()
    worldgen.generate_world(self.world, self.player, seed=self._seed)
    dead = False
    reward = 0
    info = {
        'player_action': None,
        'inventory': self.player.inventory.copy(),
        'achievements': self.player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'local_token': self._local_token_view.local_token_view(self.player),
        'player_pos': self.player.pos,
        'reward': reward,
        'large_obs': self.render(size=(512, 512))
    }
    obs = self.obs()
    return obs, info

  def step(self, action):
    self._step += 1
    self._update_time()
    if isinstance(action, int):
      self.player.action = constants.actions[action]
    else:
      # tuple actions for harder env
      ac = self.get_action_name(action)
      self.player.action = ac
    for obj in self.world.objects:
      if self.player.distance(obj) < 2 * max(self._view):
        if obj == self.player:
          action_success, eval_success = obj.update()
        else:
          obj.update()
    if self._step % 10 == 0:
      for chunk, objs in self.world.chunks.items():
        self._balance_chunk(chunk, objs)
    obs = self.obs()
    health_reward = (self.player.health - self._last_health) / 10
    self._last_health = self.player.health
    unlocked = {
        name for name, count in self.player.achievements.items()
        if count > 0 and name not in self._unlocked}
    unlock_reward = 0
    if unlocked:
      self._unlocked |= unlocked
      unlock_reward += 1.0
    if self.dying:
      dead = self.player.health <= 0
    else:
      health_reward = 0
      dead = False
    health_reward = 0
    over = self._length and self._step >= self._length
    done = dead or over
    info = {
        'inventory': self.player.inventory.copy(),
        'achievements': self.player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'player_pos': self.player.pos,
        'reward': unlock_reward + health_reward,
        'health_reward': health_reward,
        'action_success' : action_success,
        'eval_success' : eval_success,
        'player_action': self.player.action,
        'inventory': self.player.inventory.copy(),
        'local_token': self._local_token_view.local_token_view(self.player),
        'large_obs': self.render(size=(512, 512))
    }

    reward = unlock_reward + health_reward
    return obs, reward, done, info

  def convert_to_old_actions(self, action):
      return action

  def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    local_view = self._local_view(self.player, unit)
    item_view = self._item_view(self.player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def obs(self):
    return self.render()

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    progress = (self._step / 300) % 1 + 0.3
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self.world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
    light = self.world.daylight
    self._balance_object(
        chunk, objs, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
        lambda pos: objects.Zombie(self.world, pos, self.player),
        lambda num, space: (
            0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
    self._balance_object(
        chunk, objs, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
        lambda pos: objects.Skeleton(self.world, pos, self.player),
        lambda num, space: (0 if space < 6 else 1, 2))
    self._balance_object(
        chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
        lambda pos: objects.Cow(self.world, pos),
        lambda num, space: (0 if space < 30 else 1, 1.5 + light))

  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    xmin, xmax, ymin, ymax = chunk
    random = self.world.random
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    mask = self.world.mask(*chunk, material)
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      i = random.randint(0, len(xs))
      pos = np.array((xs[i], ys[i]))
      empty = self.world[pos][1] is None
      away = self.player.distance(pos) >= span_dist
      if empty and away:
        self.world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      away = self.player.distance(obj.pos) >= despan_dist
      if away:
        self.world.remove(obj)
        