# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import datetime
import io
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return len(episode['action']) - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, env, data_specs, replay_dir):
        self.env = env
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._frame_stack = data_specs[0]['obs'].shape[0]
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            if type(spec) is dict:
                spec_name = 'observation'
            else:
                spec_name = spec.name
            value = getattr(time_step, spec_name)
            if np.isscalar(value):
                value = np.array(value, dtype=spec.dtype)
            if type(spec) is dict:
                value = value.copy()
                for k in spec.keys():
                    if k == 'obs': # remove frame stack
                        value[k] = np.expand_dims(np.array(value[k][-1]), axis=0)
                        assert spec[k].shape[1:] == value[k].shape[1:], (k, spec[k].shape, value[k].shape)
                    else:
                        value[k] = np.array(value[k])
                        assert spec[k].shape == value[k].shape, (k, spec[k].shape, value[k].shape)
            else:
                try:
                    assert spec.shape == value.shape and spec.dtype == value.dtype
                except:
                    import pdb; pdb.set_trace()
            self._current_episode[spec_name].append(value)

    def store_episode(self):
        episode = self._format_for_store(self._current_episode)
        self._current_episode = defaultdict(list)
        self._store_episode(episode)

    def _format_for_store(self, curr_episode):
        episode = dict()
        for spec in self._data_specs:
            if type(spec) is dict:
                value = curr_episode['observation']
                new_dict = {}
                example = value[0]
                for k in example.keys():
                    if k in spec.keys():
                        new_dict[k] = np.array([v[k] for v in value], dtype=example[k].dtype)
                episode['observation'] = new_dict
            else:
                value = curr_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
        return episode


    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 clip_reward, fetch_every, save_snapshot, frame_stack):

        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = num_workers
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._clip_reward = clip_reward
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._frame_stack = frame_stack

    def _sample_episode(self):
        eps_fn = np.random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            print("didn't load correctly", eps_fn)
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except Exception as e:
            traceback.print_exc()
            raise e
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, max(1, episode_len(episode) - self._nstep + 1)) + 1
        o = episode['observation'].item()
        obs = {k: v[idx - 1] for k, v in o.items()}
        action = episode['action'][idx]
        next_obs = {k: v[min(episode_len(episode), idx + self._nstep) - 1] for k, v in o.items()}
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx]).astype(np.float32)
        
        for i in range(1, self._frame_stack):
            obs['obs'] = np.concatenate([o['obs'][max(idx - 1 - i, 0)], obs['obs']], axis=0)
            next_obs['obs'] = np.concatenate([o['obs'][max(min(idx + self._nstep - 1 - i, episode_len(episode)),0)], next_obs['obs']], axis=0)

        for i in range(self._nstep):
            if idx + i > episode_len(episode):
                break
            step_reward = episode['reward'][idx + i]
            if self._clip_reward:
                step_reward = np.clip(step_reward, -1, 1)
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        assert episode['discount'][-1] == 0, episode['discount']
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, seq_len, discount, downsample_rate):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._seq_len = seq_len
        self._num_workers = num_workers
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._downsample_rate = downsample_rate
        self._loaded = False

    def _load(self):
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'))
        # subsampling
        eps_fns = eps_fns[::self._downsample_rate]
        for eps_idx, eps_fn in enumerate(eps_fns):
            if self._size > self._max_size:
                break
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode['action'].shape[0]
        
    def _sample_episode(self):
        if not self._loaded:
            self._load()
            self._loaded = True
        eps_fn = np.random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode['observation'].shape[0] - self._seq_len + 1)
        
        obses = episode['observation'][idx: idx + self._seq_len]
        actions = episode['action'][idx: idx + self._seq_len]
        returns = np.flip(np.cumsum(np.flip(episode['reward'][idx:])))
        returns = returns[:self._seq_len].copy()
        timesteps = np.arange(self._seq_len, dtype=np.int64) + idx

        return (obses, actions, returns, timesteps)

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id, seed=None):
    seed = torch.initial_seed() % 2**32 #np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, clip_reward, seed, frame_stack):
    max_size_per_worker = max_size // num_workers

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            clip_reward,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            frame_stack = frame_stack)
    g = torch.Generator()
    g.manual_seed(0)
    
    if num_workers == 1: 
        num_workers = 0 # prevent parallelization 

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn,
                                         generator=g)
    return loader


def make_offline_replay_loader(replay_dir, max_size, batch_size, seq_len,
                               num_workers, discount, downsample_rate, seed):
    max_size_per_worker = max_size // num_workers

    iterable = OfflineReplayBuffer(replay_dir,
                                   max_size_per_worker,
                                   num_workers,
                                   seq_len,
                                   discount,
                                   downsample_rate)
    
    g = torch.Generator()
    g.manual_seed(seed)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn,
                                         generator=g)
    return loader
