import string

import os
import h5py
import time
import numpy as np
from functools import partial
from torch.multiprocessing import Pool

from stable_baselines3.common import base_class
import crafter
import json

import torch
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.base_class import BaseAlgorithm


def prepare_info_dict(info):
    # remove large (image-size) data
    del info['semantic']
    del info['obs72']
    info['player_pos'] = info['player_pos'].tolist()
    if 'terminal_observation' in info:
        del info['terminal_observation']
    info = json.dumps(info, ensure_ascii=False).encode('utf8')
    return info


def sample_rollout_seq_len_append(crf_size, model, min_seq_len, max_seq_len, wrap_normalizeimg, n_stack, n_skip):
    env = crafter.Env(size=(crf_size, crf_size), double_render=True)
    env = BaseAlgorithm._wrap_env(env, wrap_normalizeimg=wrap_normalizeimg)  # According to BaseAlgorithm wrapping envs in init
    env = VecFrameStack(env, n_stack=n_stack, n_skip=n_skip)
    while True:
        ep_obs64 = []
        ep_obs72 = []
        ep_act_det = []
        ep_act_sto = []
        ep_act_logits = []
        ep_info = []
        ep_done = []
        states = None
        obs64 = env.reset()
        done = False
        while len(ep_obs64) < max_seq_len and not done:
            obs64_tensor = obs_as_tensor(obs64, model.device)

            # with torch.no_grad():
            #     obs64_tensor = obs_as_tensor(obs64, model.device)
            #     actions, values, log_probs = model(obs64_tensor, deterministic=True)

            # Do policy evaluation
            act_det, states = model.predict(obs64, state=states, deterministic=True)
            act_sto, states = model.predict(obs64, state=states, deterministic=False)

            # get logits
            with torch.no_grad():
                latent_pi, latent_vf = model._get_latent(obs64_tensor)
                act_logits = model.action_net(latent_pi)

            # Step the environment
            obs64, rewards, done, info = env.step(act_sto)

            # De-vectorize and append
            ep_obs64.append(np.transpose(obs64[0], (1, 2, 0)))
            ep_obs72.append(info[0]['obs72'])
            ep_act_det.append(act_det[0])
            ep_act_sto.append(act_sto[0])
            ep_act_logits.append(act_logits[0].cpu().numpy())
            ep_info.append(prepare_info_dict(info[0]))
            ep_done.append(done[0])

        print('Ep len', len(ep_obs64))
        if len(ep_obs64) >= min_seq_len:
            if len(ep_obs64) < max_seq_len:
                # Repeat frames (taking tuples from the back of the rollout where the most interesting stuff happens)
                diff = max_seq_len - len(ep_obs64)
                print(f'Required BC seq len: {min_seq_len}-{max_seq_len}; Seq len current {len(ep_obs64)} => appending {diff} frames.')
                idx = len(ep_obs64) - diff
                for i in range(idx, idx+diff):
                    ep_obs64.append(ep_obs64[i])
                    ep_obs72.append(ep_obs72[i])
                    ep_act_det.append(ep_act_det[i])
                    ep_act_sto.append(ep_act_sto[i])
                    ep_act_logits.append(ep_act_logits[i])
                    ep_info.append(ep_info[i])
                    ep_done.append(True)

            assert len(ep_obs64) == max_seq_len

            ep_obs64 = np.array(ep_obs64, dtype=np.uint8)
            ep_obs72 = np.array(ep_obs72, dtype=np.uint8)
            ep_act_det = np.array(ep_act_det, dtype=np.uint8)
            ep_act_sto = np.array(ep_act_sto, dtype=np.uint8)
            ep_act_logits = np.array(ep_act_logits, dtype=np.float16)
            ep_info = np.array(ep_info)
            ep_done = np.array(ep_done, dtype=bool)
            return ep_obs64, ep_obs72, ep_act_det, ep_act_sto, ep_act_logits, ep_info, ep_done


def sample_rollout_mp(index, crf_size, model, min_seq_len, max_seq_len, wrap_normalizeimg, n_stack, n_skip):
    """Wraps generate_rollout for multi-processing."""
    print(index)
    return sample_rollout_seq_len_append(crf_size=crf_size, model=model, min_seq_len=min_seq_len, max_seq_len=max_seq_len, wrap_normalizeimg=wrap_normalizeimg, n_stack=n_stack, n_skip=n_skip)


def sample_rollouts(crf_size, model, pool_size, n_seq, min_seq_len, max_seq_len, wrap_normalizeimg, n_stack, n_skip):
    """Generates a dataset of n_seq in parallel."""

    # Init argument of sample_rollout function and pool.
    partial_sample_rollout_mp = partial(sample_rollout_mp, crf_size=crf_size, model=model, min_seq_len=min_seq_len, max_seq_len=max_seq_len, wrap_normalizeimg=wrap_normalizeimg, n_stack=n_stack, n_skip=n_skip)
    pool = Pool(pool_size)

    # Perform rollouts.
    t = time.time()
    obs64, obs72, act_det, act_sto, act_logits, info, done = zip(*pool.map(partial_sample_rollout_mp, range(n_seq)))
    print("Simulation time: {}".format(time.time() - t))

    return np.stack(obs64), np.stack(obs72), np.stack(act_det), np.stack(act_sto), np.stack(act_logits), np.stack(info), np.stack(done)


def generate_behavioral_cloning_dataset(
    model: "base_class.BaseAlgorithm",
    crf_size,
    pool_size,
    n_seq,
    min_seq_len,
    max_seq_len,
    wandb_id,
    wrap_normalizeimg,
    n_stack,
    n_skip,
):
    
    # Note: Code below has been moved to main, due to errors in CUDA fork when using wandb 
    # and multithreading to generate BC data. Enable it there.
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass

    print(f'Generating BC dataset with {n_seq} rollouts of min seq len {min_seq_len} and max seq len {max_seq_len} for wandb_id {wandb_id}.')
    ds_obs64, ds_obs72, ds_act_det, ds_act_sto, ds_act_logits, ds_info, ds_done = sample_rollouts(
        crf_size,
        model.policy,
        pool_size,
        n_seq,
        min_seq_len,
        max_seq_len,
        wrap_normalizeimg,
        n_stack,
        n_skip,
    )

    # Save data
    print('Observations 64 Shape:', ds_obs64.shape)
    print('Observations 72 Shape:', ds_obs72.shape)
    print('Actions Det Shape    :', ds_act_det.shape)
    print('Actions Sto Shape    :', ds_act_sto.shape)
    print('Actions Logits Shape :', ds_act_logits.shape)
    print('Info Shape           :', ds_info.shape)
    assert ds_obs64.shape[0] == ds_obs72.shape[0]
    assert ds_obs64.shape[0] == ds_act_det.shape[0]
    assert ds_obs64.shape[0] == ds_act_sto.shape[0]
    assert ds_obs64.shape[0] == ds_act_logits.shape[0]
    assert ds_obs64.shape[0] == ds_info.shape[0]
    assert ds_obs64.shape[0] == ds_done.shape[0]
    HOME_DATA_DIR = os.path.join(os.getenv("HOME"), "data")
    data_root = os.path.join(HOME_DATA_DIR, "crafter")
    os.makedirs(data_root, exist_ok=True)
    output_h5_file = os.path.join(data_root, f'bc_{wandb_id}_ns{n_seq}_minsl{min_seq_len}_maxsl{max_seq_len}.h5')
    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('obs64', data=ds_obs64)
        f.create_dataset('obs72', data=ds_obs72)
        f.create_dataset('act_det', data=ds_act_det)
        f.create_dataset('act_sto', data=ds_act_sto)
        f.create_dataset('act_logits', data=ds_act_logits)
        f.create_dataset('info', data=ds_info)
        f.create_dataset('done', data=ds_done)
    print(f'Written dataset to {output_h5_file}')
