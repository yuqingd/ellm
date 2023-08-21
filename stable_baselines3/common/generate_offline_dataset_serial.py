import string

import os
import h5py
import gym
import numpy as np

from stable_baselines3.common import base_class


def generate_behavioral_cloning_dataset_serial(
    model: "base_class.BaseAlgorithm",
    env: gym.Env,
    n_seq: int = 10,
    seq_len: int = 10,
    wandb_id: string = '',
    deterministic: bool = True,
):
    
    print(f'Generating BC dataset with {n_seq} rollouts of len {seq_len} for wandb_id {wandb_id}.')
    dataset_obs = []
    dataset_act = []
    ep_cnt = 0
    while ep_cnt < n_seq:
        print(f'Generating episode [{ep_cnt}/{n_seq}]:', end=' ')
        ep_obs = []
        ep_act = []
        states = None
        obs = env.reset()
        done = False
        ep_step = 0
        while len(ep_obs) < seq_len and not done:
            ep_step += 1
            print(ep_step, end=' ')
            act, states = model.predict(obs, state=states, deterministic=deterministic)
            obs, rewards, done, info = env.step(act)
            obs_large = env._render_large

            ep_obs.append(obs_large)
            ep_act.append(act)

            if done and len(ep_obs) < seq_len:
                print(f'Finished before {seq_len} frames reached.')

            if len(ep_obs) == seq_len:
                print()
                ep_cnt += 1
                ep_obs = np.array(ep_obs, dtype=np.uint8)
                ep_act = np.array(ep_act, dtype=np.uint8)
                dataset_obs.append(ep_obs)
                dataset_act.append(ep_act)

    dataset_obs = np.array(dataset_obs, dtype=np.uint8)
    dataset_act = np.array(dataset_act, dtype=np.uint8)

    # Save data
    print('Observations Shape:', dataset_obs.shape)
    print('Actions Shape     :', dataset_act.shape)
    assert dataset_obs.shape[0] == dataset_act.shape[0], "The number of observations and actions does not match"
    HOME_DATA_DIR = os.path.join(os.getenv("HOME"), "data")
    data_root = os.path.join(HOME_DATA_DIR, "crafter")
    output_h5_file = os.path.join(data_root, f'offline_{wandb_id}_ns{n_seq}_sl{seq_len}.h5')
    with h5py.File(output_h5_file, 'w') as f:
        f.create_dataset('obs', data=dataset_obs)
        f.create_dataset('act', data=dataset_act)
    print(f'Written dataset to {output_h5_file}')
