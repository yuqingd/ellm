from rlf.run_settings import RunSettings
from rlf.rl.loggers.wb_logger import WbLogger
from rlf.rl.loggers.base_logger import BaseLogger
from rlf import BaseAlgo
from rlf import PPO
from rlf import DistActorCritic
import orp_env_adapter
import numpy as np
from rlf import run_policy
from rlf.rl.model import MLPBase, Flatten, BaseNet, IdentityBase
import torch.nn as nn
from orp.env import get_env
from method.state_mp import StateModularPolicy
from method.multi_modular_policy import MultiStateModularPolicy
import yaml
from orp.env import TASK_CONFIGS_DIR, AGENT_CONFIGS_DIR
import os.path as osp
try:
    import wandb
except:
    pass

class ImgEncoder(BaseNet):
    def __init__(self, obs_shape, hidden_size=256):
        super().__init__(False, hidden_size, hidden_size)
        self.n_channels = obs_shape[0]

        self.net = nn.Sequential(
                nn.Conv2d(self.n_channels, 32, kernel_size=8, stride=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                Flatten(),
                # For 128 x 128
                #nn.Linear(2592, hidden_size))
                # For 64 x 64
                nn.Linear(288, hidden_size))

    def forward(self, inputs, rnn_hxs, masks):
        tmp = self.net(inputs)
        return tmp, rnn_hxs


def get_base_encoder(i_shape):
    if len(i_shape) == 1:
        return IdentityBase(i_shape)
    else:
        return ImgEncoder(i_shape)


class VizPlanRunSettings(RunSettings):
    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def _get_env_interface(self, args, task_id=None):
        base_dir = 'orp'
        with open(osp.join(base_dir, TASK_CONFIGS_DIR, args.hab_env_config + '.yaml'), 'r') as f:
            env_config = yaml.safe_load(f)
        with open(osp.join(base_dir, AGENT_CONFIGS_DIR, args.hab_agent_config + '.yaml'), 'r') as f:
            agent_config = yaml.safe_load(f)
        if not self.base_args.no_wb:
            wandb.config.update(env_config)
            wandb.config.update(agent_config)

        return super()._get_env_interface(args, task_id)

    def get_config_file(self):
        return './config.yaml'

    def get_policy(self):
        if self.base_args.alg == 'smp':
            return StateModularPolicy()
        elif self.base_args.alg == 'csmp':
            return MultiStateModularPolicy()
        else:
            ag = self.base_args.hab_agent_config
            env = self.base_args.hab_env_config
            fuse_states = []
            if 'state' not in ag:
                fuse_states = ['joint', 'ee_pos']

            return DistActorCritic(
                    fuse_states=fuse_states,
                    get_base_net_fn=get_base_encoder
                    )

    def get_algo(self):
        if self.base_args.alg == 'smp':
            return BaseAlgo()
        else:
            return PPO()

    def get_add_args(self, parser):
        parser.add_argument('--no-wb', default=False, action='store_true')
        parser.add_argument('--alg', type=str, default='smp')
        parser.add_argument('--hab-agent-config', type=str, default='arm')
        parser.add_argument('--hab-env-config', type=str, default='rearrang')


if __name__ == '__main__':
    run_policy(VizPlanRunSettings())
