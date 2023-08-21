#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import traceback
import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from text_housekeep.habitat_lab.habitat_baselines.common.baseline_registry import baseline_registry
from text_housekeep.habitat_lab.habitat_baselines.common.utils import Flatten, ResizeCenterCropper
from text_housekeep.habitat_lab.habitat_baselines.rl.ddppo.policy import resnet
from text_housekeep.habitat_lab.habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from text_housekeep.habitat_lab.habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from text_housekeep.habitat_lab.habitat_baselines.rl.ppo import Net, Policy

from text_housekeep.habitat_lab.habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder


@baseline_registry.register_policy
class AllObjectResNetPolicy(Policy):
    def __init__(
            self,
            config,
            observation_space,
            action_space,
            hidden_size=512,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            resnet_baseplanes=32,
            backbone="resnet50",
            normalize_visual_inputs=False,
            obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa : B008
            force_blind_policy=False,
            **kwargs
    ):
        super().__init__(
            MultipleObjectResNetEncoder(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                config=config
            ),
            action_space.n,
        )

    @classmethod
    def from_config(cls, config, envs, **kwargs):
        return cls(
            observation_space=envs.observation_spaces[0],
            action_space=envs.action_spaces[0],
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in envs.observation_spaces[0].spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            config=config,
            **kwargs
        )


class MultipleObjectResNetEncoder(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            config,
            **kwargs
    ):
        super().__init__()

        self.object_sensor_uuid = config.TASK_CONFIG.TASK.NEXT_OBJECT_SENSOR_UUID

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._n_input_goal = 3
        self.goal_embedding = nn.Linear(self._n_input_goal, 32)
        self.pos_embedding = nn.Linear(self._n_input_goal, 32)

        self._n_input_goal = 32 * 5
        self._n_input_pos = 32 * 5

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_input_pos + self._n_prev_action
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=None
        )
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

        self.state_encoder = RNNStateEncoder(
            self._hidden_size + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def is_blind(self):
        return False

    def get_tgt_encoding(self, observations):
        num_envs, num_objs = observations['all_object_positions'].shape[0:2]

        goal_observations = observations['all_object_positions'].reshape(num_envs * num_objs, -1)
        pos_observations = observations['all_object_goals'].reshape(num_envs * num_objs, -1)

        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )
        goal_emb = self.goal_embedding(goal_observations).reshape(num_envs, num_objs, -1)

        pos_observations = torch.stack(
            [
                pos_observations[:, 0],
                torch.cos(-pos_observations[:, 1]),
                torch.sin(-pos_observations[:, 1]),
            ],
            -1,
        )
        pos_emb = self.pos_embedding(pos_observations).reshape(num_envs, num_objs, -1)

        return goal_emb.reshape(num_envs, -1), pos_emb.reshape(num_envs, -1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if "visual_features" in observations:
            visual_feats = observations["visual_features"]
        else:
            visual_feats = self.visual_encoder(observations)

        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        goal_encoding, pos_encoding = self.get_tgt_encoding(observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x += [goal_encoding, pos_encoding, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states