#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from copy import deepcopy
import yaml
import numpy as np
import torch
from text_housekeep.habitat_lab.habitat_baselines.common.baseline_registry import baseline_registry
from text_housekeep.habitat_lab.habitat_baselines.common.utils import Flatten, ResizeCenterCropper, CustomFixedCategorical
from text_housekeep.habitat_lab.habitat_baselines.rl.ddppo.policy import resnet
from text_housekeep.habitat_lab.habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from text_housekeep.habitat_lab.habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder, RNNStateEncoderOld
from text_housekeep.habitat_lab.habitat_baselines.rl.ppo import Net, Policy
from torch import nn as nn
from text_housekeep.cos_eor.utils.objects_to_byte_tensor import dec_bytes2obj
from text_housekeep.habitat_lab.habitat.core.registry import registry


class KQPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()
        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)
        return scores


@baseline_registry.register_policy
class SingleObjectResNetPolicy(Policy):
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
            SingleObjectResNetEncoder(
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
            init_actions=False
        )
        # embedding dimensions
        self.embed_dim = self.net.output_size

        # load semantic-class annotations
        sem_classes_path = "cos_eor/scripts/dump/semantic_classes_amt.yaml"
        objects_data = yaml.load(open(sem_classes_path, "r"))
        sid_class_map = dict(objects_data["semantic_class_id_map"])
        self.sids = list(sid_class_map.keys())

        # build iid and sid embeddings
        assert min(self.sids) >= 0
        self.sid_embed_inds = np.arange(0, max(self.sids) + 1) * 0
        self.sid_embed_inds[self.sids] = np.arange(0, len(self.sids))
        self.sid_embeddings = nn.Embedding(len(self.sids), self.embed_dim)

        # total are never more than 200 for any scene?
        self.max_iids = 200
        self.iid_embeddings = nn.Embedding(self.max_iids, self.embed_dim)

        # fixed actions are -- [forward, left, right, look-up, look-down, stop]
        self.nav_actions = nn.Linear(self.net.output_size, self.dim_actions-1)

        # dynamic actions are iids to choose from at every step,
        # these are selected using a pointer network
        self.ptr_actions = KQPtrNet(self.net.output_size, self.embed_dim)

        # attach sid and iid embeddings to resnet
        self.net.build_obj_encoding(self.iid_embeddings, self.sid_embeddings)

    def action_distribution(self, features, observations):
        nav_logits = self.nav_actions(features)
        device = observations["rgb"].device

        # pad with max-iids
        pp_iids = observations["visible_obj_iids"]
        pp_mask = (pp_iids != 0).float()
        pp_sids = observations["visible_obj_sids"]
        pp_sids_numpy = pp_sids.cpu().int().numpy()
        pp_sids_embed_inds = torch.from_numpy(self.sid_embed_inds[pp_sids_numpy]).to(device)

        if pp_iids.max() >= self.iid_embeddings.weight.shape[0] or \
                pp_sids_embed_inds.max() >= self.sid_embeddings.weight.shape[0]:
            import pdb
            pdb.set_trace()

        # tensorize and build embeddings
        pp_iid_embeds = self.iid_embeddings(pp_iids.long())
        pp_sids_embeds = self.sid_embeddings(pp_sids_embed_inds.long())

        # sum both embeddings to generate the final dynamic targets
        pp_embeds = pp_iid_embeds + pp_sids_embeds
        pp_logits = self.ptr_actions(features, pp_embeds, pp_mask)

        logits = torch.cat([nav_logits, pp_logits], dim=-1)
        return CustomFixedCategorical(logits=logits), pp_iids

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution, iids = self.action_distribution(features, observations)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, iids

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution, iids = self.action_distribution(features, observations)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        return value, action_log_probs, distribution_entropy, rnn_hidden_states


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


class SingleObjectResNetEncoder(Net):
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

        # actions + padding?
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        self._n_input_goal = 3
        self.pos_embedding = nn.Linear(self._n_input_goal, 32)

        self._n_input_goal = 32
        self._n_input_pos = 32
        self._n_input_obj = 32
        self._hidden_size = hidden_size
        rnn_input_size = self._n_input_pos + self._n_prev_action + self._n_input_obj

        if not config.RL.USE_SEMANTIC_FRAME:
            observation_space = deepcopy(observation_space)
            observation_space.spaces.pop("semantic", None)
            print(f"Not using semantic frame")
        else:
            print(f"Using semantic frame")

        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            # obs_transform=None
        )
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Linear(
                np.prod(self.visual_encoder.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

        self.state_encoder = RNNStateEncoderOld(
            self._hidden_size + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    def build_obj_encoding(self, iid_embeddings, sid_embeddings):
        self.iid_embeddings = iid_embeddings
        self.iid_embeddings_linear = nn.Linear(iid_embeddings.weight.shape[-1], 32)

        self.sid_embeddings = sid_embeddings
        self.sid_embeddings_linear = nn.Linear(sid_embeddings.weight.shape[-1], 32)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def is_blind(self):
        return False

    def get_pos_encoding(self, observations):
        # R, sin-theta, cos-theta
        pos_obs = torch.stack(
            [
                observations["gps_compass"][:, 0],
                torch.cos(-observations["gps_compass"][:, 1]),
                torch.sin(-observations["gps_compass"][:, 1]),
            ],
            -1,
        )
        return self.pos_embedding(pos_obs)

    def get_obj_encoding(self, observations):
        replaces_ids = observations["gripped_object_id"] == -1
        gripped_sid = deepcopy(observations["gripped_sid"])
        gripped_iid = deepcopy(observations["gripped_iid"])
        gripped_sid[replaces_ids] = 0
        gripped_iid[replaces_ids] = 0

        if gripped_sid.max() >= self.sid_embeddings.weight.shape[0] \
                or gripped_iid.max() >= self.iid_embeddings.weight.shape[0]:
            import pdb
            pdb.set_trace()

        obj_embeddings = self.sid_embeddings_linear(
            self.sid_embeddings(gripped_sid.long())
        ) + \
        self.iid_embeddings_linear(
            self.iid_embeddings(gripped_iid.long())
        )
        obj_embeddings = obj_embeddings.squeeze(1)
        return obj_embeddings

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if "visual_features" in observations:
            visual_feats = observations["visual_features"]
        else:
            visual_feats = self.visual_encoder(observations)

        # BSx256
        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        # BSx32
        pos_encoding = self.get_pos_encoding(observations)

        # BSx32
        obj_encoding = self.get_obj_encoding(observations)

        # BSx32, Todo: Add iid embeddings
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x += [pos_encoding, prev_actions, obj_encoding]

        # BSx352
        x = torch.cat(x, dim=1)

        # BSx256
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
