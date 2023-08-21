import sys
# sys.path.append('/Users/harsh/Work/habitat-sim/')
# sys.path.append('/Users/harsh/Work/habitat-api/')
# import attr
# attr.set_run_validators(False)

import argparse
import os
import random
import time
import traceback
import ipdb
import json_tricks
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import shortuuid


cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from gym import spaces
from torch.optim.lr_scheduler import LambdaLR
from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage_old import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.config.default import get_config

# custom non-nested imports for registering components
from cos_eor.policy.policy import SingleObjectResNetPolicy
from cos_eor.policy.policy_multiple import AllObjectResNetPolicy
from cos_eor.policy.rank import RankModule
from cos_eor.policy.oracle_rank import OracleRankModule
from cos_eor.policy.nav import NavModule
from cos_eor.policy.explore import ExploreModule
from cos_eor.env.env import CosRearrangementRLEnv
from cos_eor.task.sensors import *
from cos_eor.task.measures import *


def load_replay_data():
    p = 'data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/replays/map_object/6T4DVDAzwC2ZzkEtGeoerS/'
    # data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/videos/map_object/dPKBtXvPV6S2RyA8HVckGZ
    files = os.listdir(p)
    j = 0

    replay_data = []
    for i, file in enumerate(files[j:]):
        with open(os.path.join(p, file), 'r') as f:
            data = json_tricks.load(f)
            replay_data.append(data)
    replay_df = pd.DataFrame(replay_data)
    return replay_df


class CosRearrangementTrainer(BaseRLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")
        self.rmap = registry.mapping
        self._static_encoder = False
        self._encoder = None
        self.debug_count = 0

    def debug_obs(self, obs):
        from PIL import Image
        def depth_to_rgb(depth_image: np.ndarray, clip_max: float = 10.0) -> np.ndarray:
            """Normalize depth image into [0, 1] and convert to grayscale rgb
            :param depth_image: Raw depth observation image from sensor output.
            :param clip_max: Max depth distance for clipping and normalization.
            :return: Clipped grayscale depth image data.
            """
            d_im = np.clip(depth_image, 0, clip_max)
            d_im /= clip_max
            # d_im = np.stack([d_im for _ in range(3)], axis=2)
            rgb_d_im = (d_im * 255).astype(np.uint8)
            return rgb_d_im
        if "rgb_3rd_person" in obs:
            rgb_3rd_person = Image.fromarray(obs['rgb_3rd_person'])
            rgb_3rd_person.save("debug-rgb-3rd.jpeg")
        if "rgb" in obs:
            rgb = Image.fromarray(obs['rgb'])
            rgb.save("debug-rgb.jpeg")
        if "semantic" in obs:
            from habitat_sim.utils.common import d3_40_colors_rgb
            semantic = obs['semantic']
            semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            semantic_img.save("debug-semantic.png")
        if "depth" in obs:
            d_im = depth_to_rgb(obs['depth'], clip_max=1.0)[:, :, 0]
            depth_map = np.stack([d_im for _ in range(3)], axis=2)
            depth = Image.fromarray(depth_map)
            depth.save("debug-depth.jpeg")

    def _setup_rank_module(self):
        if self.config.RL.POLICY.rank.name == "oracle":
            self.rank_module = OracleRankModule(self.config.RL.POLICY.rank)
        else:
            self.rank_module = RankModule(self.config.RL.POLICY.rank)

    def _setup_explore_module(self):
        self.explore_module = ExploreModule(self.config.RL.POLICY.explore, self.envs.num_envs)
        self.explore_module.to(self.device)

    def _setup_policy(self, ckpt_dict):
        # check hie or flat
        if self.config.RL.POLICY.name == "E2EPolicy":
            count_steps, count_checkpoints, update_start = self._setup_actor_critic_agent(ckpt_dict)
            self.nav_type = "ppo"
            self.policy = self.agent
        elif self.config.RL.POLICY.name == "HiePolicy":
            from cos_eor.policy.hie_policy import HiePolicy, ExploreModule
            # setup nav module either as shortest-path follower or ppo agent
            if self.config.RL.POLICY.nav.name == "OracleShortestPath":
                count_steps, count_checkpoints, update_start = [0] * 3
                self.nav_module = NavModule(self.envs)
                self.nav_type = "sp"
            elif self.config.RL.POLICY.nav.name == "SingleObjectResNetPolicy":
                count_steps, count_checkpoints, update_start = self._setup_actor_critic_agent(ckpt_dict)
                self.nav_type = "ppo"
            else:
                import pdb
                pdb.set_trace()

            self._setup_rank_module()
            self._setup_explore_module()
            debug_params = {"envs": self.envs}
            nav_param = self.actor_critic if self.nav_type == "ppo" else self.nav_module
            self.policy = HiePolicy(self.envs, nav_param, self.rank_module,
                                    self.explore_module, self.config.RL.POLICY,
                                    self.config.BASE_TASK_CONFIG.TASK, debug_params)
        else:
            import pdb
            pdb.set_trace()
        return count_steps, count_checkpoints, update_start

    def _setup_actor_critic_agent(self, ckpt_dict=None):
        """Load pretrained RGBD encoders, and train policy from scratch."""
        count_steps, count_checkpoints, update_start = 0, 0, 0
        nav_module = baseline_registry.get_policy(self.config.RL.POLICY.nav.name)
        ppo_cfg = self.config.RL.PPO
        self.actor_critic = nav_module.from_config(self.config, self.envs)
        self.actor_critic.to(self.device)

        if ckpt_dict is not None:
            logger.info("Loading Checkpoint")

            # if "state_dict" in ckpt_dict:
            #     count_steps, count_checkpoints, update_start = \
            #         [ckpt_dict["extra_state"][k] for k in ["count_steps", "count_checkpoints", "update"]]
            #     update_start+=1

            # default: True
            if self.config.RL.DDPPO.pretrained:
                self.actor_critic.load_state_dict(
                    {
                        k[len("actor_critic."):]: v
                        for k, v in ckpt_dict["state_dict"].items()
                    }
                )
            # default: False
            elif self.config.RL.DDPPO.pretrained_encoder:
                prefix = "actor_critic.net.visual_encoder."
                self.actor_critic.net.visual_encoder.load_state_dict(
                    {
                        k[len(prefix):]: v
                        for k, v in ckpt_dict["state_dict"].items()
                        if k.startswith(prefix)
                    }
                )
            # default: True
            if not self.config.RL.DDPPO.train_encoder:
                self._static_encoder = True
                for param in self.actor_critic.net.visual_encoder.parameters():
                    param.requires_grad_(False)
            # default: False
            if self.config.RL.DDPPO.reset_critic:
                torch.nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
                torch.nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

        return count_steps, count_checkpoints, update_start

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "object_to_goal_distance",
                         "agent_to_object_distance", 'pickup_order_oracle_next_object',
                         'current_obj_id_oracle_next_object', 'pickup_order_random_object',
                         'current_obj_id_random_object', 'pickup_order_closest_object',
                         'current_obj_id_closest_object', 'pickup_order_l2dist_object', 'current_obj_id_l2dist_object'
                         }

    @classmethod
    def _extract_scalars_from_info(
            cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                        v
                    ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
            cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def save_checkpoint(
            self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def actions_with_joint_logits(self, actions, iids):
        nav_actions = self.actor_critic.dim_actions - 1
        actions_list = [int(a[0].item()) for a in actions]
        actions_args = [int(iids[i][a - nav_actions].item()) if a >= nav_actions else -1 for i, a in enumerate(actions_list)]
        actions_inds = [nav_actions if a >= nav_actions else a for a in actions_list]
        # build actions for stepping envs with iids
        step_actions = [{"action": {"action": a, "action_args": {"iid": i}}} for a, i in zip(actions_inds, actions_args)]
        # replace iids with pp action idx to be stored in rollout storage
        # todo: ideally we should also pass iids as prev-actions in training the policy
        actions = torch.where(actions >= nav_actions, nav_actions, actions)
        return step_actions, actions

    def _collect_rollout_step(
            self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0
        t_sample_action = time.time()
        time_init = time.time()

        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            # get either pointgoals or actions here when using ppo nav
            # output = self.policy.act(step_observation)

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                iids
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action
        step_actions, actions = self.actions_with_joint_logits(actions, iids)
        t_step_env = time.time()
        act_time = time.time()
        outputs = self.envs.step(step_actions)
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        step_time = time.time()

        env_time += time.time() - t_step_env
        t_update_stats = time.time()
        self.debug_count += 1
        if env_time > 0.007:
            time_info = [f"|| {k}: {v}" for k,v in infos[0].items() if "time" in k]
            time_info = " ".join(ti for ti in time_info)
            print(f"Act time: {act_time - time_init} || Full Step time: {step_time - act_time} ")

        # batch the returned observations after executing actions
        batch = batch_obs(observations, device=self.device)
        batch_rewards = batch_obs(rewards, device=self.device)

        # concatenate different reward types along a new dimension then average across these types
        rewards_tensor = torch.cat([v.unsqueeze(1) for k, v in batch_rewards.items()], 1).sum(1)
        if torch.isnan(rewards_tensor).any() or torch.isinf(rewards_tensor).any():
            ipdb.set_trace()

        # todo: figure out why is this needed?
        rewards_tensor = torch.tensor(
            rewards_tensor, dtype=torch.float, device=current_episode_reward.device
        ) * self.config.RL.REWARD_SCALE
        rewards_tensor = rewards_tensor.unsqueeze(1)

        # mask rewards for envs with finished episodes
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards_tensor

        # only update stats for envs that are active
        # todo: likely a bug here!
        running_episode_stats["reward"] += masks * current_episode_reward
        running_episode_stats["count"] += masks

        # todo: fix
        sum_dones = max(np.sum(dones), 1)
        # running_episode_stats['object_to_goal_distance'] = np.sum(
        #     [r if done else 0 for done, r in zip(dones, batch_rewards['object_to_goal_dist_reward'])]
        # ) / sum_dones
        # running_episode_stats['agent_to_object_distance'] = np.sum([
        #     r if done else 0 for done, r in zip(dones, batch_rewards['agent_to_object_dist_reward'])
        # ]) / sum_dones

        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            running_episode_stats[k] += masks * v

        # YK: remove rewards from finished episodes
        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        # todo: check what is being stored for actions
        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards_tensor,
            masks,
        )
        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.envs.num_envs

    def _update_nav_module(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        rollouts.after_update()
        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy
        )

    def build_uuid(self, uuid):
        """Generates next available path for colliding experiment tags"""
        dump_folder = os.path.join(self.config.CHECKPOINT_FOLDER, uuid)
        if os.path.isdir(dump_folder):
            all_folders = os.listdir(os.path.dirname(dump_folder))
            curr_folder_name = uuid
            max_count = -1
            for fol_name in all_folders:
                if curr_folder_name in fol_name:
                    count = int(fol_name.split("_")[-1]) if "_" in fol_name else -1
                    if count > max_count:
                        max_count = count
            uuid = f"{uuid}_{max_count+1}"
        return uuid

    def handle_log_folder(self, tag):
        self.config.defrost()
        uuid = self.build_uuid(tag)
        self.config.CHECKPOINT_FOLDER = os.path.join(
            self.config.CHECKPOINT_FOLDER, uuid, "checkpoints"
        )
        self.config.TENSORBOARD_DIR = os.path.join(
            os.path.dirname(self.config.CHECKPOINT_FOLDER),
            self.config.TENSORBOARD_DIR
        )
        self.config.LOG_FILE = os.path.join(
            os.path.dirname(self.config.CHECKPOINT_FOLDER),
            self.config.LOG_FILE
        )
        self.config.freeze()
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
            os.makedirs(self.config.TENSORBOARD_DIR)

        registry.mapping["checkpoint_folder"] = self.config.CHECKPOINT_FOLDER
        logger.add_filehandler(self.config.LOG_FILE)
        logger.info(
            "UUID / Folder name: {}".format(uuid)
        )
        return uuid

    def build_obs_spaces(self, batch, observations):
        """ Edits the observations-space for rollout storage on the fly!"""
        # Todo: Refactor to reduce duplication!
        obs_space = self.envs.observation_spaces[0]
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        obs_keys = list(observations[0].keys())
        for ok in obs_keys:
            if ok not in obs_space.spaces:
                if ok == "semantic_class":
                    obs_space.spaces[ok] = deepcopy(obs_space.spaces["semantic"])
                elif ok in [
                    "visible_obj_iids",
                    "visible_obj_sids",
                    "visible_rec_iids",
                    "visible_rec_sids",
                ]:
                    obs_space.spaces[ok] = spaces.Box(low=0, high=255, shape=(200,))
                elif ok == "gps_compass":
                    obs_space.spaces[ok] = spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=(2,),
                        dtype=np.float32,
                    )
                elif ok in [
                    "gripped_object_id",
                    "gripped_iid",
                    "gripped_sid",
                    "num_visible_objs",
                    "num_visible_recs",
                ]:
                    obs_space.spaces[ok] = spaces.Discrete(255)
                else:
                    import pdb
                    pdb.set_trace()
                    raise ValueError
        return obs_space

    def train(self, ckpt_path=None, debug=False, tag=None) -> None:
        ckpt_dict = None
        if ckpt_path is not None and ckpt_path != "":
            logger.info("checkpoint_path: {}".format(ckpt_path))
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        uuid = self.handle_log_folder(tag)
        count_steps, count_checkpoints, update_start = self._setup_policy(ckpt_dict)

        if self.nav_type == "ppo":
            logger.info(
                "agent number of parameters: {} Million".format(
                    sum(param.numel() for param in self.agent.parameters()) / 1000000
                )
            )

        observations = self.envs.reset()
        # self.debug_obs(observations[0])
        # observations = self.envs.step([0])
        # self.debug_obs(observations[0][0])
        batch = batch_obs(observations, device=self.device)
        obs_space = self.build_obs_spaces(batch, observations)

        if self.config.RL.POLICY.name == "E2EPolicy":
            rollouts = RolloutStorage(
                ppo_cfg.num_steps,
                self.envs.num_envs,
                obs_space,
                self.envs.action_spaces[0],
                ppo_cfg.hidden_size,
                num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            )
            rollouts.to(self.device)
            try:
                # insert first-step observations
                for sensor in rollouts.observations:
                    rollouts.observations[sensor][0].copy_(batch[sensor])
            except:
                import pdb
                pdb.set_trace()
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
            )

            # batch and observations may contain shared PyTorch CUDA
            # tensors.  We must explicitly clear them here otherwise
            # they will be kept in memory for the entire duration of training!
            batch = None
            observations = None
            current_episode_reward = torch.zeros(self.envs.num_envs, 1)
            running_episode_stats = dict(
                count=torch.zeros(self.envs.num_envs, 1),
                reward=torch.zeros(self.envs.num_envs, 1),
            )

            # window size is 50 by default
            window_episode_stats = defaultdict(
                lambda: deque(maxlen=ppo_cfg.reward_window_size)
            )

            t_start = time.time()
            env_time = 0
            pth_time = 0

            with TensorboardWriter(
                    self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) as writer:
                for update in tqdm.tqdm(range(update_start, self.config.NUM_UPDATES), desc="Agent Update", total=self.config.NUM_UPDATES, initial=update_start):
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler.step()

                    if ppo_cfg.use_linear_clip_decay:
                        self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                            update, self.config.NUM_UPDATES
                        )

                    # collect agent rollouts
                    t_init = time.time()
                    # for step in tqdm.tqdm(range(ppo_cfg.num_steps), desc="Rollouts", total=ppo_cfg.num_steps):
                    nav_step = 0
                    while nav_step < ppo_cfg.num_steps:
                        (
                            delta_pth_time,
                            delta_env_time,
                            delta_steps,
                        ) = self._collect_rollout_step(
                            rollouts,
                            current_episode_reward,
                            running_episode_stats
                        )

                        pth_time += delta_pth_time
                        env_time += delta_env_time
                        count_steps += delta_steps
                        nav_step += 1

                    t_rollouts = time.time()

                    # update using rollouts
                    # print("Update: {}".format(str(update)))
                    (
                        delta_pth_time,
                        value_loss,
                        action_loss,
                        dist_entropy,
                    ) = self._update_nav_module(ppo_cfg, rollouts)
                    t_updates = time.time()
                    pth_time += delta_pth_time
                    for k, v in running_episode_stats.items():
                        window_episode_stats[k].append(v.clone()) if type(v) == torch.Tensor else v
                    losses = [value_loss, action_loss]

                    # add all scalars to tensorboard
                    # self.tboard_write(writer, window_episode_stats, losses, count_steps)

                    t_end = time.time()
                    # log stats
                    if True or update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "uuid: {}\tupdate: {}\tfps: {:.3f}\t".format(
                                uuid, update, count_steps / (env_time)
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )

                        # logger.info(
                        #     "Average window size: {}  {}".format(
                        #         len(window_episode_stats["count"]),
                        #         "  ".join(
                        #             "{}: {:.3f}".format(k, v / deltas["count"])
                        #             for k, v in deltas.items()
                        #             if k != "count"
                        #         ),
                        #     )
                        # )

                        logger.info(f"Time -- rollouts: {t_rollouts - t_init} and "
                              f"agent update: {t_updates - t_rollouts} and end: {t_end - t_updates}")

                    # # # checkpoint model
                    # if update % self.config.CHECKPOINT_INTERVAL == 0:
                    #     checkpoint_file = f"ckpt_step_{update}.pth"
                    #     self.save_checkpoint(
                    #         checkpoint_file, dict(count_steps=count_steps,
                    #                               update=update,
                    #                               count_checkpoints=count_checkpoints)
                    #     )
                    #     print(f"Dumped Checkpoint: {checkpoint_file}")
                    #     count_checkpoints += 1
        elif self.nav_type == "sp":
            pass
        self.envs.close()

    def tboard_write(self, writer, window_episode_stats, losses, count_steps):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()  # why?
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in window_episode_stats.items()
        }

        print(f"count: {deltas['count']}")
        deltas["count"] = max(deltas["count"], 1.0)
        writer.add_scalar(
            "reward", deltas["reward"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "episode_success", deltas["episode_success"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "object_success", deltas["objs_success"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "episode_length", deltas["episode_length"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "episode_mean_dist", deltas["episode_mean_dist"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "no_of_objects", deltas["no_of_objects"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "episode_spl", deltas["episode_spl"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "controller_spl", deltas["controller_spl"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "planner_spl", deltas["planner_spl"] / deltas["count"], count_steps
        )
        writer.add_scalar(
            "critic_loss", losses[0], count_steps
        )
        writer.add_scalar(
            "actor_loss", losses[1], count_steps
        )

    def evaluate(self, checkpoint_path):
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # replay_df = load_replay_data()
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        uuid = os.path.dirname(os.path.dirname(checkpoint_path)).split('/')[-1]

        eval_uuid = str(shortuuid.uuid())

        config.CHECKPOINT_FOLDER = os.path.join(
            config.CHECKPOINT_FOLDER, uuid, "checkpoints"
        )
        config.TENSORBOARD_DIR = os.path.join(
            os.path.dirname(config.CHECKPOINT_FOLDER),
            config.TENSORBOARD_DIR
        )
        config.LOG_FILE = os.path.join(
            os.path.dirname(config.CHECKPOINT_FOLDER),
            eval_uuid + '_' + config.LOG_FILE
        )

        config.VIDEO_DIR = os.path.join(
            os.path.dirname(config.CHECKPOINT_FOLDER),
            config.VIDEO_DIR,
            config.TASK_CONFIG.TASK.NEXT_OBJECT_SENSOR_UUID,
            eval_uuid
        )
        config.REPLAY_DIR = os.path.join(
            os.path.dirname(config.CHECKPOINT_FOLDER),
            'replays',
            config.TASK_CONFIG.TASK.NEXT_OBJECT_SENSOR_UUID,
            eval_uuid
        )
        config.EVAL_UUID = eval_uuid

        config.freeze()

        if not os.path.isdir(config.CHECKPOINT_FOLDER):
            os.makedirs(config.CHECKPOINT_FOLDER)
            os.makedirs(config.TENSORBOARD_DIR)

        os.makedirs(config.VIDEO_DIR)
        os.makedirs(config.REPLAY_DIR)

        print("Log file: {}".format(config.LOG_FILE))
        logger.add_filehandler(config.LOG_FILE)

        logger.info(
            "UUID: {}".format(uuid)
        )

        logger.info(
            "Eval UUID: {}".format(eval_uuid)
        )

        logger.info(
            "NEXT_OBJECT_SENSOR: {}".format(config.TASK_CONFIG.TASK.NEXT_OBJECT_SENSOR_UUID)
        )

        ppo_cfg = config.RL.PPO
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = config.EVAL.CONTENT_SCENES
        # config.NUM_PROCESSES = 1
        config.freeze()

        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG
            if config.VIDEO_OPTION[0] == 'disk':
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("EOR_TOP_DOWN_MAP")
            if "COLLISIONS" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            if "RGB_SENSOR" not in config.SENSORS:
                config.SENSORS.append("RGB_SENSOR")
            config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
            config.freeze()
        self.config = config.clone()

        logger.info(f"env config: {self.config}")
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        # using just depth sensor
        if 'rgb' in self.envs.observation_spaces[0].spaces:
            del self.envs.observation_spaces[0].spaces['rgb']

        self._setup_policy(ckpt_dict)
        # self.agent.load_state_dict(["state_dict"])
        # self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)

        self.actor_critic.eval()

        while (
                len(stats_episodes) < self.config.TEST_EPISODE_COUNT
                and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

                prev_actions.copy_(actions)

            at = [a[0].item() for a in actions]
            outputs = self.envs.step(at)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, self.device)

            batch_rewards = batch_obs(rewards)
            batch_rewards = torch.cat([v.unsqueeze(1) for k, v in batch_rewards.items()], 1).sum(1)

            rewards = torch.tensor(
                batch_rewards, dtype=torch.float, device=self.device
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=self.device,
            )

            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            episodes_to_skip = [] * self.envs.num_envs

            n_envs = self.envs.num_envs

            for i in range(n_envs):
                if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # if not replay_df[(
                #     (replay_df['episode_id']==str(next_episodes[i].episode_id)) &
                #     (replay_df['scene_id']==str(next_episodes[i].scene_id))
                #     )].empty:
                #     print(f"Skipping Scene: {next_episodes[i].episode_id}, {next_episodes[i].scene_id}")
                #     episodes_to_skip[i] = True
                #     self.envs.reset_at(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    # pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            scene_id=current_episodes[i].scene_id.split('/')[-1],
                            checkpoint_idx=1,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=None,
                        )

                        print("Done: {} ES: {:.3f}, OS: {:.3f}, " \
                              "No of obj: {:.3f}, E-SPL: {:.3f}, " \
                              "C-SPL: {:.3f}, P-SPL: {:.3f}, Avg Dist: {:.3f}, " \
                              "Avg length: {}, " \
                              "Oracle: {}, Given: {}".format(
                            len(stats_episodes),
                            infos[i]['episode_success'],
                            infos[i]['objs_success'],
                            infos[i]['no_of_objects'],
                            infos[i]['episode_spl'],
                            infos[i]['controller_spl'],
                            infos[i]['planner_spl'],
                            infos[i]['episode_mean_dist'],
                            infos[i]['episode_length'],
                            infos[i]['pickup_order_oracle_next_object'],
                            infos[i]['pickup_order_' + self.config.TASK_CONFIG.TASK.NEXT_OBJECT_SENSOR_UUID])
                        )
                        rgb_frames[i] = []

                        aggregated_stats = dict()
                        num_episodes = len(stats_episodes)

                        try:
                            for stat_key in next(iter(stats_episodes.values())).keys():
                                aggregated_stats[stat_key] = (
                                        sum(v[stat_key] for v in stats_episodes.values())
                                        / num_episodes
                                )
                        except Exception as e:
                            import ipdb
                            ipdb.set_trace()

                        print("\nUUID: {}, Done: {}, ES: {:.3f}, OS: {:.3f}, " \
                              "No of obj: {:.3f}, E-SPL: {:.3f}, " \
                              "C-SPL: {:.3f}, P-SPL: {:.3f}, Avg Dist: {:.3f}, " \
                              "Avg length: {} Episode Distance: {}".format(
                            eval_uuid,
                            num_episodes,
                            aggregated_stats['episode_success'],
                            aggregated_stats['objs_success'],
                            aggregated_stats['no_of_objects'],
                            aggregated_stats['episode_spl'],
                            aggregated_stats['controller_spl'],
                            aggregated_stats['planner_spl'],
                            aggregated_stats['episode_mean_dist'],
                            aggregated_stats['episode_length'],
                            aggregated_stats['episode_distance']
                        ))

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                    sum(v[stat_key] for v in stats_episodes.values())
                    / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        self.envs.close()

        print("environments closed")
        print(" Finished Evaluating {} episodes".format(num_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="checkpoint path"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
        help="debug"
    )
    parser.add_argument(
        "--time",
        default=False,
        action='store_true',
        help="time"
    )
    parser.add_argument(
        "--tag",
        required=True,
        type=str,
        help="experiment-tag"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, ckpt_path, opts=None, **kwargs):
    config = get_config(exp_config, opts)
    print(config.TASK_CONFIG.SEED, config.NUM_PROCESSES)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
    np.random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
    torch.manual_seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)  # Change this for different environment.
    registry.mapping["debug"] = kwargs['debug']
    registry.mapping["time"] = kwargs['time']

    if kwargs['debug'] and "RGB_SENSOR_3RD_PERSON" not in config.SENSORS and "kyash" in os.getcwd():
        # add 3rd person semantic sensor and running local
        config.SENSORS.append("RGB_SENSOR_3RD_PERSON")

    trainer = CosRearrangementTrainer(config)

    if run_type == "train":
        trainer.train(ckpt_path, debug=kwargs['debug'], tag=kwargs['tag'])
    elif run_type == "eval":
        # expects path to the exact model checkpoint.
        trainer.evaluate(ckpt_path)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=4,5,6 GLOG_minloglevel=2 MAGNUM_LOG="quiet" python rearrangement/baselines/agent.py --exp-config rearrangement/configs/rearrangement.yaml --run-type eval --ckpt-path data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/checkpoints/ckpt.1360.pth NUM_PROCESSES 10 SIMULATOR_GPU_ID "[0,1]" TORCH_GPU_ID 2 EVAL.SPLIT "test" EVAL.CONTENT_SCENES "['rearrangement_hard_v8_test_n=100_o=5_t=0.9_Sagerton']" TEST_EPISODE_COUNT 100
# ES: 0.375, OS: 0.629, No of obj: 5.000, E-SPL: 0.290, C-SPL: 0.498, P-SPL: 0.943, Avg Dist: 3.418, Avg length: 847.8563899868248`