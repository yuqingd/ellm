from typing import Dict, List, Any

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random
import sys

import numpy as np
import numba
import torch
import tqdm

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from habitat.core.registry import registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.utils.env_utils import construct_envs

from cos_eor.policy.rank import RankModule
from cos_eor.policy.nav import NavModule
from cos_eor.policy.oracle_rank import OracleRankModule
from cos_eor.policy.explore import ExploreModule
from cos_eor.policy.hie_policy import HiePolicy
from cos_eor.env.env import CosRearrangementRLEnv
from cos_eor.task.sensors import *
from cos_eor.task.measures import *

class HiePolicyRunner(object):
    def __init__(self, config):
        self.config = config

    def _setup_rank_module(self):
        if self.config.RL.POLICY.rank.name == "Oracle":
            self.rank_module = OracleRankModule(self.config.RL.POLICY.rank)
        elif self.config.RL.POLICY.rank.name == "LangModel":
            self.rank_module = RankModule(self.config.RL.POLICY.rank)
        else:
            raise AssertionError

    def _setup_explore_module(self):
        self.explore_module = ExploreModule(self.config.RL.POLICY.explore, self.envs.num_envs)
        self.explore_module.to(self.device)

    def _setup_policy(self):
        # check hie or flat
        if self.config.RL.POLICY.name == "E2EPolicy":
            raise ValueError
        elif self.config.RL.POLICY.name == "HiePolicy":
            # setup nav module as shortest-path follower
            if self.config.RL.POLICY.nav.name == "OracleShortestPath":
                count_steps, count_checkpoints, update_start = [0] * 3
                self.nav_module = NavModule(self.envs, task_params=self.config.BASE_TASK_CONFIG.TASK)
                self.nav_type = "sp"
            else:
                raise ValueError

            self._setup_rank_module()
            self._setup_explore_module()
            debug_params = {"envs": self.envs}
            nav_param = self.nav_module if self.nav_type == "sp" else self.actor_critic
            self.policy = HiePolicy(
                self.envs,
                nav_param,
                self.rank_module,
                self.explore_module,
                self.config.RL.POLICY,
                self.config.BASE_TASK_CONFIG.TASK,
                debug_params
            )
            if self.config.VIDEO_INTERVAL == 1:
                self.policy.debug_video = True
        else:
            raise ValueError
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

    def write_line_to_csv(self, file_handle, vals):
        file_handle.write(",".join(vals) + "\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())

    def write_metrics(self, metrics, file_handle):
        episode_stats_str_vals = map(str, metrics.values())
        self.write_line_to_csv(file_handle, episode_stats_str_vals)

    def _is_file_non_empty(self, file):
        return os.path.isfile(file) and os.path.getsize(file) > 0

    def init_csv_file(self, file_path: Path, append: bool):
        if append and self._is_file_non_empty(file_path):
            file = open(file_path, "a")
            is_header_written = True
        else:
            file_dir = file_path.parent
            file_dir.mkdir(exist_ok=True, parents=True)
            file = open(file_path, "w")
            is_header_written = False
        return file, is_header_written

    def _num_prev_finished_episodes(self, checkpoint_file):
        with open(checkpoint_file) as f:
            i = 0
            for i, _ in enumerate(f):
                pass
        # Don't count header
        return i

    def run(self, out_dir=None, resume=False, tag=None, num_eps=1000):
        self.config.defrost()

        navmesh_file = out_dir/self.config.TASK_CONFIG.SIMULATOR.NAVMESH
        navmesh_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.TASK_CONFIG.SIMULATOR.NAVMESH = str(navmesh_file)

        dataset_checkpoint_file = self.config.TASK_CONFIG.DATASET.CHECKPOINT_FILE.format(tag=tag)
        if (
            resume and
            os.path.isfile(out_dir/dataset_checkpoint_file) and
            os.path.getsize(out_dir/dataset_checkpoint_file) > 0
        ):
            self.config.TASK_CONFIG.DATASET.CHECKPOINT_FILE = str(out_dir/dataset_checkpoint_file)
            num_eps -= self._num_prev_finished_episodes(self.config.TASK_CONFIG.DATASET.CHECKPOINT_FILE)
        else:
            self.config.TASK_CONFIG.DATASET.CHECKPOINT_FILE = None
        self.config.freeze()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        count_steps, count_checkpoints, update_start = self._setup_policy()
        self.policy.reset()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        metrics_file_name = self.config.METRICS_FILE.format(tag=tag)
        metrics_file, is_metrics_header_written = self.init_csv_file(out_dir/metrics_file_name, resume)
        states_file_name = self.config.STATES_FILE.format(tag=tag)
        states_file, _ = self.init_csv_file(out_dir/states_file_name, resume)
        video_dir = out_dir/self.config.VIDEO_DIR/tag
        video_dir.mkdir(exist_ok=True, parents=True)
        replay_dir = out_dir/self.config.REPLAY_DIR/tag
        replay_dir.mkdir(exist_ok=True, parents=True)
        turn_measures_dir = out_dir/self.config.TURN_MEASURES_DIR/tag

        actions_buffer = [[] for _ in range(self.envs.num_envs)]
        states_buffer = [[] for _ in range(self.envs.num_envs)]
        turn_measures_buffer = [defaultdict(list) for _ in range(self.envs.num_envs)]

        aggregated_stats = defaultdict(int)
        all_episode_stats = []

        with tqdm.tqdm(total=num_eps) as pbar:
            while len(all_episode_stats) < num_eps:
                cur_episodes = self.envs.current_episodes()
                actions, turn_measures = self.policy.act(batch)
                states = self.policy.get_current_state()
                outputs = self.envs.step(actions)
                observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, device=self.device)

                num_dones = 0
                for env_idx, done in enumerate(dones):
                    actions_buffer[env_idx].append(actions[env_idx]["action"]["action"])

                    for turn_measure_name, turn_measure in turn_measures[env_idx].items():
                        turn_measures_buffer[env_idx][turn_measure_name].append(turn_measure)

                    state = states[env_idx]["act"]
                    state = "s" if state is None else state[0]
                    states_buffer[env_idx].append(state)

                    if not done:
                        continue

                    num_dones += 1
                    episode_stats = self._extract_scalars_from_info(infos[env_idx])
                    episode_stats.update(self.policy.get_info())
                    all_episode_stats.append(episode_stats)
                    for k, v in episode_stats.items():
                        aggregated_stats[k] += v

                    num_episodes = len(all_episode_stats)
                    cur_episode = cur_episodes[env_idx]
                    scene_id = Path(cur_episode.scene_id).stem
                    cur_episode_id = f"{scene_id}_{cur_episode.episode_id}"
                    episode_stats["episode_id"] = cur_episode_id

                    replays_file = replay_dir/f"ep_{cur_episode_id}.txt"
                    actions_str = "\n".join(map(str, actions_buffer[env_idx]))
                    with open(str(replays_file), "w") as f:
                        f.write(actions_str)
                    actions_buffer[env_idx] = []

                    for turn_measure_name, turn_measures_episode in turn_measures_buffer[env_idx].items():
                        turn_measures_str = "\n".join(map(str, turn_measures_episode))
                        measures_file = turn_measures_dir/turn_measure_name/f"ep_{cur_episode_id}.txt"
                        measures_file.parent.mkdir(exist_ok=True, parents=True)
                        with open(measures_file, "w") as f:
                            f.write(turn_measures_str)
                        turn_measures_buffer[env_idx][turn_measure_name].clear()

                    states_buffer[env_idx].append(cur_episode_id)
                    self.write_line_to_csv(states_file, states_buffer[env_idx])
                    states_buffer[env_idx] = []

                    if num_episodes % self.config.VIDEO_INTERVAL == 0:# and not episode_stats["episode_success"]:
                        self.policy.dump_vid(video_dir, f"ep_{cur_episode_id}", self.config.SAVE_VIDEO_AS_RAW_FRAMES)
                        self.policy.debug_video = False
                    if (num_episodes + 1) % self.config.VIDEO_INTERVAL == 0:
                        self.policy.debug_video = True

                    self.policy.reset()

                    if not is_metrics_header_written:
                        metrics_file.write(",".join(episode_stats.keys()) + "\n")
                        is_metrics_header_written = True
                    self.write_metrics(episode_stats, metrics_file)
                    print(f"Completed episode - idx: {num_episodes}, id: {cur_episode.episode_id}")
                    print("Starting episode", self.envs.current_episodes()[0].episode_id)

                    if num_episodes > 0 and num_episodes % 10 == 0:
                        print(f"Aggregated stats for {num_episodes} episodes:")
                        for k in aggregated_stats:
                            print(f"{k}: {aggregated_stats[k]/num_episodes}")

                pbar.update(num_dones)

        self.envs.close()
        metrics_file.close()

        for k, v in aggregated_stats.items():
            v /= len(all_episode_stats)
            print(f"Average episode {k}: {v:.4f}")

@torch.no_grad()
def run_exp(exp_config: str, out_dir: Path, resume: bool, opts=None, **kwargs):
    config = get_config(exp_config, opts)
    print(config.TASK_CONFIG.SEED, config.NUM_PROCESSES)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
    np.random.seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)
    torch.manual_seed(config.TASK_CONFIG.SEED * config.NUM_PROCESSES)  # Change this for different environment.
    registry.mapping["debug"] = kwargs['debug']
    registry.mapping["time"] = kwargs['time']

    out_dir.mkdir(parents=True, exist_ok=True)
    runner = HiePolicyRunner(config)
    runner.run(out_dir=out_dir, resume=resume, tag=kwargs["tag"], num_eps=kwargs["num_eps"])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="output directory"
    )
    parser.add_argument(
        "--num-eps",
        type=int,
        required=True,
        help="number of episodes"
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
        "--resume",
        default=False,
        action="store_true",
        help="resume from last episode according to metrics file"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()
