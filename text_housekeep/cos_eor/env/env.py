import time
from collections import defaultdict
from copy import deepcopy
from typing import Optional
from PIL import ImageDraw, Image

import numpy as np
import shortuuid
from text_housekeep.habitat_lab.habitat import Config, Dataset
from text_housekeep.habitat_lab.habitat.utils.visualizations.utils import images_to_video
from text_housekeep.habitat_lab.habitat_baselines.common.baseline_registry import baseline_registry
from text_housekeep.habitat_lab.habitat_baselines.common.environments import NavRLEnv
from text_housekeep.habitat_lab.habitat.core.registry import registry
from text_housekeep.cos_eor.task.utils import start_env_episode_distance
from text_housekeep.cos_eor.sim.sim import CosRearrangementSim
from text_housekeep.cos_eor.task.task import CosRearrangementTask
from text_housekeep.cos_eor.utils.geometry import extract_sensors
from text_housekeep.cos_eor.scripts.build_utils import get_scene_rec_names, match_rec_cat_to_instances


@baseline_registry.register_env(name="CosRearrangementRLEnv")
class CosRearrangementRLEnv(NavRLEnv):
    _sim: CosRearrangementSim
    _task: CosRearrangementTask

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        # this is stored to use compare values while giving rewards!
        self._prev_measure = {
            'gripped_object_id': -1,
            'gripped_object_count': defaultdict(int),
            'objs_success': [],
            'objs_dict': {},
            'misplaced_objs_keys': [],
            'placed_objs_keys': [],
            'misplaced_touched_objs_keys': defaultdict(int),
            'placed_touched_objs_keys': defaultdict(int),
            'pick_place': [],
            'object_pp_points': defaultdict(list),
            'agent_pos': []
        }
        super().__init__(config, dataset)
        self.next_object_sensor_uuid = self._core_env_config.TASK.NEXT_OBJECT_SENSOR_UUID
        self.full_config = config
        self.counter = 0
        self.steps = 0
        self.episode_frames = []
        self.video_counter = -1
        self._sim = self._env._sim
        self.action_list = self._env._config.TASK["POSSIBLE_ACTIONS"]


        self._task = self._env._task
        print("Next Object Sensor: {}".format(self.next_object_sensor_uuid))

        observations = super(NavRLEnv, self).reset()
        self.obj_iids = []
        for obj_key in observations['cos_eor']['objs_keys']:
            self.obj_iids.append(observations['cos_eor']['sim_obj_id_to_iid'][observations['cos_eor']['obj_key_to_sim_obj_id'][obj_key]])
        for rec_key in observations['cos_eor']['recs_keys']:
            if rec_key != 'agent':
                self.obj_iids.append(observations['cos_eor']['sim_obj_id_to_iid'][observations['cos_eor']['obj_key_to_sim_obj_id'][rec_key]])
        self._size = observations['rgb'].shape

    def reset(self):
        self._sim.reset()
        self.steps = 0
        self._first_correct = 0 
        self._previous_action = None
        self._prev_measure = {}
        # call grandparent's method
        observations = super(NavRLEnv, self).reset()
        observations = extract_sensors(observations)
        self._prev_measure.update(self.habitat_env.get_metrics())
        self._prev_measure['gripped_object_id'] = -1
        self._prev_measure['agent_pos'] = []
        agent_pos = self._sim.get_agent_state().position
        self._prev_measure['agent_pos'].append(agent_pos)
        self._prev_measure['gripped_object_count'] = defaultdict(int)
        self._prev_measure['object_pp_points'] = defaultdict(list)
        self._prev_measure['objs_success'] = self._episode_obj_success(observations)
        self._prev_measure['objs_dict'] = {}
        self._prev_measure['misplaced_objs_keys'] = []
        self._prev_measure['placed_objs_keys'] = []
        self._prev_measure['misplaced_touched_objs_keys'] = defaultdict(int)
        self._prev_measure['placed_touched_objs_keys'] = defaultdict(int)
        self._prev_measure["pick_place"] = []
        self._placement_history = {}
        self._interaction_list = []

        self._misplaced_objs_start = set(self._env._current_episode.get_misplaced_objects("start"))

        self.initial_success = self._episode_obj_success_dict(self._env._current_episode.objs_keys)
        print('init success', self.initial_success)

        #logging
        self._pick_success = 0
        self._place_success = 0

        return observations

    def step(self, action):
        
        self._previous_action = {'action' : action}

        observations = self._env.step(action)

        observations = extract_sensors(observations)

        metrics = self.habitat_env.get_metrics()

        # log for metrics
        self.steps += 1
        agent_pos = self._sim.get_agent_state().position
        self._prev_measure["agent_pos"].append(agent_pos)

        if self.action_list[self._previous_action['action']] == "GRAB_RELEASE":
            self._prev_measure["pick_place"].append(not observations["fail_action"])
            # if pick-place succeeds
            if not observations["fail_action"]:
                # if we dropped an object
                if observations["gripped_object_id"] == -1:
                    assert self._prev_measure["gripped_object_id"] != -1
                    obj_id = self._prev_measure["gripped_object_id"]
                    self._prev_measure["object_pp_points"][obj_id].append(self.steps)
                # if we picked an object
                else:
                    assert self._prev_measure["gripped_object_id"] == -1
                    obj_id = observations["gripped_object_id"]
                    self._prev_measure["object_pp_points"][obj_id].append(self.steps)
                # record interactions
                obj_key = self._env.task.sim_obj_id_to_obj_key[obj_id]
                if obj_key in self._misplaced_objs_start:
                    self._prev_measure["misplaced_touched_objs_keys"][obj_key] += 1
                elif obj_key not in self._misplaced_objs_start:
                    self._prev_measure["placed_touched_objs_keys"][obj_key] += 1

        reward = self.get_reward(observations, metrics)
        info = self.get_info(observations, metrics)
        done = self.get_done(observations)

        return observations, reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, metrics):
        reward = self._rl_config.SLACK_REWARD
        gripped_success_reward = 0.0
        episode_success_reward = 0.0
        drop_success_reward = 0.0
        gripped_dropped_fail_reward = 0.0
        agent_to_object_dist_reward = 0.0
        object_to_goal_dist_reward = 0.0
        collision_reward = 0.0

        action_name = self.action_list[self._previous_action['action']]
        objs_success = np.array(self._episode_obj_success(observations))
        observations["gripped_object_id"] = observations["cos_eor"].get("gripped_object_id", -1)
        observations["fail_action"] = observations["cos_eor"].get("fail_action", False)
        objs_keys = self._env._current_episode.objs_keys
        objs_ids = [self._env.task.obj_key_to_sim_obj_id[k] for k in objs_keys]

        # # compute dense reward, agent to seen objects
        # for obj in observations['cos_eor']['visible_obj_iids']:
        # dist_per_obj = {}
        # for obj_idx, (obj, recs_list) in enumerate(observations['cos_eor']['correct_mapping'].items()):
        #     closest_rec_dist = np.inf 
        #     obj_pos = observations['cos_eor']['recs_pos'][obj_idx]
        #     for rec in recs_list:
        #         idx = np.where(rec == np.array(observations['cos_eor']['recs_keys']))[0][0]
        #         pos = observations['cos_eor']['recs_pos'][idx]
        #         dist = np.linalg.norm(np.asarray(obj_pos) - np.asarray(pos))
        #         if dist < closest_rec_dist:
        #             closest_rec_dist = dist
        #     dist_per_obj[obj] = closest_rec_dist
        #     import pdb; pdb.set_trace()
        
        # object_to_goal_dist_reward = -np.sum(list(dist_per_obj.values())) / 10 # scaling

        # misplaced objects discovered
        misplaced_objs_found = []
        placed_objs_found = []
        for viid in observations["visible_obj_iids"]:
            if viid == 0:
                break
            voi = self._env.task.iid_to_sim_obj_id[viid]
            vkey = self._env.task.sim_obj_id_to_obj_key[voi]
            if vkey not in self._prev_measure["misplaced_objs_keys"] and vkey in self._misplaced_objs_start:
                misplaced_objs_found.append(vkey)
            if vkey not in self._prev_measure["placed_objs_keys"] and vkey not in self._misplaced_objs_start:
                placed_objs_found.append(vkey)

        # new success will be in {-1, 0, 1}
        new_success = (
                objs_success.astype(int) - self._prev_measure['objs_success'].astype(int)
        ).sum()

        objs_dict = {}
        for ok, oid, os in zip(objs_keys, objs_ids, objs_success):
            objs_dict[oid] = {"success": os, "obj_key": ok}

        # update for the first-time in the start
        if len(self._prev_measure["objs_dict"]) == 0:
            self._prev_measure["objs_dict"] = objs_dict

        # agent not holding an object and episode successful
        if (
                self._episode_success() and
                observations['gripped_object_id'] == -1
        ):
            assert observations['gripped_object_id'] == -1
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        # grip
        if action_name == "GRAB_RELEASE" and self._gripped_success(observations):
            assert not observations["fail_action"]
            obj_id = observations['gripped_object_id']
            self._prev_measure['gripped_object_count'][obj_id] += 1
            gripped_mul = 1
            # gripping successfully placed object
            if self._prev_measure["objs_dict"][obj_id]["success"] == True:
                gripped_success_reward =  -.1 * self._rl_config.GRIPPED_SUCCESS_REWARD
                assert new_success == -1
            # gripping misplaced object for the first time
            elif self._prev_measure['gripped_object_count'][obj_id] == 1:
                gripped_success_reward = self._rl_config.GRIPPED_SUCCESS_REWARD
                self._first_correct += 1
                self._pick_success += 1
            else: 
                gripped_success_reward = 0

        # drop
        if (action_name == "GRAB_RELEASE"
            and self._prev_measure['gripped_object_id'] != -1
            and observations['gripped_object_id'] == -1
        ):
            assert not observations["fail_action"]
            # first time placing
            obj_id = self._prev_measure['gripped_object_id']
            if new_success == 1:
                # if already previously successful, don't reward
                if self._prev_measure['gripped_object_count'][obj_id] == 1 and not self.initial_success[observations['cos_eor']['sim_obj_id_to_obj_key'][obj_id]]:
                    drop_success_reward = self._rl_config.DROP_SUCCESS_REWARD
                    self._first_correct += 1
                    self._place_success += 1

                else:
                    drop_success_reward = 0
            elif new_success == 0:
                # dropped into wrong location
                drop_success_reward = self._rl_config.DROP_SUCCESS_REWARD * -.1
            else:
                import pdb
                pdb.set_trace()


        # failed pick/drop
        if action_name == "GRAB_RELEASE" and observations["fail_action"]:
            gripped_dropped_fail_reward = self._rl_config.GRIPPED_DROPPED_FAIL_REWARD

        if metrics['collisions']['is_collision']:
            collision_reward = self._rl_config.COLLISION_REWARD

        self._prev_measure['objs_success'] = objs_success
        self._prev_measure['gripped_object_id'] = observations['gripped_object_id']
        self._prev_measure["objs_dict"] = objs_dict
        self._prev_measure["misplaced_objs_keys"].extend(misplaced_objs_found)
        self._prev_measure["placed_objs_keys"].extend(placed_objs_found)

        assert not(
            len(set(self._prev_measure["misplaced_objs_keys"])) != len(self._prev_measure["misplaced_objs_keys"]) or
            len(set(self._prev_measure["placed_objs_keys"])) != len(self._prev_measure["placed_objs_keys"])
        )
        # reward += (
        #         agent_to_object_dist_reward +
        #         object_to_goal_dist_reward +
        #         gripped_success_reward +
        #         gripped_dropped_fail_reward +
        #         drop_success_reward +
        #         episode_success_reward +
        #         collision_reward
        # )
        reward = (
            gripped_success_reward + drop_success_reward + gripped_dropped_fail_reward #+ collision_reward + self._rl_config.SLACK_REWARD
        )

        action = self.action_list[self._previous_action['action']]
        self.counter += 1
        debug_string = (f"ao: {round(agent_to_object_dist_reward, 2)} | "
                  f"og: {round(object_to_goal_dist_reward, 2)} |"
                  f"gd: {round(gripped_success_reward + gripped_dropped_fail_reward + drop_success_reward + gripped_dropped_fail_reward, 3)} | \n"
                  f"ep: {round(episode_success_reward, 3)} | "
                  f"c: {round(collision_reward, 2)} | "
                  f"a: {action}")

        assert not(action_name != "GRAB_RELEASE" and (gripped_success_reward + gripped_dropped_fail_reward) != 0)

        self.save_frames(observations, action_name, debug_string)
        return reward
        # return {
        # 'agent_to_object_dist_reward': agent_to_object_dist_reward,
        # 'object_to_goal_dist_reward': object_to_goal_dist_reward,
        # 'gripped_success_reward': gripped_success_reward,
        # 'drop_success_reward': drop_success_reward,
        # 'episode_success_reward': episode_success_reward,
        # "gripped_dropped_fail_reward": gripped_dropped_fail_reward,
        # "slack_reward": self._rl_config.SLACK_REWARD,
        # "collision_reward": collision_reward
        # }

    def save_frames(self, observations, action_name, debug_string):
        if action_name == "GRAB_RELEASE":
            # add a red border for collisions
            border = 10
            rgb_frame = deepcopy(observations['rgb'])
            rgb_frame[:border, :] = [0, 255, 0]
            rgb_frame[:, :border] = [0, 255, 0]
            rgb_frame[-border:, :] = [0, 255, 0]
            rgb_frame[:, -border:] = [0, 255, 0]
        else:
            rgb_frame = observations['rgb']

        rgb_frame = Image.fromarray(rgb_frame)
        ImageDraw.Draw(rgb_frame).text(
            (20, 20),  # Coordinates
            debug_string,  # Text
            (255, 0, 0)  # Color
        )
        rgb_frame = np.array(rgb_frame)
        self.episode_frames.append(rgb_frame)

    def _episode_success(self, return_obj_success=False):
        obj_success_cor, obj_success_mis = self._episode_obj_success_split()
        eps_success = all(obj_success_cor) and all(obj_success_mis)
        if return_obj_success:
            return eps_success, sum(obj_success_cor), sum(obj_success_mis)
        else:
            return eps_success

    def _episode_obj_rec_dict(self):
        objs_to_rect = {}
        obj_to_goal = self._env.get_metrics()["object_to_goal_distance"]
        for ok in self._env._current_episode.objs_keys:
            current_rec = obj_to_goal[ok]["current_rec"]
            objs_to_rect[ok] = current_rec

        return objs_to_rect

    def _episode_obj_success_dict(self, obj_keys):
        objs_success = {}
        obj_to_goal = self._env.get_metrics()["object_to_goal_distance"]
        for ok in obj_keys:
            current_rec = obj_to_goal[ok]["current_rec"]
            correct_recs = obj_to_goal[ok]["recs_keys"]
            if current_rec in correct_recs:
                objs_success[ok] = True
            else:
                objs_success[ok] = False

        return objs_success

    def _episode_obj_success_for_given_keys(self, obj_keys):
        objs_success = []
        obj_to_goal = self._env.get_metrics()["object_to_goal_distance"]
        for ok in obj_keys:
            current_rec = obj_to_goal[ok]["current_rec"]
            correct_recs = obj_to_goal[ok]["recs_keys"]
            if current_rec in correct_recs:
                objs_success.append(True)
            else:
                objs_success.append(False)
        return np.array(objs_success)

    def _episode_obj_success_split(self):
        mis_objs = self._misplaced_objs_start
        cor_objs = set(self._prev_measure["placed_touched_objs_keys"].keys())
        objs_cor_success = self._episode_obj_success_for_given_keys(cor_objs)
        objs_mis_success = self._episode_obj_success_for_given_keys(mis_objs)
        return objs_cor_success, objs_mis_success

    def _episode_obj_success(self, observations=None):
        return self._episode_obj_success_for_given_keys(self._env._current_episode.objs_keys)

    def _gripped_success(self, observations):
        return (
                observations['gripped_object_id'] >= 0 and
                observations['gripped_object_id'] != self._prev_measure['gripped_object_id']
        )

    def get_done(self, observations):
        done = False
        if (
            self._env.episode_over or
            self.action_list[self._previous_action['action']] == "STOP" or 
            (self._episode_success() and
                observations['gripped_object_id'] == -1)
        ):
            done = True
        return done

    def _add_obj_metrics(self, info, obj_rec_map, metric_suffix):
        eps = self._env._current_episode
        amt_data = self._env._task.amt_data
        soft_scores = []
        mean_rec_ranks = []

        for obj_key, rec_key in obj_rec_map.items():
            if "agent" in rec_key:
                continue
            obj_idx, rec_idx = eps.objs_keys.index(obj_key), eps.recs_keys.index(rec_key)
            obj_cat = eps.objs_cats[obj_idx]
            rec_room = get_scene_rec_names([rec_key])[0]

            human_anns = None
            if rec_room in amt_data["room_recs"] and obj_cat in amt_data["objs"]:
                obj_data_idx, rec_data_idx = amt_data["objs"].index(obj_cat), amt_data["room_recs"].index(rec_room)
                human_anns = amt_data["data"][rec_data_idx][obj_data_idx]
                soft_score = sum(human_anns > 0) / len(human_anns)
            else:
                soft_score = 0
            soft_scores.append(soft_score)

            if eps.end_matrix[rec_idx][obj_idx] == 1.0:
                if human_anns is not None:
                    pos_ranks = human_anns[human_anns > 0]
                    rec_ranks = [1/r for r in pos_ranks]
                    mean_rec_rank = sum(rec_ranks) / len(rec_ranks)
                else:
                    mean_rec_rank = 0
                mean_rec_ranks.append(mean_rec_rank)

        info["soft_score_" + metric_suffix] = sum(soft_scores)
        info["rearrange_quality_" + metric_suffix] = sum(mean_rec_ranks)

    def _add_path_efficiency(self, info, obj_keys):
        # calculate distance between each step
        if len(self._prev_measure["agent_pos"]) != self.steps + 1:
            import pdb
            pdb.set_trace()
        distance_per_step = []
        for step in range(self.steps):
            if len(distance_per_step) == 0:
                prev_distance  = 0
            else:
                prev_distance = distance_per_step[-1]
            distance_per_step.append(
                prev_distance +
                self._sim.get_dist_pos(
                    self._prev_measure["agent_pos"][step],
                    self._prev_measure["agent_pos"][step+1],
                    "l2")
            )

        # add current position if still holding object
        curr_obj_id = self._prev_measure["gripped_object_id"]
        if curr_obj_id != -1:
            self._prev_measure["object_pp_points"][curr_obj_id].append(self.steps)

        # add path lengths and geodesic lengths travelled summed across each object
        path_eff_info = {
            "path_lens": {},
            "geo_lens": {},
            "path_eff": {},
        }

        for obj_key in obj_keys:
            obj_id = self._env._task.obj_key_to_sim_obj_id[obj_key]
            if obj_id in self._prev_measure["object_pp_points"]:
                pp_points = self._prev_measure["object_pp_points"].pop(obj_id)
                self._prev_measure["object_pp_points"][obj_key] = pp_points
                if len(pp_points) % 2 != 0:
                    import pdb
                    pdb.set_trace()
                # iterate of pp-points to find the distance travelled vs geodesic
                # distance between start and end points
                path_length = 0.0
                for idx in range(len(pp_points) // 2):
                    pick_step = pp_points[2*idx]
                    drop_step = pp_points[2*idx + 1]
                    dist = distance_per_step[drop_step-1] - distance_per_step[pick_step-1]
                    path_length += dist
                geo_length = self._sim.get_dist_pos(
                    self._prev_measure["agent_pos"][pp_points[0]],
                    self._prev_measure["agent_pos"][pp_points[-1]],
                    "geo"
                )

                # geodesic distance between two points can be either smaller or equal to the distance 
                # travelled by the agent between them
                if path_length < geo_length:
                    geo_length = path_length

                # add path and geodesic distances
                if not np.isclose(max(geo_length, path_length), 0):
                    path_eff_info["path_lens"][obj_key] = path_length
                    path_eff_info["geo_lens"][obj_key] = geo_length
                    path_eff_info["path_eff"][obj_key] = geo_length / max(geo_length, path_length)
            else:
                self._prev_measure["object_pp_points"][obj_key] = []

        if len(path_eff_info["path_eff"]) == 0:
            info["path_efficiency"] = np.nan
        else:
            info["path_efficiency"] = sum(path_eff_info["path_eff"].values()) / len(path_eff_info["path_eff"])

    def _add_pick_place_efficiency(self, info):
        """
        pick place efficiency for an object =
        {2 / number of interactions if object was initially misplaced and ended up in right receptacle,
        0 otherwise} for all objects interacted with
        """
        misp_touched_objs = self._prev_measure["misplaced_touched_objs_keys"]
        obj_successes = self._episode_obj_success_for_given_keys(list(misp_touched_objs))

        efficiency = 0
        for num_pick_places, success in zip(misp_touched_objs.values(), obj_successes):
            if success:
                efficiency += 2 / num_pick_places

        normalizer = len(self._prev_measure["misplaced_touched_objs_keys"]) + len(self._prev_measure["placed_touched_objs_keys"])
        normalizer = normalizer if normalizer > 0 else 1
        info["pick_place_efficiency"] = efficiency / normalizer
        info["misplaced_objects_touched"] = len(self._prev_measure["misplaced_touched_objs_keys"])
        info["placed_objects_touched"] = len(self._prev_measure["placed_touched_objs_keys"])

    def add_metrics(self, info):
        eps = self._env._current_episode
        obj_rec_map = eps.get_mapping(state_type="current")

        obj_rec_map_mis = {}
        obj_rec_map_cor = {}
        num_placed_objs_start = 0
        placed_objs_keys = set(self._prev_measure["placed_touched_objs_keys"])
        for obj_key, rec_key in obj_rec_map.items():
            if obj_key in self._misplaced_objs_start:
                obj_rec_map_mis[obj_key] = rec_key
            else:
                num_placed_objs_start += 1
                if obj_key in placed_objs_keys:
                    obj_rec_map_cor[obj_key] = rec_key

        self._add_obj_metrics(info, obj_rec_map_mis, "misplaced")
        self._add_obj_metrics(info, obj_rec_map_cor, "placed")

        self._add_path_efficiency(info, list(obj_rec_map_mis.keys()) + list(obj_rec_map_cor.keys()))

        self._add_pick_place_efficiency(info)

        info["misplaced_objects_start"] = len(obj_rec_map_mis)
        info["placed_objects_start"] = num_placed_objs_start
        info["misplaced_objects_found"] = len(self._prev_measure["misplaced_objs_keys"])
        info["placed_objects_found"] = len(self._prev_measure["placed_objs_keys"])
        info["total_pick_place"] = len(self._prev_measure["pick_place"])
        info["success_pick_place"] = sum(self._prev_measure["pick_place"])
        info["fail_pick_place"] = info["total_pick_place"] - info["success_pick_place"]
        info["steps"] = self.steps
        info["map_coverage"] = self._sim.occupancy_info["seen_area"]

    def get_info(self, observations, metrics):
        """
        YK: Seems like a function that can save pickup and drop order, and can be used for analysis
        """
        # check if the episode is finished
        is_done = self.get_done(observations)
        info = deepcopy(metrics)
        # episode and object success metric
        info['episode_success'], info["object_success_placed"], info["object_success_misplaced"] = self._episode_success(return_obj_success=True)

        if is_done:
            # episode soft object success, rearrange quality, misplaced objects found
            self.add_metrics(info)

        return info

    def save_replay(self, info):
        path = "data/replays"
        uuid = shortuuid.uuid()
        self._env.task.save_replay(self._env.current_episode, info, path, uuid)

    def get_shortest_path_next_action(self, obj_pos, snap_to_navmesh=True):
        return self._sim.get_shortest_path_next_action(obj_pos, snap_to_navmesh)

    def get_agent_object_distance(self, obj_id, dist_type):
        return self._sim.get_or_dist(obj_id, dist_type)

    def get_object_object_distance(self, obj1_id, obj2_id, dist_type):
        return self._sim.get_dist_id(obj1_id, obj2_id, dist_type)

    def snap_id_to_navmesh(self, nav_obj_id, look_obj_id):
        return self._sim.snap_id_to_navmesh(nav_obj_id, look_obj_id)
