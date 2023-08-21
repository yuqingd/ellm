import math
from typing import List
from typing import Type
import numpy as np

import habitat_sim
from text_housekeep.habitat_lab.habitat.core.utils import try_cv2_import
from text_housekeep.habitat_lab.habitat.tasks.nav.nav import merge_sim_episode_config
from text_housekeep.habitat_lab.habitat.tasks.utils import cartesian_to_polar
from text_housekeep.habitat_lab.habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

# from cos_eor.task.measures import *
# from cos_eor.task.sensors import *
from text_housekeep.cos_eor.utils.geometry import geodesic_distance

cv2 = try_cv2_import()


def start_env_episode_distance(task, episode, pickup_order):
    pathfinder = task._simple_pathfinder

    agent_start_pos = episode.start_position
    prev_obj_end_pos = agent_start_pos

    object_positions = [obj.position for obj in episode.objects]
    rec_positions = [rec.position for rec in episode.get_receptacles()]

    pickup_order = [id-1 for id in pickup_order]
    shortest_dist = 0

    # todo: why the -1 and -0.5
    for i in range(len(pickup_order)):
        curr_idx = pickup_order[i]
        curr_obj_start_pos = object_positions[curr_idx]
        curr_obj_end_pos = rec_positions[curr_idx]
        shortest_dist += geodesic_distance(
            pathfinder, prev_obj_end_pos, [curr_obj_start_pos]
        ) - 1.0

        shortest_dist += geodesic_distance(
            pathfinder, curr_obj_start_pos, [curr_obj_end_pos]
        ) - 0.5
        prev_obj_end_pos = curr_obj_end_pos

    return shortest_dist


def merge_sim_episode_with_object_config_play(sim_config, episode):
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()

    sim_config.objects = [episode.objects.__dict__]
    sim_config.freeze()

    return sim_config


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()

    object_templates = {}
    for template in episode.object_templates:
        object_templates[template["object_key"]] = template["object_template"]

    objects = []
    for obj in episode.objects:
        obj.object_template = object_templates[obj.object_key]
        objects.append(obj)
    sim_config.objects = objects

    sim_config.freeze()

    return sim_config


def get_packer_mapping(packers, task):
    packer_mapping = {}
    for rec_id, packer in packers.items():
        obj_keys = [task.sim_obj_id_to_obj_key[obj_key] for obj_key in list(packer.matches.keys())]
        rec_key = task.sim_obj_id_to_obj_key[rec_id]
        for obj_key in obj_keys:
            packer_mapping[obj_key] = rec_key
    return packer_mapping

