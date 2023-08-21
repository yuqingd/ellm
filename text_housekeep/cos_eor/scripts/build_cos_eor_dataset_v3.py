import argparse
import csv
import glob
import gzip
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy

import pandas as pd
import yaml
from tqdm import tqdm


cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for _dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, _dir)

import text_housekeep.habitat_lab.habitat
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.physics import MotionType
from text_housekeep.orp.obj_loaders import load_articulated_objs
from text_housekeep.habitat_lab.habitat.sims import make_sim

from text_housekeep.cos_eor.task.sensors import *
from text_housekeep.cos_eor.task.measures import *
from text_housekeep.orp.utils import get_aabb

cwd = os.getcwd()
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)
from text_housekeep.cos_eor.utils.geometry import geodesic_distance, get_random_point_igib, get_vol, add_object_on_receptacle, \
    set_agent_on_floor_igib, get_bb_base, get_closest_nav_point, get_all_nav_points
import logging
from text_housekeep.cos_eor.utils.debug import debug_sim_viewer, get_bb, get_surface_point_for_placement
from habitat_sim.utils.common import quat_from_coeffs
from text_housekeep.cos_eor.scripts.orm.utils import preprocess
from igib_assemble_obj import urdf_to_obj, URDF_OBJ_CACHE
from text_housekeep.cos_eor.utils.shelf_bin_packer import ShelfBinPacker
from text_housekeep.cos_eor.scripts.build_utils import aggregate_amt_annotations, get_scene_rec_names, match_rec_cat_to_instances

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
MAX_DIST = 1.2
RECEPTACLE_DROPOUT = 0.25

def get_sim(scene, sim):
    """Habitat doesn't allow creating multiple simulators, and throws an error w/ OpenGL."""
    if not sim:
        config = text_housekeep.habitat_lab.habitat.get_config("./text_housekeep/cos_eor/configs/dataset/build_dataset.yaml")
        config = text_housekeep.habitat_lab.habitat.get_config_new([config.BASE_TASK_CONFIG])
        config.defrost()
        config.SIMULATOR.SCENE = scene
        config.freeze()
        sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    else:
        sim_config = deepcopy(sim.habitat_config)
        sim_config.defrost()
        sim_config.SCENE = scene
        sim_config.freeze()
        sim.reconfigure(sim_config)

    init_metadata = sim.init_metadata_objects()
    sim.navmesh_visualization = True
    return sim, init_metadata


def load_annotations(sim, data, only_ycb=False):
    """todo, yk: homogenize all these paths when relasing"""
    obj_attr_mgr = sim.get_object_template_manager()

    # load ycb objects
    obj_attr_mgr.load_configs("data/objects")

    # load cache handles -- igib objects
    cache_handles = [h for h in glob.glob(os.path.join(URDF_OBJ_CACHE, "**"), recursive=True) if
                     h.endswith(".object_config.json")]
    for ch in cache_handles:
        obj_attr_mgr.load_configs(ch)

    if only_ycb:
        return

    # already loaded (maybe better check is possible)
    if obj_attr_mgr.get_num_file_templates() > 400:
        return

    # load other annotations
    skipped_templates = 0
    for k in tqdm(data, desc="Loading handles"):
        if "template" in data[k]:
            handle = data[k]['template']
            if handle.endswith(".object_config.json") and os.path.exists(handle):
                obj_attr_mgr.load_configs(handle)
            else:
                skipped_templates += 1
    print(f"Skipped {skipped_templates}/ {len(data)} specified object templates!")


def check_floor():
    pass


def init_agent(sim):
    agent_pos = set_agent_on_floor_igib(sim)
    return sim.get_agent(0).get_state(), sim.agent_id


def get_rotation(sim, oid):
    if oid == -1:
        return [0.0] * 4
    quat = sim.get_rotation(oid)
    return np.array(quat.vector).tolist() + [quat.scalar]


def euclidean_distance(position_a, position_b):
    return np.linalg.norm(
        np.array(position_b) - np.array(position_a), ord=2
    )


def validate_floor_placement(sim, obj_id, obj, intersect_th=0.3):
    """
    how to prevent objects from getting spawned beneath receptacles?
    -- give agent a tour and ensure the object is visible?
    """

    obj_translation = sim.get_translation(obj_id)
    clearance_height = 0 # only check once
    clearance_step = 0.1
    check_height = 0

    obj_ids = sim.get_both_existing_object_ids()
    while check_height <= clearance_height:
        obj_bb = get_bb(sim, obj_id)
        obj_vol = get_vol(obj_bb)

        # iterate over each object check for collisions
        for oid in obj_ids["art"] + obj_ids["non_art"]:
            if oid == obj_id:
                continue
            oid_bb = get_bb(sim, oid)
            if mn.math.intersects(obj_bb, oid_bb):
                intersect_vol = get_vol(mn.math.intersect(obj_bb, oid_bb))
                intersect_factor = intersect_vol / obj_vol
                if intersect_factor > intersect_th:
                    print(f"unacceptable: {obj} intersection with {oid} factor: {intersect_factor}")
                    return False
                # else:
                #     print(f"found: {obj} intersection with {oid} factor: {intersect_factor}")

        check_height += clearance_step
        new_translation = deepcopy(list(obj_translation))
        new_translation[1] += check_height
        sim.set_translation(new_translation, obj_id)
    sim.set_translation(obj_translation, obj_id)
    # print(f"{obj} at {obj_translation}")
    return True


def init_episode_dict(scene, episode_num, agent_state):
    episode_dict = {
        'episode_id': episode_num,
        'scene_id': scene,
        'start_position': agent_state.position.tolist(),
        'start_rotation': list(agent_state.rotation.vec) + [agent_state.rotation.real],
    }
    return episode_dict


def set_object_on_top_of_surface(sim, obj_id):
    r"""
    Adds an object in front of the agent at some distance.
    """

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.00
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)

    return np.array(sim.get_translation(obj_id))


def add_object_in_scene(cat, data, sim, obj_attr_mgr, cat_data):
    if cat not in cat_data:
        import pdb
        pdb.set_trace()

    try:
        key = np.random.choice(cat_data[cat])
    except:
        import pdb
        pdb.set_trace()

    try:
        if "template" in data[key]:
            template = data[key]["template"]
            template = template.replace("./", "")
            handle = obj_attr_mgr.get_file_template_handles(template)[0]
            obj_template = obj_attr_mgr.get_template_by_handle(handle)
            obj_template.scale = np.array(data[key]['scale'])
            obj_attr_mgr.register_template(obj_template)
            rot = data[key]["rotation"]
            orientation = mn.Quaternion(mn.Vector3(*rot[:3]), rot[3])
            obj_id = sim.add_object_by_handle(handle)
            sim.set_rotation(orientation, obj_id)
        else:
            assert "urdfs" in data[key]
            idx = random.choice(range(len(data[key]["urdfs"])))
            file = data[key]["urdfs"][idx]
            handle, already_exists = urdf_to_obj(file, URDF_OBJ_CACHE, return_exist=True)
            if not already_exists:
                obj_attr_mgr.load_configs(handle)
            obj_id = sim.add_object_by_handle(handle)
            rot = data[key]["rots"][idx]
            orientation = mn.Quaternion(mn.Vector3(*rot[:3]), rot[3])
            sim.set_rotation(orientation, obj_id)
        obj_type = "non_art"
        file = handle
        assert os.path.exists(file)
    except:
        import pdb
        pdb.set_trace()

    return obj_id, obj_type, orientation, file


def set_object(id, sim, type, position, orientation):
    if type == "non_art":
        sim.set_object_motion_type(MotionType.DYNAMIC, id)
        sim.set_translation(position, id)
        sim.set_rotation(orientation, id)
        set_object_on_top_of_surface(sim, id)
    else:
        obj_bb = get_aabb(id, sim)
        scene_collision_margin = 0.00
        y_translation = mn.Vector3(0, obj_bb.size_y() / 2.0 + scene_collision_margin, 0)
        state = mn.Matrix4.from_(orientation.to_matrix(), mn.Vector3(*y_translation) + position)
        sim.set_articulated_object_root_state(id, state)
        sim.set_articulated_object_motion_type(id, MotionType.DYNAMIC)


def get_random_floor_point(sim, agent_position, obstacle_min_th=0.5, ao_max_th=100):
    object_position = None
    for oi in range(100):
        object_position = get_random_point_igib(sim.pathfinder)
        object_dist = sim.pathfinder.distance_to_closest_obstacle(object_position, max_search_radius=2.0)
        ao_geo_dist = sim.geodesic_distance(agent_position, object_position)
        if object_dist > obstacle_min_th and ao_geo_dist < ao_max_th:
            break
    return object_position


def re_rank(rec_keys, rec_ranks):
    "re-rank to lie between 1-n and sort by ranking"
    current_ranks = sorted(list(set(rec_ranks)))
    rec_keys = np.array(rec_keys)
    rec_ranks = np.array(rec_ranks)
    sorted_rk, sorted_rr = [], []
    for i, cr in enumerate(current_ranks):
        rks = rec_keys[rec_ranks == cr]
        sorted_rk.extend(rks)
        sorted_rr.extend([i+1] * len(rks))
    return sorted_rk, sorted_rr


def parse_scene_data(scene_metadata, data, cat_data):
    fil_meta_keys, ids, full_metadata, urdf_dir = scene_metadata
    scene_objs_ids, scene_objs_keys = [], []
    scene_recs_ids, scene_recs_keys = [], []
    dropout = random.random() < RECEPTACLE_DROPOUT
    top_recs_cats = {"bottom_cabinet", "shelf", "top_cabinet", "bottom_cabinet_no_top"}

    for k,i in zip(fil_meta_keys, ids):
        dk = "_".join(k.split("_")[:-2])

        if dropout and dk in top_recs_cats:
            continue

        # add dummy entries into annotations data
        if dk not in data:
            # print(f"Did not find {dk} in data, adding it as a dummy receptacle")
            data[dk] = {
                "type": "receptacle",
            }

        # problem if semantic classes doesn't exists
        if dk not in cat_data:
            import pdb
            pdb.set_trace()

        # extract the type from simulator, sometimes we need to change types on fly
        type_scene_data = full_metadata["urdfs"][k]["type"]
        type_full_data = data[dk]["type"]
        if type_full_data != type_scene_data:
            data[dk]["type"] = type_scene_data

        if data[dk]["type"] == "receptacle":
            scene_recs_keys.append(k)
            scene_recs_ids.append(i)
        elif data[dk]["type"] == "object":
            scene_objs_keys.append(k)
            scene_objs_ids.append(i)
        else:
            raise AssertionError

    # # not adding floor
    # scene_recs_keys.append("floor")
    # scene_recs_ids.append(-1)

    # build state matrix
    start_matrix = np.zeros(shape=(len(scene_recs_keys), len(scene_objs_keys)))

    # for o, r in full_metadata["default_mapping"]:
    #     io, ir = scene_objs_keys.index(o), scene_recs_keys.index(r)
    #     start_matrix[ir, io] = 1
    # can't use with packers
    assert len(full_metadata["default_mapping"]) == 0 and len(scene_objs_keys) == 0

    try:
        assert all(start_matrix.sum(0) == 1)
    except:
        import pdb
        pdb.set_trace()

    # add rooms to scene_recs_keys
    scene_room_recs_keys = []
    for k in scene_recs_keys:
        room = full_metadata["urdfs"][k]['room']
        key = f'{room}-{k}'
        scene_room_recs_keys.append(key)

    return (scene_objs_ids, scene_objs_keys), (scene_recs_ids, scene_room_recs_keys), start_matrix


def get_avg_th_ranks(rr_data, thresh=6):
    assert rr_data.shape[-1] == 10 and len(rr_data.shape) == 2

    num_pos_ranks = (rr_data > 0).sum(-1)
    rr_copy = deepcopy(rr_data)
    rr_copy[rr_copy <= 0] = 0
    avg_pos_ranks = rr_copy.sum(-1) / num_pos_ranks
    avg_pos_ranks = np.nan_to_num(avg_pos_ranks, nan=0)

    num_neg_ranks = (rr_data < 0).sum(-1)
    rr_copy = deepcopy(rr_data)
    rr_copy[rr_copy >= 0] = 0
    avg_neg_ranks = rr_copy.sum(-1) / num_neg_ranks
    avg_neg_ranks = np.nan_to_num(avg_neg_ranks, nan=0)

    num_zero_ranks = (rr_data == 0).sum(-1)

    assert (num_pos_ranks + num_neg_ranks + num_zero_ranks).sum() == np.prod(rr_data.shape)

    # place ranks with threshold
    final_pos_ranks = np.where(num_pos_ranks >= thresh, avg_pos_ranks, 0)
    final_neg_ranks = np.where(num_neg_ranks >= thresh, avg_neg_ranks, 0)
    final_ranks = final_pos_ranks + final_neg_ranks

    return final_ranks


def debug_print(candidate_objects, sampled_init_recs, candidate_recs, annotations):
    f = open("debug_orm.txt", 'w')
    for obj_idx in range(len(candidate_objects)):
        print(
            f"Object: {candidate_objects[obj_idx]}: \n"
            f"\t Misplace: {[(annotations['room_recs'][idx_score[0]], idx_score[-1]) for idx_score in sampled_init_recs[obj_idx]]} \n"
            f"\t Place: {[(annotations['room_recs'][idx_score[0]], idx_score[-1]) for idx_score in candidate_recs[obj_idx]]}"
            f"\n\n",
            file=f
        )
    f.close()


def filter_annotations(annotations, scene_recs_keys, cat_data):
    # filter the global-mapping based on scene receptacles
    scene_rr = get_scene_rec_names(scene_recs_keys)
    global_rns = annotations["room_recs"]
    delete_rr = []
    for ind, rn in enumerate(global_rns):
        if rn not in scene_rr:
            delete_rr.append(rn)
    for rr in delete_rr:
        # assert len(annotations["recs"]) == len(annotations["rooms"]) == len(annotations["room_recs"]) == annotations["data"].shape[1]
        ind = annotations["room_recs"].index(rr)
        annotations["room_recs"].pop(ind)
        annotations["data"] = np.delete(annotations["data"], ind, axis=0)

    rooms, recs = [rr.split("-")[0] for rr in annotations["room_recs"]], [rr.split("-")[1] for rr in annotations["room_recs"]]
    annotations["rooms"], annotations["recs"] = list(set(rooms)), list(set(recs))

    if len(scene_recs_keys) < 10:
        import pdb
        pdb.set_trace()

    print(f"\n\n\n Total iGibson receptacles inserted: {len(scene_recs_keys)} "
          f"// iGibson receptacles usage: {(annotations['data'].shape[0])} ")

    # assert right shapes
    assert len(annotations["room_recs"]) == annotations["data"].shape[0] == len(set(annotations["room_recs"]))
    assert len(annotations["objs"]) == annotations["data"].shape[1] == len(set(annotations["objs"]))

    candidate_objects, candidate_recs, init_recs = [], [], []
    misplace_recs = []

    # iterate over every object and generate place/misplaced locations
    for col_idx, obj_key in enumerate(annotations["objs"]):
        # we don't have objects belonging to this category!
        if len(cat_data[obj_key]) == 0:
            print(f"Missing object-category models: {obj_key}")
            continue
        # matrix of room-recs x user-ranks filled with 0/1/-1
        ann_recs = annotations["data"][:, col_idx, :]
        avg_ranks = get_avg_th_ranks(ann_recs, thresh=6)
        obj_init_recs, obj_candidate_recs = [], []
        obj_misplace_recs = []
        # consider for misplacement
        for rec_idx, avg_rank in enumerate(avg_ranks):
            if avg_rank < 0:
                obj_init_recs.append((rec_idx, avg_rank))
                obj_misplace_recs.append((rec_idx, avg_rank))
            elif avg_rank > 0:
                obj_candidate_recs.append((rec_idx, avg_rank))
            elif avg_rank == 0:
                # add neutral receptacles as well!
                obj_init_recs.append((rec_idx, -5.0))

        # we have a initialization and misplace target for the object
        if len(obj_init_recs) and len(obj_candidate_recs):
            candidate_objects.append(obj_key)
            # sort place recs
            obj_candidate_recs = sorted(obj_candidate_recs, key=lambda x: x[1])
            candidate_recs.append(obj_candidate_recs)

            # sort misplace recs
            obj_init_recs = sorted(obj_init_recs, key=lambda x: x[1], reverse=True)
            obj_misplace_recs = sorted(obj_misplace_recs, key=lambda x: x[1], reverse=True)
            init_recs.append(obj_init_recs)
            misplace_recs.append(obj_misplace_recs)

    # shuffle objects present
    shuffle_inds = list(range(len(candidate_objects)))
    random.shuffle(shuffle_inds)

    # shuffle all lists created using indices
    candidate_recs = [candidate_recs[i] for i in shuffle_inds]
    candidate_objects = [candidate_objects[i] for i in shuffle_inds]
    init_recs = [init_recs[i] for i in shuffle_inds]
    misplace_recs = [misplace_recs[i] for i in shuffle_inds]

    sampled_init_recs = []
    # weighted sampling (re-shuffling) based on avg-ranks of receptacles
    for recs, crecs in zip(init_recs, candidate_recs):
        rec_inds = [r[0] for r in recs]
        rec_avg_ranks = np.array([r[-1] for r in recs])
        def softmax_with_temp(x, tau):
            """ Returns softmax probabilities with temperature tau
                Input:  x -- 1-dimensional array
                Output: s -- 1-dimensional array
            """
            e_x = np.exp(x / tau)
            return e_x / e_x.sum()
        rec_probs = softmax_with_temp(rec_avg_ranks, tau=3)
        rec_inds_sampled = np.random.choice(rec_inds, size=len(rec_inds), p=rec_probs, replace=False)
        sampled_init_recs.append(rec_inds_sampled)

        crecs = [r[0] for r in crecs]
        if set(rec_inds_sampled).intersection(set(crecs)) != set():
            import pdb
            pdb.set_trace()

    # print orm matches
    # debug_print(candidate_objects, misplace_recs, candidate_recs, annotations)

    # remove avg-ranks from place recs
    candidate_recs_inds = []
    for inds_scores in candidate_recs:
        inds = [idx_score[0] for idx_score in inds_scores]
        candidate_recs_inds.append(inds)

    return candidate_objects, sampled_init_recs, candidate_recs_inds


def record_placement_attempt(sim, success, scene_objs_ids, scene_objs_keys,
                             scene_recs_ids, scene_recs_keys, episode_obj_files,
                             state_matrix, end_matrix, obj_counter,
                             obj_id, obj_cat, init_rec, end_recs, obj_file, obj_steps, rec_steps, annotations, recs_packers):
    """
            (
            scene_objs_ids, scene_objs_keys,
            scene_recs_ids, scene_recs_keys,
            state_matrix, end_matrix, obj_counter, episode_obj_files
        ) = record_placement_attempt(sim, success, scene_objs_ids, scene_objs_keys, scene_recs_ids,
                                     scene_recs_keys, episode_obj_files, state_matrix, end_matrix,
                                     obj_counter, obj_id, obj_cat, rec_key, end_rec, obj_file, obj_steps,
                                     rec_steps, annotations)
    """
    if success:
        # add inserted object to object-logs
        obj_counter[obj_cat] += 1
        obj_key = f"{obj_cat}_{obj_counter[obj_cat]}"
        if obj_file.endswith(".urdf"):
            raise ValueError  # no urdfs are used!
        scene_objs_keys.append(obj_key)
        scene_objs_ids.append(obj_id)
        episode_obj_files.append(obj_file.replace("./", ""))
        try:
            # assert unique elements
            assert len(set(scene_objs_ids)) == len(scene_objs_ids)
            assert len(set(scene_objs_keys)) == len(scene_objs_keys)
        except:
            import pdb
            pdb.set_trace()
        # add inserted object to state matrices
        obj_col = np.zeros(shape=(state_matrix.shape[0], 1))
        # add to init-matrix
        init_rec_idx = scene_recs_keys.index(init_rec)
        obj_col[init_rec_idx] = 1
        state_matrix = np.concatenate([state_matrix, obj_col], axis=-1)
        obj_col *= 0
        # add to end-matrix
        end_recs = [annotations["room_recs"][i] for i in end_recs]
        for end_rec in end_recs:
            rec_inds, _, _ = match_rec_cat_to_instances(end_rec, scene_recs_keys, scene_recs_ids)
            obj_col[rec_inds] = 1
        end_matrix = np.concatenate([end_matrix, obj_col], axis=-1)
        try:
            assert state_matrix.sum() == state_matrix.shape[1]
            assert state_matrix.shape == end_matrix.shape
            assert_mapping_consistency(recs_packers, state_matrix, scene_objs_ids, scene_recs_ids)
        except:
            import pdb
            pdb.set_trace()
        print(f"Initialized {obj_key} on {init_rec}, which should go on: {end_recs}")
    else:
        sim.remove_objects([obj_id])

    return (
        scene_objs_ids, scene_objs_keys,
        scene_recs_ids, scene_recs_keys,
        state_matrix, end_matrix, obj_counter, episode_obj_files
    )


def check_shortest_path_exists(sim, obj_id, reset=True, reset_on_fail=False, return_steps=False, max_dist=1.2, max_steps=1000):
    # follower = ShortestPathFollower(sim, goal_radius=0.5, return_one_hot=False)
    # follower.mode = "geodesic_path"
    # obj_nav_pos = get_closest_nav_point(sim, obj_pos, ignore_y=True)
    # obj_pos = sim.get_translation(obj_id)
    obj_nav_pos = sim.snap_id_to_navmesh(obj_id, obj_id)

    oas = sim.get_agent_state()
    dist = sim.get_or_dist(obj_id, 'l2')
    count = max_steps
    while count > 0:
        # ap = sim.get_agent_state().position
        # act = follower.get_next_action(obj_nav_pos)
        act = sim.get_shortest_path_next_action(obj_nav_pos, snap_to_navmesh=False)
        dist = sim.get_or_dist(obj_id, 'l2')
        if act == 0:
            break
        obs = sim.step(act)
        count = count - 1
        # print(f"dist: {dist} and max: {max_dist}, {obj_pos - ap}")

    # reset agent back to original state
    if reset:
        sim.set_agent_state(oas.position, oas.rotation)
    elif reset_on_fail and dist > max_dist:
        sim.set_agent_state(oas.position, oas.rotation)

    if return_steps:
        return dist < max_dist, max_steps - count
    else:
        return dist < max_dist


def check_solvable(sim, obj_id, end_recs, annotations, scene_recs_keys, scene_recs_ids, recs_packers, curr_rec_id, max_recs=5):
    """We do not pick/place objects here"""
    # initial state
    init_as = sim.get_agent_state()
    # obj_shortest_path, obj_steps = check_shortest_path_exists(sim, obj_id, reset=False, return_steps=True, max_dist=1.2)
    obj_shortest_path, obj_steps = check_shortest_path_exists(sim, curr_rec_id, reset=False, return_steps=True, max_dist=1.2)

    if not obj_shortest_path:
        sim.set_agent_state(init_as.position, init_as.rotation)
        return obj_shortest_path, False, -1, -1

    # near object state
    obj_as = sim.get_agent_state()
    end_recs = [annotations["room_recs"][i] for i in end_recs]
    end_recs_ids = []
    for end_rec in end_recs:
        _, _, rec_ids = match_rec_cat_to_instances(end_rec, scene_recs_keys, scene_recs_ids)
        end_recs_ids.extend(rec_ids)

    # remove duplicates
    end_recs_ids = list(set(end_recs_ids))

    # get l2 distances from all receptacles and sort the ids
    dist = [sim.get_or_dist(rec_id, 'l2') for rec_id in end_recs_ids]
    sort_inds = np.argsort(dist)
    rec_ids_sorted = [end_recs_ids[idx] for idx in sort_inds]

    # try to go from the object to receptacle
    num_tries = max_recs
    rec_shortest_path = False

    # remove from init receptacle
    if not recs_packers[curr_rec_id].remove(obj_id):
        import pdb
        pdb.set_trace()

    while num_tries > 0 and len(rec_ids_sorted) > 0:
        rec_id = rec_ids_sorted.pop(0)
        rec_shortest_path, rec_steps = check_shortest_path_exists(sim, rec_id, reset=False,
                                                                  reset_on_fail=True, return_steps=True, max_dist=1.2)

        if rec_shortest_path:
            # check if placeable
            placeable = add_object_on_receptacle(obj_id, rec_id, sim, recs_packers)

            # remove if placed
            if placeable:
                if not recs_packers[rec_id].remove(obj_id):
                    import pdb
                    pdb.set_trace()
                # add back to original receptacle
                if not add_object_on_receptacle(obj_id, curr_rec_id, sim, recs_packers):
                    import pdb
                    pdb.set_trace()

                return obj_shortest_path, rec_shortest_path and placeable, obj_steps, rec_steps

    # add back to original receptacle
    if not add_object_on_receptacle(obj_id, curr_rec_id, sim, recs_packers):
        import pdb
        pdb.set_trace()

    return obj_shortest_path, rec_shortest_path, -1, -1


def assert_mapping_consistency(packers, state_matrix, scene_objs_ids, scene_recs_ids):
    rec_inds, obj_inds = state_matrix.nonzero()
    state_mapping = {scene_objs_ids[obj_ind]: scene_recs_ids[rec_ind] for obj_ind, rec_ind in
                     zip(list(obj_inds), list(rec_inds))}

    packer_mapping = {}
    for rec_id, packer in packers.items():
        obj_ids = list(packer.matches.keys())
        for obj_id in obj_ids:
            packer_mapping[obj_id] = rec_id

    if state_mapping != packer_mapping:
        import pdb
        pdb.set_trace()


def build_episode(sim, data, object_info, scene, episode, cat_data, scene_metadata, annotations, checks_threshold=50):
    obj_attr_mgr = sim.get_object_template_manager()
    agent_init_state = sim.get_agent_state()
    num_objects = np.random.choice(object_info["objects"])
    num_mis_objects = np.random.choice(object_info["mis_objects"])

    # parse data from scene and initialize state matrix
    (scene_objs_ids, scene_objs_keys), (scene_recs_ids, scene_recs_keys), state_matrix \
        = parse_scene_data(scene_metadata, data, cat_data)

    # initialize receptacle packers
    recs_packers = {}
    for rid in scene_recs_ids:
        recs_packers[rid] = ShelfBinPacker(get_bb_base(get_bb(sim, rid)))

    # sample objects their start and end receptacles from annotations
    objs_cats, init_recs, end_recs = filter_annotations(annotations, scene_recs_keys, cat_data)
    # objs_cats, init_recs, end_recs = np.array(objs_cats), np.array(init_recs), np.array(end_recs)
    # idx = np.where(objs_cats == "guitar")[0][0]
    # inds = [idx, scene_objs_ids.index(145)]
    # objs_cats, init_recs, end_recs = objs_cats[inds], init_recs[inds], end_recs[inds]
    # idx2 = 145


    # same object can be inserted multiple times, keep a counter
    end_matrix = deepcopy(state_matrix)
    obj_counter = Counter()
    rec_counter = Counter()
    episode_obj_files = deepcopy(scene_objs_keys)
    # original state_matrix shape (to record original no. of receptacles/objects)
    default_matrix_shape = state_matrix.shape
    obj_count = 0
    mis_obj_count = 0
    oracle_obj_steps, oracle_rec_steps = [], []

    for obj_cat, init_rec, end_rec in zip(objs_cats, init_recs, end_recs):
        start_time = time.time()

        if obj_count == num_objects and mis_obj_count == num_mis_objects:
            break

        obj_id, obj_type, obj_orientation, obj_file = add_object_in_scene(obj_cat, data, sim, obj_attr_mgr, cat_data)
        obj_add_time = time.time()

        # first initialize the misplaced objects then correct objects
        init_type = "mis" if mis_obj_count < num_mis_objects else "correct"

        # find ids and keys of end receptacles by matching category to scene-keys
        end_rec_cats = [annotations["room_recs"][i] for i in end_rec]
        end_rec_ids, end_rec_keys = [], []
        for er_cat in end_rec_cats:
            rec_inds, rec_keys, rec_ids = match_rec_cat_to_instances(er_cat, scene_recs_keys, scene_recs_ids)
            end_rec_ids.extend(rec_ids)
            end_rec_keys.extend(rec_keys)

        # find ids and keys of init receptacles by matching category to scene-keys
        init_rec_cats = [annotations["room_recs"][i] for i in init_rec]
        init_rec_ids, init_rec_keys = [], []
        for ir_cat in init_rec_cats:
            rec_inds, rec_keys, rec_ids = match_rec_cat_to_instances(ir_cat, scene_recs_keys, scene_recs_ids)
            init_rec_ids.extend(rec_ids)
            init_rec_keys.extend(rec_keys)

        # based on type of initialization, fill-in rec_keys and rec_ids to be used
        if init_type == "mis":
            # place on incorrect receptacle only
            rec_keys, rec_ids = init_rec_keys, init_rec_ids
        else:
            # place on correct (end) receptacle only
            rec_keys, rec_ids = end_rec_keys, end_rec_ids

        # attempt_order = list(range(len(rec_keys)))
        # random.shuffle(attempt_order)
        obj_steps, rec_steps = -1, -1
        shuffle_time = time.time()

        loop_add_obj_time = 0
        loop_check_time = 0

        for _idx, (rec_id, rec_key) in enumerate(zip(rec_ids, rec_keys)):
            success, added = [False] * 2
            if rec_counter[rec_key] >= 2:
                # print(f"Skipping {rec_key} because too much clutter on one receptacle!")
                continue

            _loop_start_time = time.time()
            # rec_key, rec_id = rec_keys[rec_idx], rec_ids[rec_idx]

            added = add_object_on_receptacle(obj_id, rec_id, sim, recs_packers)
            _loop_add_obj_time = time.time()
            loop_add_obj_time += (_loop_add_obj_time - _loop_start_time)

            if added:
                if init_type == "mis":
                    solvable_info = check_solvable(sim, obj_id, end_rec, annotations, scene_recs_keys, scene_recs_ids, recs_packers, rec_id)
                    obj_reachable, rec_reachable, obj_steps, rec_steps = solvable_info
                    success = obj_reachable and rec_reachable
                else:
                    success = check_shortest_path_exists(sim, obj_id)
                    obj_steps, rec_steps = 0, 0

            # calculate loop times
            _loop_check_time = time.time()
            loop_check_time += (_loop_check_time - _loop_add_obj_time)

            if success:
                oracle_obj_steps.append(obj_steps)
                oracle_rec_steps.append(rec_steps)
                if init_type == "mis":
                    mis_obj_count += 1
                obj_count += 1
                if rec_counter[rec_key] == 2:
                    import pdb
                    pdb.set_trace()
                rec_counter[rec_key] += 1
                break

            # remove from packers
            if added:
                try:
                    assert recs_packers[rec_id].remove(obj_id)
                except:
                    import pdb
                    pdb.set_trace()

            if _idx > checks_threshold:
                print(f"Skipping {obj_cat} because too much effort!")
                break
            # if not reachable:
            #     print(f"Skipping because not reachable!")
            # if not solvable:
            #     print(f"Skipping because not solvable!")
            # if not success:
            #     print(f"Skipping because couldn't place!")
        loop_out_time = time.time()

        # save if successful placement
        (
            scene_objs_ids, scene_objs_keys,
            scene_recs_ids, scene_recs_keys,
            state_matrix, end_matrix, obj_counter, episode_obj_files
        ) = record_placement_attempt(sim, success, scene_objs_ids, scene_objs_keys, scene_recs_ids,
                                     scene_recs_keys, episode_obj_files, state_matrix, end_matrix,
                                     obj_counter, obj_id, obj_cat, rec_key, end_rec, obj_file, obj_steps,
                                     rec_steps, annotations, recs_packers)

        # consistency check
        if len(oracle_obj_steps) != len(scene_objs_ids):
            import pdb
            pdb.set_trace()

        record_time = time.time()
        # print(
        #     f"obj_add: {obj_add_time - start_time} "
        #     f"shuffle: {shuffle_time - obj_add_time} "
        #     f"loop_add_obj: {loop_add_obj_time} "
        #     f"loop_check: {loop_check_time}"
        #     f"record: {record_time - loop_out_time} "
        # )

    # measure how many misplaced objects are inserted, and filter
    misplaced_count = ((state_matrix - end_matrix) * state_matrix).sum()

    # start from initial point and solve each rearrangement
    oracle_steps_solve = sum(oracle_obj_steps) + sum(oracle_rec_steps)

    print(f"Misplaced objects: {misplaced_count} // Inserted objects: {obj_count} "
          f"// Inserted misplaced: {mis_obj_count} // oracle-steps: {oracle_steps_solve}")

    # ensure packers are in sync with actual inserted objects
    packers_ids = [list(packer.matches.keys()) for _, packer in recs_packers.items()]
    for pids in packers_ids:
        if len(pids) > 2:
            import pdb
            pdb.set_trace()
    packers_ids = list(itertools.chain.from_iterable(packers_ids))
    try:
        assert len(packers_ids) == obj_count == state_matrix.shape[1]
    except:
        import pdb
        pdb.set_trace()

    if misplaced_count not in object_info["mis_objects"] or obj_count not in object_info["objects"]:
        return None

    # pack and ship this episode in a dictionary
    episode_dict = init_episode_dict(scene, episode, agent_init_state)

    # return position / rotation of newly placed objects // start-matrix // end-matrix
    scene_recs_pos = [np.array(sim.get_translation(id)).tolist() if id != -1 else [0.0]*3 for id in scene_recs_ids]
    scene_objs_pos = [np.array(sim.get_translation(id)).tolist() for id in scene_objs_ids]

    # get closest navigable points to receptacles
    nav_recs_pos = [np.array(get_closest_nav_point(sim, pos, ignore_y=True)).tolist() for pos in scene_recs_pos]

    # get categories of every receptacle and object
    scene_recs_cats = ["_".join(k.split("-")[1].split("_")[:-2]) for k in scene_recs_keys]
    scene_objs_cats = ["_".join(k.split("_")[:-1]) for k in scene_objs_keys]

    # sanity check
    for cat in scene_objs_cats + scene_recs_cats:
        if cat not in cat_data:
            import pdb
            pdb.set_trace()

    # assert consistency
    assert_mapping_consistency(recs_packers, state_matrix, scene_objs_ids, scene_recs_ids)

    # replace ids in recs_packers with keys
    id_to_key = {id:k for id, k in zip(scene_objs_ids, scene_objs_keys)}
    recs_packers = {rk: recs_packers[ri].to_dict(keep="key", id_to_key=id_to_key) for ri, rk in zip(scene_recs_ids, scene_recs_keys)}

    episode_dict.update({
        "default_matrix_shape": default_matrix_shape,
        "start_matrix": state_matrix.tolist(),
        "end_matrix": end_matrix.tolist(),
        "objs_keys": scene_objs_keys,
        "objs_cats": scene_objs_cats,
        "recs_keys": scene_recs_keys,
        "recs_cats": scene_recs_cats,
        "objs_pos": scene_objs_pos,
        "objs_rot": [get_rotation(sim, id) for id in scene_objs_ids],
        "recs_pos": scene_recs_pos,
        "recs_rot": [get_rotation(sim, id) for id in scene_recs_ids],
        "nav_recs_pos": nav_recs_pos,
        "misplaced_count": misplaced_count,
        "objects_count": obj_count,
        "objs_files": episode_obj_files,
        "recs_packers": recs_packers,
        "oracle_steps_solve": oracle_steps_solve,
        "oracle_paths": None,
    })

    # remove episode specific objects before init next episode
    inserted_object_ids = scene_objs_ids[default_matrix_shape[-1]:]
    # assert len(inserted_object_ids) == num_objects
    sim.remove_objects(inserted_object_ids)
    # print(f"Len: {len(sim.get_both_existing_object_ids()['non_art'])}")
    return episode_dict


def rewrite_hack(art_data, k, hack=True):
    """ rewrite urdfs on fly if paths don't match """
    paths = []
    if "urdfs" in art_data[k]:
        for urdf_path in art_data[k]["urdfs"]:
            if "kyash" in urdf_path and hack:
                split_paths = urdf_path.split("cos-hab2/")
                cwd = os.getcwd()
                urdf_file = os.path.join(cwd, split_paths[-1])
                paths.append(urdf_file)
                urdf_lines = open(urdf_file, "r").readlines()
                new_lines = []
                for line in urdf_lines:
                    if "kyash" in line:
                        line = line.replace("/home/kyash/Documents/cos-hab2", cwd)
                    new_lines.append(line)
                write_lines = open(urdf_file, "w").writelines(new_lines)

        # only if the paths don't match
        if len(paths) > 0:
            art_data[k]["urdfs"] = paths


def build_episodes(scenes, episode_num, combined_data, object_info, save_dir, append, debug, annotations, cat_data):
    sim = None
    write_freq = 20

    # for scene_id, scene in tqdm(enumerate(scenes), desc="Building Dataset", total=len(scenes)):
    for scene_id, scene in enumerate(scenes):
        scene_name = os.path.basename(scene).split(".glb")[0]
        # save_dir = f"data/datasets/cos_eor_v11/{split}"
        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{scene_name}.json.gz"
        save_path = os.path.join(save_dir, save_file)

        if os.path.exists(save_path):
            episodes = json.load(gzip.open(save_path, "r"))
            episode = len(episodes["episodes"])
            if not append:
                save_file = f"{scene_name}_again.json.gz"
            save_path = os.path.join(save_dir, save_file)
        else:
            episodes = {'episodes': []}
            episode = 0

        import pdb; pdb.set_trace()
        sim, scene_metadata = get_sim(scene, sim)
        
        # load all object-cofigs here -- from cache and from data
        load_annotations(sim, combined_data, False)

        episode_specific_ids = []
        pbar = tqdm(total=episode_num, initial=episode, desc="Building Episodes")
        while episode < episode_num:
            start_state, agent_object_id = init_agent(sim)
            episode_specific_ids.append(agent_object_id)
            if debug:
                debug_sim_viewer(sim)
            episode_dict = build_episode(
                sim, deepcopy(combined_data), object_info,
                scene, episode, cat_data, scene_metadata, deepcopy(annotations)
            )
            if episode_dict is not None:
                episodes['episodes'].append(episode_dict)
                print("\nFinished Episode {}/{} in scene {}".format(episode + 1, episode_num, scene), end=" ")
                episode += 1
                pbar.update()
            else:
                print(f"Skipped None Episode")

            if episode % write_freq == 0 or episode_num == episode:
                # build folder
                with gzip.open(save_path, "wt") as f:
                    json.dump(episodes, f)
                print(f"\n\n Dumped {episode} episodes for scene {scene} at {save_file}")


def aggregate_annotations(return_recs=True):
    csv_folder = "./text_housekeep/cos_eor/scripts/dump/csvs"
    csv_files = glob.glob(f"{csv_folder}/**", recursive=True)
    csv_files = [file for file in csv_files if file.endswith(".csv")]
    csvs = [(os.path.basename(cf).split(".")[0], pd.read_csv(cf)) for cf in csv_files]
    agg_data = {
        "room_recs": [],
        "recs": [],
        "rooms": [],
        "objs": [],
        "anns": [],
        "data": None
    }
    rooms = csvs[0][1]["Rooms"]
    recs = csvs[0][1]["Receptacles/Objects"]
    if return_recs:
        recs = [r.replace(" ", "_") for r in recs]
        return list(set(recs))
    agg_data["room_recs"] = [f"{room}-{rec}" for room, rec in zip(rooms, recs)]
    agg_data["recs"] = list(set(recs))
    agg_data["rooms"] = list(set(rooms))
    agg_data["objs"] = list(set(csvs[0][1].columns[2:]))
    agg_data["anns"] = [a for a, _ in csvs]
    agg_data["data"] = np.stack([df.values[:, 2:] for _, df in csvs], axis=0).astype(int)
    assert agg_data["data"].shape == (len(agg_data["anns"]), len(agg_data["room_recs"]), len(agg_data["objs"]))
    print(f"Total object-classes: {len(agg_data['objs'])}, receptacle-classes: {len(agg_data['recs'])}")

    # build semantic-sensor ids!
    semantic_classes = agg_data["recs"] + agg_data["objs"]
    semantic_classes = list(set([preprocess(sc) for sc in semantic_classes]))
    # insert custom-classes
    semantic_classes.extend(["door", "window"])
    semantic_classes.sort()
    assert len(semantic_classes) == len(set(semantic_classes))
    semantic_class_id_map = list(enumerate(semantic_classes, start=1))
    sem_classes_path = "cos_eor/scripts/dump/semantic_classes.yaml"
    yaml.dump({"semantic_class_id_map": semantic_class_id_map}, open(sem_classes_path, "w"))
    print(f"Dumped Semantic class-id map: {sem_classes_path}")
    return agg_data


def build_or_load_scene_object_splits_amt():
    splits_info_path = "./text_housekeep/cos_eor/scripts/orm/amt_data/splits.yaml"
    import pdb; pdb.set_trace()
    print(f"Loading AMT splits info!")
    splits_info = yaml.load(open(splits_info_path, "r"))
    for split in ["val", "test"]:
        splits_info["objects"][f"{split}_unseen"] = splits_info["objects"][split]
        splits_info["objects"][f"{split}_seen"] = splits_info["objects"]["train"]
        del splits_info["objects"][split]
        splits_info["scenes"][f"{split}_unseen"] = splits_info["scenes"][split]
        splits_info["scenes"][f"{split}_seen"] = splits_info["scenes"][split]
        del splits_info["scenes"][split]

    print("Categories -- "
        f"Train: {len(splits_info['objects']['train'])}, "
        f"Val: {len(splits_info['objects']['val_unseen'])}, "
        f"Test: {len(splits_info['objects']['test_unseen'])}"
    )

    return splits_info


def build_or_load_scene_object_splits(scenes, object_categories):
    splits_info_path = "cos_eor/scripts/dump/splits.yaml"
    import pdb; pdb.set_trace()
    if not os.path.exists(splits_info_path):
        print(f"Running splits info!")
        splits_info = {}
        # scene splits
        np.random.shuffle(scenes)
        splits_info["scenes"] = {
            "train": scenes[:8],
            "val": scenes[8:10],
            "test": scenes[10:],
        }
        # object splits -- hold out 20% only for testing
        np.random.shuffle(object_categories)
        seen_ratio = 0.8
        trainval_split = int(len(object_categories) * seen_ratio)
        splits_info["objects"] = {
            "train": object_categories[:trainval_split],
            "val": object_categories[:trainval_split],
            "test": object_categories,
        }
        # dump info and return
        yaml.dump(splits_info, open(splits_info_path, "w"))
        return splits_info
    else:
        print(f"Loading splits info!")
        return yaml.load(open(splits_info_path, "r"))


def load_scale_rots(reset=False, annotations=None):
    scale_rots_cache = "./text_housekeep/cos_eor/scripts/dump/scale_rots_all.npy"

    if os.path.exists(scale_rots_cache) and not reset:
        data = np.load(scale_rots_cache, allow_pickle=True).item()
    else:
        print(f"Rebuilding scale-rots!")
        # ycb
        ycb = yaml.load(open('./text_housekeep/cos_eor/scripts/dump/ycb_scale_rotation.yaml'))["accepted"]

        # rcad
        rcad = np.load("./text_housekeep/cos_eor/scripts/dump/rcad_scale_fil.npy", allow_pickle=True).item()["data"]

        # gso
        gso = np.load("./text_housekeep/cos_eor/scripts/dump/gso_scale_fil.npy", allow_pickle=True).item()["data"]

        # ab
        ab_data = np.load("./text_housekeep/cos_eor/scripts/dump/ab_manual_scale_fil.npy", allow_pickle=True).item()["data"]
        ab_cat_rots = np.load('./text_housekeep/cos_eor/scripts/dump/ab_scale_rotation.npy', allow_pickle=True).item()

        for item in ab_data:
            cat = item['product_type'][0]['value']
            assert cat in ab_cat_rots
            item["rot"] = ab_cat_rots[cat]["rot"]
            item["cat"] = cat
            item["source"] = "ab"

        for item in gso:
            item["source"] = "gso"

        for item in rcad:
            item["source"] = "rcad"

        # igibson furniture rotations and scaling
        if "sky" in os.getcwd():
            igib = yaml.load(open("cos_eor/scripts/dump/igib_scale_rotation_sky.yaml"))["accepted"]
        else:
            igib = yaml.load(open("cos_eor/scripts/dump/igib_scale_rotation.yaml"))["accepted"]
        # rewrite hack for fixing path issues
        for k in tqdm(igib, total=len(igib), desc="Rewriting Files"):
            rewrite_hack(igib, k)

        # combine everything in a single data dict
        data = {}
        for t in ycb:
            assert os.path.exists(t)
            _t = t.split("/")[-1].split(".")[0]
            data[_t] = {'pickable': True, 'type': 'object', 'sidenote': '', 'template': t, 'path': t, "cat": _t, "source": "ycb"}
            data[_t].update(ycb[t])

        for item in tqdm(ab_data + rcad + gso):
            t = item['path']
            try:
                assert os.path.exists(t)
            except:
                import pdb
                pdb.set_trace()
            _t = t.split("/")[-1].split(".")[0]
            data[_t] = {'pickable': True, 'type': 'object', 'sidenote': '', 'template': t, "cat": item['cat']}
            data[_t].update({
                "rotation": item["rot"],
                "scale": item["scale"]
            })

        # add categories to igib
        for k in igib:
            igib[k]["cat"] = k
            igib[k]["source"] = "igib"

        data.update(igib)
        np.save(scale_rots_cache, data)
        print(f"Dumped: {scale_rots_cache}")

    obj_keys = list(data.keys())
    del_datasets = []
    try:
        for ok in obj_keys:
            if data[ok]["categorised"] == False:
                print(ok)
                if "source" in data[ok]:
                    del_datasets.append(data[ok]["source"])
                del data[ok]
            elif annotations is not None:
                cats = annotations["objs"] + annotations["recs"]
                if data[ok]["cat"] not in cats:
                    if "source" in data[ok]:
                        print(ok)
                        del_datasets.append(data[ok]["source"])
                    del data[ok]
    except:
        import pdb
        pdb.set_trace()
    print(f"Loaded from: {scale_rots_cache}, Total: {len(obj_keys)}, \
     Categorized: {len(data)}, Uncategorized: {len(obj_keys) - len(data)}, Del datasets: {Counter(del_datasets).most_common()}")
    return data


def merge_data_annotations(data, annotations):
    pass


def main(args):
    # set seeds
    np.random.seed(4)
    random.seed(4)
    scenes_dir = "text_housekeep/data/scene_datasets/igibson/scenes/"
    scenes = os.listdir(scenes_dir)
    skip_scenes = ["Benevolence_0_int.glb"]
    scenes = [os.path.join(scenes_dir, p) for p in scenes if p not in skip_scenes and p.endswith(".glb")]
    assert len(scenes) == 14

    # load annotation and split related data
    all_recs = aggregate_annotations(return_recs=True)
    annotations, all_cats = aggregate_amt_annotations(all_recs)
    splits_info = build_or_load_scene_object_splits_amt()

    # filter based on split
    scenes = splits_info["scenes"][args.split]
    split_objs = splits_info["objects"][args.split]
    all_objs = deepcopy(annotations["objs"])

    for obj in all_objs:
        if obj not in split_objs:
            obj_ind = annotations["objs"].index(obj)
            annotations["objs"].pop(obj_ind)
            annotations["data"] = np.delete(annotations["data"], obj_ind, axis=1)

    try:
        assert len(split_objs) == len(annotations["objs"])
    except:
        import pdb
        pdb.set_trace()

    # load rotations and scaling data needed for custom insertion of objects
    data = load_scale_rots(args.reset_scale_rots, annotations)
    cat_data = {cat:[] for cat in all_cats}
    for dk in data:
        cat = data[dk]["cat"]
        if cat in cat_data:
            cat_data[cat].append(dk)
        else:
            raise ValueError

    import pdb; pdb.set_trace()
    object_info = {
        "mis_objects": list(range(args.min_mis_num_object, args.max_mis_num_object+1)),
        "objects": list(range(args.min_num_object, args.max_num_object+1))
    }

    # used to run on a single scene via job script
    if args.scene_id != -1:
        scenes = [scenes[args.scene_id]]
    import pdb; pdb.set_trace()
    print(f"Using scenes: ")
    for sc in scenes:
        print(sc)

    # build episodes now
    build_episodes(scenes, args.episode_num, combined_data=data, object_info=object_info, save_dir=args.save_dir,
                   append=args.append, debug=args.debug, annotations=annotations, cat_data=cat_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--split', dest='split', type=str, help='split')
    parser.add_argument('--save_dir', dest='save_dir', type=str, help='save_dir')
    parser.add_argument('--append', dest='append', action='store_true', help='append')
    parser.add_argument('--num_eps', dest='episode_num', type=int, help='episode_num')
    parser.add_argument('--max_objects', dest='max_num_object', type=int, help='num_object')
    parser.add_argument('--min_objects', dest='min_num_object', type=int, help='num_object')
    parser.add_argument('--max_mis_objects', dest='max_mis_num_object', type=int, help='num_object')
    parser.add_argument('--min_mis_objects', dest='min_mis_num_object', type=int, help='num_object')
    parser.add_argument('--debug', dest="debug", action='store_true')
    parser.add_argument('--reset', dest="reset_scale_rots", action='store_true')
    parser.add_argument('--scene_id', dest="scene_id", type=int, default=-1)
    args = parser.parse_args()
    main(args)

