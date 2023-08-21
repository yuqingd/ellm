import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy

import yaml
from tqdm import tqdm

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

import habitat
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.physics import MotionType
from orp.obj_loaders import load_articulated_objs
from habitat.sims import make_sim

from cos_eor.task.sensors import *
from cos_eor.task.measures import *
from orp.utils import get_aabb

cwd = os.getcwd()
assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)
import logging
from cos_eor.utils.debug import debug_sim_viewer

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def get_sim(scene, sim, debug=False):
    """Habitat doesn't allow creating multiple simulators, and throws an error w/ OpenGL."""
    if not sim:
        config = habitat.get_config("cos_eor/configs/dataset/build_dataset.yaml")
        config = habitat.get_config_new([config.BASE_TASK_CONFIG])
        config.defrost()
        config.SIMULATOR.SCENE = scene
        config.freeze()
        sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    else:
        # reset so that agent is spawned on the floor
        sim_config = deepcopy(sim.habitat_config)
        sim_config.defrost()
        sim_config.SCENE = scene
        sim_config.freeze()
        sim.reconfigure(sim_config)

    return sim


def adjust_assets(scenes):
    sim = None
    global_mapping_path = "cos_eor/utils/global_mapping_v3_local.yaml"
    if os.path.exists(global_mapping_path) and False:
        global_mapping = yaml.load(open(global_mapping_path, "r"), Loader=yaml.BaseLoader)
    else:
        global_mapping = {
            "mapping_igib": {},
            "scenes_parsed": [],
            "version_notes": "Excluding pictures, floor lamps. Also, removed floating parts from articulate objects",
            "receptacles": [],
            "objects": []
        }

    # add non_art data
    non_art_data_path = 'cos_eor/utils/non_art_scale_rotation_v2.yaml'
    if os.path.exists(non_art_data_path):
        with open(non_art_data_path, "r") as f:
            non_art_data = yaml.load(f, Loader=yaml.BaseLoader)["accepted"]

    for template in non_art_data:
        global_mapping["objects"].append(non_art_data[template]["global_object"])

    # adjust and add used igib data
    for scene_id, scene in tqdm(enumerate(scenes), desc="Adjusting iGibson", total=len(scenes)):
        print(f"Scene: {scene}")
        scene_key = os.path.split(scene)[-1].split('.')[0]
        if scene_key in global_mapping["scenes_parsed"]:
            print(f"Skipping: {scene_key}")
            continue
        sim = get_sim(scene, sim)
        meta_keys, object_ids, metadata, metadata_dir = sim.init_metadata_objects(metadata_file="metadata_v2.yaml")
        if "urdfs" in metadata:
            metadata = metadata['urdfs']
        debug_sim_viewer(sim, False, meta_keys, object_ids, metadata, metadata_dir, global_mapping, adjust_igib=True)

    # add room-names to global-mapping from art-scale-rots yaml
    scale_rot_data = yaml.load(open("cos_eor/utils/art_scale_rotation_v3_sky.yaml"))
    global_mapping["object_room_map"] = {}

    for obj in scale_rot_data["accepted"].keys():
        try:
            rooms = set(scale_rot_data["accepted"][obj]["rooms"])
            rooms = ["_".join(r.split("_")[:-1]) for r in rooms if r is not None]
        except:
            print("not found", obj)
            continue
        global_mapping["object_room_map"][obj] = rooms

    # dump global-mapping
    with open(global_mapping_path, 'w') as f:
        yaml.dump(global_mapping, f)
    print(f"Dumped: {global_mapping_path}")


def main():
    import pdb; pdb.set_trace()
    scenes_dir = "data/scene_datasets/igibson/scenes/"
    scenes = os.listdir(scenes_dir)
    scenes = [os.path.join(scenes_dir, p) for p in scenes]
    adjust_assets(scenes)


if __name__ == '__main__':
    main()

