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


def get_sim(scene, sim, config):
    """Habitat doesn't allow creating multiple simulators, and throws an error w/ OpenGL."""
    if not sim:
        config = habitat.get_config(config)
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

def dump_visuals(scene):
    sim = None
    # adjust assets on the sofa within "Rs_int"
    sim = get_sim(scene, sim, "cos_eor/configs/dataset/build_dataset_viz.yaml")
    meta_keys, object_ids, metadata, metadata_dir = sim.init_metadata_objects()
    debug_sim_viewer(sim, False, meta_keys, object_ids, metadata, metadata_dir, dump_visuals="ycb")


def main():
    import pdb; pdb.set_trace()
    scenes_dir = "data/scene_datasets/igibson/scenes/"
    scenes = os.listdir(scenes_dir)
    scene = [os.path.join(scenes_dir, p) for p in scenes if "Rs_int" in p][0]
    dump_visuals(scene)
    # originally dumped at non_art_data_path = 'cos_eor/utils/non_art_scale_rotation_v2.yaml'
    # now at 'cos_eor/scripts/dump/ycb_scale_rotations.yaml"
    # run: py cos_eor/scripts/ycb_objects_scale_rots_manual.py

if __name__ == '__main__':
    main()

