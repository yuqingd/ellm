import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy

import pandas as pd
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
from text_housekeep.habitat.sims import make_sim

from text_housekeep.cos_eor.task.sensors import *
from text_housekeep.cos_eor.task.measures import *
from orp.utils import get_aabb
import logging
from text_housekeep.cos_eor.utils.debug import debug_sim_viewer
import json
import os
from collections import Counter
from tqdm import tqdm
import requests
import yaml
from collections import Counter
from shutil import copyfile

cwd = os.getcwd()
assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def rewrite_metadata():
    gso_dir = "data/google_object_dataset"
    glob_dump = "data/gso_glob.npy"
    gso_meta_path = "cos_eor/scripts/dump/gso_dump.npy"
    gso_metadata = {}

    if os.path.exists(glob_dump):
        paths = list(np.load(glob_dump, allow_pickle=True))
    else:
        paths = glob.glob(gso_dir + "/**", recursive=True)
        np.save(glob_dump, paths)

    paths = [pth for pth in paths if pth.endswith(".pbtxt")]
    gso_meta = [read_pbtxt(open(pth).readlines(), 0) for pth in tqdm(paths, desc="Reading pbtexts") if pth.endswith(".pbtxt")]

    # dump a dictionary
    for p,m in zip(paths, gso_meta):
        obj_name = os.path.basename(os.path.split(p)[0])
        gso_metadata[obj_name] = m

    # save the metadata
    np.save(gso_meta_path, gso_metadata)


def clean_configs():
    glob_dump = "data/gso_glob.npy"
    paths = list(np.load(glob_dump, allow_pickle=True))
    rm_paths = [p for p in paths if p.endswith("model.obj.object_config.json")]
    for p in tqdm(rm_paths, desc="Cleaning files"):
        os.remove(p)

def read_pbtxt(lines, li):
    data = {}
    ci = li
    while ci < len(lines):
        l = lines[ci].strip()
        ci += 1
        if "{" in l:
            key = l.split("{")[0].strip()
            ci, data[key] = read_pbtxt(lines, ci)
        elif ":" in l:
            colon_splits = l.split(":")
            key = colon_splits[0]
            value = ":".join(colon_splits[1:])
            key, value = key.strip(), eval(value.strip())
            data[key] = value
        elif "}" in l:
            return ci, data
    return data


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
    debug_sim_viewer(sim, False, meta_keys, object_ids, metadata, metadata_dir, dump_visuals="gso")


def main():
    # run below code once
    # rewrite_metadata()
    # clean_configs()
    import pdb; pdb.set_trace()

    scenes_dir = "data/scene_datasets/igibson/scenes/"
    scenes = os.listdir(scenes_dir)
    scene = [os.path.join(scenes_dir, p) for p in scenes if "Rs_int" in p][0]
    dump_visuals(scene)


if __name__ == '__main__':
    main()

