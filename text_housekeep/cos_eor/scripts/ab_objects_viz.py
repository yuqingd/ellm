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
from text_housekeep.habitat_lab.habitat.sims import make_sim

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
# assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")
sys.path.insert(1, os.path.dirname(cwd))
sys.path.insert(1, cwd)
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
    debug_sim_viewer(sim, False, meta_keys, object_ids, metadata, metadata_dir, dump_visuals="ab")


def read_global_metadata(redo=False):
    # read 3d specific metadata
    file_3d = f"data/amazon-berkeley/3dmodels/metadata/3dmodels.csv"
    df_3d = pd.read_csv(file_3d)
    # read global metadata
    metadata_dir = "data/amazon-berkeley/listings/metadata"
    meta_path = os.path.join(metadata_dir)
    meta_files = os.listdir(meta_path)
    meta_files = [os.path.join(metadata_dir, file) for file in meta_files if not file.endswith("json.gz")]
    dump_path = "cos_eor/scripts/dump/ab_only_3d.npy"

    # read image metadata
    file_img = f"data/amazon-berkeley/images/metadata/images.csv.gz"
    df_img = pd.read_csv(file_img)

    if os.path.exists(dump_path) and not redo:
        return np.load(dump_path, allow_pickle=True).item()

    # filter data and store both information
    fil_data = []
    for file_3d in tqdm(meta_files):
        page = open(file_3d, "r")
        lines = page.readlines()
        for line in lines:
            obj_data = json.loads(line)
            if "3dmodel_id" not in obj_data:
                continue

            row_3d = df_3d[df_3d["3dmodel_id"] == obj_data["item_id"]]
            info_3d = row_3d.to_dict(orient="records")[0]
            keys = list(info_3d.keys())
            for k in keys:
                if "3d" not in k:
                    info_3d[f"3d_{k}"] = info_3d[k]
            obj_data.update(info_3d)
            if "main_image_id" in obj_data:
                row_img_main = df_img[df_img["image_id"] == obj_data["main_image_id"]]
                info_img_main = row_img_main.to_dict(orient="records")[0]
                obj_data["image_main"] = info_img_main
            else:
                print(f"main image id not found")
            fil_data.append(obj_data)

    import pdb
    pdb.set_trace()

    # check types
    fil_types = []
    rej_types = []
    obj_types = [i["product_type"][0]["value"] for i in fil_data]
    type_count = Counter(obj_types)
    for idx, (typ, cnt) in enumerate(type_count.items()):
        inp = ""
        while inp.lower() not in ["y", "n"]:
            inp = input(f"Index: {idx+1}/{len(type_count)} Type: {typ} with count {cnt}, keep?: ")
        if inp == "y":
            fil_types.append(typ)
        else:
            rej_types.append(typ)

    store_data = {
        "keep_types": fil_types,
        "reject_types": rej_types,
        "data": fil_data,
        "note": "Filtered manually by Yash by only looking at the 'product_type' field, and no visuals."
    }
    np.save(dump_path, store_data)
    print(f"Dumped {dump_path} with {len(fil_data)} entries")
    return fil_data


def add_obj_configs():
    models_dir = f"data/amazon-berkeley/3dmodels/original"
    models = glob.glob(models_dir + "/**", recursive=True)
    models = [m for m in models if m.endswith(".glb")]
    for m in tqdm(models):
        dir, fname = os.path.split(m)
        obj_cfg_data = f'"render_asset": "{fname}", "mass": 9 \n'
        obj_cfg_data = "{" + obj_cfg_data + "}"
        fname = fname.split(".")[0]
        obj_cfg = os.path.join(dir, f"{fname}.object_config.json")
        with open(obj_cfg, "w") as f:
            f.write(obj_cfg_data)


def remove_fil(data):
    keep_types = data["keep_types"]
    fil_data = []
    for item in data["data"]:
        if item["product_type"][0]["value"] in keep_types:
            fil_data.append(item)
    print(f"From {len(data['data'])} items, kept {len(fil_data)}")
    data["data"] = fil_data
    return data


def second_manual_filter(data, redo=False):
    dump_path = "cos_eor/scripts/dump/ab_manual_fil.npy"

    if os.path.exists(dump_path):
        return np.load(dump_path, allow_pickle=True).item()

    skip_types = [
        'HOME_FURNITURE_AND_DECOR',
        'ELECTRIC_FAN',
        'PROFESSIONAL_HEALTHCARE',
        'SOUND_AND_RECORDING_EQUIPMENT',
        'PLACEMAT',
        'CLOCK',
        'DEHUMIDIFIER'

    ]
    keep_types = [
        'LAMP',
        'PILLOW',
        'VASE',
        'ELECTRONIC_CABLE',
        'PORTABLE_ELECTRONIC_DEVICE_STAND',
        'STORAGE_BOX',
        'DRINK_COASTER',
        'MULTIPORT_HUB',
        'MOUSE_PAD',
        'PORTABLE_ELECTRONIC_DEVICE_COVER',
        'OUTDOOR_LIVING'
    ]
    skip_regex = ["wall", "utility cart", "stool", "chair", "floor mat", "mirror", "picture frame", "wagon"]
    keep_regex = ["pillow", "canister", "dumbbell", "notebook"]

    img_dir = "data/amazon-berkeley/images/original/"
    obj_types = [i["product_type"][0]["value"] for i in data["data"]]
    type_count = Counter(obj_types)

    fil_data = []
    residual_data = []

    for item in data["data"]:
        item_name = [d["value"] for d in item["item_name"] if "en" in d["language_tag"]]
        item_type = item['product_type'][0]['value']

        # apply regex filtering
        if len(item_name) > 0:
            item_name = item_name[0]
        else:
            item_name = item["item_name"][0]["value"]
        found = False
        for rex in skip_regex:
            if rex in item_name.lower():
                found = True
                break
        # skip only if found
        if found:
            print(f"Skipping: {item_name} due to regex filter")
            continue

        # apply type filtering
        if item_type in skip_types:
            print(f"Skipping: {item_name} due to type filter")
            continue
        if item_type in keep_types:
            fil_data.append(item)
            continue

        # add residual data for next stage
        residual_data.append(item)

    # manual filtering on remaining
    for idx, item in enumerate(residual_data):
        while True:
            # skip if already excluded
            if item["product_type"][0]["value"] in skip_types:
                keep = False
                break

            # add if already included
            if item["product_type"][0]["value"] in keep_types:
                keep = True
                break

            # get item name in english
            item_name = [d["value"] for d in item["item_name"] if "en" in d["language_tag"]]
            if len(item_name) > 0:
                item_name = item_name[0]
            else:
                item_name = item["item_name"][0]["value"]
            item_type = item['product_type'][0]['value']

            # show image
            img_path = item["image_main"]["path"] if "image_main" in item else None
            if img_path is not None:
                disp_path = "debug-data/ab_disp.jpg"
                img_path = os.path.join(img_dir, img_path)
                if os.path.exists(img_path):
                    copyfile(img_path, disp_path)
                else:
                    print(f"No Image available")
            else:
                print("No image available!")

            # show information
            print(f"{idx+1} / {len(residual_data)} \n Type: {item_type} \n Name: {item_name} \n Type Count: {type_count[item_type]}")
            inp = input("y: keep || n: skip || yt: keep type || nt: skip type || d: debug || s:save -- ")

            if inp == "y":
                keep = True
                break
            elif inp == "n":
                keep = False
                break
            elif inp == "yt":
                keep = True
                keep_types.append(item_type)
                break
            elif inp == "nt":
                keep = False
                skip_types.append(item_type)
                break
            elif inp == "d":
                import pdb
                pdb.set_trace()
            elif inp == "s":
                data["data"] = fil_data
                data["manual_skip_types"] = skip_types
                data["manual_keep_types"] = skip_types
                data["last_idx"] = idx-1
                np.save(dump_path, data)
                print(f"Dumped at {dump_path}")

        assert keep in [True, False]
        if keep:
            fil_data.append(item)

    data["data"] = fil_data
    data["manual_skip_types"] = skip_types
    data["manual_keep_types"] = keep_types
    data["manual_skip_regex"] = skip_regex
    data["manual_keep_regex"] = keep_regex

    np.save(dump_path, data)
    print(f"Dumped at {dump_path}")

    return data


def main():
    # one-time runs below
    # data = read_global_metadata(redo=False)
    # data_fil1 = remove_fil(data)
    # data_fil2 = second_manual_filter(-1)
    add_obj_configs()

    # scenes_dir = "data/scene_datasets/igibson/scenes/"
    # scenes = os.listdir(scenes_dir)
    # scene = [os.path.join(scenes_dir, p) for p in scenes if "Rs_int" in p][0]
    # dump_visuals(scene)
    # originally dumped at non_art_data_path = 'cos_eor/utils/non_art_scale_rotation_v2.yaml'
    # now at 'cos_eor/scripts/dump/ycb_scale_rotations.yaml"
    # run: py cos_eor/scripts/ycb_objects_scale_rots_manual.py





if __name__ == '__main__':
    main()

