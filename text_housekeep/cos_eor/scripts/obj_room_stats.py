import argparse
from collections import defaultdict
import json
import os
import sys

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from habitat_baselines.config.default import get_config
from habitat import make_dataset

from cos_eor.dataset.dataset import CosRearrangementDatasetV0, CosRearrangementEpisode
from cos_eor.scripts.orm.utils import preprocess

def get_room_from_rec_key(rec_key):
    return preprocess(rec_key.split("-")[0].rsplit("_", 1)[0])

def get_obj_key_to_cat_map(episode: CosRearrangementEpisode):
    key_to_cat = {}
    for obj_key, obj_cat in zip(episode.objs_keys, episode.objs_cats):
        key_to_cat[obj_key] = preprocess(obj_cat)
    return key_to_cat

def get_scene_stats(config_file, stats):
    config = get_config(config_file)
    scene = config.BASE_TASK_CONFIG.DATASET.CONTENT_SCENES[0]
    print("Starting scene", scene)
    dataset: CosRearrangementDatasetV0 = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    for episode in dataset.episodes:
        correct_mapping = episode.get_correct_mapping()
        obj_key_to_cat = get_obj_key_to_cat_map(episode)
        for obj_key, recs in correct_mapping.items():
            obj_cat = obj_key_to_cat[obj_key]
            for rec_key in recs:
                room = get_room_from_rec_key(rec_key)
                stats[obj_cat][room] += 1

def main(config_files, out_file):
    stats = defaultdict(lambda: defaultdict(int))
    for config_file in config_files:
        get_scene_stats(config_file, stats)
    with open(out_file, "w") as f:
        json.dump(stats, f)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-files",
    nargs="+",
    required=True,
    help="List of config files to use for loading datasets"
)

parser.add_argument(
    "--out-file",
    required=True
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
