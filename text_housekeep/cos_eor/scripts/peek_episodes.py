import argparse
import csv
import glob
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy

import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np

splits = ["train", "val", "test"]

import pdb
pdb.set_trace()

for split in splits:
    print(f"Peeking into: {split}")
    save_dir = f"data/datasets/cos_eor/{split}"
    scenes = glob.glob(save_dir + "/**")
    for scene in scenes:
        with gzip.open(scene, "rt") as fp:
            episodes = json.load(fp)["episodes"]
        ### add your logic
        for eps in tqdm(episodes, desc=f"Episodes for {scene}, split: {split}"):
            avail_keys = eps.keys()
            objs_files = eps["objs_files"]
            default_objs = eps["default_matrix_shape"][-1]

            for of in objs_files:
                if of.endswith(".urdf"):
                    import pdb
                    pdb.set_trace()



