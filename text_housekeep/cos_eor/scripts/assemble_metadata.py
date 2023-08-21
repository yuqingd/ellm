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

IGIB_ASSETS = "data/scene_datasets/igibson/assets/"
scenes = os.listdir(IGIB_ASSETS)

import pdb
pdb.set_trace()

for sc in tqdm(scenes):
    sc_path = os.path.join(IGIB_ASSETS, sc)
    meta_v2 = os.path.join(sc_path, "metadata_v2.yaml")
    meta_v2_readj = os.path.join(sc_path, "metadata_v2_readjusted.yaml")
    meta_v2_readj_doors = os.path.join(sc_path, "metadata_v2_readjusted_with_outer_doors.yaml")
    meta_assemble_path = os.path.join(sc_path, "metadata_assembled.yaml")
    try:
        assert os.path.exists(meta_v2) and os.path.exists(meta_v2_readj_doors)
    except:
        import pdb
        pdb.set_trace()

    meta_v2 = yaml.load(open(meta_v2))
    meta_v2_readj_doors = yaml.load(open(meta_v2_readj_doors))
    meta_v2_readj = yaml.load(open(meta_v2_readj))

    meta_assemble = {
        "default_mapping": meta_v2_readj_doors["default_mapping"],
        "urdfs": {}
    }

    # add receptacles from readjusted, but positions from v2
    for urdf in meta_v2_readj["urdfs"]:
        meta_assemble["urdfs"][urdf] = meta_v2[urdf]

    # add doors from arun's file, but positions from v2
    for urdf in meta_v2_readj_doors["urdfs"]:
        if "door" in urdf:
            meta_assemble["urdfs"][urdf] = meta_v2[urdf]

    yaml.dump(meta_assemble, open(meta_assemble_path, "w"))
    print(f"Dumped: {meta_assemble_path}")
