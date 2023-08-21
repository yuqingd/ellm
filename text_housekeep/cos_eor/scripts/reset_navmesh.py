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

import pdb; pdb.set_trace()
scenes_dir = "data/scene_datasets/igibson/scenes/"
scenes = os.listdir(scenes_dir)
navmeshes = [os.path.join(scenes_dir, p) for p in scenes if p.endswith(".navmesh")]

for nm in tqdm(navmeshes, desc="Deleting navmeshes"):
    print(f"Deleting: {nm}")
    os.remove(nm)

