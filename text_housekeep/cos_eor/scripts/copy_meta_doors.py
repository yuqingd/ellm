import os
from shutil import copy2

from tqdm import tqdm

current_path = "data/metadata_with_doors"
assets_path = "data/scene_datasets/igibson/assets"
scenes = os.listdir(current_path)

import pdb
pdb.set_trace()

for sc in tqdm(scenes, desc="Copying"):
    scp = os.path.join(assets_path, sc)
    os.path.exists(scp)
    src_path = os.path.join(current_path, sc, "metadata_v2_readjusted_with_outer_doors.yaml")
    dest_path = os.path.join(assets_path, sc, "metadata_v2_readjusted_with_outer_doors.yaml")
    assert os.path.exists(src_path)
    copy2(src_path, dest_path)
