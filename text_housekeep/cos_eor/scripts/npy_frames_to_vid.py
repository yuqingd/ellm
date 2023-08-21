import argparse
from pathlib import Path
import os
import sys

import numpy as np

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from habitat.utils.visualizations.utils import images_to_video

parser = argparse.ArgumentParser()
parser.add_argument("--in-file", type=Path, required=True)
parser.add_argument("--out-file", type=Path)

def main(in_file: Path, out_file):
    if out_file is None:
        out_file = in_file
    frames = np.load(in_file, allow_pickle=True)
    images_to_video(frames, out_file.parent, out_file.stem, fps=4)
    print(f"Dumped: {out_file.stem}.mp4 video")

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
