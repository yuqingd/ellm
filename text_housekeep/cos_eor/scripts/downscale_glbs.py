import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from shutil import copyfile
import subprocess

from tqdm import tqdm

DOWNSCALE_DIM = 64

def get_files(dir: Path):
    return dir.rglob("*.glb")

def downscale(file):
    file = str(file)
    orig_file = file[:-len(".glb")] + "_original.glb"
    if not os.path.exists(orig_file):
        copyfile(file, orig_file)
        assert os.path.exists(orig_file)
    resize_results = subprocess.run([
        "node",
        "node_modules/@gltf-transform/cli/bin/cli.js",
        "resize",
        orig_file,
        file,
        "--width", str(DOWNSCALE_DIM),
        "--height", str(DOWNSCALE_DIM),
    ], stdout=subprocess.DEVNULL)
    return file, resize_results.returncode == 0

def main(in_dir: Path, tmp_dir: Path, num_proc: int):
    tmp_dir.mkdir(exist_ok=True, parents=True)
    files = get_files(in_dir)
    files = [file for file in files if not str(file).endswith("_original.glb")]
    with Pool(num_proc) as pool:
        for file, result in tqdm(pool.imap_unordered(downscale, files, chunksize=4), total=len(files)):
            if result:
                print("Downscaled", file)
            else:
                print("Failed to downscale", file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=32
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default="tmp/glb_downscale"
    )

    args = parser.parse_args()
    main(**vars(args))