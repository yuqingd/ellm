import argparse
import itertools
from multiprocessing import Pool
import os
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

import trimesh

def convert_urdf(file: Path, urdf_dir: Path, out_dir: Path):
    tree = ET.parse(file)
    root = tree.getroot()
    for mesh_node in root.findall(".//mesh"):
        obj_file = mesh_node.attrib["filename"]
        # In case obj is already converted, convert again
        obj_file = str(Path(obj_file).with_suffix(".obj"))
        glb_file = str(Path(obj_file).with_suffix(".glb"))
        mesh = trimesh.load(obj_file)
        mesh.export(file_type="glb", file_obj=glb_file)
        assert os.path.isfile(glb_file)
        mesh_node.set("filename", glb_file)
    common_prefix = os.path.commonprefix([urdf_dir, file])
    rel_file = file.relative_to(common_prefix)
    out_file = out_dir/rel_file
    tree.write(str(out_file))
    print("Completed", out_file)

def main(urdf_dir: Path, out_dir: Path, num_proc: int):
    files = urdf_dir.rglob("*urdf")
    args = list(zip(files, itertools.cycle([urdf_dir]), itertools.cycle([out_dir])))
    with Pool(num_proc) as pool:
        pool.starmap(convert_urdf, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--urdf-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=32
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path
    )

    args = parser.parse_args()
    main(**vars(args))
