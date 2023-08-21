import sys
from copy import deepcopy

sys.path.insert(0, "./")
from habitat_sim.physics import MotionType
import yaml
import logging
import json
import os
import magnum as mn
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from tqdm import tqdm
import numpy as np
import os
from shutil import copyfile

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
cwd = os.getcwd()
assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")
import os.path as osp


def load_old():
    metadata = json.load(open(f"{cwd}/cos_eor/utils/object_metadata_v2.json", "r"))
    # metadata_filled = json.load(open(f"{cwd}/cos_eor/utils/object_metadata_filled_v2.json", "r"))
    return metadata


def load_obj_dats(scene_dir):
    """Load the full scene, and only parse the objects w/o walls, ceiling, floor"""
    # Load the scene layout.
    scene = InteractiveIndoorScene(
        scene_dir, texture_randomization=False, object_randomization=False,
        custom=True)
    object_data = []

    import pdb
    pdb.set_trace()

    for k, art_obj in scene.objects_by_name.items():
        if k in ['walls', 'floors', 'ceilings']:
            continue

        # number of components of a single articulate object
        n_comps = len(art_obj.urdf_paths)

        # not considering these for now (but we need beds!)
        if n_comps > 1:
            # manually filter to keep only the main file and important file
            file_lens = []
            file_strs = []
            keep_idx = -1
            for i in range(n_comps):
                file_str = open(art_obj.urdf_paths[i], "r").read()
                file_lens.append(len(file_str))
                file_strs.append(file_str)
                # main file in beds have "normalized" in file_str
                if "normalized" in file_str:
                    # wrong logic if two files have normalized.
                    if keep_idx != -1:
                        import pdb
                        pdb.set_trace()
                    keep_idx = i
        else:
            keep_idx = 0

        # do not consider
        if keep_idx == -1:
            print(f"skipping: {art_obj.urdf_paths}")
            continue

        # only keep the first one
        i = keep_idx
        urdf_file = art_obj.urdf_paths[i]
        if not osp.exists(urdf_file):
            raise ValueError('Could not find', urdf_file)

        trans = art_obj.poses[i]
        pos = trans[:3, -1]
        room = scene.get_room_instance_by_point(pos[:-1])
        # print(f"room: {room} // urdf: {urdf_file}")
        object_data.append([urdf_file, art_obj.my_is_fixed, art_obj.poses[i], room])

    return object_data


def dump_obj_dats(object_data, metadata, assets_dir, scene_dir):
    # create dirs
    save_dir = os.path.join(assets_dir, scene_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Save the scene layout to the yaml config format
    def convert_obj_dat_to_str(obj_dat):
        """ Calculate object positon/rotation, and set its motion type."""
        # obj_dat -- urdf_file, art_obj.my_is_fixed, art_obj.poses[i]
        fname, is_fixed, trans, room = obj_dat
        rot = mn.Quaternion.from_matrix(trans[:3,:3])
        rot = mn.Quaternion.rotation(mn.Deg(-90), mn.Vector3(1,0,0)) * rot
        rot = [*rot.vector, rot.scalar]
        pos = trans[:3, -1]
        # Z is up for igibson
        pos[1], pos[2] = pos[2], pos[1]
        pos[2] *= -1.0
        if is_fixed:
            obj_type = int(MotionType.KINEMATIC)
        else:
            obj_type = int(MotionType.DYNAMIC)
        return {
            "urdf": fname,
            "pos": pos.astype(float).tolist(),
            "rot": np.array(rot).tolist(),
            "obj_type": obj_type,
            "room": room
        }

    scene_metadata = {}
    for od in object_data:
        odict = convert_obj_dat_to_str(od)
        fname = odict["urdf"]
        local_name = fname.split("/")[-1]
        key_name = local_name.split("_")[:-2]
        key_name = "_".join(key_name)
        assert key_name in metadata
        # copy file
        copyfile(fname, os.path.join(save_dir, local_name))
        # add to scene metadata
        scene_metadata[local_name] = deepcopy(metadata[key_name])
        scene_metadata[local_name].update(odict)
        scene_metadata[local_name]["category"] = key_name
        # remove previous keys -- these are the only keys that differ across filled or not and they are removed!
        for k in ["rots", "urdfs"]:
            if k in scene_metadata[local_name]:
                del scene_metadata[local_name][k]
        try:
            assert len(scene_metadata[local_name]) == 9
        except:
            import pdb
            pdb.set_trace()

    metadata_file = os.path.join(save_dir, "metadata_v2.yaml")
    with open(metadata_file, 'w') as f:
        yaml.dump(scene_metadata, f)
    print(f"Dumped: {metadata_file}")


def add_rooms(object_data, assets_dir, scene_dir):
    save_dir = os.path.join(assets_dir, scene_dir)
    current_file = os.path.join(save_dir, "metadata_v2_readjusted.yaml")
    current_data = yaml.load(current_file)

    import pdb
    pdb.set_trace()

    for od in object_data:
        fname = odict["urdf"]
        local_name = fname.split("/")[-1]
        current_data[local_name]["room"] = od[-1]

    import pdb
    pdb.set_trace()

    new_file = os.path.join(save_dir, "metadata_v2_readjusted_room.yaml")
    with open(new_file, 'w') as f:
        yaml.dump(new_file, f)
        print(f"Dumped: {new_file}")


# todo: add room-names to every object here!
if __name__ == '__main__':
    import pdb; pdb.set_trace()
    scenes_dir = "data/scene_datasets/igibson/scenes/"
    assets_dir = "data/scene_datasets/igibson/assets/"

    # load the manual annotations
    metadata = load_old()
    scenes = list(os.listdir(scenes_dir))

    for scene_dir in tqdm(scenes):
        scene_name = scene_dir.split(".glb")[0]
        if "Rs_int" not in scene_dir:
            continue
        # print(f"############# Extracting from: {scene_dir} #############")
        obj_dats = load_obj_dats(scene_name)
        dump_obj_dats(obj_dats, deepcopy(metadata), assets_dir, scene_name)
        # add_rooms(obj_dats, assets_dir, scene_name)