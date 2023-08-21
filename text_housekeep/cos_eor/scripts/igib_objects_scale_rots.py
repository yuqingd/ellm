import glob
import json
import os
import magnum as mn
import yaml
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from tqdm import tqdm
import numpy as np

cwd = os.getcwd()
assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")
import os.path as osp

ig_obj_path = f"{cwd}/iGibson/gibson2/data/ig_dataset/objects"
cat_specs = json.load(open(f"{ig_obj_path}/avg_category_specs.json", "r"))
save_path = f"{cwd}/cos_eor/utils/object_metadata_v2.json"
ig_dataset = "iGibson/gibson2/data/ig_dataset/"


def load_old():
    metadata = json.load(open(save_path, "r"))
    # remove unwanted parts
    for key in metadata:
        if "rots" in metadata[key]:
            del metadata[key]["rots"]
        if "urdfs" in metadata[key]:
            del metadata[key]["urdfs"]
    return metadata


metadata = load_old()
scenes_dir = osp.join(ig_dataset, 'scenes')
scene_dirs = list(os.listdir(scenes_dir))
scene_dirs = [d for d in scene_dirs if "int" in d]


for scene_dir in tqdm(scene_dirs):
    print(f"Reading from {scene_dir}")

    # Load the scene layout.
    scene = InteractiveIndoorScene(
        scene_dir, texture_randomization=False, object_randomization=False,
        custom=True)

    object_data = []
    for k, art_obj in scene.objects_by_name.items():
        if k in ['walls', 'floors', 'ceilings']:
            continue
        n_comps = len(art_obj.urdf_paths)

        # if n_comps > 1:
        #     # manually filter to keep only the main file and important file
        #     file_lens = []
        #     file_strs = []
        #     for i in range(n_comps):
        #         file_str = open(art_obj.urdf_paths[i], "r").read()
        #         file_lens.append(len(file_str))
        #         file_strs.append(file_str)
        #
        #     keep_idx = file_lens.index(max(file_lens))
        #     print(file_strs[keep_idx])
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     keep_idx = 0

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

                if "normalized" in file_str:
                    if keep_idx != -1:
                        import pdb
                        pdb.set_trace()
                    keep_idx = i
        else:
            keep_idx = 0

        # do not consider
        if keep_idx == -1:
            continue

        # only keep the first one
        i = keep_idx
        urdf_file = art_obj.urdf_paths[i]
        if not osp.exists(urdf_file):
            raise ValueError('Could not find', urdf_file)
        object_data.append([urdf_file, art_obj.my_is_fixed, art_obj.poses[i]])

    for fname, is_fixed, trans in object_data:
        rot = mn.Quaternion.from_matrix(trans[:3,:3])
        rot = mn.Quaternion.rotation(mn.Deg(-90), mn.Vector3(1,0,0)) * rot
        rot = [*rot.vector, rot.scalar]
        pos = trans[:3, -1]
        room = scene.get_room_instance_by_point(pos[:-1])
        # print(f"{os.path.split(fname)[-1]} is placed in {room}")
        # map fname to metadata key and put the rotation in there.
        key_name = fname.split("/")[-1].split("_")[:-2]
        key_name = "_".join(key_name)
        assert key_name in metadata

        if "floor" in key_name or "picture" in key_name:
            metadata[key_name]["type"] = "receptacle"

        if "rot" in metadata[key_name]:
            del metadata[key_name]["rot"]

        # keeping the first found rotation.
        if "rots" in metadata[key_name]:
            metadata[key_name]["rots"].append(rot)
            metadata[key_name]["urdfs"].append(fname)
        else:
            metadata[key_name]["rots"] = [rot]
            metadata[key_name]["urdfs"] = [fname]

        # keeping the first found rotation.
        if "rooms" in metadata[key_name]:
            metadata[key_name]["rooms"].append(room)
        else:
            metadata[key_name]["rooms"] = [room]

if "sky" in os.getcwd():
    new_save_path = f"{cwd}/cos_eor/utils/art_scale_rotation_v3_sky.yaml"
else:
    new_save_path = f"{cwd}/cos_eor/utils/art_scale_rotation_v3.yaml"

save_data = {
    "accepted": metadata,
    "rejected": []
}

yaml.dump(save_data, open(new_save_path, "w"))
print(f"Dumped: {new_save_path}")
