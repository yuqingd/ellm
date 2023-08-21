import yaml
import numpy as np
import os

from text_housekeep.cos_eor.scripts.orm.utils import preprocess


def match_rec_cat_to_instances(rec_cat, scene_recs_keys, scene_recs_ids):
    rec_inds, rec_keys, rec_ids = [], [], []
    scene_rr = get_scene_rec_names(scene_recs_keys)
    for ind, (rn, rk, rid) in enumerate(zip(scene_rr, scene_recs_keys, scene_recs_ids)):
        if rec_cat in rn:
            rec_keys.append(rk)
            rec_ids.append(rid)
            rec_inds.append(ind)
    return rec_inds, rec_keys, rec_ids


def get_scene_rec_names(scene_recs_keys):
    scene_recs_names = []
    for k in scene_recs_keys:
        room, k = k.split("-")
        # if k == "floor":
        #     scene_recs_names.append(k)
        #     continue
        # rn = k.split("_")[:-2]
        # rn = "_".join(rn)
        rn = preprocess(k)
        rn = rn.replace(" ", "_")
        room = room.split("_")
        room = "_".join(room[:-1])
        if rn not in scene_recs_names:
            scene_recs_names.append(f"{room}-{rn}")
    return scene_recs_names


def aggregate_amt_annotations(all_recs=None, only_amt=False):
    agg_data = {
        "room_recs": [],
        "recs": [],
        "rooms": [],
        "objs": [],
        "anns": [],
        "data": None
    }

    amt_data = np.load(os.getcwd().split('/exp_local')[0] + '/text_housekeep/cos_eor/scripts/orm/amt_data/data.npy', allow_pickle=True).item()
    agg_data["room_recs"] = [str(rr.replace("|", "-")) for rr in amt_data["room_receptacles"]]
    agg_data["objs"] = [str(obj) for obj in amt_data["objects"]]
    agg_data["recs"] = list(set([str(rr.split("|")[-1]) for rr in amt_data["room_receptacles"]]))
    agg_data["rooms"] = list(set([str(rr.split("|")[0]) for rr in amt_data["room_receptacles"]]))
    agg_data["data"] = amt_data["ranks"]
    print(f"Total object-classes from AMT: {len(agg_data['objs'])}, receptacle-classes: {len(agg_data['recs'])}")

    if only_amt:
        return agg_data

    # build semantic-sensor ids!
    semantic_classes = agg_data["recs"] + agg_data["objs"] + all_recs
    # insert custom-classes
    semantic_classes.extend(["door", "window"])
    semantic_classes = list(set(semantic_classes))
    semantic_classes.sort()
    assert len(semantic_classes) == len(set(semantic_classes))
    semantic_class_id_map = list(enumerate(semantic_classes, start=1))
    sem_classes_path = "./text_housekeep/cos_eor/scripts/dump/semantic_classes_amt.yaml"
    yaml.dump({"semantic_class_id_map": semantic_class_id_map}, open(sem_classes_path, "w"))
    print(f"Dumped Semantic class-id map: {sem_classes_path}")
    return agg_data, semantic_classes


