import glob
import json
import os
cwd = os.getcwd()
assert cwd.endswith("cos-hab2") or cwd.endswith("p-viz-plan")

ig_obj_path = f"{cwd}/iGibson/gibson2/data/ig_dataset/objects"
cat_specs = json.load(open(f"{ig_obj_path}/avg_category_specs.json", "r"))
save_path = f"{cwd}/cos_eor/utils/object_metadata.json"
_continue = False

def pull_videos(obj):
    path = f"{ig_obj_path}/{obj}/**"
    files = glob.glob(path, recursive=True)
    for file in files:
        if file.endswith(".mp4"):
            link_path = f"{cwd}/cos_eor/scripts/igib_obj_viz/{obj}"
            os.makedirs(link_path, exist_ok=True)
            import subprocess
            subprocess.run(f"ln -s {file} {link_path}", shell=True)


def load_old():
    old_metadata = json.load(open(save_path, "r"))
    return old_metadata


if _continue:
    metadata = load_old()
    objs = list(set(cat_specs.keys()) - set(metadata.keys()))
else:
    metadata = {}
    objs = cat_specs.keys()

print(f"We are labeling: {len(objs)} objects!")

for obj in objs:
    pull_videos(obj)
    metadata[obj] = {}
    run = True
    while run:
        p, t, sn = input(f"{obj} is pickable (y/n): "), \
                   input(f"{obj} is of type (o/r/or): "), \
                   input("Sidenote (optional): ")
        if p in ["y", "n"] and t in ["o", "r", "or"]:
            run = False

    metadata[obj]["pickable"] = True if p == "y" else False

    # anything that's not pickable is a receptacle
    if p == "n":
        assert t == "r"

    if t == "o":
        t = "object"
    elif t == "or":
        t = "object_receptacle"
    else:
        t = "receptacle"
    metadata[obj]["type"] = t
    metadata[obj]["sidenote"] = sn

    with open(save_path, 'w') as f:
        json.dump(metadata, f)
