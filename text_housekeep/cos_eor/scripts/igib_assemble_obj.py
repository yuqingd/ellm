import json
import xml.etree.ElementTree as ET
import os
import os.path as osp
import trimesh
import glob
import shutil
from tqdm import tqdm

IGIB_ASSETS = "data/scene_datasets/igibson/assets/"
IGIB_ASSETS_ASSEMBLE = "data/scene_datasets/igibson/assets_assemble/"
interactive_scenes = sorted(glob.glob(IGIB_ASSETS + "/*"))
URDF_OBJ_CACHE = "data/urdf_obj_cache"


def str_to_tup(str):
    return  eval(str.replace(" ", ","))


def parse_visual_link(link):
    origin = link.findall("origin")
    assert len(origin) == 1 or len(origin) == 0

    if len(origin):
        xyz = str_to_tup(origin[0].attrib["xyz"])
        rpy = str_to_tup(origin[0].attrib["rpy"])
    else:
        xyz = [0.0] * 3
        rpy = [0.0] * 3
    geometry = link.findall("geometry")
    mesh = geometry[0].findall("mesh")
    mesh_file = mesh[0].attrib["filename"]
    scale = str_to_tup(mesh[0].attrib["scale"])
    assert len(geometry) == 1
    return xyz, rpy, scale, mesh_file


def urdf_to_obj(urdf, dump_dir, return_exist=False):
    obj_name = os.path.split(urdf)[-1].split(".urdf")[0]
    obj_dir = os.path.join(dump_dir, obj_name)
    config_path = os.path.join(obj_dir, f"{obj_name}.object_config.json")

    if os.path.exists(config_path) and return_exist:
        return config_path, True

    os.makedirs(obj_dir, exist_ok=True)
    print(f"URDF: {obj_name} | NEW DIR: {obj_dir}")
    robot = ET.parse(urdf).getroot()
    robot_links = robot.findall("link")
    robot_vis_links = []
    for link in robot_links:
        robot_vis_links.extend(link.findall("visual"))

    obj_paths = []
    obj_info = []
    for vis_link in robot_vis_links:
        xyz, rpy, scale, mesh_file = parse_visual_link(vis_link)
        obj_info.append((xyz, rpy, scale, mesh_file))
        assert os.path.exists(mesh_file)
        obj_paths.append(mesh_file)

    visual_dir = os.path.dirname(obj_paths[0])
    obj_paths_2 = [p for p in glob.glob(visual_dir + "/**")
                   if p.endswith(".obj")]
    mtl_file = os.path.join(visual_dir, "default.mtl")
    try:
        # assert all paths are covered
        assert set(obj_paths_2) == set(obj_paths)
    except:
        print(f"Mismatched links: {obj_name}")
    assert os.path.exists(mtl_file)

    # assemble and export obj file
    obj_mesh = []
    for oi in obj_info:
        xyz, rpy, scale, mesh_file = oi
        mesh = trimesh.load(
            mesh_file,
            resolver=trimesh.visual.resolvers.FilePathResolver(
                osp.dirname(mesh_file)
            )
        )
        mesh.apply_scale(scale)
        obj_mesh.append(mesh)

    obj_mesh = trimesh.util.concatenate(obj_mesh)
    obj_mesh.export(os.path.join(obj_dir, "model.obj"))

    # reading and copying useful parts of default.mtl
    mtl_lines = [l.strip() for l in open(mtl_file, "r").readlines()][1:9]

    if 'window' not in obj_dir and 'door' not in obj_dir:
        mtl_lines[2] = "Kd 0.300000 0.300000 0.300000"

    # creating new material file (to be used by the newly assembled objs)
    mtl_file = os.path.join(obj_dir, "material_0.mtl")


    # adding custom lines to point to right color texture file
    mtl_lines.insert(0, "newmtl material_0")
    mtl_lines.append("map_Kd material_0.png")

    # pasting into new file
    with open(mtl_file, "w") as f:
        f.writelines("\n".join(mtl_lines) + "\n")

    # add object_config.json file for loading
    config_path = os.path.join(obj_dir, f"{obj_name}.object_config.json")
    data = {
        "render_asset": "model.obj",
        "use_bounding_box_for_collision": False,
        "requires_lighting": True,
        "margin": 0
    }
    with open(config_path, "w") as file:
        file.write(json.dumps(data))
    return config_path, False


if __name__ == "__main__":
    os.makedirs(IGIB_ASSETS_ASSEMBLE, exist_ok=True)
    for scene in tqdm(interactive_scenes, desc="Scenes Processed"):
        # if "Rs_int" not in scene or "coll" in scene:
        #     continue
        sc_dir = os.path.join(IGIB_ASSETS_ASSEMBLE, os.path.split(scene)[-1])
        os.makedirs(sc_dir, exist_ok=True)
        urdfs = [p for p in glob.glob(scene + "/**") if p.endswith(".urdf")]
        for urdf in tqdm(urdfs, desc="URDFs"):
            urdf_to_obj(urdf, sc_dir)
