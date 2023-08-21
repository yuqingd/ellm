import os
import os.path as osp
import sys
import glob
from tqdm import tqdm
import json

base_obj_dir = 'data/objects'
convex_obj_dir = 'data/objects/configs_convex'

def gen_minimal_phys_config_files(path_to_dir, args):
    """
    Scans a directory for files of a desired type and creates minimal <name>.phys_properties.json files for each.

    :param path_to_dir: The directory to scan and create files in
    """
    END_CONFIG = '.object_config.json'
    if not os.path.exists(path_to_dir):
        print(
            "gen_minimal_phys_prop_files error! Directory does not exist: "
            + str(path_to_dir)
        )
        return

    files = glob.glob(path_to_dir + '/**/*.obj', recursive=True)
    for f in tqdm(files):
        obj_dir = '/'.join(f.split('/')[:-1])
        obj_name = f.split('/')[-1].split('.')[0]

        sep = ",\n    "

        data_dict = {
                "friction_coefficient": 1.0,
                "mass": 30.0,
                }
        if 'cracker' in obj_name:
            data_dict['friction_coefficient'] = 1.0
            data_dict['mass'] = 30.0
        if args.bb:
            data_dict["use_bounding_box_for_collision"] = True
        if args.cv_decomp:
            data_dict["join_collision_meshes"] = False

        base_name = f.split('/')[-3]
        if args.cv_decomp:
            obj_num = base_name.split('_')[0]
            matches = [cv_f for cv_f in os.listdir(convex_obj_dir) if cv_f.startswith(obj_num)]
            if len(matches) == 0:
                print('Could not find cv match for ', base_name, 'skipping')
                continue
            assert len(matches) == 1, f"matches are {matches} for {obj_num}"
            coll_asset = osp.join('configs_convex', matches[0])
            full_coll_name = osp.join('data/objects', coll_asset)
            assert osp.exists(full_coll_name), f"CV {full_coll_name} does not exist"
            data_dict["collision_asset"] = coll_asset

        config_filename = osp.join(base_obj_dir, base_name + END_CONFIG)
        obj_dir_no_base = '/'.join(obj_dir.split('/')[2:])
        render_asset = osp.join(obj_dir_no_base, obj_name + '.obj')

        data_dict['render_asset'] = render_asset
        data_dict['requires_lighting'] = True

        with open(config_filename, 'w') as f:
            json.dump(data_dict, f, indent=2)

if __name__== "__main__":
    """
    - Argument for cv decomp
    - arg for BB collision mesh.
    """
    #parse each provided directory
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bb', action='store_true', default=False)
    parser.add_argument('--cv-decomp', action='store_true', default=False)
    args = parser.parse_args()
    gen_minimal_phys_config_files('data/objects/ycb', args)
