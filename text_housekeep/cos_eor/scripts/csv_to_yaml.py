import glob
import pandas as pd
import yaml
import os
import csv


global_mapping_path = "global_mapping.yaml"
csv_folder = "cos_eor/utils/csvs"
csv_files = glob.glob(f"{csv_folder}/**", recursive=True)
csv_files = [file for file in csv_files if file.endswith(".csv")]
csvs = [(os.path.basename(cf), pd.read_csv(cf)) for cf in csv_files if "kyash" not in cf]

# prelimnary analysis
objs = csvs[0][1].columns[2:]
rooms = csvs[0][1]["Rooms"]
recs = csvs[0][1]["Receptacles/Objects"]
no_recs_objs = []
out_file = open(f"{csv_folder}/stats.txt", "w")

import pdb
pdb.set_trace()

for obj in objs:
    obj_stat = []
    for ann, csv in csvs:
        obj_rooms = rooms[csv[obj]]
        obj_recs = recs[csv[obj]]
        obj_room_recs = [f"{rec} in {room}" for room, rec in zip(obj_rooms, obj_recs)]
        obj_stat.append((ann, obj, obj_room_recs))

    # print(f"Object: {obj}")
    common = set(obj_stat[0][-1])
    for stat in obj_stat:
        common = common.intersection(set(stat[-1]))

    if len(common):
        print(f"All agree {obj} can go on/at: {common}", file=out_file)
    else:
        no_recs_objs.append(obj)

    # for stat in obj_stat:
    #     ann_rr = set(stat[-1]) - common
    #     if len(ann_rr):
    #         print(f"{stat[0]} says {obj} can go at/on: {list(ann_rr)}")
    print("\n", file=out_file)
print(f"No overlap for recs for: {no_recs_objs}", file=out_file)

import pdb
pdb.set_trace()
