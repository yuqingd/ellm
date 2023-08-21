import yaml
import os
import csv


# generic code to load all receptacles and objects
global_mapping_path = "cos_eor/utils/global_mapping_v3.yaml"
global_mapping = yaml.load(open(global_mapping_path, "r"), Loader=yaml.BaseLoader)
igib_objs = list(set([item[0] for item in global_mapping["objects"] if item[-1] == 'igib']))
ycb_objs = list(set([item[0] for item in global_mapping["objects"] if item[-1] == 'ycb_or_ab_adjusted']))
igib_rooms = []
for rooms in global_mapping["object_room_map"].values():
    igib_rooms.extend(rooms)
igib_rooms = list(set(igib_rooms))
igib_recs = list(set([item[0] for item in global_mapping["receptacles"] if item[-1] == 'igib']))
ycb_recs = list(set([item[0] for item in global_mapping["receptacles"] if item[-1] == 'ycb_or_ab_adjusted']))
igib_rooms = sorted(igib_rooms)
igib_recs = sorted(igib_recs)
ycb_recs = sorted(ycb_recs)

global_mapping_csv_path = "cos_eor/utils/global_mapping_v3.csv"
csv_first_row = ["Rooms", "Receptacles/Objects"] + igib_objs + ycb_objs
recs = igib_recs + ycb_recs
with open(global_mapping_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(csv_first_row)

    # iterate over all rooms
    for room in igib_rooms:
        # objs/recs that were found in this room
        room_recs = [item for item in global_mapping["object_room_map"] \
                     if (room in global_mapping["object_room_map"][item]) and (item in recs)]
        for rec in room_recs:
            row = [room, rec] + [0] * len(igib_objs + ycb_objs)
            if rec in global_mapping['mapping_igib']:
                # paired objects in default mapping
                for obj_item in global_mapping["mapping_igib"][rec]:
                    obj = "_".join(obj_item[0].split("_")[:-2])
                    row_idx = csv_first_row.index(obj)
                    row[row_idx] = 1
            writer.writerow(row)

print(f"Dumped: {global_mapping_csv_path}")

