import json
import os
from collections import Counter

from tqdm import tqdm

dataset_path = "/coc/pskynet1/ykant3/amazon-berkeley/"
metadata_dir = os.path.join(dataset_path, "listings/metadata")

object_types = []
for file in tqdm(os.listdir(metadata_dir), "processing"):
    if file.endswith(".json.gz"):
        continue
    lines = open(os.path.join(metadata_dir, file), "r").readlines()
    for line in lines:
        obj_data = json.loads(line)
        if "3dmodel_id" not in obj_data:
            continue
        if len(obj_data["product_type"]) > 1:
            print(obj_data["product_type"])
        obj_type = obj_data["product_type"][0]['value']
        object_types.append(obj_type)

count_types = list(Counter(object_types).items())
print_lines = [str(item) + "\n" for item in count_types]

dump_path = "amazon_berkeley_types.txt"
with open(dump_path, "w") as file:
    file.writelines(print_lines)
    file.writelines([f"Total: {len(object_types)} \n"])
