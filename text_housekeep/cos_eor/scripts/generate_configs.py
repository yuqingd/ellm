import argparse
from pathlib import Path
from string import Template

scenes = [
    ("Rs_int", "train"),
    ("Beechwood_0_int", "train"),
    ("Beechwood_1_int", "test"),
    ("Benevolence_1_int", "test"),
    ("Benevolence_2_int", "train"),
    ("Ihlen_0_int", "test"),
    ("Ihlen_1_int", "val"),
    ("Merom_0_int", "test"),
    ("Merom_1_int", "val"),
    ("Pomaria_0_int", "train"),
    ("Pomaria_1_int", "train"),
    ("Pomaria_2_int", "train"),
    ("Wainscott_0_int", "train"),
    ("Wainscott_1_int", "train"),
]

# rank_types = ["Oracle", "LangModel"]
# explore_modes = ["phasic", "oracle"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-file",
    type=Path,
    required=True,
    help="path to config template file"
)

parser.add_argument(
    "--out-dir",
    type=Path,
    required=True,
    help="output directory"
)

parser.add_argument(
    "--explore-type",
    type=str,
    default="oracle",
    help="explore strategy"
)

parser.add_argument(
    "--rank-type",
    type=str,
    default="Oracle",
    help="ranking strategy"
)

def main():
    args = parser.parse_args()
    with open(args.config_file) as f:
        config_template = Template(f.read())
    # file_name_base = args.config_file.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Writing to directory", args.out_dir)
    for scene, split in scenes:
        config = config_template.substitute(
            scene=scene,
            split=split,
            rank_type=args.rank_type,
            explore_type=args.explore_type
        )
        config_file_name = f"{scene.lower()}.yaml"
        with open(args.out_dir/config_file_name, "w") as f:
            f.write(config)
        print("Generated file", config_file_name)
    print("Done!")

if __name__ == "__main__":
    main()
