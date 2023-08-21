from typing import List

import argparse
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import yaml

import pandas as pd

@dataclass
class Split:
    name: str
    num_episodes: List[int]
    num_good_episodes: List[int]
    scenes: List[str]

# splits = [
#     Split("val", [150, 150], [100, 100], ["Ihlen_1_int", "Merom_1_int"]),
#     Split("test", [300, 300], [200, 200], ["Beechwood_1_int", "Benevolence_1_int", "Ihlen_0_int", "Merom_0_int"]),
# ]

# splits = [
#     Split("val", [150, 150], [100, 100], ["Ihlen_1_int", "Merom_1_int"]),
#     Split("test", [300, 300], [200, 200], ["Beechwood_1_int", "Benevolence_1_int", "Ihlen_0_int"]),
# ]

splits = [
    Split("train", [2000], [1000], ["Rs_int", "Beechwood_0_int", "Benevolence_2_int", "Pomaria_0_int", "Pomaria_1_int", "Pomaria_2_int", "Wainscott_0_int", "Wainscott_1_int"])
]

# splits = [
#     Split("test", [300, 300, 100, 100], [179, 179, 21, 21], ["Merom_0_int"]),
# ]

def prune_episodes(episodes, scene, metrics, num_good_episodes):
    good_episodes = []
    for episode in episodes:
        episode_full_id = f"{scene}_{episode['episode_id']}"
        try:
            episode_stats = metrics.loc[episode_full_id]
        except KeyError:
            continue
        if not math.isclose(episode_stats["episode_success"], 1):
            continue
        good_episodes.append(episode)
        if len(good_episodes) == num_good_episodes:
            break
    assert len(good_episodes) == num_good_episodes
    return good_episodes

def merom_0_int_hack(episodes):
    rearranged_episodes = []
    i = 0
    for ep in episodes[:179]:
        ep["episode_id"] = i
        rearranged_episodes.append(ep)
        i += 1
    for ep in episodes[179+179:179+179+21]:
        ep["episode_id"] = i
        rearranged_episodes.append(ep)
        i += 1
    for ep in episodes[179:179+179]:
        ep["episode_id"] = i
        rearranged_episodes.append(ep)
        i += 1
    for ep in episodes[179+179+21:]:
        ep["episode_id"] = i
        rearranged_episodes.append(ep)
        i += 1

    assert len(rearranged_episodes) == 400
    import pdb; pdb.set_trace()
    splits_info = yaml.safe_load(open("cos_eor/scripts/orm/amt_data/splits.yaml", "r"))
    seen_cats = set(splits_info["objects"]["train"])
    for ep in rearranged_episodes[:200]:
        assert (
            len(ep["recs_cats"]) ==
            len(ep["recs_keys"]) ==
            len(ep["start_matrix"]) ==
            len(ep["end_matrix"]) ==
            len(ep["recs_pos"]) ==
            len(ep["nav_recs_pos"]) ==
            len(ep["recs_packers"])
        )
        for cat in ep["objs_cats"]:
            assert cat in seen_cats
    for ep in rearranged_episodes[200:]:
        assert (
            len(ep["recs_cats"]) ==
            len(ep["recs_keys"]) ==
            len(ep["start_matrix"]) ==
            len(ep["end_matrix"]) ==
            len(ep["recs_pos"]) ==
            len(ep["nav_recs_pos"]) ==
            len(ep["recs_packers"])
        )
        for cat in ep["objs_cats"]:
            assert cat not in seen_cats

    return rearranged_episodes

def main(metrics_path, episodes_path, out_dir: Path):
    for split in splits:
        for scene in split.scenes:
            episodes_file = episodes_path/split.name/f"{scene}.json.gz"
            with gzip.open(episodes_file, "rt") as file:
                episodes = json.load(file)

            metrics_file = metrics_path/f"metrics-{scene.lower()}.csv"
            metrics = pd.read_csv(metrics_file, index_col="episode_id").astype(float)

            good_episodes = []
            num_eps_covered = 0
            for num_eps, num_good_eps in zip(split.num_episodes, split.num_good_episodes):
                good_episodes.extend(
                    prune_episodes(episodes["episodes"][num_eps_covered : num_eps_covered + num_eps], scene, metrics, num_good_eps)
                )
                num_eps_covered += num_eps

            # good_episodes = merom_0_int_hack(good_episodes)

            episode_ids = {ep["episode_id"] for ep in good_episodes}
            assert len(episode_ids) == sum(split.num_good_episodes)

            for id, episode in enumerate(good_episodes):
                episode["episode_id"] = id

            new_episodes_file = out_dir/split.name/f"{scene}.json.gz"
            new_episodes_file.parent.mkdir(exist_ok=True, parents=True)
            with gzip.open(new_episodes_file, "wt") as file:
                json.dump({"episodes": good_episodes}, file)

            print("Pruned", episodes_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--episodes-path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(**vars(args))
