import argparse
import math

import pandas as pd

parser = argparse.ArgumentParser()

final_metrics = [
    "episode_success",
    "object_success",
    "avg_soft_score",
    "rearrange_quality",
    "map_coverage",
    "misplaced_objects_start",
    "misplaced_objects_found",
    "steps",
    "pick_place_efficiency",
    "success_pick_place",
    "success_look_at",
    "path_efficiency",
]

main_metrics = [
    "episode_success",
    "object_success",
    "avg_soft_score",
    "pick_place_efficiency",
    "misplaced_objects_found",
]

parser.add_argument(
    "--metrics-files",
    nargs="+",
    required=True,
    help="Path to metrics files"
)

parser.add_argument(
    "--subsplit",
    default="all",
    choices=["all", "seen", "unseen"]
)

parser.add_argument(
    "--out-file",
    help="Path to write aggregated metrics file to"
)

def pretty_print_metrics(mean, std, count):
    print("MEAN")
    print(mean[main_metrics])
    print("SEM")
    print(std[main_metrics]/math.sqrt(count))

def get_split(metrics, split):
    if split == "all":
        return metrics

    metrics["id"] = metrics.index.str.rsplit(pat="_", n=1).str[1].astype(int)
    if metrics["id"].max() > len(metrics):
        split_id = math.floor(len(metrics) * 1.5)/2
    else:
        split_id = len(metrics)/2

    if split == "seen":
        return metrics[metrics["id"] < split_id].drop(columns="id")
    elif split == "unseen":
        return metrics[metrics["id"] >= split_id].drop(columns="id")
    else:
        raise ValueError

def main(args):
    metrics = [pd.read_csv(file, index_col="episode_id") for file in args.metrics_files]
    metrics = [get_split(m, args.subsplit) for m in metrics]
    metrics = pd.concat(metrics)
    metrics = metrics.astype(float)
    num_rows = len(metrics)
    num_objs_touch = metrics["misplaced_objects_start"] + metrics["placed_objects_touched"]
    metrics["object_success"] = (metrics["object_success_placed"] + metrics["object_success_misplaced"]) / num_objs_touch
    metrics["avg_soft_score"] = (metrics["soft_score_placed"] + metrics["soft_score_misplaced"]) / num_objs_touch
    metrics["rearrange_quality"] = (metrics["rearrange_quality_placed"] + metrics["rearrange_quality_misplaced"]) / num_objs_touch
    metrics["misplaced_objects_found"] = metrics["misplaced_objects_found"] / metrics["misplaced_objects_start"]
    # metrics["pick_place_energy"] = metrics["total_pick_place"] / (2 * metrics["misplaced_objects_start"])
    metrics["success_pick_place"] = metrics["success_pick_place"] / metrics["total_pick_place"]
    metrics["success_look_at"] = metrics["success_look_at"] / (metrics["success_look_at"] + metrics["fail_look_at"])
    mean_metrics = metrics[final_metrics].mean()
    std_metrics = metrics[final_metrics].std()
    print("Number of episodes:", num_rows)
    # print("MEAN METRICS:")
    # print(mean_metrics)
    # print("STD DEV METRICS:")
    # print(std_metrics)
    if args.out_file is None:
        pretty_print_metrics(mean_metrics, std_metrics, num_rows)
    else:
        metrics = pd.concat((mean_metrics.reset_index().T, (std_metrics/math.sqrt(num_rows)).reset_index().T))
        metrics = pd.concat((metrics.iloc[:2], metrics.iloc[3:]))
        metrics.to_csv(args.out_file, header=False, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
