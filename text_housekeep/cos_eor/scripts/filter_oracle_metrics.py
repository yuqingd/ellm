import argparse
from pathlib import Path

import pandas as pd

def main(ref_metrics_files, metrics_files, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    ref_metrics = (pd.read_csv(file) for file in ref_metrics_files)
    ref_metrics = pd.concat(ref_metrics)
    header = ",".join(ref_metrics.columns.values) + "\n"
    ref_episodes = set(ref_metrics["episode_id"])

    for metrics_file in metrics_files:
        filtered_metrics = [header]
        with open(metrics_file) as f:
            for episode_metrics in f:
                episode_metrics = episode_metrics
                episode_id = episode_metrics.rsplit(",", 1)[1].strip()
                if episode_id not in ref_episodes:
                    continue
                filtered_metrics.append(episode_metrics)
        assert len(filtered_metrics) == 400
        with open(out_dir/metrics_file.name, "w") as f:
            f.writelines(filtered_metrics)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--metrics-files",
    nargs="+",
    type=Path,
    required=True
)

parser.add_argument(
    "--ref-metrics-files",
    nargs="+",
    type=Path,
    required=True
)

parser.add_argument(
    "--out-dir",
    type=Path,
    required=True
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
