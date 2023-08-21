from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

@dataclass
class Experiment:
    id: str
    num: int
    rank_module: str
    explore_module: str

full_table_line_tmpl = (
    "& \\scriptsize "
    "\\texttt{{{num}}} & "
    "\\texttt{{{rank_module}}} & "
    "\\texttt{{{explore_module}}} & "
    "{episode_success:.2f} \\confint{{{episode_success_err:.2f}}} & "
    "{object_success:.2f} \\confint{{{object_success_err:.2f}}} & "
    "{avg_soft_score:.2f} \\confint{{{avg_soft_score_err:.2f}}} & "
    "{rearrange_quality:.2f} \\confint{{{rearrange_quality_err:.2f}}} & "
    "{map_coverage:.0f} \\confint{{{map_coverage_err:.0f}}} & "
    "{misplaced_objects_found:.2f} \\confint{{{misplaced_objects_found_err:.2f}}} & "
    "{pick_place_efficiency:.2f} \\confint{{{pick_place_efficiency_err:.2f}}} \\\\ "
)

simple_table_line_tmpl = (
    "\\scriptsize "
    "\\texttt{{{num}}} & "
    "\\texttt{{{explore_module}}} & "
    "{object_success:.2f} \\confint{{{object_success_err:.2f}}} & "
    "{map_coverage:.0f} \\confint{{{map_coverage_err:.0f}}} & "
    "{misplaced_objects_found:.2f} \\confint{{{misplaced_objects_found_err:.2f}}} & "
    "{pick_place_efficiency:.2f} \\confint{{{pick_place_efficiency_err:.2f}}} \\\\ "
)

rearrange_table_line_tmpl = (
    "\\scriptsize "
    "\\texttt{{{num}}} & "
    "\\texttt{{{explore_module}}} & "
    "{object_success:.2f} \\confint{{{object_success_err:.2f}}} & "
    "{pick_place_efficiency:.2f} \\confint{{{pick_place_efficiency_err:.2f}}} \\\\ "
)

nan_pattern = re.compile(r"& nan \\confint{nan}")
def create_line(line_template, params):
    line = line_template.format_map(params)
    line = nan_pattern.sub("& --", line)
    return line

def create_table(experiments, metrics_path, subsplits, line_template):
    for subsplit in subsplits:
        print("Split", subsplit)
        for experiment in experiments:
            metrics_file = metrics_path.format(experiment_id=experiment.id, subsplit=subsplit)
            metrics = pd.read_csv(metrics_file)

            mean_metrics = metrics.to_dict("records")[0]
            metrics = metrics.add_suffix("_err")
            error_metrics = metrics.to_dict("records")[1]
            metrics = {**mean_metrics, **error_metrics}

            if experiment.explore_module == "OR":
                metrics["misplaced_objects_found"] = 1
                metrics["misplaced_objects_found_err"] = 0
                metrics["map_coverage"] = float("nan")
                metrics["map_coverage_err"] = float("nan")
                if experiment.rank_module == "OR":
                    metrics["episode_success"] = 1
                    metrics["episode_success_err"] = 0
                    metrics["object_success"] = 1
                    metrics["object_success_err"] = 0
                    # metrics["pick_place_efficiency"] = 1
                    # metrics["pick_place_efficiency_err"] = 0
            if experiment.rank_module == "OR":
                metrics["pick_place_efficiency"] = 1
                metrics["pick_place_efficiency_err"] = 0
            # print(metrics)
            metrics["num"] = experiment.num
            metrics["rank_module"] = experiment.rank_module
            metrics["explore_module"] = experiment.explore_module
            print(create_line(line_template, metrics))

def main():
    logs_dir = Path("logs/")

    print("Main table:")
    experiments = [
        Experiment("0011", 1, "OR", "OR"),
        Experiment("0012", 2, "OR", "FTR"),
        Experiment("0010", 3, "LM", "OR"),
        Experiment("0001", 4, "LM", "FTR"),
    ]
    metrics_path = f"{logs_dir}/{{experiment_id}}/metrics-agg-test-{{subsplit}}.csv"
    create_table(experiments, metrics_path, ["seen", "unseen"], full_table_line_tmpl)

    print("Num explore steps:")
    experiments = [
        Experiment("0000", 1, "", "N=0"),
        Experiment("0001", 2, "", "N=16"),
        Experiment("0002", 3, "", "N=32"),
        Experiment("0003", 4, "", "N=64"),
        Experiment("0004", 5, "", "N=128"),
        Experiment("0005", 6, "", "N=256"),
        Experiment("0006", 7, "", "N=512"),
    ]
    metrics_path = f"{logs_dir}/{{experiment_id}}/metrics-agg-val-{{subsplit}}.csv"
    create_table(experiments, metrics_path, ["all"], simple_table_line_tmpl)

    print("Rearrangement orders:")
    experiments = [
        Experiment("0001", 1, "", "DIS"),
        Experiment("0007", 2, "", "SCG"),
        Experiment("0008", 3, "", "A-O"),
        Experiment("0009", 4, "", "O-R"),
    ]
    metrics_path = f"{logs_dir}/{{experiment_id}}/metrics-agg-val-{{subsplit}}.csv"
    create_table(experiments, metrics_path, ["all"], rearrange_table_line_tmpl)

    print("Exploration algos:")
    experiments = [
        Experiment("0013", 1, "", "RND"),
        Experiment("0014", 2, "", "FWR"),
        Experiment("0005", 3, "", "FRT"),
    ]
    metrics_path = f"{logs_dir}/{{experiment_id}}/metrics-agg-val-{{subsplit}}.csv"
    create_table(experiments, metrics_path, ["all"], simple_table_line_tmpl)

if __name__ == "__main__":
    main()
