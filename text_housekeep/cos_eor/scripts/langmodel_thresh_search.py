import argparse
from collections import OrderedDict
from multiprocessing import Pool
# from multiprocessing.dummy import Pool
import itertools
import os
import signal
import sys

import numpy as np

cwd = os.getcwd()
pwd = os.path.dirname(cwd)
ppwd = os.path.dirname(pwd)

for dir in [cwd, pwd, ppwd]:
    sys.path.insert(1, dir)

from habitat_baselines.config.default import get_config
from habitat import make_dataset

from cos_eor.policy.rank import RankModule
from cos_eor.dataset.dataset import CosRearrangementDatasetV0, CosRearrangementEpisode
from cos_eor.scripts.orm.utils import preprocess

def create_init_worker(pool_class):
    import multiprocessing
    import multiprocessing.pool
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    def init_worker_dummy():
        pass
    if pool_class == multiprocessing.Pool:
        return init_worker
    elif pool_class == multiprocessing.pool.ThreadPool:
        return init_worker_dummy

def build_metrics_arrays(dims):
    metrics_keys = ["true_positives", "positives", "ground_truth_positives", "counts"]
    metrics = {}
    for k in metrics_keys:
        metrics[k] = np.zeros(dims)
    return metrics

def safe_divide(a, b):
    return np.divide(a, b, np.full_like(a, np.nan, dtype=float), where=b!=0)

def get_recs_and_objs(episode: CosRearrangementEpisode):
    rec_rooms, objs = OrderedDict(), OrderedDict()
    for ri, (rec, sem_class) in enumerate(zip(episode.recs_keys, episode.recs_cats)):
        room = rec.split("-")[0].rsplit("_", 1)[0]
        rec_rooms[ri] = {
            "sem_class": sem_class,
            "room": preprocess(room),
        }
    for oi, sem_class in enumerate(episode.objs_cats):
        objs[oi] = {"sem_class": sem_class}
    return rec_rooms, objs

def measure_dataset(config, score_threshold, aggregate_threshold):
    rank_module = RankModule(config.RL.POLICY.rank)
    dataset: CosRearrangementDatasetV0 = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    metrics = build_metrics_arrays((len(rank_module.data["rooms"]), len(rank_module.data["objects"])))
    for episode in dataset.episodes:
        rec_rooms, objs = get_recs_and_objs(episode)

        rank_module.reset()
        rank_module.rerank(None, rec_rooms, objs, True)

        langmodel_mapping = rank_module.scores > score_threshold
        # Remove agent from receptacles
        correct_mapping = episode.end_matrix[:-1]
        true_positives = (langmodel_mapping == correct_mapping) & (langmodel_mapping == 1)
        ground_truth_positives = np.isclose(correct_mapping, 1)

        for oi, obj in objs.items():
            obj_class = preprocess(obj["sem_class"])
            obj_id = rank_module.key_to_idx["objects"][obj_class]
            # For an obj, only consider rooms with at least 1 correct receptacle
            good_rooms = {preprocess(rec["room"]) for rec in rec_rooms.values()}
            for ri, rec in rec_rooms.items():
                room = rec["room"] 
                if room == "none":
                    continue
                if room not in good_rooms:
                    continue
                room_id = rank_module.key_to_idx["rooms"][room]
                metrics["positives"][room_id, obj_id] += langmodel_mapping[ri, oi]
                metrics["true_positives"][room_id, obj_id] += true_positives[ri, oi]
                metrics["ground_truth_positives"][room_id, obj_id] += ground_truth_positives[ri, oi]
                metrics["counts"] += 1

    if aggregate_threshold:
        for k in metrics.keys():
            metrics[k] = metrics[k].sum(keepdims=True)

    precision = safe_divide(metrics["true_positives"], metrics["positives"])
    recall = safe_divide(metrics["true_positives"], metrics["ground_truth_positives"])
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return (
        np.array((precision, recall, f1)),
        np.array((
            metrics["true_positives"],
            metrics["positives"],
            metrics["ground_truth_positives"],
            metrics["counts"]
        ))
    )

def eval_dataset(config, score_thresholds, mean_score_threshold, aggregate_threshold):
    rank_module = RankModule(config.RL.POLICY.rank)
    dataset: CosRearrangementDatasetV0 = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    metrics = build_metrics_arrays((len(rank_module.data["rooms"]), len(rank_module.data["objects"])))
    for episode in dataset.episodes:
        rec_rooms, objs = get_recs_and_objs(episode)

        rank_module.reset()
        rank_module.rerank(None, rec_rooms, objs, True)

        # Remove agent from receptacles
        correct_mapping = episode.end_matrix[:-1]

        for oi, obj in objs.items():
            obj_class = preprocess(obj["sem_class"])
            obj_id = rank_module.key_to_idx["objects"][obj_class]
            # For an obj, only consider rooms with at least 1 correct receptacle
            good_rooms = {preprocess(rec["room"]) for rec in rec_rooms.values()}
            for ri, rec in rec_rooms.items():
                room = rec["room"] 
                if room == "none":
                    continue
                if room not in good_rooms:
                    continue
                room_id = rank_module.key_to_idx["rooms"][room]

                score = rank_module.scores[ri, oi]
                if aggregate_threshold:
                    score_threshold = score_thresholds
                else:
                    score_threshold = score_thresholds[room_id, obj_id]
                if score_threshold == -1:
                    score_threshold = mean_score_threshold
                langmodel_mapping = score > score_threshold

                metrics["positives"][room_id, obj_id] += langmodel_mapping
                metrics["true_positives"][room_id, obj_id] += (langmodel_mapping == correct_mapping[ri, oi] == True)
                metrics["ground_truth_positives"][room_id, obj_id] += correct_mapping[ri, oi]

    if aggregate_threshold:
        for k in metrics.keys():
            metrics[k] = metrics[k].sum(keepdims=True)

    precision = safe_divide(metrics["true_positives"], metrics["positives"])
    recall = safe_divide(metrics["true_positives"], metrics["ground_truth_positives"])
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return (
        np.array((precision, recall, f1)),
        np.array((
            metrics["true_positives"],
            metrics["positives"],
            metrics["ground_truth_positives"])
        )
    )

def init_search_matrix(num_search_elts, init_vals, init_idx):
    matrix_shape = (num_search_elts,) + init_vals.shape
    matrix = np.zeros(matrix_shape)
    matrix[init_idx] = init_vals
    return matrix

def search_threshold_multi(score_thresholds, config_files, aggregate_threshold, num_jobs):
    configs = (get_config(config_file) for config_file in config_files)
    args = list(itertools.product(configs, score_thresholds, [aggregate_threshold]))
    score_thresh_to_idx = {thresh: idx for idx, thresh in enumerate(score_thresholds)}
    metrics, stats = None, None
    try:
        with Pool(min(num_jobs, len(args)), create_init_worker(Pool)) as pool:
            for task_args, (dataset_metrics, dataset_stats) in zip(args, pool.starmap(measure_dataset, args)):
                _, score_threshold, _ = task_args
                thresh_idx = score_thresh_to_idx[score_threshold]
                if metrics is not None:
                    metrics[thresh_idx] += dataset_metrics
                    stats[thresh_idx] += dataset_stats
                else:
                    metrics = init_search_matrix(len(score_thresholds), dataset_metrics, thresh_idx)
                    stats = init_search_matrix(len(score_thresholds), dataset_stats, thresh_idx)
                    # metrics_shape = (len(score_thresholds),) + dataset_metrics.shape
                    # metrics = np.zeros(metrics_shape)
                    # metrics[thresh_idx] = dataset_metrics
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        return None

    metrics /= len(config_files)
    return metrics, stats

def search_threshold(score_thresholds, config_files):
    # metrics = np.zeros((len(score_thresholds), 3))
    metrics = None
    for config_file in config_files:
        config = get_config(config_file)
        scene = config.BASE_TASK_CONFIG.DATASET.CONTENT_SCENES[0]
        prune_thresholds = False
        print("Starting scene", scene)
        for i, score_threshold in enumerate(score_thresholds):
            print("Starting score threshold", score_threshold)
            dataset_metrics = measure_dataset(config, score_threshold)
            # if np.isnan(dataset_metrics).any():
            #     prune_thresholds = True
            #     break
            # if (~np.isnan(dataset_metrics)).any():
            #     import pdb
            #     pdb.set_trace()
            if metrics is not None:
                metrics[i] += dataset_metrics
            else:
                metrics_shape = (len(score_thresholds),) + dataset_metrics.shape
                metrics = np.zeros(metrics_shape)
                metrics[i] = dataset_metrics
            print(dataset_metrics[-1])
        if prune_thresholds:
            score_thresholds = score_thresholds[:i]
    # for i, score_threshold in enumerate(score_thresholds):
    #     print(score_threshold, metrics[i]/len(config_files))
    metrics /= len(config_files)
    return metrics

def train(config_files, aggregate_threshold, beta, num_jobs):
    # score_thresholds = [np.linspace(10**(-i-1), 10**(-i), 91) for i in range(10)]
    score_thresholds = np.geomspace(1e-10, 1, 11)
    score_thresholds = np.array(score_thresholds).flatten()
    # score_thresholds = np.linspace(1e-2, 1e-1, 91)
    metrics, stats = search_threshold_multi(score_thresholds, config_files, aggregate_threshold, num_jobs)

    # get f1 score
    metrics = metrics[:, -1, ...]

    # create nan masks where metrics are nan for all scores
    nan_idxs = np.isnan(metrics).all(axis=0)

    # get best thresholds for each obj, room pair
    nan_idxs_rep = np.repeat(nan_idxs[np.newaxis, ...], metrics.shape[0], axis=0)
    metrics[nan_idxs_rep] = 0
    best_thresh_idxs = np.nanargmax(metrics, axis=0)
    best_scores = score_thresholds[best_thresh_idxs]
    # use -1 to denote pesky nan metrics
    best_scores[nan_idxs] = -1

    # get stats corresponding to best threshold for each obj, room pair
    best_thresh_idxs_rep = np.repeat(best_thresh_idxs[np.newaxis, ...], stats.shape[1], axis=0)[np.newaxis, ...]
    best_stats = np.take_along_axis(stats, best_thresh_idxs_rep, axis=0).squeeze(axis=0)
    nan_idxs_rep = np.repeat(nan_idxs[np.newaxis, ...], best_stats.shape[0], axis=0)
    best_stats[nan_idxs_rep] = np.nan

    # aggregate f1 score
    true_positives, positives, gt_positives, counts = best_stats[0], best_stats[1], best_stats[2], best_stats[3]
    precision = np.nansum(true_positives) / np.nansum(positives)
    recall = np.nansum(true_positives) / np.nansum(gt_positives)
    f1 = (2 * precision * recall) / (precision + recall)

    # calculate mean weighted score threshold
    counts[nan_idxs] = 0
    mean_score = (best_scores * counts).sum()/counts.sum()

    print(best_scores)
    print(np.unique(best_scores, return_counts=True))
    print(precision, recall, f1)

    return best_scores, mean_score

def simple_train(config_files, aggregate_threshold, beta, num_jobs):
    score_thresholds = [np.linspace(10**(-i-1), 10**(-i), 91) for i in [3, 2, 1, 0]]
    score_thresholds = np.array(score_thresholds).flatten()
    metrics, stats = search_threshold_multi(score_thresholds, config_files, aggregate_threshold, num_jobs)

    # import pdb
    # pdb.set_trace()

    metrics = metrics[:, -1, ...].squeeze(axis=(-1, -2))
    stats = stats.squeeze(axis=(-1, -2))

    true_positives, positives, gt_positives = stats[:, 0], stats[:, 1], stats[:, 2]
    precision = true_positives/positives
    recall = true_positives/gt_positives
    beta_sq = beta**2
    fb = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)

    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F_BETA:", fb)

    import pdb
    pdb.set_trace()

    best_idx = np.nanargmax(fb)
    best_score = np.array([score_thresholds[best_idx]])
    print("BEST SCORE:", best_score)
    return best_score, best_score

def eval(config_files, best_scores, mean_score, aggregate_threshold, beta):
    if aggregate_threshold:
        best_scores = best_scores.squeeze().item()

    _, stats = eval_dataset(get_config(config_files[0]), best_scores, mean_score, aggregate_threshold)

    true_positives, positives, gt_positives = stats[0], stats[1], stats[2]
    precision = true_positives.sum() / positives.sum()
    recall = true_positives.sum() / gt_positives.sum()
    f1 = (2 * precision * recall) / (precision + recall)
    beta_sq = beta**2
    fb = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)

    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1:", f1)
    print("F_BETA:", fb)

def main(config_files, eval_config_files, aggregate_threshold, beta, scores_dump_file, scores_checkpoint_file, num_jobs):
    if scores_checkpoint_file is None:
        if aggregate_threshold:
            train_fn = simple_train
        else:
            train_fn = train
        best_scores, mean_score = train_fn(config_files, aggregate_threshold, beta, num_jobs)
        if scores_dump_file is not None:
            np.save(scores_dump_file, {"scores": best_scores, "mean_score": mean_score})
    else:
        scores_data = np.load(scores_checkpoint_file, allow_pickle=True).item()
        best_scores, mean_score = scores_data["scores"], scores_data["mean_score"]
    eval(eval_config_files, best_scores, mean_score, aggregate_threshold, beta)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-files",
    nargs="+",
    required=True,
    help="List of config files to measure score thresholds for"
)

parser.add_argument(
    "--eval-config-files",
    nargs="+",
    required=True,
    help="List of config files to evaluate score threshold(s) on"
)

parser.add_argument(
    "--beta",
    type=float,
    default=1,
    help="Beta to use in F_beta score. Higher means recall is more important"
)

parser.add_argument(
    "--aggregate-threshold",
    action="store_true",
    help="Set flag to calculate a common threshold for all (obj, room) pairs instead of separate thresholds"
)

parser.add_argument(
    "--scores-dump-file",
    help="File to dump tuned score thresholds to"
)

parser.add_argument(
    "--scores-checkpoint-file",
    help="File to read tuned scores from"
)

parser.add_argument(
    "--num-jobs",
    type=int,
    help="Number of jobs to spawn to search for score threshold"
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
