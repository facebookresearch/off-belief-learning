# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import pprint
import json
from collections import defaultdict
import numpy as np

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
import common_utils
import utils


def find_best_models(
    root, num_game, seed, eval_num_game, eval_seed, num_player, overwrite
):
    runs = [os.path.join(root, f) for f in os.listdir(root)]
    runs = [f for f in runs if os.path.isdir(f)]
    for run in runs:
        output = os.path.join(run, "best_model.json")
        if not overwrite and os.path.exists(output):
            print(f"skip {run}")
            continue
        models = common_utils.get_all_files(run, "pthw")
        models = [m for m in models if "epoch" not in m]

        best_model = None
        best_score = -1
        for m in models:
            _, _, _, scores, _ = evaluate_saved_model(
                [m] * num_player, num_game, seed, 0
            )
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_model = m
                best_score = mean_score

        score, sem, perfect, _, _ = evaluate_saved_model(
            [best_model] * num_player, eval_num_game, eval_seed, 0
        )
        result = {
            "best_model": best_model,
            "score": score,
            "sem": sem,
            "perfect": perfect,
        }
        with open(output, "w") as f:
            json.dump(result, f)


def aggregate_result(root):
    runs = [os.path.join(root, f) for f in os.listdir(root)]
    runs = [f for f in runs if os.path.isdir(f)]
    run_results = defaultdict(list)
    for run in runs:
        output = os.path.join(run, "best_model.json")
        if not os.path.exists(output):
            continue
        result = json.load(open(output, "r"))
        run_name = run.rsplit("_", 1)[0].split("/")[-1]
        run_results[run_name].append(result)

    agg_results = {}
    for run, results in run_results.items():
        scores = [r["score"] for r in results]
        perfects = [r["perfect"] for r in results]
        models = [r["best_model"] for r in results]

        mean_score = np.mean(scores)
        sem = np.std(scores) / np.sqrt(len(scores) - 1)
        mean_perfect = np.mean(perfects)
        max_score = np.max(scores)
        max_idx = np.argmax(scores)
        max_perfect = perfects[max_idx]
        best_model = models[max_idx]
        agg_results[run] = {
            "mean_score": mean_score,
            "sem": sem,
            "mean_perfect": mean_perfect,
            "max": max_score,
            "max_perfect": max_perfect,
            "best_model": best_model,
        }
    return agg_results
