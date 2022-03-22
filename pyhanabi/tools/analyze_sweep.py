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
import os
import argparse
import pprint
from math import isnan
import numpy as np
from tabulate import tabulate

from parse_log import *


def analyze_sweep(
    root, max_epoch, min_epoch, include, exclude, new_log, short, full_name
):
    logs = parse_from_root(root, max_epoch, min_epoch, include, exclude, new_log)

    l = list(logs.items())
    l = sorted(l, key=lambda x: x[0])
    summary = []

    def get_score(d):
        if not isnan(d["final_score"]):
            return d["final_score"]
        if not isnan(d["final_xent_pred"]):
            return d["final_xent_pred"]
        if not isnan(d["final_loss"]):
            return d["final_loss"]

    for ll in l:
        entry = [
            shorten_name(ll[0]) if not full_name else ll[0],
            ll[1]["epoch"],
            ll[1]["act_rate"],
            ll[1]["train_rate"],
            ll[1]["buffer_rate"],
            get_score(ll[1]),
            ll[1]["final_perfect"],
        ]
        # print(ll[1].keys())
        if "selfplay" in ll[1]:
            entry.append(ll[1]["selfplay"])
        summary.append(entry)

    header = ["name", "epoch", "act", "train", "buffer", "score", "perfect"]
    if "selfplay" in l[0][1]:
        header.append("selfplay")

    print(tabulate(summary, headers=header))

    if short:
        return

    print("\n=====avg, stderr======")
    avg_scores = {k: v["scores"] for k, v in logs.items()}
    avg_scores = average_across_seed(avg_scores)

    l = list(avg_scores.items())
    l = sorted(l, key=lambda x: x[0])
    summary = [
        (
            shorten_name(ll[0]),
            len(ll[1][0]),
            np.mean(ll[1][0][-10:]),
        )
        for ll in l
    ]
    header = ["name", "epoch", "score"]
    print(tabulate(summary, headers=header))

    print("\n======best over seed======")
    scores = {k: v["scores"] for k, v in logs.items()}
    best_scores = max_across_seed(scores)
    for k, (s, loc) in best_scores.items():
        print("%s: %.2f" % (k, s))
        print("\tat: ", loc)
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--max_epoch", type=int, default=0)
    parser.add_argument("--min_epoch", type=int, default=0)
    parser.add_argument("--include", type=str, default="", nargs="+")
    parser.add_argument("--exclude", type=str, default="", nargs="+")
    parser.add_argument("--new_log", type=int, default=1)
    parser.add_argument("--short", default=False, action="store_true")
    parser.add_argument("--full_name", default=False, action="store_true")
    args = parser.parse_args()

    analyze_sweep(
        args.root,
        args.max_epoch,
        args.min_epoch,
        args.include,
        args.exclude,
        args.new_log,
        args.short,
        args.full_name,
    )
