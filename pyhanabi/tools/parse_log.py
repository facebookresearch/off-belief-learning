# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from collections import defaultdict
import numpy as np


def shorten_name(name):
    name = name.replace("PREFIX_SEED", "PS")
    name = name.replace("FEED_SEED", "FS")
    name = name.replace("SEED", "S")
    name = name.replace("NUM_PLAYER", "NP")
    name = name.replace("TRAIN_BOMB", "TBOMB")
    name = name.replace("EVAL_BOMB", "EBOMB")
    name = name.replace("FIXED_EPS", "FEPS")
    name = name.replace("GREEDY_EXTRA", "GA")
    name = name.replace("PRED_RATIO", "PRED")
    name = name.replace("17779999", "179")
    name = name.replace("GAME_PER_THREAD", "GPT")
    name = name.replace("METHODbest_response_", "")
    name = name.replace("METHOD", "")
    name = name.replace("DATA_SIZE", "data")
    name = name.replace("ACT_DEVICEcuda:1,cuda:2,cuda:3,cuda:4", "AGPU4")
    name = name.replace("ACT_DEVICEcuda:1,cuda:2,cuda:3", "AGPU3")
    name = name.replace("ACT_DEVICEcuda:1,cuda:2", "AGPU2")
    name = name.replace("ACT_DEVICEcuda:1", "AGPU1")
    name = name.replace("BATCHSIZE", "BSIZE")
    name = name.replace("ACT_BASE_EPS", "EPS")
    name = name.replace("ACT_EPS_ALPHA", "EA")
    name = name.replace("SHUFFLE_OBS", "SO")
    name = name.replace("HIDE_ACTION", "HA")
    name = name.replace("NUM_EPOCH", "E")
    name = name.replace("EPOCH_LEN", "EL")
    name = name.replace("RNN_HID_DIM", "RDIM")
    return name


def parse_new_log(filename, max_epoch):
    lines = open(filename, "r").readlines()
    times = []
    scores = []
    clone_scores = []
    perfects = []
    train_rate = []
    buffer_rate = []
    act_rate = []
    self_score = []
    aux = []
    aux1 = []
    aux2 = []
    xent_pred = []
    xent_v0 = []
    loss = []

    def get_val_from_line(l):
        a = float(l.split()[3][:-1])
        return a

    for i, l in enumerate(lines):
        if "Time spent =" in l:
            t = float(l.split()[-2])
            if len(times) == 0:
                times.append(t)
            else:
                times.append(times[-1] + t)
        if "Speed" in l:
            split = l.split()
            if "act" in l:
                train = float(split[2][:-1])
                act = float(split[4][:-1])
                buf = float(split[6][:-1])
            else:
                train = float(split[2][:-1])
                act = 0
                buf = float(split[4][:-1])
            train_rate.append(train)
            act_rate.append(act)
            buffer_rate.append(buf)
        if ("eval score:" in l or "eval_score:" in l) and "clone bot" not in l:
            split = l.split()
            score = float(split[4][:-1])
            scores.append(score)
        if "clone bot" in l:
            split = l.split()
            score = float(split[-1])
            clone_scores.append(float(score))
        if "eval score" in l:
            perfect = float(l.split()[6][:-1])
            perfects.append(perfect)
        if "eval: self," in l:
            score = float(l.split()[-3][:-1])
            self_score.append(score)
        if ":aux" in l and "avg" in l:
            aux.append(get_val_from_line(l))
        if ":aux1" in l and "avg" in l:
            aux1.append(get_val_from_line(l))
        if ":aux2" in l and "avg" in l:
            aux2.append(get_val_from_line(l))
        if ":xent_pred" in l and "avg" in l:
            a = float(l.split("avg:")[1].strip().split()[0][:-1])
            xent_pred.append(a)
        if ":xent_v0" in l and "avg" in l:
            a = float(l.split()[3][:-1])
            xent_v0.append(a)
        if ":loss" in l and "avg" in l:
            loss.append(get_val_from_line(l))

        if max_epoch > 0 and (len(scores) == max_epoch or len(xent_pred) == max_epoch):
            break

    if len(scores):
        epoch = len(scores)
    elif len(xent_pred):
        epoch = len(xent_pred)
    elif len(loss):
        epoch = len(loss)
    else:
        epoch = 0

    if len(act_rate):
        avg_act_rate = int(np.mean(act_rate[-10:]))
    else:
        avg_act_rate = 0
    if len(train_rate):
        avg_train_rate = int(np.mean(train_rate[-10:]))
    else:
        avg_train_rate = 0
    if len(buffer_rate):
        avg_buffer_rate = int(np.mean(buffer_rate[-10:]))
    else:
        avg_buffer_rate = 0
    times = [t / 60 / 60 for t in times]

    final_xent_pred = 0 if len(xent_pred) == 0 else np.mean(xent_pred[-10:])
    info = {
        "id": filename,
        "epoch": epoch,
        "act_rate": avg_act_rate,
        "train_rate": avg_train_rate,
        "buffer_rate": avg_buffer_rate,
        "final_score": np.mean(scores[-10:]),
        "scores": scores,
        "clone_scores": clone_scores,
        "final_perfect": np.mean(perfects[-10:]),
        "perfects": perfects,
        "aux": aux,
        "aux1": aux1,
        "aux2": aux2,
        "xent_pred": xent_pred,
        "xent_v0": xent_v0,
        "loss": loss,
        "final_xent_pred": final_xent_pred,
        "final_loss": np.mean(loss[-10:]),
        "times": times,
    }
    if len(self_score) > 0:
        info["selfplay"] = np.mean(self_score[-10:])

    return info


def average_across_seed(logs):
    new_logs = defaultdict(list)
    for k, v in logs.items():
        s = k.rsplit("_", 1)
        if len(s) == 2:
            name, seed = s
        elif len(s) == 1:
            name = "default"
            seed = s[0]
        if not seed.startswith("SEED"):
            name = k
        new_logs[name].append(v)

    for k in new_logs:
        vals = new_logs[k]
        means = []
        sems = []
        max_len = np.max([len(v) for v in vals])
        for i in range(max_len):
            nums = []
            for v in vals:
                if len(v) > i:
                    nums.append(v[i])
            means.append(np.mean(nums))
            if len(nums) == 1:
                sems.append(0)
            else:
                sems.append(np.std(nums) / np.sqrt(len(nums)))
        new_logs[k] = (means, sems)

    return new_logs


def max_across_seed(logs):
    new_logs = {}
    for k, v in logs.items():
        s = k.rsplit("_", 1)
        if len(s) == 2:
            name, seed = s
        elif len(s) == 1:
            name = "default"
            seed = s[0]
        if not seed.startswith("SEED"):
            # print("no multiple seeds, omit maxing: ", name)
            name = k
        if name not in new_logs or np.mean(v[-10:]) > new_logs[name][0]:
            new_logs[name] = (np.mean(v[-10:]), k)

    return new_logs


def parse_from_root(root, max_epoch, min_epoch, include, exclude, new_log):
    """
    include means include all, &&
    exclude means exclude any, ||
    """
    logs = {}
    root = os.path.abspath(root)
    for exp in os.listdir(root):
        if include:
            skip = False
            for s in include:
                if s not in exp:
                    skip = True
                    break
            if skip:
                continue

        skip = False
        for i, s in enumerate(exclude):
            if s in exp:
                skip = True
                break
        if skip:
            continue

        exp_folder = os.path.join(root, exp)
        log_file = os.path.join(exp_folder, "train.log")
        if not os.path.exists(log_file):
            log_file = os.path.join(exp_folder, "std.out")
        if os.path.exists(log_file):
            # try:
            if new_log:
                log = parse_new_log(log_file, max_epoch)
            else:
                log = parse_log(log_file, max_epoch)
            if min_epoch > 0 and log["epoch"] < min_epoch:
                print(
                    "%s is dropped due to being too short\n\t%d vs %d"
                    % (log_file, log["epoch"], min_epoch)
                )
            else:
                logs[exp] = log
        # except:
        #     print("something is wrong with %s" % log_file)

    return logs
