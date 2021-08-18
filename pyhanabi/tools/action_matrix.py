# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys
import argparse
import pprint
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import common_utils
from parse_handshake import *
from parse_log import *


plt.rc("image", cmap="viridis")
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)
plt.rc("axes", labelsize=10)
plt.rc("axes", titlesize=10)


def render_plots(plot_funcs, rows, cols):
    fig = plt.figure(figsize=(cols * 8, rows * 8))
    ax = fig.subplots(rows, cols)
    for i, func in enumerate(plot_funcs):
        r = i // cols
        c = i % cols
        func(fig, ax[r][c])
    return fig, ax


def plot(mat, title, num_player, *, fig=None, ax=None, savefig=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(mat)
    ax.set_title(title)
    if num_player == 2:
        ax.set_xticks(range(20))
        ax.set_xticklabels(idx2action)
        ax.set_yticks(range(20))
        ax.set_yticklabels(idx2action)
    elif num_player == 3:
        ax.set_xticks(range(30))
        ax.set_xticklabels(idx2action_p3)
        ax.set_yticks(range(30))
        ax.set_yticklabels(idx2action_p3)

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)


def plot_factory(dataset, name, epoch):
    normed_p0_p1, _ = analyze(dataset)
    title = "%s_%d" % (name, epoch)

    def f(fig, ax):
        plot(normed_p0_p1, title, fig=fig, ax=ax)

    return f


def render_folder(sweep_folder, savefig=None):
    models = common_utils.get_all_files(sweep_folder, "pthw", contain="model0")
    models = sorted(models)
    logs = parse_from_root(sweep_folder, -1, -1, [], [], True)
    model_infos = {}
    for i, m in enumerate(models):
        name = m.split("/")[-2]
        print("M%2d: %s" % (i, name))
        model_infos["M%2d" % i] = {"path": m, "log": logs[name]}

    # get dataset
    datasets = {}
    for k, v in model_infos.items():
        dset, _, context = create_dataset(v["path"])
        datasets[k] = dset
        context.terminate()

    # render plots
    render_funcs = []
    for m_idx in sorted(model_infos.keys()):
        #     print('generate rendering function for %s' % m_idx)
        epoch = model_infos[m_idx]["log"]["epoch"]
        dset = datasets[m_idx]
        name = shorten_name(model_infos[m_idx]["log"]["id"].split("/")[-2])
        name = name.replace("_PRED0.25", "")
        name = name.replace("S9999999", "A")
        name = name.replace("S9", "B")
        name = name.replace("EPS_ALPHA", "EA")
        rf = plot_factory(dset, name, epoch)
        render_funcs.append(rf)

    print(len(render_funcs))
    cols = 2
    rows = len(render_funcs) // 2 + int(bool((len(render_funcs) % 2)))
    render_plots(render_funcs, rows, cols)
    if savefig is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(savefig)
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.model is not None:
        dataset, _, _ = create_dataset_new(args.model)
        normed_p0_p1, _ = analyze(dataset, args.num_player)
        plot(normed_p0_p1, "action_matrix", args.num_player, savefig=args.output)
    elif args.folder is not None:
        render_folder(args.folder, args.output)
