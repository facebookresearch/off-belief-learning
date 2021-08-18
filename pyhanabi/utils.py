# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
from collections import OrderedDict
import json
import torch
import numpy as np

import r2d2
from create import create_envs
import common_utils
from supervised_model import SupervisedAgent


def load_supervised_agent(weight_file, device):
    # this is a bit hard-coded, works for now
    print("loading file from: ", weight_file)
    try:
        cfg = get_train_config(weight_file)
    except:
        cfg = {}
    state_dict = torch.load(weight_file)

    if "dataset" in cfg:
        # latest version from online training
        agent = SupervisedAgent(
            device,
            cfg["priv_in_dim"],
            cfg["publ_in_dim"],
            cfg["rnn_hid_size"],
            cfg["num_action"],
            cfg["lstm_layers"],
            cfg["net"],
            cfg["dropout"],
        )
        agent.load_state_dict(state_dict)
        return agent

    if cfg.get("net", "publ-lstm") == "lstm":
        priv_in = state_dict["net.net.0.weight"].size(1)
        out, hid = state_dict["net.out_layer.weight"].size()
        agent = SupervisedAgent(device, priv_in, 0, hid, out, 1, "lstm", 0.0)
        agent.load_state_dict(state_dict)
        return agent

    if not next(iter(state_dict)).startswith("net"):
        new_state_dict = type(state_dict)()
        for k, v in state_dict.items():
            new_state_dict["net." + k] = v
        state_dict = new_state_dict

    priv_in = state_dict["net.priv_net.0.weight"].size(1)
    publ_in = state_dict["net.publ_net.0.weight"].size(1)
    if "net.final_ff.weight" in state_dict:
        state_dict["net.out_layer.weight"] = state_dict["net.final_ff.weight"]
        state_dict["net.out_layer.bias"] = state_dict["net.final_ff.bias"]
        state_dict.pop("net.final_ff.weight")
        state_dict.pop("net.final_ff.bias")

    out, hid = state_dict["net.out_layer.weight"].size()
    agent = SupervisedAgent(device, priv_in, publ_in, hid, out, 1, "publ-lstm", 0.0)
    agent.load_state_dict(state_dict)
    return agent


def parse_first_dict(lines):
    config_lines = []
    open_count = 0
    for i, l in enumerate(lines):
        if l.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(l)
        if l.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and len(config_lines) != 0:
            break

    config = "".join(config_lines).replace("'", '"')
    config = config.replace("True", "true")
    config = config.replace("False", "false")
    config = config.replace("None", "null")
    config = json.loads(config)
    return config, lines[i + 1 :]


def parse_hydra_dict(lines):
    conf = OmegaConf.create(lines[0])
    cfg = {
        "num_player": conf.env.num_player,
        "train_bomb": conf.env.train_bomb,
        "max_len": conf.env.max_len,
        "sad": conf.env.sad,
        "shuffle_obs": conf.env.shuffle_obs,
        "shuffle_color": conf.env.shuffle_color,
        "hide_action": conf.env.hide_action,
        "rnn_hid_dim": conf.agent.params.hid_dim,
        "num_lstm_layer": conf.agent.params.num_lstm_layer,
        "boltzmann_act": conf.agent.params.boltzmann_act,
        "multi_step": conf.agent.params.multi_step,
        "gamma": conf.agent.params.gamma,
    }

    return cfg


def get_train_config(weight_file):
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        return None

    lines = open(log, "r").readlines()
    try:
        cfg, rest = parse_first_dict(lines)
    except json.decoder.JSONDecodeError as e:
        cfg = parse_hydra_dict(lines)
    return cfg


def flatten_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_dict)
        else:
            new_dict[k] = v


def load_agent(weight_file, overwrite):
    """
    overwrite has to contain "device"
    """
    if weight_file == "legacy":
        from legacy_agent import load_legacy_agent

        return load_legacy_agent()
    if "arxiv" in weight_file:
        from legacy_agent import load_legacy_agent

        return load_legacy_agent(weight_file)

    print("loading file from: ", weight_file)
    cfg = get_train_config(weight_file)
    assert cfg is not None

    if "core" in cfg:
        new_cfg = {}
        flatten_dict(cfg, new_cfg)
        cfg = new_cfg

    hand_size = cfg.get("hand_size", 5)

    game = create_envs(
        1,
        1,
        cfg["num_player"],
        cfg["train_bomb"],
        cfg["max_len"],
        hand_size=hand_size,
    )[0]

    config = {
        "vdn": overwrite["vdn"] if "vdn" in overwrite else cfg["method"] == "vdn",
        "multi_step": overwrite.get("multi_step", cfg["multi_step"]),
        "gamma": overwrite.get("gamma", cfg["gamma"]),
        "eta": 0.9,
        "device": overwrite["device"],
        "in_dim": game.feature_size(cfg["sad"]),
        "hid_dim": cfg["hid_dim"] if "hid_dim" in cfg else cfg["rnn_hid_dim"],
        "out_dim": game.num_action(),
        "num_lstm_layer": cfg.get("num_lstm_layer", overwrite.get("num_lstm_layer", 2)),
        "boltzmann_act": overwrite.get(
            "boltzmann_act", cfg.get("boltzmann_act", False)
        ),
        "uniform_priority": overwrite.get("uniform_priority", False),
        "net": cfg.get("net", "publ-lstm"),
        "off_belief": overwrite.get("off_belief", cfg.get("off_belief", False)),
    }
    if cfg.get("net", None) == "transformer":
        config["nhead"] = cfg["nhead"]
        config["nlayer"] = cfg["nlayer"]
        config["max_len"] = cfg["max_len"]

    agent = r2d2.R2D2Agent(**config).to(config["device"])
    load_weight(agent.online_net, weight_file, config["device"])
    agent.sync_target_with_online()
    return agent, cfg


def log_explore_ratio(games, expected_eps):
    explore = []
    for g in games:
        explore.append(g.get_explore_count())
    explore = np.stack(explore)
    explore = explore.sum(0)  # .reshape((8, 10)).sum(1)

    step_counts = []
    for g in games:
        step_counts.append(g.get_step_count())
    step_counts = np.stack(step_counts)
    step_counts = step_counts.sum(0)  # .reshape((8, 10)).sum(1)

    factor = []
    for i in range(len(explore)):
        if step_counts[i] == 0:
            factor.append(1.0)
        else:
            f = expected_eps / max(1e-5, (explore[i] / step_counts[i]))
            f = max(0.5, min(f, 2))
            factor.append(f)

    explore = explore.reshape((8, 10)).sum(1)
    step_counts = step_counts.reshape((8, 10)).sum(1)

    print("exploration:")
    for i in range(len(explore)):
        ratio = 100 * explore[i] / max(step_counts[i], 0.1)
        factor_ = factor[i * 10 : (i + 1) * 10]
        print(
            "\tbucket [%2d, %2d]: %5d, %5d, %2.2f%%: mean factor: %.2f"
            % (
                i * 10,
                (i + 1) * 10,
                explore[i],
                step_counts[i],
                ratio,
                np.mean(factor_),
            )
        )

    for g in games:
        g.reset_count()

    return factor


class Tachometer:
    def __init__(self):
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self):
        self.t = time.time()

    def lap(self, replay_buffer, num_train, factor):
        t = time.time() - self.t
        self.total_time += t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        print(
            "Speed: train: %.1f, buffer_add: %.1f, buffer_size: %d"
            % (train_rate, buffer_rate, replay_buffer.size())
        )
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Time: %s, %ds"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        print(
            "Total Sample: train: %s, buffer: %s"
            % (
                common_utils.num2str(self.num_train),
                common_utils.num2str(self.num_buffer),
            )
        )


def load_weight(model, weight_file, device, *, state_dict=None):
    if state_dict is None:
        state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()

    if not set(state_dict.keys()).intersection(set(target_state_dict.keys())):
        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            if "online_net" in k:
                new_k = k[len("online_net.") :]
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict

    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded [not found in file]" % k)
            state_dict[k] = v
        elif state_dict[k].size() != v.size():
            print(
                "warnning: %s not loaded\n[size mismatch %s (in net) vs %s (in file)]"
                % (k, v.size(), state_dict[k].size())
            )
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            print("removing: %s not used" % k)
        else:
            source_state_dict[k] = state_dict[k]

    model.load_state_dict(source_state_dict)
    return


# returns the number of steps in all actors
def get_num_acts(actors):
    total_acts = 0
    for actor in actors:
        if isinstance(actor, list):
            total_acts += get_num_acts(actor)
        else:
            total_acts += actor.num_act()
    return total_acts


def generate_explore_eps(base_eps, alpha, num_env):
    if num_env == 1:
        if base_eps < 1e-6:
            base_eps = 0
        return [base_eps]

    eps_list = []
    for i in range(num_env):
        eps = base_eps ** (1 + i / (num_env - 1) * alpha)
        if eps < 1e-6:
            eps = 0
        eps_list.append(eps)
    return eps_list


def generate_log_uniform(min_val, max_val, n):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uni = np.linspace(log_min, log_max, n)
    uni_exp = np.exp(uni)
    return uni_exp.tolist()
