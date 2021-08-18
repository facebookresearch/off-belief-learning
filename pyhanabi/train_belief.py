# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint
import pickle

import numpy as np
import torch

from act_group import ActGroup
from create import create_envs, create_threads
import r2d2
import common_utils
import rela
import utils
import belief_model


def parse_args():
    parser = argparse.ArgumentParser(description="train belief model")
    parser.add_argument("--save_dir", type=str, default="exps/dev_belief")
    parser.add_argument("--load_model", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--fc_only", type=int, default=0)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:1")

    # load policy config
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--explore", type=int, default=1)
    parser.add_argument("--rand", type=int, default=0)
    parser.add_argument("--clone_bot", type=int, default=0)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # load from dataset setting
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--inf_data_loop", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--shuffle_color", type=int, default=0)

    args = parser.parse_args()
    return args


def create_rl_context(args):
    agent_overwrite = {
        "vdn": False,
        "device": args.train_device,  # batch runner will create copy on act device
        "uniform_priority": True,
    }

    if args.clone_bot:
        agent = utils.load_supervised_agent(args.policy, args.train_device)
        cfgs = {
            "act_base_eps": 0.1,
            "act_eps_alpha": 7,
            "num_game_per_thread": 80,
            "num_player": 2,
            "train_bomb": 0,
            "max_len": 80,
            "sad": 0,
            "shuffle_color": 0,
            "hide_action": 0,
            "multi_step": 1,
            "gamma": 0.999,
        }
    else:
        agent, cfgs = utils.load_agent(args.policy, agent_overwrite)

    assert cfgs["shuffle_color"] == False
    assert args.explore

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        1.0,  # priority exponent
        0.0,  # priority weight
        args.prefetch,
    )

    if args.rand:
        explore_eps = [1]
    elif args.explore:
        # use the same exploration config as policy learning
        explore_eps = utils.generate_explore_eps(
            cfgs["act_base_eps"], cfgs["act_eps_alpha"], cfgs["num_game_per_thread"]
        )
    else:
        explore_eps = [0]

    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))
    if args.clone_bot or not agent.boltzmann:
        print("no boltzmann act")
        boltzmann_t = []
    else:
        boltzmann_beta = utils.generate_log_uniform(
            1 / cfgs["max_t"], 1 / cfgs["min_t"], cfgs["num_t"]
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        cfgs["num_player"],
        cfgs["train_bomb"],
        cfgs["max_len"],
    )

    act_group = ActGroup(
        args.act_device,
        agent,
        args.seed,
        args.num_thread,
        args.num_game_per_thread,
        cfgs["num_player"],
        explore_eps,
        boltzmann_t,
        "iql",
        cfgs["sad"],
        cfgs["shuffle_color"] if not args.rand else False,
        cfgs["hide_action"],
        False,  # not trinary, need full hand for prediction
        replay_buffer,
        cfgs["multi_step"],  # not used
        cfgs["max_len"],
        cfgs["gamma"],  # not used
        False,  # turn off off-belief rewardless of how it is trained
        None,  # belief_model
    )

    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )
    return agent, cfgs, replay_buffer, games, act_group, context, threads


def create_sl_context(args):
    games = pickle.load(open(args.dataset, "rb"))
    print(f"total num game: {len(games)}")
    if args.shuffle_color:
        # to make data generation speed roughly the same as consumption
        args.num_thread = 10
        args.inf_data_loop = 1

    if args.replay_buffer_size < 0:
        args.replay_buffer_size = len(games) * args.num_player
    if args.burn_in_frames < 0:
        args.burn_in_frames = len(games) * args.num_player

    # priority not used
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        priority_exponent,
        priority_weight,
        args.prefetch,
    )
    data_gen = hanalearn.CloneDataGenerator(
        replay_buffer,
        args.num_player,
        args.max_len,
        args.shuffle_color,
        False,
        args.num_thread,
    )
    game_params = {
        "players": str(args.num_player),
        "random_start_player": "0",
        "bomb": "0",
    }
    data_gen.set_game_params(game_params)
    for i, g in enumerate(games):
        data_gen.add_game(g["deck"], g["moves"])
        if (i + 1) % 10000 == 0:
            print(f"{i+1} games added")

    return data_gen, replay_buffer


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 2)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.dataset is None or len(args.dataset) == 0:
        (
            agent,
            cfgs,
            replay_buffer,
            games,
            act_group,
            context,
            threads,
        ) = create_rl_context(args)
        act_group.start()
        context.start()
    else:
        data_gen, replay_buffer = create_sl_context(args)
        data_gen.start_data_generation(args.inf_data_loop, args.seed)
        # only for getting feature size
        games = create_envs(1, 1, args.num_player, 0, args.max_len)
        cfgs = {"sad": False}

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    if args.load_model:
        belief_config = utils.get_train_config(cfgs["belief_model"])
        print("load belief model from:", cfgs["belief_model"])
        model = belief_model.ARBeliefModel.load(
            cfgs["belief_model"],
            args.train_device,
            5,
            0,
            belief_config["fc_only"],
        )
    else:
        model = belief_model.ARBeliefModel(
            args.train_device,
            games[0].feature_size(cfgs["sad"])[1],
            args.hid_dim,
            5,  # hand_size
            25,  # bits per card
            0,  # num_sample
            fc_only=args.fc_only,
        ).to(args.train_device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    for epoch in range(args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()

        for batch_idx in range(args.epoch_len):
            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            assert weight.max() == 1
            loss, xent, xent_v0, _ = model.loss(batch)
            loss = loss.mean()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad()
            replay_buffer.update_priority(torch.Tensor())

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["xent_pred"].feed(xent.detach().mean().item())
            stat["xent_v0"].feed(xent_v0.detach().mean().item())

        print("EPOCH: %d" % epoch)

        if args.dataset is None or len(args.dataset) == 0:
            scores = [g.last_episode_score() for g in games]
            print("mean score: %.2f" % np.mean(scores))

        count_factor = 1
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)

        force_save_name = None
        if epoch > 0 and epoch % 100 == 0:
            force_save_name = "model_epoch%d" % epoch
        saver.save(
            None,
            model.state_dict(),
            -stat["loss"].mean(),
            True,
            force_save_name=force_save_name,
        )
        stat.summary(epoch)
        print("===================")
