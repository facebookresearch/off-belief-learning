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
import time
import os
import sys
import argparse
import pprint
import copy
import datetime

import numpy as np
import torch
from torch import nn
import concurrent.futures as futures

from act_group import ActGroup, BRActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--mode", type=str, default="selfplay")
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--boltzmann_act", type=int, default=0)
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--num_fict_sample", type=int, default=10)

    parser.add_argument("--coop_agents", type=str, default=None, nargs="+")
    parser.add_argument("--coop_sync_freq", type=int, default=0)

    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--clone_bot", type=str, default="", help="behavior clone loss")
    parser.add_argument("--clone_weight", type=float, default=0.0)
    parser.add_argument("--clone_t", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    assert args.method in ["iql"]
    assert args.mode in ["br", "klr"]
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

    if args.boltzmann_act:
        boltzmann_beta = utils.generate_log_uniform(
            1 / args.max_t, 1 / args.min_t, args.num_t
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))
    else:
        boltzmann_t = []
        print("no boltzmann")

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )

    agent = r2d2.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        games[0].feature_size(args.sad),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.boltzmann_act,
        False,  # uniform priority
        False,  # off belief not supported for best response
    )
    agent.sync_target_with_online()

    if args.load_model and args.load_model != "None":
        print("*****loading pretrained model*****")
        print(args.load_model)
        utils.load_weight(agent.online_net, args.load_model, args.train_device)
        print("*****done*****")

    # use clone bot for additional bc loss
    if args.clone_bot and args.clone_bot != "None":
        clone_bot = utils.load_supervised_agent(args.clone_bot, args.train_device)
    else:
        clone_bot = None

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)
    eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    if args.coop_sync_freq:
        sync_pool = futures.ThreadPoolExecutor(max_workers=1)
        save_future, coop_future = None, None
        save_ckpt = common_utils.ModelCkpt(args.save_dir)
        print(
            f"{datetime.datetime.now()}, save the initial model to ckpt: {save_ckpt.prefix}"
        )
        utils.save_intermediate_model(agent.online_net.state_dict(), save_ckpt)

    coop_agents = None
    coop_eval_agents = []
    if args.coop_agents is not None:
        coop_ckpts = []
        for coop_pth in args.coop_agents:
            coop_ckpts.append(common_utils.ModelCkpt(coop_pth))
        coop_agents = utils.load_coop_agents(coop_ckpts, device="cpu")

    act_group_args = {
        "devices": args.act_device,
        "agent": agent,
        "seed": args.seed,
        "num_thread": args.num_thread,
        "num_game_per_thread": args.num_game_per_thread,
        "num_player": args.num_player,
        "explore_eps": explore_eps,
        "boltzmann_t": boltzmann_t,
        "method": args.method,
        "sad": args.sad,
        "shuffle_color": args.shuffle_color,
        "hide_action": args.hide_action,
        "trinary": True,  # trinary, 3 bits for aux task
        "replay_buffer": replay_buffer,
        "multi_step": args.multi_step,
        "max_len": args.max_len,
        "gamma": args.gamma,
    }

    act_group_cls = BRActGroup
    if coop_agents is not None or args.mode == "klr":
        if args.mode == "klr" and coop_agents is None:
            print("Going to make BR act group for KLR level 1")
        act_group_args["coop_agents"] = coop_agents

    act_group = act_group_cls(**act_group_args)

    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )

    act_group.start()
    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    for epoch in range(args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent)
            if args.coop_sync_freq and num_update % args.coop_sync_freq == 0:
                print(f">>>step {num_update}, sync models")
                if save_future is None or save_future.done():
                    save_future = sync_pool.submit(
                        utils.save_intermediate_model,
                        copy.deepcopy(agent.online_net.state_dict()),
                        save_ckpt,
                    )
                if coop_agents is not None and (
                    coop_future is None or coop_future.done()
                ):
                    coop_future = sync_pool.submit(
                        utils.update_intermediate_coop_agents, coop_ckpts, act_group
                    )
                print(f"<<<step {num_update}, sync done")

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")

            loss, priority, online_q = agent.loss(batch, args.aux_weight, stat)
            if clone_bot is not None and args.clone_weight > 0:
                bc_loss = agent.behavior_clone_loss(
                    online_q, batch, args.clone_t, clone_bot, stat
                )
                loss = loss + bc_loss * args.clone_weight
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()
        stat.summary(epoch)

        eval_seed = (9917 + epoch * 999999) % 7777777
        eval_agent.load_state_dict(agent.state_dict())

        eval_agents = [eval_agent for _ in range(args.num_player)]
        if coop_agents is not None:
            coop_eval_agents = utils.load_coop_agents(
                coop_ckpts,
                overwrites={"vdn": False, "boltzmann_act": False},
                device=args.train_device,
            )
            eval_idxs = np.random.choice(
                len(coop_eval_agents), args.num_player - 1, replace=False
            )
            eval_agents = [eval_agent]
            for idx in eval_idxs:
                eval_agents.append(coop_eval_agents[idx])

        score, perfect, *_ = evaluate(
            eval_agents,
            1000,
            eval_seed,
            args.eval_bomb,
            0,  # explore eps
            args.sad,
            args.hide_action,
        )

        force_save_name = None
        if epoch > 0 and epoch % 100 == 0:
            force_save_name = "model_epoch%d" % epoch
        model_saved = saver.save(
            None, agent.online_net.state_dict(), score, force_save_name=force_save_name
        )
        print(
            "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
            % (epoch, score, perfect * 100, model_saved)
        )

        if clone_bot is not None:
            score, perfect, *_ = evaluate(
                [clone_bot] + [eval_agent for _ in range(args.num_player - 1)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
            )
            print(f"clone bot score: {np.mean(score)}")

        print("==========")
