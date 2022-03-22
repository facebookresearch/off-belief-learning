# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import pprint
import torch
import numpy as np

# c++ backend
import set_path

set_path.append_sys_path()
import rela
import hanalearn
import utils
import common_utils


def run(seed, actors, search_actor_idx, num_search, threshold, num_thread):
    params = {
        "players": str(len(actors)),
        "seed": str(seed),
        "bomb": str(0),
        "hand_size": str(5),
        "random_start_player": str(0),  # do not randomize start_player
    }
    game = hanalearn.GameSimulator(params)
    step = 0
    moves = []
    while not game.terminal():
        print("================STEP %d================" % step)
        print(game.state().to_string())

        cur_player = game.state().cur_player()

        actors[search_actor_idx].update_belief(game, num_thread)
        for i, actor in enumerate(actors):
            actor.observe(game)

        for i, actor in enumerate(actors):
            print(f"---Actor {i} decide action---")
            action = actor.decide_action(game)
            if i == cur_player:
                move = game.get_move(action)

        # run sparta, this may change the move
        if cur_player == search_actor_idx:
            move = actors[search_actor_idx].sparta_search(
                game, move, num_search, threshold
            )

        print(f"Acitve Player {cur_player} pick action: {move.to_string()}")
        moves.append(move)
        game.step(move)
        step += 1

    print(f"Final Score: {game.get_score()}, Seed: {seed}")
    return moves, game.get_score()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--save_dir", type=str, default="exps/sparta")
    parser.add_argument("--num_search", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument("--weight_file", type=str, default=None)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--search_player", type=int, default=1)
    parser.add_argument("--seed", type=int, default=200191)
    parser.add_argument("--game_seed", type=int, default=19)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)

    if "fc_v.weight" in torch.load(args.weight_file).keys():
        bp, config = utils.load_agent(args.weight_file, {"device": args.device})
        assert not config["hide_action"]
        assert not config["boltzmann_act"]
    else:
        bp = utils.load_supervised_agent(args.weight_file, args.device)
    bp.train(False)

    bp_runner = rela.BatchRunner(bp, args.device, 2000, ["act"])
    bp_runner.start()

    seed = args.seed
    actors = []
    for i in range(args.num_player):
        actor = hanalearn.SpartaActor(i, bp_runner, seed)
        seed += 1
        actors.append(actor)

    actors[args.search_player].set_partners(actors)

    moves, score = run(
        args.game_seed,
        actors,
        args.search_player,
        args.num_search,
        args.threshold,
        args.num_thread,
    )
