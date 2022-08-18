import os
import sys
import argparse
import torch
import numpy as np
import time
import random
import pprint

# c++ backend
import set_path

set_path.append_sys_path()
import rela
import hanalearn
import utils
import common_utils
from tools import model_zoo
from tools.play_and_find_finesse import find_prompts_and_finesses


def print_state(one_step, finesse_step, step):
    if not args.one_step:
        return True
    if step >= args.finesse_step and step <= args.finesse_step + 2:
        return True
    return False


def run(seed, actors, args):
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
        print(common_utils.get_mem_usage())
        if print_state(args.one_step, args.finesse_step, step):
            print(f"{game.state().to_string()}")

        cur_player = game.state().cur_player()

        for i, actor in enumerate(actors):
            if args.one_step and step > args.finesse_step:
                continue
            print(f"---Actor {i} update belief---")
            actor.update_belief(game, args.num_thread)

        if (args.one_step and step == args.finesse_step) or (
            not args.one_step and step >= args.finesse_step
        ):
            print(f"---Actor {cur_player} try finesse---")
            # current player try finesse
            t = time.time()
            actors[cur_player].finesse(
                game,
                args.num_p2_hand,
                args.num_p1_hand,
                args.num_thread,
                0.05,
                args.argmax,
                args.beta,
                args.vote_based,
                args.compute_twice,
            )
            print("time taken:", time.time() - t)

        for i, actor in enumerate(actors):
            # print(f"---Actor {i} observe---")
            actor.observe(game)

        actions = []
        for i, actor in enumerate(actors):
            # print(f"---Actor {i} decide action---")
            action = actor.decide_action(game)
            actions.append(action)

        move = game.get_move(actions[cur_player])
        moves.append(move)
        print(f"Acitve Player {cur_player} pick action: {move.to_string()}")
        game.step(move)
        step += 1

    print(f"Final Score: {game.get_score()}, Seed: {seed}")
    return moves, game.get_score()


def parse_args():
    clone_file = (
        "/checkpoint/hengyuan/hanabi_sl/player3_op_publ_lstm/"
        "NUM_PLAYER3_NETpubl-lstm_OP1_RBS100000_RNN_HID512_LSTM_LAYER2_DROPOUT0.5/model1.pthw"
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--save_dir", type=str, default="exps/finesse")
    parser.add_argument("--seed", type=int, default=200191)
    parser.add_argument("--game_seed", type=int, default=19)
    parser.add_argument("--device", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--num_thread", type=int, default=80)
    parser.add_argument("--argmax", type=int, default=1)
    parser.add_argument("--beta", type=float, default=10)
    parser.add_argument("--finesse_step", type=int, default=5)
    parser.add_argument("--one_step", type=int, default=0)
    parser.add_argument("--num_p2_hand", type=int, default=400)
    parser.add_argument("--num_p1_hand", type=int, default=100)
    parser.add_argument("--vote_based", type=int, default=0)
    parser.add_argument("--compute_twice", type=int, default=0)
    parser.add_argument("--weight_file", type=str, default=clone_file)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    pprint.pprint(vars(args))
    devices = args.device.split(",")

    state_dict = torch.load(args.weight_file)
    if "fc_v.weight" in state_dict.keys():
        bp, _ = utils.load_agent(args.weight_file, {"device": "cuda:0"})
    else:
        bp = utils.load_supervised_agent(args.weight_file, "cuda:0")
    bp.train(False)
    bp_runners = []
    for device in devices:
        bp_runners.append(rela.BatchRunner(bp.clone(device), device, 2500, ["act"]))
        bp_runners[-1].start()

    common_utils.set_all_seeds(args.seed)
    logger_path = os.path.join(args.save_dir, "run.log")
    sys.stdout = common_utils.Logger(logger_path)

    actors = []
    for i in range(3):
        seed = random.randint(1, 100000)
        actor = hanalearn.FinesseActor(i, bp_runners, seed)
        actors.append(actor)
    for i in range(len(actors)):
        actors[i].set_partners(actors)

    moves, ref_score = run(args.game_seed, actors, args)
    prompt_and_finesse, score = find_prompts_and_finesses(
        3, args.game_seed, moves, False
    )
    # assert(score == ref_score)
    pprint.pprint(prompt_and_finesse)
    finesse = [
        f
        for f in prompt_and_finesse
        if f["is_finesse"] and f["targets_immediate_next_player"]
    ]
    if len(finesse):
        for f in finesse:
            s = f"Find finesse at step {f['step']}"
            if args.one_step:
                s += f", orginal finesse step: {args.finesse_step}"
            print(s)
            pprint.pprint(f)
    else:
        print("Fail to finesse")
