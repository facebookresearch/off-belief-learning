import os
import sys
import torch

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from create import *
import utils
import r2d2
import pprint

################## copied from find_prompts_and_finesses ############
import argparse
import getpass
import os
import sys
import pickle
import logging
import copy
import collections
import json
import functools

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import set_path
set_path.append_sys_path()

import torch
import rela
import hanalearn
import common_utils

# from extract_human_data import Game, Move, Card
# import run_human_game

color_codes = [
    "\033[31m",
    "\033[33m",
    "\033[32m",
    "\033[37m",
    "\033[34m",
]
color_chars = [
    "R",
    "Y",
    "G",
    "W",
    "B",
]
resetcolor = "\033[0m"

def _card_to_str(card):
    return (
        color_codes[card.color()] +
        str(card.rank()+1) +
        resetcolor
    )

def _card_to_str_with_id(card):
    return (
        f"({card.id()})"+
        color_codes[card.color()] +
        str(card.rank()+1) +
        resetcolor
    )

def _fireworks_to_str(fireworks):
    s = ""
    for color in range(len(fireworks)):
        rank = fireworks[color]
        s += (
            color_codes[color] +
            str(rank) +
            resetcolor
        )
    return s

def _hand_to_str(hand):
    s = ""
    for card in hand.cards():
        s += _card_to_str(card)
    return s

def _move_to_str(env, obs, move):
    mt = move.move_type()
    pla = env.get_current_player()
    if mt == hanalearn.MoveType.Play:
        pos = move.card_index()
        card = obs.hands()[env.get_current_player()].cards()[pos]
        card_str = _card_to_str(card)
        return f"P{pla} Play card {card_str} pos {pos}"
    elif mt == hanalearn.MoveType.Discard:
        pos = move.card_index()
        card = obs.hands()[env.get_current_player()].cards()[pos]
        card_str = _card_to_str(card)
        return f"P{pla} Discard card {card_str} pos {pos}"
    elif mt == hanalearn.MoveType.RevealColor:
        target = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        color = move.color()
        color_str = color_codes[color] + color_chars[color] + resetcolor
        return f"P{pla} Hint P{target} {color_str}"
    elif mt == hanalearn.MoveType.RevealRank:
        target = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        rank = move.rank()
        rank_str = str(rank+1)
        return f"P{pla} Hint P{target} {rank_str}"
    return "Unknown move"

def _movetype_to_json(mt):
    if mt == hanalearn.MoveType.Play:
        return "Play"
    elif mt == hanalearn.MoveType.Discard:
        return "Discard"
    elif mt == hanalearn.MoveType.RevealColor:
        return "RevealColor"
    elif mt == hanalearn.MoveType.RevealRank:
        return "RevealRank"
    return "Unknown"


def print_game_state_and_move(important_steps, env, move):
    obs = env.get_obs_show_cards()

    fireworks = obs.fireworks()
    fireworks_str = _fireworks_to_str(fireworks)

    hands = obs.hands()
    hand_strs = [_hand_to_str(hand) for hand in hands]
    allhands_str = " | ".join(hand_strs)

    hints = obs.information_tokens()
    life = obs.life_tokens()
    deck_size = obs.deck_size()

    discards = obs.discard_pile()
    discards_str = "".join([_card_to_str(card) for card in discards])

    move_str = _move_to_str(env,obs,move)
    step = env.get_step()

    print(f"{fireworks_str} || {allhands_str} | Hints {hints} | Life {life} | Deck {deck_size} | Step {step} | Discards {discards_str}")
    print(move_str)
    if int(step) in important_steps:
        print(env.get_hle_state().to_string())


def find_cards_hinted(env,obs,move):
    mt = move.move_type()
    cards_hinted = []
    if mt == hanalearn.MoveType.RevealColor:
        pla = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        color = move.color()
        cards_hinted.extend(card for card in obs.hands()[pla].cards() if card.color() == color)

    elif mt == hanalearn.MoveType.RevealRank:
        pla = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        rank = move.rank()
        cards_hinted.extend(card for card in obs.hands()[pla].cards() if card.rank() == rank)
    return cards_hinted


def find_prompts_and_finesses(num_player, game_seed, moves, verbose):
    # if game["num_players"] <= 2:
    #     return []
    possible_pfs = []
    def record_possible_prompts_and_finesses(env, move):
        obs = env.get_obs_show_cards()
        cards_hinted = find_cards_hinted(env,obs,move)

        # Check that no cards hinted are playable and at least one card hinted is one away.
        fireworks = obs.fireworks()
        any_playable = False
        cards_one_away = []
        for card in cards_hinted:
            if card.rank() == fireworks[card.color()]:
                any_playable = True
                break
            if card.rank() == fireworks[card.color()]+1:
                cards_one_away.append(card)

        if any_playable:
            return
        if len(cards_one_away) <= 0:
            return

        # For each card one away, look for prompts and finesses
        prompt_targets = []
        finesse_targets = []
        bluff_targets = []
        targets_immediate_next_player = False
        for card in cards_one_away:
            for p,hand in enumerate(obs.hands()):
                current_player = env.get_current_player()
                if p == current_player:
                    continue
                is_immediate_next_player = (p == ((current_player + 1) % env.get_num_players()))

                handcards = hand.cards()
                know = hand.knowledge_()
                for pos,c in enumerate(handcards):
                    if c.color() == card.color() and c.rank() == card.rank()-1:
                        color_hinted = know[pos].color_hinted()
                        rank_hinted = know[pos].rank_hinted()
                        if pos == len(handcards)-1 and not color_hinted and not rank_hinted:
                            finesse_targets.append(c)
                            targets_immediate_next_player = True
                        if color_hinted or rank_hinted:
                            prompt_targets.append(c)
                            targets_immediate_next_player = True
                    # Require bluffs to be of the immediate next player
                    if is_immediate_next_player and pos == len(handcards)-1 and fireworks[c.color()] == c.rank():
                        color_hinted = know[pos].color_hinted()
                        rank_hinted = know[pos].rank_hinted()
                        if not color_hinted and not rank_hinted:
                            bluff_targets.append(c)
                            targets_immediate_next_player = True

        targets = None
        if prompt_targets:
            targets = prompt_targets
            is_prompt = True
            is_finesse = False
            is_bluff = False
        elif finesse_targets:
            targets = finesse_targets
            is_prompt = False
            is_finesse = True
            is_bluff = False
        elif bluff_targets:
            targets = bluff_targets
            is_prompt = False
            is_finesse = False
            is_bluff = True

        if targets:
            target_ids = [target.id() for target in targets]
            pf = dict(
                game_seed=game_seed, # game_id=game["game_id"],
                step=env.get_step(),
                move_str=_move_to_str(env,obs,move),
                move=dict(
                    move_type=_movetype_to_json(move.move_type()),
                    card_index=move.card_index(),
                    color=move.color(),
                    rank=move.rank(),
                    target_offset=move.target_offset(),
                ),
                target_ids=target_ids,
                is_prompt=is_prompt,
                is_finesse=is_finesse,
                is_bluff=is_bluff,
                targets_immediate_next_player=targets_immediate_next_player,
            )
            possible_pfs.append(pf)

    run_fixed_game(num_player, game_seed, moves, record_possible_prompts_and_finesses)

    # Dict mapping card id -> steps it was touched by hint
    executed_pfs = []
    def record_executed_prompts_and_finesses(env, move):
        nonlocal possible_pfs

        obs = env.get_obs_show_cards()
        cards_hinted = find_cards_hinted(env,obs,move)
        step = env.get_step()
        for card in cards_hinted:
            cid = card.id()

            # If a target was hinted again after the pf, filter out the pf
            new_possible_pfs = []
            for pf in possible_pfs:
                # Require bluffs to be played immediately
                if pf["is_bluff"] and step == pf["step"] + 2:
                    continue
                if pf["step"] >= step:
                    new_possible_pfs.append(pf)
                    continue
                rehinted = any(target_id == cid for target_id in pf["target_ids"])
                if not rehinted:
                    new_possible_pfs.append(pf)
                    continue
            possible_pfs = new_possible_pfs

        mt = move.move_type()
        pla = env.get_current_player()
        if mt == hanalearn.MoveType.Play:
            pos = move.card_index()
            card = obs.hands()[env.get_current_player()].cards()[pos]
            played_id = card.id()
            new_possible_pfs = []
            for pf in possible_pfs:
                if any(target_id == played_id for target_id in pf["target_ids"]):
                    pf["step_played"] = step
                    executed_pfs.append(pf)
                    # print(f"Executed: {pf}")
                else:
                    new_possible_pfs.append(pf)
            possible_pfs = new_possible_pfs

    score = run_fixed_game(num_player, game_seed, moves, record_executed_prompts_and_finesses)
    important_steps = [int(f["step"]) for f in executed_pfs]

    if verbose:
        run_fixed_game(
            num_player,
            game_seed,
            moves,
            functools.partial(print_game_state_and_move, important_steps)
        )
    return executed_pfs, score


######################### copy end #######################


def run_fixed_game(num_player, seed, moves, detect_func):
    params = {
        "players": str(num_player),
        "seed": str(seed),
        "bomb": str(0),
        "random_start_player": str(0),
    }
    game = hanalearn.HanabiEnv(params, -1, False)  # max_len  # verbose
    game.reset()

    for move in moves:
        detect_func(game, move)
        game.step(move)

    return game.get_score()


def run_game(agents, seed):
    params = {
        "players": str(len(agents)),
        "seed": str(seed),
        "bomb": str(0),
        "random_start_player": str(0),
    }
    game = hanalearn.HanabiEnv(params, -1, False)  # max_len  # verbose

    game.reset()
    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            if isinstance(agents[0], r2d2.R2D2Agent):
                h[k] = v.cuda().unsqueeze(0)  # add batch dim
            else:
                h[k] = v.cuda()

    moves = []
    while not game.terminated():
        actions = []
        new_hids = []
        for i, (agent, hid) in enumerate(zip(agents, hids)):
            # Note: argument here is (game_state, player_idx, hide_action)
            # make sure to specify the correct hide_action value
            obs = hanalearn.observe(game.get_hle_state(), i, False)
            priv_s = obs["priv_s"].cuda().unsqueeze(0)
            publ_s = obs["publ_s"].cuda().unsqueeze(0)
            legal_move = obs["legal_move"].cuda().unsqueeze(0)

            action, new_hid = agent.greedy_act(priv_s, publ_s, legal_move, hid)
            if i == 0:
                actions.append([action.item()])
            else:
                actions[-1].append(action.item())
            new_hids.append(new_hid)

        hids = new_hids
        cur_player = game.get_current_player()
        move = game.get_move(actions[-1][cur_player])
        moves.append(move)

        game.step(move)

    return (seed, moves, game.get_score())



model_zoo = {
    "clone_bot": "/checkpoint/hengyuan/hanabi_sl/player3_op/NUM_PLAYER3_OP1_RBS100000_RNN_HID512_LSTM_LAYER2_DROPOUT0.5/model0.pthw",
    "clone_bot_publ_lstm": "/checkpoint/hengyuan/hanabi_sl/player3_op_publ_lstm/NUM_PLAYER3_NETpubl-lstm_OP1_RBS100000_RNN_HID512_LSTM_LAYER2_DROPOUT0.5/model1.pthw",
    "best_sad": "/checkpoint/hengyuan/hanabi_benchmark/vani_sad_aux_op_3p/NETlstm_NUM_PLAYER3_SAD1_SEEDh/model1.pthw",
    # "best_vdn": "/checkpoint/hengyuan/hanabi_benchmark/vani_sad_aux_op_3p/NETlstm_NUM_PLAYER3_SEEDe/model4.pthw",
    "best_op": "/checkpoint/hengyuan/hanabi_benchmark/vani_sad_aux_op_3p/NETlstm_NUM_PLAYER3_PRED0.25_OP1_SEEDh/model2.pthw",
    "best_aux": "/checkpoint/hengyuan/hanabi_benchmark/vani_sad_aux_op_3p/NETlstm_NUM_PLAYER3_PRED0.25_OP1_SEEDa/model0.pthw",
    "rl_cloneA": "/checkpoint/hengyuan/hanabi_benchmark/rl_bc_p3/NETlstm_NUM_PLAYER3_PRED0.25_CLONE_WEIGHT0.1_CLONE_T0.1_SEEDa/model0.pthw",
    "rl_cloneB": "/checkpoint/hengyuan/hanabi_benchmark/rl_bc_p3/NETlstm_NUM_PLAYER3_PRED0.25_CLONE_WEIGHT0.1_CLONE_T0.125_SEEDa/model0.pthw",
    # obl
    "obl1": "/checkpoint/hengyuan/hanabi_benchmark_obl/p3_obl1_publ_lstm/NETpubl-lstm_NUM_PLAYER3_LOAD0_SEEDb/model0.pthw",
    "obl2": "/checkpoint/hengyuan/hanabi_benchmark_obl/p3_obl2_publ_lstm_b200/NETpubl-lstm_NUM_PLAYER3_LOAD1_SEEDa/model0.pthw",
    "obl3": "/checkpoint/hengyuan/hanabi_benchmark_obl/p3_obl3_publ_lstm/NETpubl-lstm_NUM_PLAYER3_LOAD1_NUM_EPOCH300_SEEDa/model0.pthw",
    "obl4": "/checkpoint/hengyuan/hanabi_benchmark_obl/p3_obl4_publ_lstm/NETpubl-lstm_NUM_PLAYER3_LOAD1_NUM_EPOCH500_SEEDa/model0.pthw",
    "obl5": "/checkpoint/hengyuan/hanabi_benchmark_obl/p3_obl5_publ_lstm/NETpubl-lstm_NUM_PLAYER3_LOAD1_NUM_EPOCH500_SEEDa/model0.pthw",
}

if __name__ == "__main__":
    import time
    import numpy as np

    # main program
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_file", default=None, type=str, required=True)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_game", default=1, type=int)
    parser.add_argument("--verbose", default=1, type=int)
    args = parser.parse_args()

    num_player = 3
    if args.weight_file in model_zoo:
        args.weight_file = model_zoo[args.weight_file]

    state_dict = torch.load(args.weight_file)
    if "fc_v.weight" in state_dict.keys():
        agent, cfg = utils.load_agent(args.weight_file, {"device": "cuda:0"})
    else:
        agent = utils.load_supervised_agent(args.weight_file, "cuda:0")
        cfg = {"hide_action": False}
    agent.train(False)

    agents = [agent for _ in range(num_player)]
    t = time.time()

    scores = []
    finesses = []
    for i in range(args.num_game):
        seed, moves, ref_score = run_game(agents, args.seed + i)
        finesse, score = find_prompts_and_finesses(num_player, seed, moves, args.verbose)
        assert score == ref_score
        scores.append(score)
        finesses.extend(finesse)
        if args.verbose:
            pprint.pprint(finesse)
    print("time taken:", time.time() - t)
    print(f"# game: {len(scores)}, mean: {np.mean(scores):.4f}")

    real_finesse = []
    prompt = []
    bluff = []
    for finesse in finesses:
        if finesse["is_prompt"]:
            prompt.append(finesse)
        if finesse["is_finesse"] and finesse["targets_immediate_next_player"]:
            if finesse["step_played"] - finesse["step"] == 2:
                real_finesse.append(finesse)
        if finesse["is_bluff"]:
            bluff.append(finesse)

    if not args.verbose:
        print(f"# num finesse: {len(real_finesse)}")
        print(f"# num prompt: {len(prompt)}")
        print(f"# num bluff: {len(bluff)}")
        pprint.pprint(real_finesse)
