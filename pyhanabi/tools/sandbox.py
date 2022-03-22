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
import getpass
import os
import sys
import pickle
import logging
import copy

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import set_path

set_path.append_sys_path()

import torch
import rela
import hanalearn
import common_utils
from extract_human_data import Game, Move, Card

import run_human_game

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
    return color_codes[card.color()] + str(card.rank() + 1) + resetcolor


def _fireworks_to_str(fireworks):
    s = ""
    for color in range(len(fireworks)):
        rank = fireworks[color]
        s += color_codes[color] + str(rank) + resetcolor
    return s


def _hand_to_str(hand):
    s = ""
    for card in hand.cards():
        s += _card_to_str(card)
    return s


def _move_to_str(env, obs, move):
    mt = move.move_type()
    if mt == hanalearn.MoveType.Play:
        pos = move.card_index()
        card = obs.hands()[env.get_current_player()].cards()[pos]
        card_str = _card_to_str(card)
        return f"Play card {card_str} pos {pos}"
    elif mt == hanalearn.MoveType.Discard:
        pos = move.card_index()
        card = obs.hands()[env.get_current_player()].cards()[pos]
        card_str = _card_to_str(card)
        return f"Discard card {card_str} pos {pos}"
    elif mt == hanalearn.MoveType.RevealColor:
        pla = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        color = move.color()
        color_str = color_codes[color] + color_chars[color] + resetcolor
        return f"Hint p{pla} {color_str}"
    elif mt == hanalearn.MoveType.RevealRank:
        pla = (env.get_current_player() + move.target_offset()) % env.get_num_players()
        rank = move.rank()
        rank_str = str(rank + 1)
        return f"Hint p{pla} {rank_str}"
    return "Unknown move"


def print_game_state_and_move(env, move):
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

    move_str = _move_to_str(env, obs, move)
    step = env.get_step()

    print(
        f"{fireworks_str} || {allhands_str} | Hints {hints} | Life {life} | Deck {deck_size} | Step {step} | Discards {discards_str}"
    )
    print(move_str)


def process(game):
    run_human_game.run_game(game, max_len=80, f=print_game_state_and_move)


if __name__ == "__main__":
    pkl_file = "/checkpoint/dwu/hanabi/hanabi-live-data/games.0.pkl"
    games = pickle.load(open(pkl_file, "rb"))

    for i, game in enumerate(games):
        if i >= 1:
            break
        process(game)
