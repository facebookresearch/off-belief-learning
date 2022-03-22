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
import csv
import copy

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import set_path

set_path.append_sys_path()

import torch
import rela
import hanalearn
import common_utils
from .extract_human_data import Game, Move, Card


# For the dataset with action durations this was not necessary
BGA_BAD_GAMES_PER_PKL = {
    0: [5324],
    1: [2929, 8100, 9712],
    2: [2703, 2888, 3280],
    35: [8125],
    36: [5088],
    37: [2498, 3281, 7703, 7714, 9047, 9048],
    39: [2729, 8753, 8923],
    40: [6231],
    41: [528, 2037, 7166, 9470],
    42: [2160, 6172, 7150, 8012],
    43: [6222, 9305],
    44: [3154, 5824],
    45: [3721, 7280],
}


def filter_game(history, num_players, skip_player_name_check=False):
    if history["num_players"] != num_players:
        return False
    expected_hand_size = 5 if num_players <= 3 else 4
    for hand in history["hands"]:
        if len(hand) != expected_hand_size:
            return False
    if history["abandoned"]:
        return False
    player_ids = list(set([m.player_name for m in history["moves"]]))
    if (
        not skip_player_name_check
        and len(player_ids) > 1
        and len(player_ids) != num_players
    ):
        return False

    sim_hands = [[copy.copy(c) for c in hand] for hand in history["hands"]]
    sim_deck = [copy.copy(c) for c in history["deck"]]
    hints = 8
    for move_idx, move in enumerate(history["moves"]):
        hand = sim_hands[move.player_id]
        if move.type == "playCard" or move.type == "discardCard":
            hand.pop(move.value)
            if sim_deck:
                hand.append(sim_deck.pop())
            if move.type == "discardCard":
                if hints >= 8:
                    return False
                hints += 1
        elif move.type == "hintValue":
            hints -= 1
            other_hand: List[Card] = sim_hands[move.target_player]
            if not (move.value in [x.value for x in other_hand]):
                return False
        elif move.type == "hintColor":
            hints -= 1
            other_hand: List[Card] = sim_hands[move.target_player]
            if not (move.value in [x.color for x in other_hand]):
                return False

    return True


def run_game(game, max_len, f=None):
    """Iterate through a single human hanabi game from pickle format

    Parameters:
    game: extract_human_data.Game loaded from pickle file
    max_len: Bound on max len for hanabi env to simulate
    f: Optional function, if provided, call on every turn passing it the env and the action about to be made.

    Returns:
    A tuple of (final HanabiEnv object, hle_game object)
    """
    num_player = game["num_players"]
    hle_game = {"num_player": num_player}
    hle_game = {"game_id": game["game_id"]}
    game_params = {
        "players": str(num_player),
        "random_start_player": "0",
        "bomb": "0",
    }
    env = hanalearn.HanabiEnv(game_params, max_len, False)
    deck = []
    for card in game["deck"]:
        hle_card = hanalearn.HanabiCardValue(card.color - 1, card.value - 1)
        deck.append(hle_card)

    card_in_hands = []
    for hand in game["hands"]:
        for card in hand:
            hle_card = hanalearn.HanabiCardValue(card.color - 1, card.value - 1)
            card_in_hands.append(hle_card)
    deck = deck + list(reversed(card_in_hands))
    env.reset_with_deck(deck)
    hle_game["deck"] = deck

    moves = []
    cur_player = 0
    for move in game["moves"]:
        assert move.player_id == cur_player
        target_offset = -1
        if move.type == "hintValue":
            target_offset = (move.target_player - cur_player + num_player) % num_player
            move = hanalearn.HanabiMove(
                hanalearn.MoveType.RevealRank, -1, target_offset, -1, move.value - 1
            )
        elif move.type == "hintColor":
            target_offset = (move.target_player - cur_player + num_player) % num_player
            move = hanalearn.HanabiMove(
                hanalearn.MoveType.RevealColor, -1, target_offset, move.value - 1, -1
            )
        elif move.type == "playCard":
            move = hanalearn.HanabiMove(hanalearn.MoveType.Play, move.value, -1, -1, -1)
        elif move.type == "discardCard":
            move = hanalearn.HanabiMove(
                hanalearn.MoveType.Discard, move.value, -1, -1, -1
            )
        else:
            logging.error(f"Unknown or illegal move in {debug_label}: {move}")
            return False
        moves.append(move)
        cur_player = (cur_player + 1) % num_player

    hle_game["moves"] = moves
    for move in moves:
        if f is not None:
            f(env, move)
        env.step(move)

    return env, hle_game


def process_game(game, max_len):
    env, hle_game = run_game(game, max_len)
    hle_game["score"] = env.get_score()
    if env.get_score() == int(game["score"]):
        return True, hle_game
    else:
        logging.error(
            f"Score mismatch {env.get_score()}(re-run), {game['score']}(from data)"
        )
        return False, None


def process_pkl(pkl, num_players, max_len, bga):
    idx = int(pkl.split(".")[-2])
    print(f"process {idx}, {pkl}")
    games = pickle.load(open(pkl, "rb"))
    filtered_games = []
    mismatch = []
    game_ids = []
    for i, game in enumerate(games):
        if bga and i in BGA_BAD_GAMES_PER_PKL.get(idx, []):
            continue
        if not filter_game(game, num_players):
            continue
        if len(game["moves"]) > max_len:
            logging.error(f"game too long {len(game['moves'])}")
            continue
        if len(game["moves"]) == 0:
            logging.error(f"game too short {len(game['moves'])}")
            continue
        score_match, hle_game = process_game(game, max_len)
        if score_match:
            filtered_games.append(hle_game)
            game_ids.append(game["game_id"])
        else:
            mismatch.append(game["game_id"])
    print("mismatch", mismatch)
    return filtered_games, game_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate supervised learning dataset")
    parser.add_argument("--bga", type=int, default=1)
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--dev", type=int, default=0)
    args = parser.parse_args()

    name = f"games_player{args.num_players}"
    if args.dev:
        name += "dev"

    if args.bga:
        src = "/checkpoint/hengyuan/hanabi_human_data"
        dest = f"/private/home/{getpass.getuser()}/scratch/bga_data/{name}.pkl"
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        pkl_files = common_utils.get_all_files(src, ".pkl", "games")
        games = []
        for pkl in pkl_files:
            games.extend(process_pkl(pkl, args.num_players, args.max_len, True)[0])
            if args.dev:
                break

        print(f"dumping {len(games)} games to {dest}")
        pickle.dump(games, open(dest, "wb"))
    else:
        pkl_file = "/checkpoint/dwu/hanabi/hanabi-live-data/games.0.pkl"
        games, game_ids = process_pkl(pkl_file, args.num_players, args.max_len, False)
        dest = f"/private/home/{getpass.getuser()}/scratch/hanabi_live_data/{name}.pkl"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"dumping {len(games)} games to {dest}")
        pickle.dump(games, open(dest, "wb"))

        info = csv.reader(
            open("/checkpoint/dwu/hanabi/hanabi-live-data/games.csv", "r")
        )
        game_ids = set(game_ids)
        seeds = set()
        for i, row in enumerate(info):
            if i == 0:
                continue
            if int(row[0]) in game_ids:
                seeds.add(row[5])
        print(f"num of unique seeds: {len(seeds)}")
