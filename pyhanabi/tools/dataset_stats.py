# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import pickle
import random
import json
from collections import defaultdict
import numpy as np

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from common_utils import get_all_files, set_all_seeds
from extract_human_data import Game, Move, Card
import generate_clonebot_dataset as gcd


def get_all_games(pkls, num_player):
    games = []
    for pkl in pkls:
        print(f"processing {pkl}")
        pkl_idx = int(pkl.split(".")[-2])
        data = pickle.load(open(pkl, "rb"))
        for i, g in enumerate(data):
            if i in gcd.BAD_GAMES_PER_PKL.get(pkl_idx, []):
                continue
            if g["num_players"] != num_player or len(g["hands"][0]) != 5:
                continue
            # if g["abandoned"]:
            #     continue
            player_ids = list(set([m.player_name for m in g["moves"]]))
            if len(player_ids) != num_player:
                continue
            if not gcd.filter_game(g):
                continue
            games.append(g)

    return games


def aggregate_players(games):
    players = defaultdict(list)
    for g in games:
        player_ids = list(set([m.player_name for m in g["moves"]]))
        for pid in player_ids:
            players[pid].append(g)
    return players


def split_dataset(train_percent, player_games, seed, output):
    random.seed(seed)

    train = []
    test = []
    train_count = 0
    test_count = 0

    for player, games in player_games.items():
        rand = random.random()
        if rand < train_percent:
            train.append(player)
            train_count += len(games)
        else:
            test.append(player)
            test_count += len(games)

    print("actual train percent: ", train_count / (train_count + test_count))
    train_output = os.path.join(output, f"split{train_percent}_seed{seed}_train.json")
    test_output = os.path.join(output, f"split{train_percent}_seed{seed}_test.json")
    json.dump(train, open(train_output, "w"))
    json.dump(test, open(test_output, "w"))


def get_player_avg_score(player_games, filter_players):
    player_scores = {}
    for player, games in player_games.items():
        if filter_players is not None and player not in filter_players:
            continue
        scores = [int(g["score"]) for g in games]
        abandoned = [int(g["abandoned"]) for g in games]
        abandon_rate = np.sum(abandoned) / len(abandoned)
        player_scores[player] = (np.mean(scores), len(scores), abandon_rate)
    listed = ((k, v) for k, v in player_scores.items())
    sorted_list = sorted(listed, key=lambda x: -x[1][0])
    return sorted_list


def filter_min_game(player_scores, min_num):
    filtered = []
    for p, (score, num, abandon_rate) in player_scores:
        if num >= min_num:
            filtered.append((p, (score, num, abandon_rate)))
    return filtered


def get_mean_score(player_scores):
    total = 0
    count = 0
    for _, (score, num, _) in player_scores:
        total += score * num
        count += num
    return total / count, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--train_percent", default=0.8, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--use_split", default=None, type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--min_game", default=100, type=int)

    args = parser.parse_args()

    src = "/scratch/hengyuan/hanabi_human_data"
    pkls = get_all_files(src, "pkl", "games")
    pkls = sorted(pkls, key=lambda x: int(x.split(".")[-2]))
    games = get_all_games(pkls, 2)
    print(f"num_game {len(games)}")

    players = aggregate_players(games)

    if args.split:
        split_dataset(args.train_percent, players, args.seed, src)
        exit()

    if args.use_split is not None:
        filter_players = json.load(open(args.use_split, "r"))
        filter_players = set(filter_players)
    else:
        filter_players = None
    player_scores = get_player_avg_score(players, filter_players)

    filtered_player_scores = filter_min_game(player_scores, args.min_game)

    # for score in range(23, 9, -1):
    #     players = [p for p in filtered_player_scores if p[1][0] >= score and p[1][0] < score + 1]
    #     output = os.path.join(src, f'{args.prefix}_min_game{min_game}_score{score}.pkl')
    #     print(f"dumping to {output}")
    #     pickle.dump(players, open(output, 'wb'))

    # k = 100
    # topk = filtered_player_scores[: k]
    # midk = filtered_player_scores[k:2*k]
    # botk = filtered_player_scores[2*k:3*k]

    # pickle.dump(topk, open(os.path.join(src, f'min_game{min_game}_top{k}.pkl'), 'wb'))
    # pickle.dump(midk, open(os.path.join(src, f'min_game{min_game}_mid{k}.pkl'), 'wb'))
    # pickle.dump(botk, open(os.path.join(src, f'min_game{min_game}_bot{k}.pkl'), 'wb'))
