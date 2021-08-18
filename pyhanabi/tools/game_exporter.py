# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, List
import json
import set_path

set_path.append_sys_path()

import torch  # Needed for hanalearn import to work.
import rela  # Needed for hanalearn import to work.
import hanalearn


COLORS = "abcde"


def export_game(env: hanalearn.HanabiEnv, moves: List[int]) -> Dict:
    num_players = 2
    game_json = {}

    game_json["players"] = [f"p{i}" for i in range(num_players)]

    game_json["deck"] = []
    for rank, color in env.deck_history():
        game_json["deck"].append(dict(rank=int(rank), suitIndex=COLORS.index(color)))

    max_cards = len(game_json["deck"])
    # The index of the next card to pick.
    deck_cursor = 10
    # Player hands as indices in the deck.
    hands = {0: list(range(5)), 1: list(range(5, 10))}

    # Valid actions types are:
    # - 0 for a play
    # - 1 for a discard
    # - 2 for a color clue
    # - 3 for a number clue
    # - 4 for an end game
    # Hanabi env codes:
    # 0-4 is discard, 5-9 is play
    # 10-14 is color hint, 15-19 is rank hint
    assert num_players == 2
    game_json["actions"] = []
    for i, action in enumerate(moves):
        assert 0 <= action < 20, action
        player = i % 2
        if action < 10:
            action_type = 1 if action < 5 else 0
            game_json["actions"].append(
                dict(type=action_type, target=hands[player][action % 5])
            )
            del hands[player][action % 5]
            if deck_cursor < max_cards:
                hands[player].append(deck_cursor)
                deck_cursor += 1
        elif action < 15:
            game_json["actions"].append(
                dict(type=2, target=1 - player, value=action % 5)
            )
        elif action < 20:
            # Ranks start with 1, therefore + 1.
            game_json["actions"].append(
                dict(type=3, target=1 - player, value=action % 5 + 1)
            )
        else:
            assert False, f"Bad action: {action}"

    game_json["options"] = dict(variant="No Variant")

    return game_json


if __name__ == "__main__":
    import random

    seed = 0
    params = {
        "players": str(2),
        "seed": str(seed),
        "bomb": str(0),
    }
    env = hanalearn.HanabiEnv(
        params,
        [0],  # eps
        [],  # boltzmann_temperature
        -1,  # max len
        False,  # sad
        False,  # shuffle obs
        False,  # shuffle color
        True,  # hide action
        True,  # trinary
        False,  # verbose
    )
    move_history = []
    obs = env.reset()
    while not env.terminated():
        mask = obs["legal_move"][env.get_current_player()]
        moves = torch.arange(len(mask))[mask > 0.5].tolist()
        action = torch.tensor([20, 20])
        move_history.append(random.choice(moves))
        action[env.get_current_player()] = move_history[-1]
        obs, _, _ = env.step(dict(a=action))

    print(json.dumps(export_game(env, move_history)))
