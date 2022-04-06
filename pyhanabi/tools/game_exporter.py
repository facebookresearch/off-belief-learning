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
import json


def export_game(deck, moves):
    deck = deck[::-1]
    num_players = 2
    game_json = {}
    game_json["players"] = [f"p{i}" for i in range(num_players)]
    game_json["deck"] = []
    for card in deck:
        game_json["deck"].append(dict(rank=card.rank()+1, suit=card.color()))

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
    for i, move in enumerate(moves):
        player = i % 2
        if move.move_type() == hanalearn.MoveType.Play:
            game_json["actions"].append(
                dict(type=0, target=hands[player][move.card_index()])
            )
            del hands[player][move.card_index()]
            if deck_cursor < max_cards:
                hands[player].append(deck_cursor)
                deck_cursor += 1
        elif move.move_type() == hanalearn.MoveType.Discard:
            game_json["actions"].append(
                dict(type=1, target=hands[player][move.card_index()])
            )
            del hands[player][move.card_index()]
            if deck_cursor < max_cards:
                hands[player].append(deck_cursor)
                deck_cursor += 1
        elif move.move_type() == hanalearn.MoveType.RevealColor:
            game_json["actions"].append(
                dict(type=2, target=1 - player, value=move.color())
            )
        elif move.move_type() == hanalearn.MoveType.RevealRank:
            game_json["actions"].append(
                dict(type=3, target=1 - player, value=move.rank()+1)
            )
        else:
            assert False
    game_json["options"] = dict(variant="No Variant")
    return game_json
