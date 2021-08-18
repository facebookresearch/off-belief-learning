# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
from .extract_human_data import Game, Move, Card

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
    return color_codes[card.color - 1] + str(card.value) + resetcolor


def _print_current_state(hands, piles, discards):
    for i in range(len(color_codes)):
        print(color_codes[i], end="")
        print(piles[i + 1], end="")
        print(resetcolor, end="")
    print("|", end="")

    for hand in hands:
        for card in hand:
            print(_card_to_str(card), end="")
        print("|", end="")

    for card in discards:
        print(_card_to_str(card), end="")
    print("")


def _print_move(hands, move):
    if move.type == "hintColor":
        print(
            f"{move.player_id} hints {move.target_player} {color_codes[move.value-1]}{color_chars[move.value-1]}{resetcolor}"
        )
    elif move.type == "hintValue":
        print(f"{move.player_id} hints {move.target_player} {move.value}")
    elif move.type == "playCard":
        print(
            f"{move.player_id} plays position {move.value}, card {_card_to_str(hands[move.player_id][move.value])}"
        )
    elif move.type == "discardCard":
        print(
            f"{move.player_id} discards position {move.value}, card {_card_to_str(hands[move.player_id][move.value])}"
        )


# Prints out a pickled bgg game, for debugging purposes
def print_pickled_bgg_game(game):
    num_players = game["num_players"]
    game_parameters = {
        "players": str(num_players),
        "random_start_player": "0",
        "bomb": "0",
    }
    deck = game["deck"][:]
    hands = [hand[:] for hand in game["hands"]]
    piles = defaultdict(int)
    discards = []

    for move in game["moves"]:
        _print_current_state(hands, piles, discards)
        _print_move(hands, move)

        must_deal = False
        if move.type == "hintValue":
            pass
        elif move.type == "hintColor":
            pass
        elif move.type == "playCard":
            card = hands[move.player_id].pop(move.value)
            if piles[card.color] == card.value - 1:
                piles[card.color] = card.value
            else:
                discards.append(card)
            must_deal = True
        elif move.type == "discardCard":
            card = hands[move.player_id].pop(move.value)
            discards.append(card)
            must_deal = True
        else:
            print("Error: ", move)
            break

        if must_deal and deck:
            card = deck.pop()
            hands[move.player_id].append(card)

    _print_current_state(hands, piles, discards)
