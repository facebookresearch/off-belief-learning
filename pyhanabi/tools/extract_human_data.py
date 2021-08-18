# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import json
import sys
import pprint
import glob
from recordclass import recordclass, asdict
import tarfile
import os
import time
import pickle
from datetime import datetime as dt
import traceback


def log(msg):
    print("{}     {}".format(dt.now().strftime("%Y-%m-%d %H:%M:%S.%f"), msg))
    sys.stdout.flush()


Move = recordclass(
    "Move",
    ["player_name", "player_id", "type", "value", "target_player"],
)
Card = recordclass("Card", ["id", "color", "value"])
Game = recordclass(
    "Game",
    ["game_id", "num_players", "hands", "deck", "moves", "score", "abandoned"],
)
Game.__repr__ = lambda self: pprint.pformat(asdict(self))


class QuietError(RuntimeError):
    pass


def json_to_card(j):
    return Card(int(j["id"]), int(j["type"]), int(j["type_arg"]))


def listdir(path_or_tar):
    if isinstance(path_or_tar, tarfile.TarFile):
        return [member.name for member in path_or_tar]
    elif isinstance(path_or_tar, str):
        return os.listdir(path_or_tar)


def open_file(path_or_tar, fname):
    if isinstance(path_or_tar, tarfile.TarFile):
        return path_or_tar.extractfile(fname)
    elif isinstance(path_or_tar, str):
        return open(f"{path_or_tar}/{fname}")


def extract_game(path, game_id):
    tic = time.perf_counter()

    players = None
    moves = []
    deck = []
    init_hands = {}

    htmls = []
    game_log = None
    for fname in listdir(path):
        if fname.endswith(".html"):
            htmls.append(fname)
        elif fname.endswith(".log"):
            game_log = fname

    num_players = len(htmls)

    for fname in htmls:
        suc = False
        for line in open_file(path, fname):
            if isinstance(line, bytes):
                try:
                    line = line.decode(encoding="utf-8")
                except:
                    continue

            if "gameui.completesetup" not in line:
                continue
            suc = True
            # EVIL EVIL EVIL
            begin = line.find("{")
            assert begin != -1, f"Bad begin: {line}"
            end = line.rfind(",")

            data_str = line[begin:end]

            t0 = time.perf_counter()
            data = json.loads("[" + data_str + "]")[0]
            # pprint.pprint(data)
            if players is None:
                players = [str(p) for p in data["playerorder"]]
                player_names = {str(p): data["players"][p]["name"] for p in players}
            colors = data["colors"]
            deck = [
                json_to_card(data["deck"][str(k)])
                for k in sorted([int(i) for i in data["deck"]])
            ]
            for player_idx, player in enumerate(players):
                hand_json = data["hand" + str(player)]
                hand = [
                    json_to_card(hand_json[str(k)])
                    for k in sorted([int(i) for i in hand_json])
                ]
                if hand[0].value != 6:  # not my hand
                    init_hands[player_names[player]] = hand
        assert suc

    all_cards = deck + [c for h in init_hands.values() for c in h]
    max_color = max(c.color for c in all_cards)
    if max_color != 5:
        # multicolor game, don't record it
        return None
    assert max(c.value for c in all_cards) == 5, "Invalid value for card"

    # with open_file(path, game_log) as f:
    j = json.load(open_file(path, game_log))
    abandoned = False
    score = None
    initial_time = None
    for t in j["data"]["data"]:
        if initial_time is None:
            initial_time = int(t["time"])
        for e in t["data"]:
            type_, args = e["type"], e["args"]
            if type_ == "giveValue":
                moves.append(
                    Move(
                        args["player_name"],
                        args["player_id"],
                        "hintValue",
                        int(args["value"]),
                        args["target_name"],
                        int(t["time"]) - initial_time,
                    )
                )
            elif type_ == "giveColor":
                moves.append(
                    Move(
                        args["player_name"],
                        args["player_id"],
                        "hintColor",
                        int(args["color"]),
                        args["target_name"],
                        int(t["time"]) - initial_time,
                    )
                )
            elif type_ == "playCard":
                moves.append(
                    Move(
                        args["player_name"],
                        args["player_id"],
                        "playCard",
                        int(args["card_id"]),
                        -1,
                        int(t["time"]) - initial_time,
                    )
                )
            elif type_ == "discardCard":
                moves.append(
                    Move(
                        args["player_name"],
                        args["player_id"],
                        "discardCard",
                        int(args["card_id"]),
                        -1,
                        int(t["time"]) - initial_time,
                    )
                )
            elif type_ == "missCard":
                # this is a failed play?
                moves.append(
                    Move(
                        args["player_name"],
                        args["player_id"],
                        "playCard",
                        int(args["card_id"]),
                        -1,
                        int(t["time"]) - initial_time,
                    )
                )
            elif type_ == "newScores":
                # print()
                score = list(args["newScores"].values())[0]
            if (
                type_ == "gameStateChange"
                and "name" in args
                and args["name"] == "gameEnd"
            ):
                if args["args"].get("abandon_forced_by_metasite", False):
                    abandoned = True
            if type_ == "simpleNote" and "choosed to abandon" in e["log"]:
                abandoned = True

    if abandoned and score is None:
        score = 0

    if score is None:
        raise QuietError("No score!")

    # sys.exit()# fixup player name -> id
    if len(moves) < num_players:
        return None

    player_order = [m.player_name for m in moves[:num_players]]
    for move in moves:
        player_name = move.player_name
        move.player_name = move.player_id  # In case names are not unique identifiers
        move.player_id = player_order.index(player_name)
        if move.target_player != -1:
            move.target_player = player_order.index(move.target_player)
    init_hands = [init_hands[name] for name in player_order]

    # now simulate the game for some sanity checking and placing the cards
    sim_hands = [[copy.copy(c) for c in hand] for hand in init_hands]
    sim_deck = [copy.copy(c) for c in deck]
    for move_idx, move in enumerate(moves):
        assert (
            move.player_id == move_idx % num_players
        ), f"{move.player_id} != {move_idx} % {num_players}"
        if move.type == "playCard" or move.type == "discardCard":
            hand = sim_hands[move.player_id]
            suc = False
            for card_idx, card in enumerate(hand):
                if card.id == move.value:
                    suc = True
                    move.value = card_idx
                    hand.pop(card_idx)
                    if sim_deck:
                        hand.append(sim_deck.pop())
                    break

            if not suc:
                raise QuietError(f"Invalid move: {(move, sim_hands[move.player_id])}")

    # print("Players")
    # pprint.pprint(player_order)

    # print("Init hands")
    # pprint.pprint(init_hands)

    # print("Deck")
    # pprint.pprint(deck)

    # print("Moves")
    # pprint.pprint(moves)

    game = Game(game_id, num_players, init_hands, deck, moves, score, abandoned)
    return game


if __name__ == "__main__":
    inf, outf = sys.argv[1:]
    # important: you can't operate on the entire tarfile,
    # because it has to seek to each file, which gets slower
    # and slower as you go along!
    tic = time.perf_counter()
    valid, total = 0, 0
    outsize = 10000
    file_idx = 0
    os.makedirs(outf, exist_ok=True)
    games = []
    for dirpath, dirnames, filenames in os.walk(inf):
        for fname in filenames:
            if not fname.endswith(".tgz"):
                continue
            total += 1

            game_id = os.path.basename(fname).split(".")[0]
            full_fname = os.path.join(dirpath, fname)
            try:
                with tarfile.open(full_fname) as game_tar:
                    g = extract_game(game_tar, game_id)
            except KeyboardInterrupt:
                raise
            except QuietError as e:
                log(f"Error: {e}")
                g = None
            except:
                tp, e, tb = sys.exc_info()
                log(f"Caught unknown error for {game_id}: {e}")
                g = None
                traceback.print_tb(tb)

            if g is not None:
                valid += 1
                delta = time.perf_counter() - tic
                log(
                    f"{valid:5d} : Processed {game_id} , {g.num_players} players ; "
                    f"score {g.score:2} ; abandoned {g.abandoned} ; {delta:.3f} s ;  "
                    f"{valid/total*100:.0f}% valid "
                )
                tic = time.perf_counter()
                games.append(g)

            assert len(games) <= outsize
            if len(games) == outsize:
                outfn = f"{outf}/games.{file_idx}.pkl"
                log(f"Dumping {len(games)} games to {outfn}")
                pickle.dump(games, open(outfn, "wb"))
                file_idx += 1
                games = []
