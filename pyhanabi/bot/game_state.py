import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from set_path import append_sys_path
append_sys_path()

import numpy as np
import torch
import rela
import hanalearn as hle


class ACTION:
    PLAY = 0
    DISCARD = 1
    COLOR_HINT = 2
    RANK_HINT = 3


class Card:
    def __init__(self, color, rank, order):
        if rank != -1:
            rank -= 1
        assert self.check_card(color, rank)
        self.hle_card = hle.HanabiCard(color, rank, order)
        self.order = order

    def __repr__(self):
        return '%s%s' % ('RYGBPU'[self.color], '12345U'[self.rank])

    @property
    def color(self):
        return self.hle_card.color()

    @property
    def rank(self):
        return self.hle_card.rank()

    @staticmethod
    def check_card(color, rank):
        if color == -1 and rank == -1:
            return True
        if color >= 0 and color < 5 and rank >= 0 and rank < 5:
            return True

    def is_valid(self):
        return self.hle_card.is_valid()


class Hand:
    def __init__(self, hand_size, num_color, num_rank):
        self.hand_size = hand_size
        self.num_color = num_color
        self.num_rank = num_rank

        self.hle_hand = hle.HanabiHand()
        self.cards = []

    def __len__(self):
        return len(self.cards)

    def get_with_order(self, order):
        for card in self.cards:
            if card.order == order:
                return card
        return None

    def add_card(self, card):
        assert len(self.cards) < self.hand_size
        self.cards.append(card)
        self.hle_hand.add_card(
            card.hle_card, hle.CardKnowledge(self.num_color, self.num_rank)
        )

    def remove_from_hand(self, order):
        to_remove = -1
        for i, card in enumerate(self.cards):
            if card.order == order:
                assert to_remove == -1
                to_remove = i
        assert to_remove >= 0 and to_remove < len(self.cards)
        card = self.cards.pop(to_remove)
        self.hle_hand.remove_from_hand(to_remove, [])
        return card, to_remove


class HleGameState:
    def __init__(self, players, my_name, start_player, hide_action, verbose):
        self.num_player = len(players)
        self.players = players
        self.start_player = start_player
        self.my_index = [i for i, name in enumerate(players) if name == my_name][0]

        self.hle_game = hle.HanabiGame({'players': str(len(players))})
        self.hle_state = hle.HanabiState(self.hle_game, start_player)
        self.hide_action = hide_action

        self.hands = [
            Hand(
                self.hle_game.hand_size(),
                self.hle_game.num_colors(),
                self.hle_game.num_ranks(),
            )
            for _ in range(self.num_player)
        ]
        self.encoder = hle.ObservationEncoder(self.hle_game)

        self.verbose = verbose
        if self.verbose:
            print("Create state for %d player game" % self.num_player)
            print('init hle game state done')

    ############### public functions for draw and play/discard/hint

    def draw(self, player, color, rank, order):
        if self.verbose:
            print(
                'player: %d draw a card: %s/%s at order: %s'
                % (player, color, rank, order)
            )
            print('before draw card: hand len is: ', len(self.hands[player]))

        if self.hle_state.is_terminal():
            print("game has ended, skip draw")
            return

        self.hands[player].add_card(Card(color, rank, order))
        move = hle.HanabiMove(hle.MoveType.Deal, -1, -1, color, rank-1)

        self.hle_state.apply_move(move)

    def play(self, seat, color, rank, order, success):
        card = Card(color, rank, order)
        if self.verbose:
            print(
                'player %s try to play [%s] and %s'
                % (seat, card, 'success' if success else 'fail')
            )
        removed, card_idx = self.hands[seat].remove_from_hand(order)
        assert removed.order == order

        move = hle.HanabiMove(hle.MoveType.Play, card_idx, -1, -1, -1)
        self.hle_state.apply_move(move)

    def discard(self, seat, color, rank, order):
        card = Card(color, rank, order)
        removed, card_idx = self.hands[seat].remove_from_hand(order)
        assert removed.order == order

        move = hle.HanabiMove(hle.MoveType.Discard, card_idx, -1, -1, -1)
        self.hle_state.apply_move(move)

    def hint(self, giver, target, hint_type, hint_value, hinted_card_orders):
        assert giver != target

        target_offset = (target - giver) % self.num_player
        hint_type = ['color_hint', 'rank_hint'][hint_type]
        if hint_type == 'rank_hint':
            hint_value -= 1
            move = hle.HanabiMove(hle.MoveType.RevealRank, -1, target_offset, -1, hint_value)
        else:
            move = hle.HanabiMove(hle.MoveType.RevealColor, -1, target_offset, hint_value, -1)

        self.hle_state.apply_move(move)

    def _get_observation_and_legal_move(self):
        obs = hle.HanabiObservation(self.hle_state, self.my_index, False)
        obs_vec = self.encoder.encode(obs, False, [], False, [], [], self.hide_action)
        legal_moves = obs.legal_moves()
        legal_move_vec = self._legal_moves_to_vector(legal_moves)
        return obs_vec, legal_move_vec

    def _legal_moves_to_vector(self, legal_moves):
        vec = [0 for _ in range(self.hle_game.max_moves() + 1)]
        if self.is_my_turn():
            for m in legal_moves:
                uid = self.hle_game.get_move_uid(m)
                vec[uid] = 1
        else:
            assert len(legal_moves) == 0
            vec[-1] = 1
        return vec

    def observe(self):
        obs_vec, legal_move_vec = self._get_observation_and_legal_move()

        obs = torch.tensor(obs_vec, dtype=torch.float32)
        priv_s = obs[125:].unsqueeze(0)
        publ_s = obs[125 * self.num_player:].unsqueeze(0)

        legal_move = torch.tensor(legal_move_vec, dtype=torch.float32)

        return priv_s, publ_s, legal_move

    def convert_move(self, hle_move):
        type_map = {
            hle.MoveType.Play: ACTION.PLAY,
            hle.MoveType.Discard: ACTION.DISCARD,
            hle.MoveType.RevealColor: ACTION.COLOR_HINT,
            hle.MoveType.RevealRank: ACTION.RANK_HINT,
        }

        if hle_move.move_type() in [hle.MoveType.Play, hle.MoveType.Discard]:
            card_idx = hle_move.card_index()
            assert card_idx >= 0 and card_idx < len(self.hands[self.my_index])
            card_order = self.hands[self.my_index].cards[card_idx].order
            return {
                'type': type_map[hle_move.move_type()],
                'target': card_order,
            }

        target_idx = (self.my_index + hle_move.target_offset()) % self.num_player
        if hle_move.move_type() == hle.MoveType.RevealColor:
            value = hle_move.color()
            assert value >= 0 and value < self.hle_game.num_colors()
        else:
            value = hle_move.rank()
            assert value >= 0 and value < self.hle_game.num_ranks()
            value += 1  # hanabi-live is 1 indexed for rank

        return {
            'type': type_map[hle_move.move_type()],
            'target': target_idx,
            'value': value,
        }

    def is_my_turn(self):
        return self.hle_state.cur_player() == self.my_index

    def get_score(self):
        return self.hle_state.score()

    @property
    def hint_tokens(self):
        return self.hle_state.info_tokens()


# debug utils
def print_observation(vec):
    def print_hand(hand):
        for i in range(5):
            card = hand[i * 25 : (i + 1) * 25]
            print('card: %d, sum %d, argmax %d' % (i, sum(card), np.argmax(card)))

    def print_knowledge(kn):
        for i in range(5):
            card = kn[i*35 : i*35+25]
            color = kn[i*35+25:i*35+30]
            rank = kn[i*35+30:i*35+35]
            # print('card: %d, sum %d, argmax %d', (i, sum(card), np.argmax(card)))
            print('*****')
            card_print = []
            for i in range(5):
                c = ','.join([str(i)[:5] for i in card[5*i:5*(i+1)]])
                card_print.append(c)
            print(card_print)
            print(color)
            print(rank)

    print('--my hand--')
    print_hand(vec[:125])
    print('--your hand--')
    print_hand(vec[125:250])
    print('--hand info--')
    print(vec[250:252])
    print('--deck--: sum: %d' % sum(vec[252:292]))
    print([int(x) for x in vec[252:292]])
    print('--firework--: sum: %d' % sum(vec[292:317]))
    for i in range(5):
        print([int(x) for x in vec[292+5*i:292+5*(i+1)]])
    print('--info--: sum %d' % sum(vec[317:325]))
    print([int(x) for x in vec[317:325]])
    print('--life--: sum %d' % sum(vec[325:328]))
    print([int(x) for x in vec[325:328]])
    print('--discard--')
    for i in range(5):
        print([int(x) for x in vec[328+10*i:328+10*(i+1)]])
    print('--last action--')
    print(sum(vec[378:433]))
    print('--card knowledge--')
    print('--my knowledge--')
    print_knowledge(vec[433:433+175])
    print('--your knowledge--')
    print_knowledge(vec[433+175:])
