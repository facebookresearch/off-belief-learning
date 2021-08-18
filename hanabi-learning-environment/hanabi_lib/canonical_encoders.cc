// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "canonical_encoders.h"

namespace hanabi_learning_env {

// namespace {

// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int>& shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<int>());
}

const HanabiHistoryItem* GetLastNonDealMove(
    const std::vector<HanabiHistoryItem>& past_moves) {
  auto it = std::find_if(
      past_moves.begin(), past_moves.end(), [](const HanabiHistoryItem& item) {
        return item.move.MoveType() != HanabiMove::Type::kDeal;
      });
  return it == past_moves.end() ? nullptr : &(*it);
}

int BitsPerCard(const HanabiGame& game) {
  return game.NumColors() * game.NumRanks();
}

// The card's one-hot index using a color-major ordering.
int CardIndex(int color,
              int rank,
              int num_ranks,
              bool shuffle_color,
              const std::vector<int>& color_permute) {
  if (shuffle_color) {
    // std::cout << "mapping: " << color << " to " << color_permute[color] << std::endl;
    color = color_permute[color];
  }
  return color * num_ranks + rank;
}

int HandsSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() * BitsPerCard(game) +
         game.NumPlayers();
}

// Enocdes cards in all other player's hands (excluding our unknown hand),
// and whether the hand is missing a card for all players (when deck is empty.)
// Each card in a hand is encoded with a one-hot representation using
// <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
// Returns the number of entries written to the encoding.
int EncodeHands(const HanabiGame& game,
                const HanabiObservation& obs,
                int start_offset,
                bool show_own_cards,
                const std::vector<int>& order,
                bool shuffle_color,
                const std::vector<int>& color_permute,
                std::vector<float>* encoding) {
  int bits_per_card = BitsPerCard(game);
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiCard>& cards = hands[player].Cards();
    int num_cards = 0;

    // for (const HanabiCard& card : cards) {
    for (int i = 0; i < cards.size(); ++i) {
      int card_i = i;
      if (player != 0 && order.size() > 0) {
        card_i = order[i];
      }
      const auto& card = cards[card_i];
      // Only a player's own cards can be invalid/unobserved.
      // assert(card.IsValid());
      assert(card.Color() < game.NumColors());
      assert(card.Rank() < num_ranks);
      if (player == 0) {
        if (show_own_cards) {
          assert(card.IsValid());
          // std::cout << offset << CardIndex(card.Color(), card.Rank(), num_ranks) << std::endl;
          // std::cout << card.Color() << ", " << card.Rank() << ", " << num_ranks << std::endl;
          auto card_idx = CardIndex(
              card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
          (*encoding).at(offset + card_idx) = 1;
        } else {
          assert(!card.IsValid());
          // (*encoding).at(offset + CardIndex(card.Color(), card.Rank(), num_ranks)) = 0;
        }
      } else {
        assert(card.IsValid());
        auto card_idx = CardIndex(
            card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
        (*encoding).at(offset + card_idx) = 1;
      }

      ++num_cards;
      offset += bits_per_card;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset += (hand_size - num_cards) * bits_per_card;
    }
  }

  // For each player, set a bit if their hand is missing a card.
  for (int player = 0; player < num_players; ++player) {
    if (hands[player].Cards().size() < game.HandSize()) {
      (*encoding)[offset + player] = 1;
    }
  }
  offset += num_players;

  assert(offset - start_offset == HandsSectionLength(game));
  return offset - start_offset;
}

int BoardSectionLength(const HanabiGame& game) {
  return game.MaxDeckSize() - game.NumPlayers() * game.HandSize() +  // deck
         game.NumColors() * game.NumRanks() +  // fireworks
         game.MaxInformationTokens() +         // info tokens
         game.MaxLifeTokens();                 // life tokens
}

// Encode the board, including:
//   - remaining deck size
//     (max_deck_size - num_players * hand_size bits; thermometer)
//   - state of the fireworks (<num_ranks> bits per color; one-hot)
//   - information tokens remaining (max_information_tokens bits; thermometer)
//   - life tokens remaining (max_life_tokens bits; thermometer)
// We note several features use a thermometer representation instead of one-hot.
// For example, life tokens could be: 000 (0), 100 (1), 110 (2), 111 (3).
// Returns the number of entries written to the encoding.
int EncodeBoard(const HanabiGame& game,
                const HanabiObservation& obs,
                int start_offset,
                bool shuffle_color,
                // const std::vector<int>& color_permute,
                const std::vector<int>& inv_color_permute,
                std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();
  int max_deck_size = game.MaxDeckSize();

  int offset = start_offset;
  // Encode the deck size
  for (int i = 0; i < obs.DeckSize(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  // std::cout << "max_deck_size: " << max_deck_size
  //           << ", deck_size: " << obs.DeckSize() << std::endl;
  offset += (max_deck_size - hand_size * num_players);  // 40 in normal 2P game

  // fireworks
  // assert(false);
  const std::vector<int>& fireworks = obs.Fireworks();
  // std::cout << "normal order:" << std::endl;
  // for (auto q : fireworks) {
  //   std::cout << q << ", ";
  // }
  // std::cout << std::endl;

  // std::cout << "actual" << ", shuffle? " << shuffle_color << std::endl;
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    int color = c;
    if (shuffle_color) {
      color = inv_color_permute[c];
    }

    if (fireworks[color] > 0) {
      (*encoding)[offset + fireworks[color] - 1] = 1;
    }
    // std::cout << fireworks[color] << ", ";
    offset += num_ranks;
  }
  // std::cout << std::endl;
  // std::cout << "Order: " << std::endl;
  // for (auto q : color_permute) {
  //   std::cout << q << ", ";
  // }
  // std::cout << std::endl;

  // info tokens
  assert(obs.InformationTokens() >= 0);
  assert(obs.InformationTokens() <= game.MaxInformationTokens());
  for (int i = 0; i < obs.InformationTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxInformationTokens();

  // life tokens
  assert(obs.LifeTokens() >= 0);
  assert(obs.LifeTokens() <= game.MaxLifeTokens());
  for (int i = 0; i < obs.LifeTokens(); ++i) {
    (*encoding)[offset + i] = 1;
  }
  offset += game.MaxLifeTokens();

  assert(offset - start_offset == BoardSectionLength(game));
  return offset - start_offset;
}

int DiscardSectionLength(const HanabiGame& game) { return game.MaxDeckSize(); }

// Encode the discard pile. (max_deck_size bits)
// Encoding is in color-major ordering, as in kColorStr ("RYGWB"), with each
// color and rank using a thermometer to represent the number of cards
// discarded. For example, in a standard game, there are 3 cards of lowest rank
// (1), 1 card of highest rank (5), 2 of all else. So each color would be
// ordered like so:
//
//   LLL      H
//   1100011101
//
// This means for this color:
//   - 2 cards of the lowest rank have been discarded
//   - none of the second lowest rank have been discarded
//   - both of the third lowest rank have been discarded
//   - one of the second highest rank have been discarded
//   - the highest rank card has been discarded
// Returns the number of entries written to the encoding.
int EncodeDiscards(const HanabiGame& game,
                   const HanabiObservation& obs,
                   int start_offset,
                   bool shuffle_color,
                   const std::vector<int>& color_permute,
                   std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  int offset = start_offset;
  std::vector<int> discard_counts(num_colors * num_ranks, 0);
  for (const HanabiCard& card : obs.DiscardPile()) {
    ++discard_counts[CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute)];
  }

  for (int c = 0; c < num_colors; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      // "discard_counts" has been permuted, and the order of discard is fixed as in the pile
      int num_discarded = discard_counts[CardIndex(c, r, num_ranks, false, color_permute)];
      for (int i = 0; i < num_discarded; ++i) {
        (*encoding)[offset + i] = 1;
      }
      offset += game.NumberCardInstances(c, r);
    }
  }

  assert(offset - start_offset == DiscardSectionLength(game));
  return offset - start_offset;
}

// Encode the last player action (not chance's deal of cards). This encodes:
//  - Acting player index, relative to ourself (<num_players> bits; one-hot)
//  - The MoveType (4 bits; one-hot)
//  - Target player index, relative to acting player, if a reveal move
//    (<num_players> bits; one-hot)
//  - Color revealed, if a reveal color move (<num_colors> bits; one-hot)
//  - Rank revealed, if a reveal rank move (<num_ranks> bits; one-hot)
//  - Reveal outcome (<hand_size> bits; each bit is 1 if the card was hinted at)
//  - Position played/discarded (<hand_size> bits; one-hot)
//  - Card played/discarded (<num_colors> * <num_ranks> bits; one-hot)
// Returns the number of entries written to the encoding.
int EncodeLastAction_(const HanabiGame& game,
                      const HanabiObservation& obs,
                      int start_offset,
                      const std::vector<int>& order,
                      bool shuffle_color,
                      const std::vector<int>& color_permute,
                      std::vector<float>* encoding) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const HanabiHistoryItem* last_move = GetLastNonDealMove(obs.LastMoves());
  if (last_move == nullptr) {
    offset += LastActionSectionLength(game);
  } else {
    HanabiMove::Type last_move_type = last_move->move.MoveType();

    // player_id
    // Note: no assertion here. At a terminal state, the last player could have
    // been me (player id 0).
    (*encoding)[offset + last_move->player] = 1;
    offset += num_players;

    // move type
    switch (last_move_type) {
      case HanabiMove::Type::kPlay:
        (*encoding)[offset] = 1;
        break;
      case HanabiMove::Type::kDiscard:
        (*encoding)[offset + 1] = 1;
        break;
      case HanabiMove::Type::kRevealColor:
        (*encoding)[offset + 2] = 1;
        break;
      case HanabiMove::Type::kRevealRank:
        (*encoding)[offset + 3] = 1;
        break;
      default:
        std::abort();
    }
    offset += 4;

    // target player (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      int8_t observer_relative_target =
          (last_move->player + last_move->move.TargetOffset()) % num_players;
      (*encoding)[offset + observer_relative_target] = 1;
    }
    offset += num_players;

    // color (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealColor) {
      int color = last_move->move.Color();
      if (shuffle_color) {
        color = color_permute[color];
      }
      (*encoding)[offset + color] = 1;
    }
    offset += num_colors;

    // rank (if hint action)
    if (last_move_type == HanabiMove::Type::kRevealRank) {
      (*encoding)[offset + last_move->move.Rank()] = 1;
    }
    offset += num_ranks;

    if (!order.empty()) {
      // when there are 2 players, we do not need to permute outcome
      // as hinted cards are always our cards
      // if there are more than 1 players, then this may not be true
      // we have not implemented the case for >2 players, so assert here
      assert(num_players == 2);
    }
    // outcome (if hinted action)
    if (last_move_type == HanabiMove::Type::kRevealColor ||
        last_move_type == HanabiMove::Type::kRevealRank) {
      for (int i = 0, mask = 1; i < hand_size; ++i, mask <<= 1) {
        if ((last_move->reveal_bitmask & mask) > 0) {
          (*encoding)[offset + i] = 1;
        }
      }
    }
    offset += hand_size;

    // position (if play or discard action)
    // play & discard should always be permuted,
    // because it is always partner's action, revealing info on partner's hand
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      if (order.size() > 0) {
        // does nothing
        // std::cout << "hand idx" << hand_idx << std::endl;
        // hand_idx = order[hand_idx];
      } else {
        // in normal mode, tells you which card was played/discarded
        int hand_idx = last_move->move.CardIndex();
        (*encoding)[offset + hand_idx] = 1;
      }
    }
    offset += hand_size;

    // card (if play or discard action)
    if (last_move_type == HanabiMove::Type::kPlay ||
        last_move_type == HanabiMove::Type::kDiscard) {
      assert(last_move->color >= 0);
      assert(last_move->rank >= 0);
      int card_idx = CardIndex(
          last_move->color, last_move->rank, num_ranks, shuffle_color, color_permute);
      (*encoding)[offset + card_idx] = 1;
    }
    offset += BitsPerCard(game);

    // was successful and/or added information token (if play action)
    if (last_move_type == HanabiMove::Type::kPlay) {
      if (last_move->scored) {
        (*encoding)[offset] = 1;
      }
      if (last_move->information_token) {
        (*encoding)[offset + 1] = 1;
      }
    }
    offset += 2;
  }

  assert(offset - start_offset == LastActionSectionLength(game));
  return offset - start_offset;
}

int CardKnowledgeSectionLength(const HanabiGame& game) {
  return game.NumPlayers() * game.HandSize() *
         (BitsPerCard(game) + game.NumColors() + game.NumRanks());
}

// Encode the common card knowledge.
// For each card/position in each player's hand, including the observing player,
// encode the possible cards that could be in that position and whether the
// color and rank were directly revealed by a Reveal action. Possible card
// values are in color-major order, using <num_colors> * <num_ranks> bits per
// card. For example, if you knew nothing about a card, and a player revealed
// that is was green, the knowledge would be encoded as follows.
// R    Y    G    W    B
// 0000000000111110000000000   Only green cards are possible.
// 0    0    1    0    0       Card was revealed to be green.
// 00000                       Card rank was not revealed.
//
// Similarly, if the player revealed that one of your other cards was green, you
// would know that this card could not be green, resulting in:
// R    Y    G    W    B
// 1111111111000001111111111   Any card that is not green is possible.
// 0    0    0    0    0       Card color was not revealed.
// 00000                       Card rank was not revealed.
// Uses <num_players> * <hand_size> *
// (<num_colors> * <num_ranks> + <num_colors> + <num_ranks>) bits.
// Returns the number of entries written to the encoding.
int EncodeCardKnowledge(const HanabiGame& game,
                        const HanabiObservation& obs,
                        int start_offset,
                        const std::vector<int>& order,
                        bool shuffle_color,
                        const std::vector<int>& color_permute,
                        std::vector<float>* encoding) {
  int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  int offset = start_offset;
  const std::vector<HanabiHand>& hands = obs.Hands();
  assert(hands.size() == num_players);
  for (int player = 0; player < num_players; ++player) {
    const std::vector<HanabiHand::CardKnowledge>& knowledge =
        hands[player].Knowledge();
    int num_cards = 0;

    for (int i = 0; i < knowledge.size(); ++i) {
    // for (const HanabiHand::CardKnowledge& card_knowledge : knowledge) {
      int card_idx = i;
      if (player != 0 && order.size() > 0) {
        card_idx = order[i];
      }
      const auto& card_knowledge = knowledge[card_idx];
      // Add bits for plausible card.
      for (int color = 0; color < num_colors; ++color) {
        if (card_knowledge.ColorPlausible(color)) {
          for (int rank = 0; rank < num_ranks; ++rank) {
            if (card_knowledge.RankPlausible(rank)) {
              int card_idx = CardIndex(color, rank, num_ranks, shuffle_color, color_permute);
              (*encoding)[offset + card_idx] = 1;
            }
          }
        }
      }
      offset += bits_per_card;

      // Add bits for explicitly revealed colors and ranks.
      if (card_knowledge.ColorHinted()) {
        int color = card_knowledge.Color();
        if (shuffle_color) {
          color = color_permute[color];
        }
        (*encoding)[offset + color] = 1;
      }
      offset += num_colors;
      if (card_knowledge.RankHinted()) {
        (*encoding)[offset + card_knowledge.Rank()] = 1;
      }
      offset += num_ranks;

      ++num_cards;
    }

    // A player's hand can have fewer cards than the initial hand size.
    // Leave the bits for the absent cards empty (adjust the offset to skip
    // bits for the missing cards).
    if (num_cards < hand_size) {
      offset +=
          (hand_size - num_cards) * (bits_per_card + num_colors + num_ranks);
    }
  }

  assert(offset - start_offset == CardKnowledgeSectionLength(game));
  return offset - start_offset;
}

int EncodeV0Belief_(const HanabiGame& game,
                    const HanabiObservation& obs,
                    int start_offset,
                    const std::vector<int>& order,
                    bool shuffle_color,
                    const std::vector<int>& color_permute,
                    std::vector<float>* encoding,
                    std::vector<int>* ret_card_count,
                    bool publ) {
  // int bits_per_card = BitsPerCard(game);
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  // compute public card count
  std::vector<int> card_count = ComputeCardCount(
      game, obs, shuffle_color, color_permute, publ);
  if (ret_card_count != nullptr) {
    *ret_card_count = card_count;
  }

  // card knowledge
  const int len = EncodeCardKnowledge(
      game, obs, start_offset, order, shuffle_color, color_permute, encoding);
  const int player_offset = len / num_players;
  const int per_card_offset = len / hand_size / num_players;
  assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

  auto ref_encoding = *encoding;
  const std::vector<HanabiHand>& hands = obs.Hands();
  for (int player_id = 0; player_id < num_players; ++player_id) {
    int num_cards = hands[player_id].Cards().size();
    for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
      float total = 0;
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        // std::cout << offset << ", " << len << std::endl;
        assert(offset - start_offset < len);
        (*encoding)[offset] *= card_count[i];
        total += (*encoding)[offset];
      }
      if (total <= 0) {
        // const std::vector<HanabiHand>& hands = obs.Hands();
        std::cout << "publ? " << publ << std::endl;
        std::cout << "encoding size: " << encoding->size() << std::endl;
        std::cout << hands[0].Cards().size() << std::endl;
        std::cout << hands[1].Cards().size() << std::endl;
        std::cout << "player idx: " <<  player_id
                  << ", card idx: " << card_idx
                  << ", hand: " << std::endl;
        std::cout << hands[player_id].ToString()
                  << std::endl;
        std::cout << "total = 0 " << std::endl;
        std::cout << "card count" << std::endl;
        for (size_t x = 0; x < num_colors * num_ranks; ++x) {
          std::cout << card_count[x] << ", ";
          if ((x+1) % 5 == 0) {
            std::cout << std::endl;
          }
        }
        std::cout << "ck" << std::endl;
        for (size_t x = 0; x < num_colors * num_ranks; ++x) {
          int offset = (start_offset
                        + player_offset * player_id
                        + card_idx * per_card_offset
                        + x);
          std::cout << ref_encoding[offset] << ", ";
          if ((x+1) % 5 == 0) {
            std::cout << std::endl;
          }
        }
        std::cout << "Game Seed: " << game.Seed() << std::endl;
        std::cout << "OBS:" << std::endl;
        std::cout << obs.ToString() << std::endl;

        assert(false);
      }
      for (int i = 0; i < num_colors * num_ranks; ++i) {
        int offset = (start_offset
                      + player_offset * player_id
                      + card_idx * per_card_offset
                      + i);
        (*encoding)[offset] /= total;
      }
    }
    if (!publ) {
      break;
    }
  }
  return len;
}

int LastActionSectionLength(const HanabiGame& game) {
  return game.NumPlayers() +  // player id
         4 +                  // move types (play, dis, rev col, rev rank)
         game.NumPlayers() +  // target player id (if hint action)
         game.NumColors() +   // color (if hint action)
         game.NumRanks() +    // rank (if hint action)
         game.HandSize() +    // outcome (if hint action)
         game.HandSize() +    // position (if play action)
         BitsPerCard(game) +  // card (if play or discard action)
         2;                   // play (successful, added information token)
}

std::vector<int> CanonicalObservationEncoder::Shape() const {
  int l = HandsSectionLength(*parent_game_) +
          BoardSectionLength(*parent_game_) +
          DiscardSectionLength(*parent_game_) +
          LastActionSectionLength(*parent_game_) +
          (parent_game_->ObservationType() == HanabiGame::kMinimal
               ? 0
               : CardKnowledgeSectionLength(*parent_game_));
  return {l};
}

std::vector<float> CanonicalObservationEncoder::EncodeLastAction(
    const HanabiObservation& obs,
    const std::vector<int>& order,
    bool shuffle_color,
    const std::vector<int>& color_permute) const {
  std::vector<float> encoding(LastActionSectionLength(*parent_game_), 0);
  int offset = 0;
  offset += EncodeLastAction_(
      *parent_game_, obs, offset, order, shuffle_color, color_permute, &encoding);
  assert(offset == encoding.size());
  return encoding;
}

std::tuple<std::vector<float>, std::vector<int>>
CanonicalObservationEncoder::EncodePrivateV0Belief(
    const HanabiObservation& obs,
    const std::vector<int>& order,
    bool shuffle_color,
    const std::vector<int>& color_permute) const {
  int size = CardKnowledgeSectionLength(*parent_game_);
  int myBeliefSize = size / parent_game_->NumPlayers();

  std::vector<float> encoding(size, 0);
  std::vector<int> cardCount;
  int codeLen = EncodeV0Belief_(
      *parent_game_,
      obs,
      0,
      order,
      shuffle_color,
      color_permute,
      &encoding,
      &cardCount,
      false);
  (void)codeLen;
  assert(codeLen == (int)encoding.size());
  std::vector<float> privateV0(encoding.begin(), encoding.begin() + myBeliefSize);
  return {privateV0, cardCount};
}

std::vector<float> CanonicalObservationEncoder::Encode(
    const HanabiObservation& obs,
    bool show_own_cards,
    const std::vector<int>& order,
    bool shuffle_color,
    const std::vector<int>& color_permute,
    const std::vector<int>& inv_color_permute,
    bool hide_action) const {
  // Make an empty bit string of the proper size.
  std::vector<float> encoding(FlatLength(Shape()), 0);

  // This offset is an index to the start of each section of the bit vector.
  // It is incremented at the end of each section.
  int offset = 0;

  offset += EncodeHands(
      *parent_game_, obs, offset, show_own_cards, order, shuffle_color, color_permute, &encoding);
  offset += EncodeBoard(
      *parent_game_, obs, offset, shuffle_color, inv_color_permute, &encoding);
  offset += EncodeDiscards(
      *parent_game_, obs, offset, shuffle_color, color_permute, &encoding);
  if (hide_action) {
    offset += LastActionSectionLength(*parent_game_);
  } else {
    offset += EncodeLastAction_(
        *parent_game_, obs, offset, order, shuffle_color, color_permute, &encoding);
  }
  if (parent_game_->ObservationType() != HanabiGame::kMinimal) {
    offset += EncodeV0Belief_(
        *parent_game_, obs, offset, order, shuffle_color, color_permute, &encoding, nullptr, true);
  }

  assert(offset == encoding.size());
  return encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeOwnHandTrinary(
    const HanabiObservation& obs) const {
  // hard code 5 cards, empty slot will be all zero
  int len = parent_game_->HandSize() * 3;
  std::vector<float> encoding(len, 0);
  int bits_per_card = 3; // BitsPerCard(game);
  int num_ranks = parent_game_->NumRanks();
  (void)num_ranks;

  int offset = 0;
  const std::vector<HanabiHand>& hands = obs.Hands();
  const int player = 0;
  const std::vector<HanabiCard>& cards = hands[player].Cards();

  const std::vector<int>& fireworks = obs.Fireworks();
  for (const HanabiCard& card : cards) {
    // Only a player's own cards can be invalid/unobserved.
    // assert(card.IsValid());
    assert(card.Color() < parent_game_->NumColors());
    assert(card.Rank() < num_ranks);
    assert(card.IsValid());
    auto firework = fireworks[card.Color()];
    if (card.Rank() == firework) {
      encoding[offset] = 1;
    } else if (card.Rank() < firework) {
      encoding[offset + 1] = 1;
    } else {
      encoding[offset + 2] = 1;
    }

    offset += bits_per_card;
  }

  assert(offset <= len);
  return encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeOwnHand(
    const HanabiObservation& obs,
    bool shuffle_color,
    const std::vector<int>& color_permute
) const {
  int bits_per_card =  BitsPerCard(*parent_game_);
  int len = parent_game_->HandSize() * bits_per_card;
  std::vector<float> encoding(len, 0);

  const std::vector<HanabiCard>& cards = obs.Hands()[0].Cards();
  const int num_ranks = parent_game_->NumRanks();

  int offset = 0;
  for (const HanabiCard& card : cards) {
    // Only a player's own cards can be invalid/unobserved.
    assert(card.IsValid());
    int idx = CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
    encoding[offset + idx] = 1;
    offset += bits_per_card;
  }

  assert(offset == cards.size() * bits_per_card);
  return encoding;
}

std::vector<float> CanonicalObservationEncoder::EncodeAllHand(
    const HanabiObservation& obs,
    bool shuffle_color,
    const std::vector<int>& color_permute
) const {
  int bits_per_card =  BitsPerCard(*parent_game_);
  int len = obs.Hands().size() * parent_game_->HandSize() * bits_per_card;
  std::vector<float> encoding(len, 0);

  int offset = 0;
  for (int player_idx = 0; player_idx < obs.Hands().size(); ++player_idx) {
    const std::vector<HanabiCard>& cards = obs.Hands()[player_idx].Cards();
    const int num_ranks = parent_game_->NumRanks();

    for (const HanabiCard& card : cards) {
      // Only a player's own cards can be invalid/unobserved.
      assert(card.IsValid());
      int idx = CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
      encoding[offset + idx] = 1;
      offset += bits_per_card;
    }
    offset += bits_per_card * (parent_game_->HandSize() - cards.size());
  }

  assert(offset == len);
  return encoding;
}

std::vector<int> ComputeCardCount(
    const HanabiGame& game,
    const HanabiObservation& obs,
    bool shuffle_color,
    const std::vector<int>& color_permute,
    bool publ) {
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();

  std::vector<int> card_count(num_colors * num_ranks, 0);
  int total_count = 0;
  // full deck card count
  for (int color = 0; color < game.NumColors(); ++color) {
    for (int rank = 0; rank < game.NumRanks(); ++rank) {
      auto count = game.NumberCardInstances(color, rank);
      card_count[CardIndex(color, rank, num_ranks, shuffle_color, color_permute)] = count;
      total_count += count;
    }
  }
  // remove discard
  for (const HanabiCard& card : obs.DiscardPile()) {
    --card_count[CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute)];
    --total_count;
  }

  // remove firework
  const std::vector<int>& fireworks = obs.Fireworks();
  for (int c = 0; c < num_colors; ++c) {
    // fireworks[color] is the number of successfully played <color> cards.
    // If some were played, one-hot encode the highest (0-indexed) rank played
    if (fireworks[c] > 0) {
      for (int rank = 0; rank < fireworks[c]; ++rank) {
        --card_count[CardIndex(c, rank, num_ranks, shuffle_color, color_permute)];
        --total_count;
      }
    }
  }

  if (publ) {
    return card_count;
  }

  // {
  //   // sanity check
  //   const std::vector<HanabiHand>& hands = obs.Hands();
  //   int total_hand_size = 0;
  //   for (const auto& hand : hands) {
  //     total_hand_size += hand.Cards().size();
  //   }
  //   if(total_count != obs.DeckSize() + total_hand_size) {
  //     std::cout << "size mismatch: " << total_count
  //               << " vs " << obs.DeckSize() + total_hand_size << std::endl;
  //     assert(false);
  //   }
  // }

  // convert to private count
  for (int i = 1; i < obs.Hands().size(); ++i) {
    const auto& hand = obs.Hands()[i];
    for (auto card : hand.Cards()) {
      int index = CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
      --card_count[index];
      assert(card_count[index] >= 0);
    }
  }

  return card_count;
}

std::vector<float> CanonicalObservationEncoder::EncodeARV0Belief(
    const HanabiObservation& obs,
    const std::vector<int>& order,
    bool shuffle_color,
    const std::vector<int>& color_permute) const {
  auto& game = *parent_game_;
  int num_colors = game.NumColors();
  int num_ranks = game.NumRanks();
  int num_players = game.NumPlayers();
  int hand_size = game.HandSize();

  std::vector<std::vector<int>> ar_card_counts;
  {
    // compute private card count
    std::vector<int> card_count = ComputeCardCount(
        game, obs, shuffle_color, color_permute, false);
    auto& myCards = obs.Hands()[0].Cards();
    auto ar_card_count = card_count;
    for (int i = 0; i < myCards.size(); ++i) {
      ar_card_counts.push_back(ar_card_count);
      auto card = myCards[i];
      int index = CardIndex(card.Color(), card.Rank(), num_ranks, shuffle_color, color_permute);
      --ar_card_count[index];
      assert(ar_card_count[index] >= 0);
    }
  }

  int size = CardKnowledgeSectionLength(*parent_game_);
  int myBeliefSize = size / parent_game_->NumPlayers();
  std::vector<float> encoding(size);

  // card knowledge
  const int len = EncodeCardKnowledge(
      game, obs, 0, order, shuffle_color, color_permute, &encoding);
  const int player_offset = len / num_players;
  const int per_card_offset = len / hand_size / num_players;
  assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

  const std::vector<HanabiHand>& hands = obs.Hands();
  int player_id = 0;
  int num_cards = hands[player_id].Cards().size();
  for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
    auto card_count = ar_card_counts[card_idx];
    float total = 0;
    for (int i = 0; i < num_colors * num_ranks; ++i) {
      int offset = player_offset * player_id + card_idx * per_card_offset + i;
      // std::cout << offset << ", " << len << std::endl;
      assert(offset < len);
      encoding[offset] *= card_count[i];
      total += encoding[offset];
    }
    if (total <= 0) {
      // const std::vector<HanabiHand>& hands = obs.Hands();
      std::cout << hands[0].Cards().size() << std::endl;
      std::cout << hands[1].Cards().size() << std::endl;
      std::cout << "player idx: " <<  player_id
                << ", card idx: " << card_idx << std::endl;
      std::cout << "total = 0 " << std::endl;
      assert(false);
    }
    for (int i = 0; i < num_colors * num_ranks; ++i) {
      int offset = player_offset * player_id + card_idx * per_card_offset + i;
      encoding[offset] /= total;
    }
  }

  std::vector<float> ret(encoding.begin(), encoding.begin() + myBeliefSize);
  return ret;
}

}  // namespace hanabi_learning_env
