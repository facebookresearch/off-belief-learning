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

#include "hanabi_observation.h"

#include <algorithm>
#include <cassert>

#include "util.h"

namespace hanabi_learning_env {

namespace {
// Returns the offset of player ID pid relative to player ID observer_pid,
// or pid for negative values. That is, offset such that for a non-negative
// player id pid, we have (observer_pid + offset) % num_players == pid.
int PlayerToOffset(int pid, int observer_pid, int num_players) {
  return pid >= 0 ? (pid - observer_pid + num_players) % num_players : pid;
}

// Switch members from absolute player indices to observer-relative offsets,
// including player indices within the contained HanabiMove.
void ChangeHistoryItemToObserverRelative(int observer_pid, int num_players,
                                         bool show_cards,
                                         HanabiHistoryItem* item) {
  if (item->move.MoveType() == HanabiMove::kDeal) {
    assert(item->player < 0 && item->deal_to_player >= 0);
    item->deal_to_player =
        (item->deal_to_player - observer_pid + num_players) % num_players;
    if (item->deal_to_player == 0 && !show_cards) {
      // Hide cards dealt to observer if they shouldn't be able to see them.
      item->move = HanabiMove(HanabiMove::kDeal, -1, -1, -1, -1);
    }
  } else {
    assert(item->player >= 0);
    item->player = (item->player - observer_pid + num_players) % num_players;
  }
}
}  // namespace

HanabiObservation::HanabiObservation(const HanabiState& state,
                                     int observing_player,
                                     bool show_cards)
    : cur_player_offset_(PlayerToOffset(state.CurPlayer(), observing_player,
                                        state.ParentGame()->NumPlayers())),
      observing_player_(observing_player),
      discard_pile_(state.DiscardPile()),
      fireworks_(state.Fireworks()),
      deck_size_(state.Deck().Size()),
      information_tokens_(state.InformationTokens()),
      life_tokens_(state.LifeTokens()),
      legal_moves_(state.LegalMoves(observing_player)),
      parent_game_(state.ParentGame()) {
  REQUIRE(observing_player >= 0 &&
          observing_player < state.ParentGame()->NumPlayers());
  hands_.reserve(state.Hands().size());
  const bool hide_knowledge =
      state.ParentGame()->ObservationType() == HanabiGame::kMinimal;
  show_cards = (show_cards || state.ParentGame()->ObservationType() == HanabiGame::kSeer);
  hands_.push_back(
      HanabiHand(state.Hands()[observing_player], !show_cards, hide_knowledge));
  for (int offset = 1; offset < state.ParentGame()->NumPlayers(); ++offset) {
    hands_.push_back(HanabiHand(state.Hands()[(observing_player + offset) %
                                              state.ParentGame()->NumPlayers()],
                                false, hide_knowledge));
  }

  const auto& history = state.MoveHistory();
  auto start = std::find_if(history.begin(), history.end(),
                            [](const HanabiHistoryItem& item) {
                              return item.player != kChancePlayerId;
                            });
  std::reverse_iterator<decltype(start)> rend(start);
  for (auto it = history.rbegin(); it != rend; ++it) {
    last_moves_.push_back(*it);
    ChangeHistoryItemToObserverRelative(observing_player,
                                        state.ParentGame()->NumPlayers(),
                                        show_cards,
                                        &last_moves_.back());
    if (it->player == observing_player) {
      break;
    }
  }
}

HanabiObservation::HanabiObservation(
    int cur_player,
    int observing_player,
    // hands[observing_player] is for the observing player
    const std::vector<HanabiHand>& hands,
    const std::vector<HanabiCard>& discard_pile,
    const std::vector<int>& fireworks,
    const std::vector<HanabiHistoryItem>& last_moves,
    // number of remaining cards
    int deck_size,
    int information_tokens,
    int life_tokens,
    const std::vector<HanabiMove>& legal_moves,
    const HanabiGame* parent_game)
    : cur_player_offset_(PlayerToOffset(cur_player, observing_player, parent_game->NumPlayers())),
      observing_player_(observing_player),
      discard_pile_(discard_pile),
      fireworks_(fireworks),
      deck_size_(deck_size),
      information_tokens_(information_tokens),
      life_tokens_(life_tokens),
      legal_moves_(legal_moves),
      parent_game_(parent_game) {
  REQUIRE(observing_player >= 0 && observing_player < parent_game_->NumPlayers());
  hands_.reserve(hands.size());
  const bool hide_knowledge = parent_game_->ObservationType() == HanabiGame::kMinimal;
  auto show_cards = false;
  assert(!hide_knowledge && !show_cards);
  hands_.push_back(HanabiHand(hands[observing_player], !show_cards, hide_knowledge));

  const int num_players = parent_game_->NumPlayers();
  for (int offset = 1; offset < num_players; ++offset) {
    auto player_idx = (observing_player + offset) % num_players;
    auto hand = HanabiHand(hands[player_idx], false, hide_knowledge);
    hands_.push_back(std::move(hand));
  }

  const auto& history = last_moves;
  for (auto it = history.rbegin(); it != history.rend(); ++it) {
    last_moves_.push_back(*it);
    ChangeHistoryItemToObserverRelative(observing_player,
                                        parent_game_->NumPlayers(),
                                        show_cards,
                                        &last_moves_.back());
    if (it->player == observing_player) {
      break;
    }
  }
}

std::string HanabiObservation::ToString() const {
  std::string result;
  result += "Life tokens: " + std::to_string(LifeTokens()) + "\n";
  result += "Info tokens: " + std::to_string(InformationTokens()) + "\n";
  result += "Fireworks: ";
  for (int i = 0; i < ParentGame()->NumColors(); ++i) {
    result += ColorIndexToChar(i);
    result += std::to_string(fireworks_[i]) + " ";
  }
  result += "\nHands:\n";
  for (int i = 0; i < hands_.size(); ++i) {
    if (i > 0) {
      result += "-----\n";
    }
    if (i == CurPlayerOffset()) {
      result += "Cur player\n";
    }
    result += hands_[i].ToString();
  }
  result += "Deck size: " + std::to_string(DeckSize()) + "\n";
  result += "Discards:";
  for (int i = 0; i < discard_pile_.size(); ++i) {
    result += " " + discard_pile_[i].ToString();
  }
  result += "\nLastMoves: \n";
  for (int i = 0; i < LastMoves().size(); ++i) {
    result += " " + LastMoves()[i].ToString() + "\n";
  }

  return result;
}

bool HanabiObservation::CardPlayableOnFireworks(int color, int rank) const {
  if (color < 0 || color >= ParentGame()->NumColors()) {
    return false;
  }
  return rank == fireworks_[color];
}

}  // namespace hanabi_learning_env
