// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rlcc/utils.h"

namespace hle = hanabi_learning_env;

class HanabiEnv {
 public:
  HanabiEnv(
      const std::unordered_map<std::string, std::string>& gameParams,
      int maxLen,
      bool verbose)
      : game_(gameParams)
      , state_(nullptr)
      , maxLen_(maxLen)
      , verbose_(verbose)
      , lastActivePlayer_(-1)
      , lastMove_(hle::HanabiMove::kInvalid, -1, -1, -1, -1)
      , lastEpisodeScore_(-1) {
    auto params = game_.Parameters();

    if (verbose_) {
      std::cout << "Hanabi game created, with parameters:\n";
      for (const auto& item : params) {
        std::cout << "  " << item.first << "=" << item.second << "\n";
      }
    }
  }

  std::tuple<int, int, int> featureSize(bool sad) const {
    auto encoder = hle::CanonicalObservationEncoder(&game_);
    int size = encoder.Shape()[0];
    if (sad) {
      size += hle::LastActionSectionLength(game_);
    }

    int priv = size - handFeatureSize();
    int publ = size - game_.NumPlayers() * handFeatureSize();
    // TODO: remove the "size" part of the return value
    return std::make_tuple(size, priv, publ);
  }

  int numAction() const {
    return game_.MaxMoves() + 1;
  }

  int noOpUid() const {
    return numAction() - 1;
  }

  int handFeatureSize() const {
    return game_.HandSize() * game_.NumColors() * game_.NumRanks();
  }

  int getLastAction() const {
    return game_.GetMoveUid(lastMove_);
  }

  hle::HanabiMove getMove(int uid) const {
    return game_.GetMove(uid);
  }

  hle::HanabiObservation getObsShowCards() const {
    int player = 0;
    bool show = true;
    auto obs = hle::HanabiObservation(*state_, player, show);
    return obs;
  }

  void reset() {
    assert(terminated());
    state_ = std::make_unique<hle::HanabiState>(&game_);
    // chance player
    while (state_->CurPlayer() == hle::kChancePlayerId) {
      state_->ApplyRandomChance();
    }
    numStep_ = 0;
  }

  void resetWithDeck(const std::vector<hle::HanabiCardValue>& deck) {
    assert(terminated());
    state_ = std::make_unique<hle::HanabiState>(&game_);
    state_->SetDeckOrder(deck);
    // chance player
    while (state_->CurPlayer() == hle::kChancePlayerId) {
      state_->ApplyRandomChance();
    }
    numStep_ = 0;
  }

  void step(hle::HanabiMove move) {
    assert(maxLen_ < 0 || numStep_ < maxLen_);

    ++numStep_;
    lastActivePlayer_ = state_->CurPlayer();
    lastMove_ = move;

    auto [r, t] = applyMove(*state_, move, numStep_ == maxLen_);
    if (t) {
      lastEpisodeScore_ = state_->Score();
    }

    if (colorReward_ > 0 && move.MoveType() == hle::HanabiMove::kRevealColor) {
      r += colorReward_;
    }

    stepReward_ = r;
  }

  float stepReward() const {
    return stepReward_;
  }

  int lastActivePlayer() const {
    assert(lastActivePlayer_ != -1);
    return lastActivePlayer_;
  }

  hle::HanabiMove lastMove() const {
    return lastMove_;
  }

  int numStep() const {
    return numStep_;
  }

  bool terminated() const {
    if (state_ == nullptr) {
      return true;
    }

    if (state_->IsTerminal() || (maxLen_ > 0 && numStep_ == maxLen_)) {
      return true;
    }

    return false;
  }

  int getCurrentPlayer() const {
    assert(state_ != nullptr);
    return state_->CurPlayer();
  }

  int lastEpisodeScore() const {
    return lastEpisodeScore_;
  }

  std::vector<std::string> deckHistory() const {
    return state_->DeckHistory();
  }

  const hle::HanabiState& getHleState() const {
    assert(state_ != nullptr);
    return *state_;
  }

  const hle::HanabiGame& getHleGame() const {
    return game_;
  }

  int getNumPlayers() const {
    return game_.NumPlayers();
  }

  int getScore() const {
    return state_->Score();
  }

  int getLife() const {
    return state_->LifeTokens();
  }

  int getInfo() const {
    return state_->InformationTokens();
  }

  std::vector<int> getFireworks() const {
    return state_->Fireworks();
  }

  void setColorReward(float colorReward) {
    colorReward_ = colorReward;
  }

 protected:
  const hle::HanabiGame game_;
  std::unique_ptr<hle::HanabiState> state_;
  const int maxLen_;
  const bool verbose_;

  int numStep_;
  float stepReward_;
  int lastActivePlayer_;
  hle::HanabiMove lastMove_;
  int lastEpisodeScore_;

  float colorReward_ = -1;
};
