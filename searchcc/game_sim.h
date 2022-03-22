// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rela/batch_runner.h"
#include "rela/tensor_dict.h"
#include "rlcc/utils.h"

namespace hle = hanabi_learning_env;

namespace search {

struct SimHand {
  int index;
  std::vector<hle::HanabiCardValue> cards;

  SimHand(int index, const std::vector<hle::HanabiCardValue>& cards)
      : index(index)
      , cards(cards) {
  }
};

class GameSimulator {
 public:
  GameSimulator(const std::unordered_map<std::string, std::string>& params)
      : game_(params)
      , state_(&game_) {
    while (state_.CurPlayer() == hle::kChancePlayerId) {
      state_.ApplyRandomChance();
    }
  }

  GameSimulator(const hle::HanabiState& refState, const std::vector<SimHand>& simHands)
      : game_(*refState.ParentGame())
      , state_(refState) {
    reset(refState, simHands);
  }

  GameSimulator(
      const hle::HanabiState& refState, const std::vector<SimHand>& simHands, int newSeed)
      : game_(*refState.ParentGame())
      , state_(refState) {
    game_.SetSeed(newSeed);
    reset(refState, simHands);
  }

  GameSimulator(const GameSimulator& sim)
      : game_(sim.game_)
      , state_(sim.state_) {
    state_.SetGame(&game_);
  };

  GameSimulator& operator=(const GameSimulator&) = delete;
  GameSimulator(GameSimulator&&) = delete;
  GameSimulator& operator=(GameSimulator&&) = delete;

  void reset(const hle::HanabiState& refState, const std::vector<SimHand>& simHands) {
    state_ = refState;
    state_.SetGame(&game_);
    terminal_ = false;
    reward_ = 0;

    for (const auto& simHand : simHands) {
      const auto& realCards = state_.Hands()[simHand.index].Cards();
      auto& deck = state_.Deck();
      deck.PutCardsBack(realCards);
    }
    for (const auto& simHand : simHands) {
      auto& deck = state_.Deck();
      deck.DealCards(simHand.cards);

      auto& hand = state_.Hands()[simHand.index];
      if (!hand.CanSetCards(simHand.cards)) {
        std::cout << "cannot set hand:" << std::endl;
        std::cout << "real hand: " << std::endl;
        std::cout << hand.ToString() << std::endl;
        std::cout << "sim hand: ";
        for (auto& c : simHand.cards) {
          std::cout << c.ToString() << ", ";
        }
        std::cout << std::endl;
      }
      hand.SetCards(simHand.cards);
    }
  }

  void step(hle::HanabiMove move) {
    std::tie(reward_, terminal_) = applyMove(state_, move, false);
  }

  hle::HanabiMove getMove(int uid) const {
    return game_.GetMove(uid);
  }

  const hle::HanabiState& state() const {
    return state_;
  }

  const hle::HanabiGame& game() const {
    return game_;
  }

  float reward() const {
    return reward_;
  }

  bool terminal() const {
    return terminal_;
  }

  void setTerminal(bool terminal) {
    terminal_ = terminal;
  }

  int score() const {
    return state_.Score();
  }

 private:
  hle::HanabiGame game_;
  hle::HanabiState state_;

  bool terminal_ = false;
  float reward_ = 0;
};

}  // namespace search
