// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rlcc/utils.h"

#include "rela/thread_loop.h"

#include "searchcc/game_sim.h"
#include "searchcc/hand_dist.h"
#include "searchcc/sim_actor.h"

namespace hle = hanabi_learning_env;

namespace search {

// the threadloop for collecting data for RL training
class SearchThreadLoop : public rela::ThreadLoop {
 public:
  SearchThreadLoop(
      const hle::HanabiState& state,
      const HandDistribution& myHandDist,
      const HandDistribution& partnerHandDist,
      std::vector<std::vector<std::unique_ptr<SimulationActor>>>&& actors,
      std::shared_ptr<std::vector<std::vector<std::vector<hle::HanabiCardValue>>>>
          simHands,
      bool useSimHands,
      int myIndex,
      int seed,
      int numGame,
      bool jointSample);

  virtual void mainLoop() override;

  const std::vector<float>& getScores() const {
    return scores_;
  }

 private:
  std::vector<SimHand> sampleHands() const;

  const int numEnv_;
  const hle::HanabiState refState_;
  const hle::HanabiGame game_;
  const HandDistribution& myHandDist_;
  const HandDistribution& partnerHandDist_;
  const int myIndex_;
  const bool jointSample_;

  std::vector<int> cardCount_;

  int numGame_;  // -1 means inf
  std::vector<GameSimulator> envs_;
  std::vector<std::vector<std::unique_ptr<SimulationActor>>> actors_;
  std::shared_ptr<std::vector<std::vector<std::vector<hle::HanabiCardValue>>>> simHands_;
  bool useSimHands_;
  std::vector<float> scores_;
  mutable std::mt19937 rng_;
};

}  // namespace search
