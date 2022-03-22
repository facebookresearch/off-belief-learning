// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "searchcc/game_sim.h"
#include "searchcc/hand_dist.h"
#include "searchcc/hybrid_model.h"

namespace search {

class SpartaActor {
 public:
  SpartaActor(int index, std::shared_ptr<rela::BatchRunner> bpRunner, int seed)
      : index(index)
      , rng_(seed)
      , prevModel_(index)
      , model_(index) {
    assert(bpRunner != nullptr);
    model_.setBpModel(bpRunner, getH0(*bpRunner, 1));
  }

  void setPartners(std::vector<std::shared_ptr<SpartaActor>> partners) {
    partners_ = std::move(partners);
  }

  void updateBelief(const GameSimulator& env, int numThread) {
    assert(callOrder_ == 0);
    ++callOrder_;

    const auto& state = env.state();
    int curPlayer = state.CurPlayer();
    int numPlayer = env.game().NumPlayers();
    assert((int)partners_.size() == numPlayer);
    int prevPlayer = (curPlayer - 1 + numPlayer) % numPlayer;
    std::cout << "prev player: " << prevPlayer << std::endl;

    auto [obs, lastMove, cardCount, myHand] =
        observeForSearch(env.state(), index, hideAction, false);

    search::updateBelief(
        prevState_,
        env.game(),
        lastMove,
        cardCount,
        myHand,
        partners_[prevPlayer]->prevModel_,
        index,
        handDist_,
        numThread);
  }

  void observe(const GameSimulator& env) {
    // assert(callOrder_ == 1);
    ++callOrder_;

    const auto& state = env.state();
    model_.observeBeforeAct(env, 0);

    if (prevState_ == nullptr) {
      prevState_ = std::make_unique<hle::HanabiState>(state);
    } else {
      *prevState_ = state;
    }
  }

  int decideAction(const GameSimulator& env) {
    // assert(callOrder_ == 2);
    callOrder_ = 0;

    prevModel_ = model_;  // this line can only be in decide action
    return model_.decideAction(env, false);
  }

  // should be called after decideAction
  hle::HanabiMove spartaSearch(
      const GameSimulator& env, hle::HanabiMove bpMove, int numSearch, float threshold);

  const int index;
  const bool hideAction = false;

 private:
  mutable std::mt19937 rng_;

  HybridModel prevModel_;
  HybridModel model_;
  HandDistribution handDist_;

  std::vector<std::shared_ptr<SpartaActor>> partners_;
  std::unique_ptr<hle::HanabiState> prevState_ = nullptr;

  int callOrder_ = 0;
};

}  // namespace search
