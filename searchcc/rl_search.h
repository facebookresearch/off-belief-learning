// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/context.h"
#include "rela/prioritized_replay.h"

#include "searchcc/game_sim.h"
#include "searchcc/hand_dist.h"
#include "searchcc/hybrid_model.h"
#include "searchcc/thread_loop.h"

namespace search {

class RLSearchActor {
 public:
  RLSearchActor(
      int index,
      std::shared_ptr<rela::BatchRunner> bpRunner,
      std::shared_ptr<rela::BatchRunner> rlRunner,
      std::shared_ptr<rela::BatchRunner> beliefRunner,
      int num_samples,
      bool publBelief,
      bool jointSearch,
      const std::vector<float>& epsList,
      int nStep,
      float gamma,
      int seed)
      : index(index)
      , beliefRunner_(beliefRunner)
      , num_samples_(num_samples)
      , publBelief_(publBelief)
      , jointSearch_(jointSearch)
      , epsList_(epsList)
      , nStep_(nStep)
      , gamma_(gamma)
      , rng_(seed)
      , prevModel_(index)
      , model_(index) {
    assert(bpRunner != nullptr);
    model_.setBpModel(bpRunner, getH0(*bpRunner, 1));
    if (rlRunner != nullptr) {
      model_.setRlModel(rlRunner, getH0(*rlRunner, 1));
    }
    if (beliefRunner_ != nullptr) {
      beliefRnnHid_ = getH0(*beliefRunner_, 1);
      model_.setBeliefHid(beliefRnnHid_);
    }
  }

  void setPartner(std::shared_ptr<RLSearchActor> partner) {
    partner_ = std::move(partner);
  }

  void setComputeConfig(int numThread, int numEnvPerThread) {
    assert(numThread_ == -1);
    numThread_ = numThread;
    numEnvPerThread_ = numEnvPerThread;
  }

  void resetRlRnn() {
    if (model_.getRlStep() > 0) {
      model_.setRlStep(0);
    }
    model_.setRlHid(model_.getBpHid());
  }

  void setUseRL(int numStep) {
    // assert(callOrder_ == 1);
    assert(model_.getRlStep() == 0);
    model_.setRlStep(numStep);
  }

  int usingRL() const {
    return model_.getRlStep();
  }

  void updateBelief(const GameSimulator& env) {
    assert(callOrder_ == 0);
    ++callOrder_;

    auto [obs, lastMove, cardCount, myHand] =
        observeForSearch(env.state(), index, hideAction, publBelief_);
    search::updateBelief(
        prevState_,
        env.game(),
        lastMove,
        cardCount,
        myHand,
        partner_->prevModel_,
        index,
        handDist_,
        numThread_);
  }

  void updateBeliefHid(const GameSimulator& env) {

    auto [obs, lastMove, cardCount, myHand] =
        observeForSearch(env.state(), index, hideAction, publBelief_);
    applyModel(obs, *beliefRunner_, beliefRnnHid_, "observe");
    model_.setBeliefHid(beliefRnnHid_);
  }

  std::vector<std::vector<std::vector<hle::HanabiCardValue>>> sampleHands(
      const hle::HanabiState& state, int numSample);

  rela::TensorDict getBeliefHidden();
  rela::TensorDict getModelBeliefHidden();

  void startDataGeneration(
      const GameSimulator& env,
      std::shared_ptr<rela::RNNPrioritizedReplay> replay,
      int numRlStep,
      std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
      bool useSimHands,
      bool beliefMode) const;

  void stopDataGeneration() const;

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
    return model_.decideAction(env, true);
  }

  // should be called after decideAction?
  hle::HanabiMove spartaSearch(
      const GameSimulator& env, hle::HanabiMove bpMove, int numSearch, float threshold);

  std::vector<float> runSimGames(
      const GameSimulator& env,
      int numGame,
      int numRLStep,
      int seed,
      std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
      bool useSimHands) const;

  const int index;
  const bool hideAction = false;

 private:
  std::vector<std::shared_ptr<SearchThreadLoop>> startDataGeneration_(
      const GameSimulator& env,
      std::shared_ptr<rela::RNNPrioritizedReplay> replay,
      int numRLStep,
      const std::vector<float>& epsList,
      std::mt19937& rng,
      int numThread,
      int numEnvPerThread,
      int numGame,
      bool train,
      std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
      bool useSimHands,
      bool beliefMode) const;

  std::vector<std::vector<std::vector<hle::HanabiCardValue>>> sampleHandsPubl_(
      const hle::HanabiState& state, int numSample);
  std::vector<std::vector<std::vector<hle::HanabiCardValue>>> sampleHandsPriv_(
      const hle::HanabiState& state, int numSample);

  // for learned model
  std::shared_ptr<rela::BatchRunner> beliefRunner_;
  rela::TensorDict beliefRnnHid_;
  int num_samples_;

  const bool publBelief_;
  const bool jointSearch_;
  const std::vector<float> epsList_;
  const int nStep_;
  const float gamma_;
  mutable std::mt19937 rng_;

  int numThread_ = -1;
  int numEnvPerThread_ = -1;

  HybridModel prevModel_;
  HybridModel model_;
  HandDistribution handDist_;

  std::shared_ptr<RLSearchActor> partner_;
  std::unique_ptr<hle::HanabiState> prevState_ = nullptr;

  // related to parallel simulation
  mutable std::unique_ptr<rela::Context> context_;
  int callOrder_ = 0;
};

}  // namespace search
