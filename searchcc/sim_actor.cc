// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "searchcc/sim_actor.h"

namespace search {

void SimulationActor::observeBeforeAct(const GameSimulator& env) {
  assert(callOrder_ == 0);
  assert(!needReset_);
  ++callOrder_;
  ++numObserve_;

  if (r2d2Buffer_ == nullptr) {
    return model_.observeBeforeAct(env, eps_);
  }

  rela::TensorDict feat;
  model_.observeBeforeAct(env, eps_, &feat);
  r2d2Buffer_->pushObs(feat);
}

int SimulationActor::decideAction(const GameSimulator& env) {
  assert(callOrder_ == 1);
  ++callOrder_;

  if (r2d2Buffer_ == nullptr) {
    return model_.decideAction(env, false);
  }

  rela::TensorDict action;
  auto aid = model_.decideAction(env, false, &action);
  r2d2Buffer_->pushAction(action);
  return aid;
}

void SimulationActor::observeAfterAct(const GameSimulator& env) {
  assert(callOrder_ == 2);
  ++callOrder_;

  if (r2d2Buffer_ == nullptr) {
    return;
  }

  float reward = env.reward();
  bool terminal = env.terminal();
  if (numObserve_ < 2 * initRlStep_ || terminal) {
    r2d2Buffer_->pushReward(reward);
    r2d2Buffer_->pushTerminal(terminal);
    return;
  }

  // end of trajectory
  if (!terminal) {
    futTarget_ = model_.asyncComputeTarget(env, reward, terminal);
  }
  // if (numAct_ < maxNumAct_) {
  //   r2d2Buffer_->pushReward(reward);
  //   r2d2Buffer_->pushTerminal(terminal);
  //   return;
  // }

  // // numAct_ == maxNumAct_
  // // final move has been finished
  // accReward_ += env.reward();
  // if (env.state().CurPlayer() == model_.index && !terminal) {
  //   // my partner just finished final move, next state is my turn, bootstrap
  //   futTarget_ = model_.asyncComputeTarget(env, accReward_, terminal);
  // }
}

bool SimulationActor::maybeEndEpisode(const GameSimulator& env) {
  assert(callOrder_ == 3);
  callOrder_ = 0;

  if (r2d2Buffer_ == nullptr) {
    return false;
  }

  if (!futPriority_.isNull()) {
    auto priority = futPriority_.get()["priority"].item<float>();
    replayBuffer_->add(std::move(lastEpisode_), priority);
  }

  bool terminal = env.terminal();
  bool endEpisode = false;

  if (terminal) {
    // natural termination, no target should be computed
    endEpisode = true;
    assert(futTarget_.isNull());
    if (r2d2Buffer_->len() != numObserve_) {
      std::cout << r2d2Buffer_->len() << ", " << numObserve_ << std::endl;
    }
    assert(r2d2Buffer_->len() == numObserve_);
  } else {
    // we have gathered enough timestep
    if (numObserve_ == 2 * initRlStep_) {
      endEpisode = true;
      auto reward = futTarget_.get().at("target").item<float>();
      r2d2Buffer_->pushReward(reward);
      r2d2Buffer_->pushTerminal(true);
    }
  }

  if (endEpisode) {
    lastEpisode_ = r2d2Buffer_->popTransition();
    auto input = lastEpisode_.toDict();
    futPriority_ = model_.asyncComputePriority(input);
    needReset_ = true;
  }
  return endEpisode;
}
}  // namespace search
