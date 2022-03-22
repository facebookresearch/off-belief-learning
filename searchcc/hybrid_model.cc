// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "searchcc/hybrid_model.h"

namespace search {

rela::Future HybridModel::asyncComputeAction(const GameSimulator& env) const {
  auto input = observe(env.state(), index, hideAction);
  if (rlStep_ > 0) {
    addHid(input, rlHid_);
    input["eps"] = torch::tensor(std::vector<float>{0});
    return rlModel_->call("act", input);
  } else {
    addHid(input, bpHid_);
    return bpModel_->call("act", input);
  }
}

// compute bootstrap target/value using blueprint
rela::Future HybridModel::asyncComputeTarget(
    const GameSimulator& env, float reward, bool terminal) const {
  auto feat = observe(env.state(), index, false);
  feat["reward"] = torch::tensor(reward);
  feat["terminal"] = torch::tensor((float)terminal);
  addHid(feat, bpHid_);
  return bpModel_->call("compute_target", feat);
}

// compute priority with rl model
rela::Future HybridModel::asyncComputePriority(const rela::TensorDict& input) const {
  assert(rlModel_ != nullptr);
  return rlModel_->call("compute_priority", input);
}

// observe before act
void HybridModel::observeBeforeAct(
    const GameSimulator& env, float eps, rela::TensorDict* retFeat) {
  auto feat = observe(env.state(), index, hideAction);
  if (retFeat != nullptr) {
    *retFeat = feat;
  }

  // forward bp regardless of whether rl is used
  {
    auto input = feat;
    addHid(input, bpHid_);
    futBp_ = bpModel_->call("act", input);
  }

  // maybe forward rl
  if (rlStep_ > 0) {
    feat["eps"] = torch::tensor(std::vector<float>{eps});
    auto input = feat;
    addHid(input, rlHid_);
    futRl_ = rlModel_->call("act", input);
  }
}

int HybridModel::decideAction(
    const GameSimulator& env, bool verbose, rela::TensorDict* retAction) {
  // first get results from the futures, to update hid
  int action = -1;
  auto bpReply = futBp_.get();
  updateHid(bpReply, bpHid_);

  if (rlStep_ > 0) {
    auto rlReply = futRl_.get();
    updateHid(rlReply, rlHid_);
    action = rlReply.at("a").item<int64_t>();
    if (env.state().CurPlayer() == index) {
      --rlStep_;
    }

    if (verbose) {
      auto bpAction = bpReply.at("a").item<int64_t>();
      std::cout << "rl picks " << action << ", bp picks " << bpAction
                << ", remaining rl steps: " << rlStep_ << std::endl;
    }

    if (retAction != nullptr) {
      *retAction = rlReply;
    }
  } else {
    assert(futRl_.isNull());
    action = bpReply.at("a").item<int64_t>();
    if (verbose) {
      std::cout << "in act, action: " << action << std::endl;
    }

    // assert(retAction == nullptr);
    // technically this is not right, we should never return action from bp
    // for training purpose, but in this case it will be skip anyway.
    if (retAction != nullptr) {
      assert(action == env.game().MaxMoves());
      *retAction = bpReply;
    }
  }

  if (env.state().CurPlayer() != index) {
    assert(action == env.game().MaxMoves());
  }
  return action;
}
}  // namespace search
