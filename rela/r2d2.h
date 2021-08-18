// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

class R2D2Buffer {
 public:
  R2D2Buffer(int multiStep, int seqLen, float gamma)
      : multiStep_(multiStep)
      , maxSeqLen_(seqLen)
      , gamma_(gamma)
      , obs_(seqLen)
      , action_(seqLen)
      , reward_(seqLen)
      , terminal_(seqLen)
      , bootstrap_(seqLen)
      , accReward_(seqLen)
      , seqLen_(0)
      , callOrder_(0) {
  }

  void init(const TensorDict& h0) {
    h0_ = h0;
  }

  int len() const {
    return seqLen_;
  }

  TensorDict& obsBack() {
    if (callOrder_ == 0) {
      assert(seqLen_ > 0);
      return obs_[seqLen_ - 1];
    } else {
      return obs_[seqLen_];
    }
  }

  void pushObs(const TensorDict& obs) {
    assert(callOrder_ == 0);
    ++callOrder_;

    assert(seqLen_ < maxSeqLen_);
    obs_[seqLen_] = obs;
  }

  void pushAction(const TensorDict& action) {
    assert(callOrder_ == 1);
    ++callOrder_;
    action_[seqLen_] = action;
  }

  void pushReward(float r) {
    assert(callOrder_ == 2);
    ++callOrder_;
    reward_[seqLen_] = r;
  }

  void pushTerminal(float t) {
    assert(callOrder_ == 3);
    callOrder_ = 0;
    terminal_[seqLen_] = t;
    ++seqLen_;
  }

  RNNTransition popTransition() {
    assert(callOrder_ == 0);
    // episode has to terminate
    assert(terminal_[seqLen_ - 1] == 1.0f);

    // acc reward
    for (int i = 0; i < seqLen_; ++i) {
      float factor = 1;
      float acc = 0;
      for (int j = 0; j < multiStep_; ++j) {
        if (i + j >= seqLen_) {
          break;
        }
        acc += factor * reward_[i + j];
        factor *= gamma_;
      }
      accReward_[i] = acc;
    }

    for (int i = 0; i < maxSeqLen_; ++i) {
      if (i < seqLen_ - multiStep_) {
        bootstrap_[i] = 1.0f;
      } else {
        bootstrap_[i] = 0.0f;
      }
    }

    // padding
    for (int i = seqLen_; i < maxSeqLen_; ++i) {
      obs_[i] = tensor_dict::zerosLike(obs_[seqLen_ - 1]);
      action_[i] = tensor_dict::zerosLike(action_[seqLen_ - 1]);
      reward_[i] = 0.f;
      terminal_[i] = 1.0f;
      accReward_[i] = 0.0f;
    }

    RNNTransition transition;
    transition.obs = tensor_dict::stack(obs_, 0);
    transition.action = tensor_dict::stack(action_, 0);
    transition.reward = torch::tensor(accReward_);
    transition.terminal = torch::tensor(terminal_);
    transition.bootstrap = torch::tensor(bootstrap_);
    transition.seqLen = torch::tensor(float(seqLen_));
    transition.h0 = h0_;

    seqLen_ = 0;
    callOrder_ = 0;
    return transition;
  }

 private:
  const int multiStep_;
  const int maxSeqLen_;
  const float gamma_;

  TensorDict h0_;
  std::vector<TensorDict> obs_;
  std::vector<TensorDict> action_;
  std::vector<float> reward_;
  std::vector<float> terminal_;

  // derived
  std::vector<float> bootstrap_;
  std::vector<float> accReward_;

  int seqLen_;
  int callOrder_;
};

}  // namespace rela
