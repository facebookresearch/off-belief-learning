// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <torch/extension.h>

#include "tensor_dict.h"

namespace rela {

class FFTransition {
 public:
  FFTransition() = default;

  FFTransition(
      TensorDict& obs,
      TensorDict& action,
      torch::Tensor& reward,
      torch::Tensor& terminal,
      torch::Tensor& bootstrap,
      TensorDict& nextObs)
      : obs(obs)
      , action(action)
      , reward(reward)
      , terminal(terminal)
      , bootstrap(bootstrap)
      , nextObs(nextObs) {
  }

  FFTransition index(int i) const;

  FFTransition padLike() const;

  std::vector<torch::jit::IValue> toVectorIValue(const torch::Device& device) const;

  TensorDict toDict();

  TensorDict obs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  TensorDict nextObs;
};

class RNNTransition {
 public:
  RNNTransition() = default;

  RNNTransition index(int i) const;

  TensorDict toDict();

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor terminal;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;
};

FFTransition makeBatch(
    const std::vector<FFTransition>& transitions, const std::string& device);

RNNTransition makeBatch(
    const std::vector<RNNTransition>& transitions, const std::string& device);

TensorDict makeBatch(
    const std::vector<TensorDict>& transitions, const std::string& device);

}  // namespace rela
