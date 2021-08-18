// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <cassert>
#include <thread>

#include "rela/batcher.h"
#include "rela/tensor_dict.h"

namespace rela {

class BatchRunner {
 public:
  BatchRunner(
      py::object pyModel,
      const std::string& device,
      int maxBatchsize,
      const std::vector<std::string>& methods)
      : pyModel_(pyModel)
      , jitModel_(pyModel_.attr("_c").cast<torch::jit::script::Module*>())
      , device_(torch::Device(device))
      , batchsizes_(methods.size(), maxBatchsize)
      , methods_(methods) {
  }

  BatchRunner(py::object pyModel, const std::string& device)
      : pyModel_(pyModel)
      , jitModel_(pyModel_.attr("_c").cast<torch::jit::script::Module*>())
      , device_(torch::Device(device)) {
  }

  BatchRunner(const BatchRunner&) = delete;
  BatchRunner& operator=(const BatchRunner&) = delete;

  ~BatchRunner() {
    stop();
  }

  void setLogFreq(int logFreq) {
    logFreq_ = logFreq;
  }

  void addMethod(const std::string& method, int batchSize) {
    batchsizes_.push_back(batchSize);
    methods_.push_back(method);
  }

  FutureReply call(const std::string& method, const TensorDict& t) const;

  void start();

  void stop();

  void updateModel(py::object agent) {
    std::lock_guard<std::mutex> lk(mtxUpdate_);
    pyModel_.attr("load_state_dict")(agent.attr("state_dict")());
  }

  const torch::jit::script::Module& jitModel() {
    return *jitModel_;
  }

  // for debugging
  rela::TensorDict blockCall(const std::string& method, const TensorDict& t);

 private:
  void runnerLoop(const std::string& method);

  py::object pyModel_;
  torch::jit::script::Module* const jitModel_;
  const torch::Device device_;
  std::vector<int> batchsizes_;
  std::vector<std::string> methods_;

  // ideally this mutex should be 1 per device, thus global
  std::mutex mtxDevice_;
  std::mutex mtxUpdate_;

  mutable std::map<std::string, std::unique_ptr<Batcher>> batchers_;
  std::vector<std::thread> threads_;

  int logFreq_ = -1;
};
}  // namespace rela
