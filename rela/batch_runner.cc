// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "batch_runner.h"

namespace rela {

FutureReply BatchRunner::call(const std::string& method, const TensorDict& t) const {
  auto batcherIt = batchers_.find(method);
  if (batcherIt == batchers_.end()) {
    std::cerr << "Error: Cannot find method: " << method << std::endl;
    std::cerr << "avail methods are:" << std::endl;
    for (auto& kv : batchers_) {
      std::cerr << kv.first << std::endl;
    }
    assert(false);
  }
  return batcherIt->second->send(t);
}

void BatchRunner::start() {
  for (size_t i = 0; i < methods_.size(); ++i) {
    batchers_.emplace(methods_[i], std::make_unique<Batcher>(batchsizes_[i]));
  }

  for (auto& kv : batchers_) {
    threads_.emplace_back(&BatchRunner::runnerLoop, this, kv.first);
  }
}

void BatchRunner::stop() {
  // batchers_.clear();
  for (auto& kv : batchers_) {
    kv.second->exit();
  }

  for (auto& v : threads_) {
    if (v.joinable()) {
      v.join();
    }
  }
}

// for debugging
rela::TensorDict BatchRunner::blockCall(const std::string& method, const TensorDict& t) {
  torch::NoGradGuard ng;
  std::vector<torch::jit::IValue> input;
  input.push_back(tensor_dict::toIValue(t, device_));
  torch::jit::IValue output;
  {
    std::lock_guard<std::mutex> lk(mtxUpdate_);
    output = jitModel_->get_method(method)(input);
  }
  return tensor_dict::fromIValue(output, torch::kCPU, true);
}

void BatchRunner::runnerLoop(const std::string& method) {
  auto batcherIt = batchers_.find(method);
  if (batcherIt == batchers_.end()) {
    std::cerr << "Error: RunnerLoop, Cannot find method: " << method << std::endl;
    assert(false);
  }
  auto& batcher = *(batcherIt->second);

  int aggSize = 0;
  int aggCount = 0;

  while (!batcher.terminated()) {
    auto batch = batcher.get();
    if (batch.empty()) {
      assert(batcher.terminated());
      break;
    }

    if (logFreq_ > 0) {
      aggSize += (batch.begin()->second.size(0));
      aggCount += 1;

      if (aggCount % logFreq_ == 0) {
        std::cout << method << ", average batchsize: " << aggSize / (float)aggCount
                  << ", call count: " << aggCount << std::endl;
        aggSize = 0;
        aggCount = 0;
      }
    }

    {
      std::lock_guard<std::mutex> lk(mtxDevice_);

      torch::NoGradGuard ng;
      std::vector<torch::jit::IValue> input;
      input.push_back(tensor_dict::toIValue(batch, device_));
      torch::jit::IValue output;
      {
        std::lock_guard<std::mutex> lk(mtxUpdate_);
        output = jitModel_->get_method(method)(input);
      }
      batcher.set(tensor_dict::fromIValue(output, torch::kCPU, true));
    }
  }
}

}  // namespace rela
