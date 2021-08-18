// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "rela/tensor_dict.h"
#include "rela/utils.h"

namespace rela {

inline TensorDict allocateBatchStorage(const TensorDict& data, int size) {
  TensorDict storage;
  for (const auto& kv : data) {
    auto t = kv.second.sizes();
    std::vector<int64_t> sizes;
    sizes.push_back(size);
    for (size_t i = 0; i < t.size(); ++i) {
      sizes.push_back(t[i]);
    }

    storage[kv.first] = torch::zeros(sizes, kv.second.dtype());
  }
  return storage;
}

class FutureReply_;

class FutureReply {
 public:
  FutureReply()
      : fut_(nullptr)
      , slot(-1) {
  }

  FutureReply(std::shared_ptr<FutureReply_> fut, int slot)
      : fut_(std::move(fut))
      , slot(slot) {
  }

  TensorDict get();

  bool isNull() const {
    return fut_ == nullptr;
  }

 private:
  std::shared_ptr<FutureReply_> fut_;
  int slot;
};

using Future = FutureReply;

class Batcher {
 public:
  Batcher(int batchsize);

  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  ~Batcher() {
    exit();
  }

  void exit() {
    {
      std::unique_lock<std::mutex> lk(mNextSlot_);
      exit_ = true;
    }
    cvGetBatch_.notify_all();
  }

  bool terminated() {
    return exit_;
  }

  // send data into batcher
  FutureReply send(const TensorDict& t);

  // get batch input from batcher
  TensorDict get();

  // set batch reply for batcher
  void set(TensorDict&& t);

 private:
  const int batchsize_;

  int nextSlot_;
  int numActiveWrite_;
  std::condition_variable cvNextSlot_;

  TensorDict fillingBuffer_;
  std::shared_ptr<FutureReply_> fillingReply_;

  TensorDict filledBuffer_;
  std::shared_ptr<FutureReply_> filledReply_;

  bool exit_ = false;
  std::condition_variable cvGetBatch_;
  std::mutex mNextSlot_;
};

}  // namespace rela
