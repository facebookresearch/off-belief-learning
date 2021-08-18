// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "rela/batcher.h"
#include "rela/utils.h"

namespace rela {

class FutureReply_ {
 public:
  FutureReply_()
      : ready_(false) {
  }

  TensorDict get(int slot) {
    std::unique_lock<std::mutex> lk(mReady_);
    cvReady_.wait(lk, [this] { return ready_; });
    lk.unlock();

    TensorDict e;
    for (const auto& kv : data_) {
      assert(slot >= 0 && slot < kv.second.size(0));
      e[kv.first] = kv.second[slot];
    }
    return e;
  }

  void set(TensorDict&& t) {
    {
      std::lock_guard<std::mutex> lk(mReady_);
      ready_ = true;
      data_ = std::move(t);
    }
    cvReady_.notify_all();
  }

 private:
  // no need for protection, only set() can set it
  TensorDict data_;

  std::mutex mReady_;
  bool ready_;
  std::condition_variable cvReady_;
};

TensorDict FutureReply::get() {
  assert(fut_ != nullptr);
  auto ret = fut_->get(slot);
  fut_ = nullptr;
  return ret;
}

Batcher::Batcher(int batchsize)
    : batchsize_(batchsize)
    , nextSlot_(0)
    , numActiveWrite_(0)
    , fillingReply_(std::make_shared<FutureReply_>())
    , filledReply_(nullptr) {
  assert(batchsize_ > 0);
}

// send data into batcher
FutureReply Batcher::send(const TensorDict& t) {
  std::unique_lock<std::mutex> lk(mNextSlot_);

  // init buffer
  if (fillingBuffer_.empty()) {
    assert(filledBuffer_.empty());
    fillingBuffer_ = allocateBatchStorage(t, batchsize_);
    filledBuffer_ = allocateBatchStorage(t, batchsize_);
  } else {
    if (t.size() != fillingBuffer_.size()) {
      std::cout << "key in buffer: " << std::endl;
      utils::printMapKey(fillingBuffer_);
      std::cout << "key in data: " << std::endl;
      utils::printMapKey(t);
      assert(false);
    }
  }

  assert(nextSlot_ <= batchsize_);
  // wait if current batch is full and not extracted
  cvNextSlot_.wait(lk, [this] { return nextSlot_ < batchsize_; });

  int slot = nextSlot_;
  ++nextSlot_;
  ++numActiveWrite_;
  lk.unlock();

  // this will copy
  for (const auto& kv : t) {
    if (fillingBuffer_[kv.first][slot].sizes() != kv.second.sizes()) {
      std::cout << "cannot batch data, batcher need size: "
                << fillingBuffer_[kv.first][slot].sizes()
                << ", get: " << kv.second.sizes() << std::endl;
    }
    fillingBuffer_[kv.first][slot] = kv.second;
  }

  // batch has not been extracted yet
  assert(numActiveWrite_ > 0);
  assert(fillingReply_ != nullptr);
  auto reply = fillingReply_;
  lk.lock();
  --numActiveWrite_;
  lk.unlock();
  if (numActiveWrite_ == 0) {
    cvGetBatch_.notify_one();
  }
  return FutureReply(reply, slot);
}

// get batch input from batcher
TensorDict Batcher::get() {
  std::unique_lock<std::mutex> lk(mNextSlot_);
  cvGetBatch_.wait(
      lk, [this] { return (nextSlot_ > 0 && numActiveWrite_ == 0) || exit_; });

  if (exit_) {
    return TensorDict();
  }

  int bsize = nextSlot_;
  nextSlot_ = 0;
  // assert previous reply has been handled
  assert(filledReply_ == nullptr);
  std::swap(fillingBuffer_, filledBuffer_);
  std::swap(fillingReply_, filledReply_);
  fillingReply_ = std::make_shared<FutureReply_>();

  lk.unlock();
  cvNextSlot_.notify_all();

  TensorDict batch;
  for (const auto& kv : filledBuffer_) {
    batch[kv.first] = kv.second.narrow(0, 0, bsize).contiguous();
  }

  return batch;
}

// set batch reply for batcher
void Batcher::set(TensorDict&& t) {
  for (const auto& kv : t) {
    assert(kv.second.device().is_cpu());
  }
  filledReply_->set(std::move(t));
  filledReply_ = nullptr;
}
}  // namespace rela
