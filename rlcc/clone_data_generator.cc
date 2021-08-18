// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "rlcc/clone_data_generator.h"
#include "rlcc/hanabi_env.h"

void DataGenLoop::shuffleColor(const hle::HanabiGame game) {
  for (int i = 0; i < numPlayer_; ++i) {
    auto& colorPermute = colorPermutes_[i];
    auto& invColorPermute = invColorPermutes_[i];
    colorPermute.clear();
    invColorPermute.clear();
    for (int i = 0; i < game.NumColors(); ++i) {
      colorPermute.push_back(i);
      invColorPermute.push_back(i);
    }
    std::shuffle(colorPermute.begin(), colorPermute.end(), rng_);
    std::sort(invColorPermute.begin(), invColorPermute.end(), [&](int i, int j) {
      return colorPermute[i] < colorPermute[j];
    });
    for (int i = 0; i < (int)colorPermute.size(); ++i) {
      assert(invColorPermute[colorPermute[i]] == i);
    }
  }
}

void DataGenLoop::mainLoop() {
  assert(gameDatas_.size() > 0);
  std::vector<size_t> idxsLeft;
  while (!terminated()) {
    if (idxsLeft.size() <= 0) {
      if (!infLoop_) {
        if (epoch_ == 0) {
          ++epoch_;
        } else {
          break;
        }
      }
      idxsLeft.resize(gameDatas_.size());
      std::iota(idxsLeft.begin(), idxsLeft.end(), 0);
      std::shuffle(idxsLeft.begin(), idxsLeft.end(), rng_);
    }
    size_t idx = idxsLeft.back();
    idxsLeft.pop_back();
    auto gameData = gameDatas_[idx];

    HanabiEnv env(gameParams_, maxLen_, false);
    env.resetWithDeck(gameData.deck_);
    auto& state = env.getHleState();

    if (shuffleColor_) {
      shuffleColor(env.getHleGame());
    }

    for (size_t midx = 0; midx < gameData.moves_.size(); ++midx) {
      auto move = gameData.moves_[midx];
      int curPlayer = env.getCurrentPlayer();
      for (int i = 0; i < numPlayer_; ++i) {
        auto obs = observe(
            state,
            i,
            shuffleColor_,
            colorPermutes_[i],
            invColorPermutes_[i],
            false,     // hideAction
            trinary_,  // trinary for aux task
            false);    // sad
        r2d2Buffers_[i].pushObs(obs);
        int action = -1;
        if (i == curPlayer) {
          if (shuffleColor_ && move.MoveType() == hle::HanabiMove::kRevealColor) {
            auto shuffledMove = move;
            shuffledMove.SetColor(colorPermutes_[i][move.Color()]);
            action = env.getHleGame().GetMoveUid(shuffledMove);
          } else {
            action = env.getHleGame().GetMoveUid(move);
          }
        } else {
          action = env.noOpUid();
        }
        r2d2Buffers_[i].pushAction({{"a", torch::tensor(action)}});
      }

      env.step(move);
      float reward = env.stepReward();
      float terminal = env.terminated();
      if (midx == gameData.moves_.size() - 1) {
        terminal = true;
      }
      for (int i = 0; i < numPlayer_; ++i) {
        r2d2Buffers_[i].pushReward(reward);
        r2d2Buffers_[i].pushTerminal(terminal);
      }
    }

    for (int i = 0; i < numPlayer_; ++i) {
      replayBuffer_->add(r2d2Buffers_[i].popTransition(), 1.0);
    }
  }  // while (!terminated())
};

void CloneDataGenerator::startDataGeneration(bool infLoop, int seed) {
  std::mt19937 rng(seed);
  context_ = std::make_unique<rela::Context>();
  for (int i = 0; i < numThread_; ++i) {
    int seed = (int)rng();
    auto thread = std::make_shared<DataGenLoop>(
        replayBuffer_,
        gameParams_,
        gameDatas_[i],
        numPlayer_,
        maxLen_,
        infLoop,
        shuffleColor_,
        trinary_,
        seed);
    context_->pushThreadLoop(thread);
    threads_.push_back(thread);
  }
  context_->start();
}
