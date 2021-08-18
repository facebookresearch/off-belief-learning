// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rela/context.h"
#include "rela/prioritized_replay.h"
#include "rela/r2d2.h"
#include "rela/thread_loop.h"

namespace hle = hanabi_learning_env;

struct GameData {
  GameData(
      const std::vector<hle::HanabiCardValue>& deck,
      const std::vector<hle::HanabiMove>& moves)
      : deck_(deck)
      , moves_(moves) {
  }

  std::vector<hle::HanabiCardValue> deck_;
  std::vector<hle::HanabiMove> moves_;
};

class DataGenLoop : public rela::ThreadLoop {
 public:
  DataGenLoop(
      std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer,
      const std::unordered_map<std::string, std::string>& gameParams,
      const std::vector<GameData>& gameDatas,
      int numPlayer,
      int maxLen,
      bool infLoop,
      bool shuffleColor,
      bool trinary,
      int seed)
      : replayBuffer_(replayBuffer)
      , gameParams_(gameParams)
      , gameDatas_(gameDatas)
      , numPlayer_(numPlayer)
      , maxLen_(maxLen)
      , infLoop_(infLoop)
      , shuffleColor_(shuffleColor)
      , trinary_(trinary)
      , rng_(seed)
      , colorPermutes_(numPlayer)
      , invColorPermutes_(numPlayer) {
    for (int i = 0; i < numPlayer_; ++i) {
      r2d2Buffers_.emplace_back(1, maxLen_, 1);
    }
  }

  virtual void mainLoop() override;

 private:
  void shuffleColor(const hle::HanabiGame game);

  std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;
  const std::unordered_map<std::string, std::string> gameParams_;
  const std::vector<GameData> gameDatas_;
  const int numPlayer_;
  const int maxLen_;
  const bool infLoop_;
  const bool shuffleColor_;
  const bool trinary_;
  std::mt19937 rng_;

  int epoch_ = 0;
  std::vector<std::vector<int>> colorPermutes_;
  std::vector<std::vector<int>> invColorPermutes_;
  std::vector<rela::R2D2Buffer> r2d2Buffers_;
};

class CloneDataGenerator {
 public:
  CloneDataGenerator(
      std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer,
      int numPlayer,
      int maxLen,
      bool shuffleColor,
      bool trinary,
      int numThread)
      : gameDatas_(numThread)
      , replayBuffer_(replayBuffer)
      , numPlayer_(numPlayer)
      , maxLen_(maxLen)
      , shuffleColor_(shuffleColor)
      , trinary_(trinary)
      , numThread_(numThread) {
  }

  void setGameParams(const std::unordered_map<std::string, std::string>& gameParams) {
    gameParams_ = gameParams;
  }

  void addGame(
      const std::vector<hle::HanabiCardValue>& deck,
      const std::vector<hle::HanabiMove>& moves) {
    gameDatas_[nextGameThread_].emplace_back(deck, moves);
    nextGameThread_ = (nextGameThread_ + 1) % numThread_;
  }

  void startDataGeneration(bool infLoop, int seed);

  void terminate() {
    context_ = nullptr;
  }

 protected:
  std::vector<std::vector<GameData>> gameDatas_;
  std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;
  std::unique_ptr<rela::Context> context_;
  std::vector<std::shared_ptr<DataGenLoop>> threads_;

  int numPlayer_;
  int maxLen_;
  bool shuffleColor_;
  bool trinary_;
  int numThread_;

  std::unordered_map<std::string, std::string> gameParams_;
  int nextGameThread_ = 0;
};
