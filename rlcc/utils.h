// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_observation.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_state.h"

#include "rela/batch_runner.h"
#include "rela/tensor_dict.h"

namespace hle = hanabi_learning_env;

inline int cardToIndex(const hle::HanabiCardValue& card, int numRank) {
  return card.Color() * numRank + card.Rank();
}

inline hle::HanabiCardValue indexToCard(int index, int numRank) {
  return hle::HanabiCardValue(index / numRank, index % numRank);
}

inline rela::TensorDict splitPrivatePublic(
    const std::vector<float>& feat, const hle::HanabiGame& game) {
  int bitsPerHand = game.HandSize() * game.NumColors() * game.NumRanks();
  // remove my hand, should be zero anyway
  std::vector<float> vPriv(feat.begin() + bitsPerHand, feat.end());
  // remove all private observation
  std::vector<float> vPubl(feat.begin() + bitsPerHand * game.NumPlayers(), feat.end());
  return {{"priv_s", torch::tensor(vPriv)}, {"publ_s", torch::tensor(vPubl)}};
}

inline rela::TensorDict convertSad(
    const std::vector<float>& feat,
    const std::vector<float>& sad,
    const hle::HanabiGame& game) {
  int bitsPerHand = game.HandSize() * game.NumColors() * game.NumRanks();
  std::vector<float> vPriv = feat;
  std::fill(vPriv.begin(), vPriv.begin() + bitsPerHand, 0);
  vPriv.insert(vPriv.end(), sad.begin(), sad.end());
  auto ret = splitPrivatePublic(vPriv, game);
  // // for compatibility with legacy model
  // ret["s"] = torch::tensor(vPriv);
  return ret;
}

// return reward, terminal
std::tuple<float, bool> applyMove(
    hle::HanabiState& state, hle::HanabiMove move, bool forceTerminal);

rela::TensorDict observe(
    const hle::HanabiState& state,
    int playerIdx,
    bool shuffleColor,
    const std::vector<int>& colorPermute,
    const std::vector<int>& invColorPermute,
    bool hideAction,
    bool trinary,
    bool sad);

inline rela::TensorDict observe(
    const hle::HanabiState& state, int playerIdx, bool hideAction) {
  return observe(
      state,
      playerIdx,
      false,
      std::vector<int>(),
      std::vector<int>(),
      hideAction,
      true,
      false);
}

inline rela::TensorDict observeSAD(const hle::HanabiState& state, int playerIdx) {
  return observe(
      state, playerIdx, false, std::vector<int>(), std::vector<int>(), false, true, true);
}

inline std::unique_ptr<hle::HanabiHistoryItem> getLastNonDealMove(
    const std::vector<hle::HanabiHistoryItem>& past_moves) {
  auto it = std::find_if(
      past_moves.begin(), past_moves.end(), [](const hle::HanabiHistoryItem& item) {
        return item.move.MoveType() != hle::HanabiMove::Type::kDeal;
      });
  if (it == past_moves.end()) {
    return nullptr;
  }
  return std::make_unique<hle::HanabiHistoryItem>(*it);
}

inline std::unique_ptr<hle::HanabiHistoryItem> getLastNonDealMoveReverse(
    const std::vector<hle::HanabiHistoryItem>& past_moves) {
  auto it = std::find_if(
      past_moves.rbegin(), past_moves.rend(), [](const hle::HanabiHistoryItem& item) {
        return item.move.MoveType() != hle::HanabiMove::Type::kDeal;
      });
  if (it == past_moves.rend()) {
    return nullptr;
  }
  return std::make_unique<hle::HanabiHistoryItem>(*it);
}

inline std::tuple<
    rela::TensorDict,
    std::unique_ptr<hle::HanabiHistoryItem>,
    std::vector<int>,
    hle::HanabiHand>
observeForSearch(
    const hle::HanabiState& state, int playerIdx, bool hideAction, bool publCount) {
  assert(!state.IsTerminal());
  auto feat = ::observe(state, playerIdx, hideAction);
  auto obs = hle::HanabiObservation(state, playerIdx, false);
  auto lastMove = getLastNonDealMove(obs.LastMoves());
  auto cardCount = hle::ComputeCardCount(
      *state.ParentGame(), obs, false, std::vector<int>(), publCount);
  hle::HanabiHand myHand = state.Hands()[playerIdx];

  return {feat, std::move(lastMove), cardCount, myHand};
}

std::tuple<rela::TensorDict, std::vector<int>, std::vector<float>> beliefModelObserve(
    const hle::HanabiState& state,
    int playerIdx,
    bool shuffleColor,
    const std::vector<int>& colorPermute,
    const std::vector<int>& invColorPermute,
    bool hideAction,
    bool publ=false);

inline std::vector<float> createOneHot(int oneHotValue, int oneHotLength) {
  assert(oneHotValue < oneHotLength);
  std::vector<float> oneHot;
  for (int i = 0; i < oneHotLength; ++i) {
    if (i == oneHotValue) {
      oneHot.push_back(1);
    } else {
      oneHot.push_back(0);
    }
  }
  return oneHot;
}

inline rela::TensorDict getH0(rela::BatchRunner& runner, int batchsize) {
  std::vector<torch::jit::IValue> input{batchsize};
  auto model = runner.jitModel();
  auto output = model.get_method("get_h0")(input);
  auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
  return h0;
}

// apply model on a single observation, hid will be in-place updated
rela::TensorDict applyModel(
    const rela::TensorDict& obs,
    rela::BatchRunner& runner,
    rela::TensorDict& hid,
    const std::string& method);

std::vector<std::vector<float>> extractPerCardBelief(
    const std::vector<float>& encoding, const hle::HanabiGame& game, const int handSize);
