// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "rlcc/utils.h"

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"

// return reward, terminal
std::tuple<float, bool> applyMove(
    hle::HanabiState& state, hle::HanabiMove move, bool forceTerminal) {
  float prevScore = state.Score();
  state.ApplyMove(move);

  bool terminal = state.IsTerminal();
  float reward = state.Score() - prevScore;
  if (forceTerminal) {
    reward = 0 - prevScore;
    terminal = true;
  }

  if (!terminal) {
    // chance player
    while (state.CurPlayer() == hle::kChancePlayerId) {
      state.ApplyRandomChance();
    }
  }

  return {reward, terminal};
}

rela::TensorDict observe(
    const hle::HanabiState& state,
    int playerIdx,
    bool shuffleColor,
    const std::vector<int>& colorPermute,
    const std::vector<int>& invColorPermute,
    bool hideAction,
    bool trinary,
    bool sad) {
  const auto& game = *(state.ParentGame());
  auto obs = hle::HanabiObservation(state, playerIdx, true);
  auto encoder = hle::CanonicalObservationEncoder(&game);

  std::vector<float> vS = encoder.Encode(
      obs,
      true,  // splitPrivatePulic/convertSad will mask out private hand
      std::vector<int>(),  // shuffle card
      shuffleColor,
      colorPermute,
      invColorPermute,
      hideAction);

  rela::TensorDict feat;
  if (!sad) {
    feat = splitPrivatePublic(vS, game);
  } else {
    // only for evaluation
    auto vA =
        encoder.EncodeLastAction(obs, std::vector<int>(), shuffleColor, colorPermute);
    feat = convertSad(vS, vA, game);
  }

  if (trinary) {
    auto vOwnHand = encoder.EncodeOwnHandTrinary(obs);
    feat["own_hand"] = torch::tensor(vOwnHand);
  } else {
    auto vOwnHand = encoder.EncodeOwnHand(obs, shuffleColor, colorPermute);
    std::vector<float> vOwnHandARIn(vOwnHand.size(), 0);
    int end = (game.HandSize() - 1) * game.NumColors() * game.NumRanks();
    std::copy(
        vOwnHand.begin(),
        vOwnHand.begin() + end,
        vOwnHandARIn.begin() + game.NumColors() * game.NumRanks());
    feat["own_hand"] = torch::tensor(vOwnHand);
    feat["own_hand_ar_in"] = torch::tensor(vOwnHandARIn);
    auto privARV0 =
        encoder.EncodeARV0Belief(obs, std::vector<int>(), shuffleColor, colorPermute);
    feat["priv_ar_v0"] = torch::tensor(privARV0);
  }

  // legal moves
  const auto& legalMove = state.LegalMoves(playerIdx);
  std::vector<float> vLegalMove(game.MaxMoves() + 1);
  for (auto move : legalMove) {
    if (shuffleColor && move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      int permColor = colorPermute[move.Color()];
      move.SetColor(permColor);
    }

    auto uid = game.GetMoveUid(move);
    vLegalMove[uid] = 1;
  }
  if (legalMove.size() == 0) {
    vLegalMove[game.MaxMoves()] = 1;
  }

  feat["legal_move"] = torch::tensor(vLegalMove);
  return feat;
}

std::tuple<rela::TensorDict, std::vector<int>, std::vector<float>> beliefModelObserve(
    const hle::HanabiState& state,
    int playerIdx,
    bool shuffleColor,
    const std::vector<int>& colorPermute,
    const std::vector<int>& invColorPermute,
    bool hideAction,
    bool publ) {
  const auto& game = *(state.ParentGame());
  auto obs = hle::HanabiObservation(state, playerIdx, true);
  auto encoder = hle::CanonicalObservationEncoder(&game);

  std::vector<float> vS = encoder.Encode(
      obs,
      true,
      std::vector<int>(),  // shuffle card
      shuffleColor,
      colorPermute,
      invColorPermute,
      hideAction);
  rela::TensorDict feat = splitPrivatePublic(vS, game);
  auto [v0, cardCount] = encoder.EncodeV0Belief(
      obs, std::vector<int>(), shuffleColor, colorPermute, publ);
  feat["v0"] = torch::tensor(v0);
  return {feat, cardCount, v0};
}

rela::TensorDict applyModel(
    const rela::TensorDict& obs,
    rela::BatchRunner& runner,
    rela::TensorDict& hid,
    const std::string& method) {
  rela::TensorDict input;
  // add batch dim
  for (auto& kv : obs) {
    input[kv.first] = kv.second.unsqueeze(0);
  }
  for (auto& kv : hid) {
    auto ret = input.emplace(kv.first, kv.second.unsqueeze(0));
    assert(ret.second);
  }

  auto reply = runner.blockCall(method, input);

  // remove batch dim
  for (auto& kv : reply) {
    reply[kv.first] = kv.second.squeeze(0);
  }

  // in-place update hid
  for (auto& kv : hid) {
    auto newHidIt = reply.find(kv.first);
    assert(newHidIt != reply.end());
    auto newHid = newHidIt->second;
    assert(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    reply.erase(newHidIt);
  }

  return reply;
}

std::vector<std::vector<float>> extractPerCardBelief(
    const std::vector<float>& encoding, const hle::HanabiGame& game, const int handSize) {
  int numColors = game.NumColors();
  int numRanks = game.NumRanks();
  int bitsPerCard = numColors * numRanks;
  int perCardLen = bitsPerCard + numColors + numRanks;
  assert(perCardLen * game.HandSize() == (int)encoding.size());

  std::vector<std::vector<float>> beliefs(handSize, std::vector<float>(bitsPerCard));
  for (int j = 0; j < handSize; ++j) {
    int start = j * perCardLen;
    int end = start + bitsPerCard;

    std::copy(encoding.begin() + start, encoding.begin() + end, beliefs[j].begin());
  }
  return beliefs;
}
