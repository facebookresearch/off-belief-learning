// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "searchcc/rl_search.h"
#include <chrono>

namespace search {

std::vector<std::shared_ptr<SearchThreadLoop>> RLSearchActor::startDataGeneration_(
    const GameSimulator& env,
    std::shared_ptr<rela::RNNPrioritizedReplay> replay,
    int numRlStep,
    const std::vector<float>& epsList,
    std::mt19937& rng,
    int numThread,
    int numEnvPerThread,
    int numGame,
    bool train,
    std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
    bool useSimHands,
    bool beliefMode) const {
  assert(context_ == nullptr);
  assert(partner_ != nullptr);
  assert(!(train && beliefMode));

  context_ = std::make_unique<rela::Context>();
  std::vector<std::shared_ptr<SearchThreadLoop>> threads;

  auto ptr =
      std::make_shared<std::vector<std::vector<std::vector<hle::HanabiCardValue>>>>(
          simHands);

  for (int i = 0; i < numThread; ++i) {
    std::vector<std::vector<std::unique_ptr<SimulationActor>>> actors(numEnvPerThread);
    for (int j = 0; j < numEnvPerThread; ++j) {
      int seed = std::abs((int)rng());
      std::unique_ptr<SimulationActor> me;
      if (train) {
        me = std::make_unique<SimulationActor>(
            model_, numRlStep, epsList, replay, nStep_, gamma_, seed);
      } else if (beliefMode) {
        me = std::make_unique<SimulationActor>(model_, replay, nStep_);
      } else {
        me = std::make_unique<SimulationActor>(model_, numRlStep);
      }
      std::unique_ptr<SimulationActor> partner;
      if (jointSearch_) {
        if (train) {
          // we also need data from out partner's perspective
          partner = std::make_unique<SimulationActor>(
              partner_->model_, numRlStep, epsList, replay, nStep_, gamma_, seed);
        } else if (beliefMode) {
          partner = std::make_unique<SimulationActor>(partner_->model_, replay, nStep_);
        } else {
          // partner also uses rl in evaluation
          partner = std::make_unique<SimulationActor>(partner_->model_, numRlStep);
        }
      } else {
        partner = std::make_unique<SimulationActor>(partner_->model_);
      }
      // in fact, order of the actor does not matter
      if (index == 1) {
        actors[j].push_back(std::move(partner));
        actors[j].push_back(std::move(me));
      } else {
        actors[j].push_back(std::move(me));
        actors[j].push_back(std::move(partner));
      }
    }

    int seed = std::abs((int)rng());

    auto thread = std::make_shared<SearchThreadLoop>(
        env.state(),
        handDist_,
        partner_->handDist_,
        std::move(actors),
        ptr,
        useSimHands,
        index,
        seed,
        numGame,
        jointSearch_);

    context_->pushThreadLoop(thread);
    threads.push_back(thread);
  }

  context_->start();
  return threads;
}

std::vector<float> RLSearchActor::runSimGames(
    const GameSimulator& env,
    int numGame,
    int numRlStep,
    int seed,
    std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
    bool useSimHands) const {
  assert(callOrder_ == 1 || useSimHands);

  std::mt19937 rng(seed);
  auto threads = startDataGeneration_(
      env,
      nullptr,
      numRlStep,
      {0},
      rng,
      numGame,
      1,
      1,
      false,
      simHands,
      useSimHands,
      false);
  while (!context_->terminated()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::vector<float> scores;
  for (const auto& t : threads) {
    const auto& tScores = t->getScores();
    scores.insert(scores.end(), tScores.begin(), tScores.end());
  }
  context_ = nullptr;
  return scores;
}

void RLSearchActor::startDataGeneration(
    const GameSimulator& env,
    std::shared_ptr<rela::RNNPrioritizedReplay> replay,
    int numRlStep,
    std::vector<std::vector<std::vector<hle::HanabiCardValue>>> simHands,
    bool useSimHands,
    bool beliefMode) const {
  assert(callOrder_ == 1 || useSimHands);
  startDataGeneration_(
      env,
      replay,
      numRlStep,
      epsList_,
      rng_,
      numThread_,
      numEnvPerThread_,
      -1,
      !beliefMode,
      simHands,
      useSimHands,
      beliefMode);
}

void RLSearchActor::stopDataGeneration() const {
  // destructor of context will terminate the ThreadLoops
  context_ = nullptr;
}

float searchMove(
    const hle::HanabiState& state,
    hle::HanabiMove move,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int> seeds,
    const HybridModel& me,
    const HybridModel& partner) {
  std::vector<std::vector<HybridModel>> players;
  if (me.index == 0) {
    players = std::vector<std::vector<HybridModel>>(hands.size(), {me, partner});
  } else {
    players = std::vector<std::vector<HybridModel>>(hands.size(), {partner, me});
  }

  std::vector<GameSimulator> games;
  games.reserve(hands.size());
  for (size_t i = 0; i < hands.size(); ++i) {
    std::vector<SimHand> simHands{
        SimHand{me.index, hands[i]},
    };
    games.emplace_back(state, simHands, seeds[i]);
  }

  size_t terminated = 0;
  std::vector<int> notTerminated;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(!games[i].terminal());
    games[i].step(move);
    if (!games[i].terminal()) {
      notTerminated.push_back(i);
    } else {
      ++terminated;
    }
  }

  while (!notTerminated.empty()) {
    std::vector<int> newNotTerminated;
    for (auto i : notTerminated) {
      assert(!games[i].state().IsTerminal());
      for (auto& actor : players[i]) {
        actor.observeBeforeAct(games[i], 0);
      }
    }

    for (auto i : notTerminated) {
      auto& game = games[i];
      int action = -1;
      for (auto& actor : players[i]) {
        int a = actor.decideAction(game, false);
        if (actor.index == game.state().CurPlayer()) {
          action = a;
        }
      }
      game.step(game.getMove(action));
      if (!game.terminal()) {
        newNotTerminated.push_back(i);
      } else {
        ++terminated;
      }
    }

    notTerminated = newNotTerminated;
  }
  assert(terminated == games.size());

  std::vector<float> scores(games.size());
  float mean = 0;
  for (size_t i = 0; i < games.size(); ++i) {
    assert(games[i].terminal());
    scores[i] = games[i].score();
    mean += scores[i];
  }
  mean = mean / scores.size();
  return mean;
}
rela::TensorDict RLSearchActor::getBeliefHidden() {
  return beliefRnnHid_;
}

rela::TensorDict RLSearchActor::getModelBeliefHidden() {
  return model_.getBeliefHid();
}

std::vector<std::vector<std::vector<hle::HanabiCardValue>>> RLSearchActor::sampleHands(
    const hle::HanabiState& state, int numSample) {
  if (publBelief_) {
    return sampleHandsPubl_(state, numSample);
  } else {
    return sampleHandsPriv_(state, numSample);
  }
}

std::vector<std::vector<std::vector<hle::HanabiCardValue>>> RLSearchActor::sampleHandsPubl_(
    const hle::HanabiState& state, int numSample) {
  const auto& game = *state.ParentGame();
  auto [obs, publCardCount, v0] = beliefModelObserve(
      state, index, false, std::vector<int>(), std::vector<int>(), hideAction, true);
  (void)v0;
  auto reply = applyModel(obs, *beliefRunner_, beliefRnnHid_, "sample");
  auto sample = reply.at("sample");
  const auto& myHandTrue = state.Hands()[index];
  int myHandSize = myHandTrue.Cards().size();
  const auto& partnerHandTrue = state.Hands()[1 - index];
  int partnerHandSize = partnerHandTrue.Cards().size();
  std::vector<std::vector<hle::HanabiCardValue>> myHands;
  std::vector<std::vector<hle::HanabiCardValue>> partnerHands;
  std::cout << "sample dim: " << sample.sizes() << std::endl;

  auto sampleAcc = sample.accessor<int64_t, 2>();
  assert(sampleAcc.size(0) >= numSample);
  int sampleExhausted = sampleAcc.size(0);

  for (int i = 0; i < sampleAcc.size(0); ++i) {
    std::vector<hle::HanabiCardValue> myCards;
    std::vector<hle::HanabiCardValue> partnerCards;
    auto cardRemain = publCardCount;

    for (int j = 0; j < partnerHandSize + myHandSize; ++j) {
      int idx = sampleAcc[i][j];
      if (cardRemain[idx] == 0) {
        break;
      }
      --cardRemain[idx];

      auto card = indexToCard(idx, game.NumRanks());
      // cards.push_back(card);

      if (j < myHandSize) {
        myCards.push_back(card);
      } else {
        partnerCards.push_back(card);
      }
    }

    if (((int)partnerCards.size() == partnerHandSize &&
         (int)myCards.size() == myHandSize && partnerHandTrue.CanSetCards(partnerCards) &&
         myHandTrue.CanSetCards(myCards))) {
      myHands.push_back(myCards);
      partnerHands.push_back(partnerCards);
      if ((int)partnerHands.size() >= numSample) {
        sampleExhausted = i + 1;
        break;
      }
    }
  }
  std::cout << "Need: " << numSample << " samples, get: " << partnerHands.size()
            << " samples." << std::endl;
  std::cout << "Success rate: " << partnerHands.size() / (float)sampleExhausted
            << std::endl;
  // Reset the hidden state because sampling modifies it!
  beliefRnnHid_ = getModelBeliefHidden();
  if (index == 0) {
    return {myHands, partnerHands};
  } else {
    return {partnerHands, myHands};
  }
}

std::vector<std::vector<std::vector<hle::HanabiCardValue>>> RLSearchActor::sampleHandsPriv_(
    const hle::HanabiState& state, int numSample) {
  const auto& game = *state.ParentGame();
  auto [obs, privCardCount, v0] = beliefModelObserve(
      state, index, false, std::vector<int>(), std::vector<int>(), hideAction, false);
  (void)v0;
  auto reply = applyModel(obs, *beliefRunner_, beliefRnnHid_, "sample");
  auto sample = reply.at("sample");
  const auto& myHandTrue = state.Hands()[index];
  const auto& partnerHandTrue = state.Hands()[1 - index];
  int myHandSize = myHandTrue.Cards().size();
  std::vector<std::vector<hle::HanabiCardValue>> myHands;
  std::vector<std::vector<hle::HanabiCardValue>> partnerHands;
  std::cout << "sample dim: " << sample.sizes() << std::endl;

  auto sampleAcc = sample.accessor<int64_t, 2>();
  assert(sampleAcc.size(0) >= numSample);
  int sampleExhausted = sampleAcc.size(0);

  for (int i = 0; i < sampleAcc.size(0); ++i) {
    std::vector<hle::HanabiCardValue> myCards;
    auto cardRemain = privCardCount;

    for (int j = 0; j < myHandSize; ++j) {
      int idx = sampleAcc[i][j];
      if (cardRemain[idx] == 0) {
        break;
      }
      --cardRemain[idx];

      auto card = indexToCard(idx, game.NumRanks());
      // cards.push_back(card);
      myCards.push_back(card);
    }

    if (((int)myCards.size() == myHandSize && myHandTrue.CanSetCards(myCards))) {
      myHands.push_back(myCards);
      partnerHands.push_back(partnerHandTrue.CardValues());
      if ((int)partnerHands.size() >= numSample) {
        sampleExhausted = i + 1;
        break;
      }
    }
  }
  std::cout << "Need: " << numSample << " samples, get: " << partnerHands.size()
            << " samples." << std::endl;
  std::cout << "Success rate: " << partnerHands.size() / (float)sampleExhausted
            << std::endl;
  // Reset the hidden state because sampling modifies it!
  beliefRnnHid_ = getModelBeliefHidden();
  if (index == 0) {
    return {myHands, partnerHands};
  } else {
    return {partnerHands, myHands};
  }
}

// should be called after decideAction?
hle::HanabiMove RLSearchActor::spartaSearch(
    const GameSimulator& env, hle::HanabiMove bpMove, int numSearch, float threshold) {
  torch::NoGradGuard ng;

  const auto& state = env.state();
  assert(state.CurPlayer() == index);
  auto legalMoves = state.LegalMoves(index);

  int numSearchPerMove = numSearch / legalMoves.size();
  std::cout << "SPARTA will run " << numSearchPerMove << " searches on "
            << legalMoves.size() << " legal moves" << std::endl;
  auto simHands = handDist_.sampleHands(numSearchPerMove, &rng_);
  std::vector<int> seeds;
  for (size_t i = 0; i < simHands.size(); ++i) {
    seeds.push_back(int(rng_()));
  }

  std::vector<std::future<float>> futMoveScores;
  for (auto move : legalMoves) {
    futMoveScores.push_back(std::async(
        std::launch::async,
        searchMove,
        state,
        move,
        simHands,
        seeds,
        model_,
        partner_->model_));
  }

  hle::HanabiMove bestMove = bpMove;
  float bpScore = -1;
  float bestScore = -1;

  std::cout << "SPARTA scores for moves:" << std::endl;
  for (size_t i = 0; i < legalMoves.size(); ++i) {
    float score = futMoveScores[i].get();
    auto move = legalMoves[i];
    if (move == bpMove) {
      assert(bpScore == -1);
      bpScore = score;
    }
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    std::cout << move.ToString() << ": " << score << std::endl;
  }

  std::cout << "SPARTA best - bp: " << bestScore - bpScore << std::endl;
  if (bestScore - bpScore >= threshold) {
    std::cout << "SPARTA changes move from " << bpMove.ToString() << " to "
              << bestMove.ToString() << std::endl;
    return bestMove;
  } else {
    return bpMove;
  }
}

}  // namespace search
