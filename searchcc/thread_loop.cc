// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "searchcc/thread_loop.h"

namespace search {

bool plausible(
    const hle::HanabiGame& game,
    std::vector<int> cardCount,
    const std::vector<std::vector<hle::HanabiCardValue>>& hands) {
  for (const auto& hand : hands) {
    for (const auto& card : hand) {
      int idx = cardToIndex(card, game.NumRanks());
      --cardCount[idx];
      if (cardCount[idx] < 0) {
        return false;
      }
    }
  }
  return true;
}

std::vector<SimHand> SearchThreadLoop::sampleHands() const {
  std::vector<SimHand> simHands;
  if (jointSample_) {
    while (true) {
      if (useSimHands_) {
        std::uniform_int_distribution<> dis(0, simHands_->at(0).size() - 1);
        int index = dis(rng_);
        std::vector<hle::HanabiCardValue> myHand = simHands_->at(myIndex_)[index];
        std::vector<hle::HanabiCardValue> partnerHand =
            simHands_->at(1 - myIndex_)[index];
        simHands.emplace_back(myIndex_, myHand);
        simHands.emplace_back(1 - myIndex_, partnerHand);
        break;
      } else {
        auto mySimHand = myHandDist_.sampleHands(1, &rng_)[0];
        auto partnerSimHand = partnerHandDist_.sampleHands(1, &rng_)[0];
        if (plausible(game_, cardCount_, {mySimHand, partnerSimHand})) {
          simHands.emplace_back(myIndex_, mySimHand);
          simHands.emplace_back(1 - myIndex_, partnerSimHand);
          break;
        }
      }
    }
  } else if (useSimHands_) {
    std::uniform_int_distribution<> dis(0, simHands_->at(0).size() - 1);
    int index = dis(rng_);
    std::vector<hle::HanabiCardValue> myHand = simHands_->at(myIndex_)[index];
    simHands.emplace_back(myIndex_, myHand);
  } else {
    auto mySimHand = myHandDist_.sampleHands(1, &rng_)[0];
    assert(plausible(game_, cardCount_, {mySimHand}));
    simHands.emplace_back(myIndex_, mySimHand);
  }
  return simHands;
}

// the threadloop for collecting data for RL training
SearchThreadLoop::SearchThreadLoop(
    const hle::HanabiState& state,
    const HandDistribution& myHandDist,
    const HandDistribution& partnerHandDist,
    std::vector<std::vector<std::unique_ptr<SimulationActor>>>&& actors,
    std::shared_ptr<std::vector<std::vector<std::vector<hle::HanabiCardValue>>>> simHands,
    bool useSimHands,
    int myIndex,
    int seed,
    int numGame,
    bool jointSample)
    : numEnv_(actors.size())
    , refState_(state)
    , game_(*state.ParentGame())
    , myHandDist_(myHandDist)
    , partnerHandDist_(partnerHandDist)
    , myIndex_(myIndex)
    , jointSample_(jointSample)
    , numGame_(numGame)
    , actors_(std::move(actors))
    , simHands_(simHands)
    , useSimHands_(useSimHands)
    , scores_(numEnv_, -1)
    , rng_(seed) {
  if (numGame_ > 0) {
    // for eval mode
    assert(numEnv_ == 1);
    // if (actors[0].beliefRunner_ != nullptr){
    // TODO sample a lot of hands from actors[0] and store them here
    // tmp = actors[0].sampleHands(state, 10);
    // }
  }

  // get deck + cards in hand
  hle::HanabiState::HanabiDeck deck = state.Deck();
  deck.PutCardsBack(state.Hands()[0].Cards());
  deck.PutCardsBack(state.Hands()[1].Cards());
  cardCount_ = deck.CardCount();

  // important: to avoid GameSimulator from being re-allocated
  envs_.reserve(numEnv_);
  for (int i = 0; i < numEnv_; ++i) {
    int seed = std::abs((int)rng_());
    auto simHands = sampleHands();
    envs_.emplace_back(refState_, simHands, seed);
  }
}

void SearchThreadLoop::mainLoop() {
  while (!terminated()) {
    for (int i = 0; i < numEnv_; ++i) {
      for (auto& actor : actors_[i]) {
        actor->observeBeforeAct(envs_[i]);
      }
    }

    for (int i = 0; i < numEnv_; ++i) {
      int curPlayer = envs_[i].state().CurPlayer();
      int action = -1;
      for (int j = 0; j < (int)actors_[i].size(); ++j) {
        auto action_ = actors_[i][j]->decideAction(envs_[i]);
        if (j == curPlayer) {
          action = action_;
        }
      }
      envs_[i].step(envs_[i].getMove(action));
    }

    for (int i = 0; i < numEnv_; ++i) {
      for (auto& actor : actors_[i]) {
        actor->observeAfterAct(envs_[i]);
      }
    }

    std::vector<bool> endEpisode(numEnv_, false);
    for (int i = 0; i < numEnv_; ++i) {
      for (auto& actor : actors_[i]) {
        endEpisode[i] = (actor->maybeEndEpisode(envs_[i]) || endEpisode[i]);
      }
    }

    for (int i = 0; i < numEnv_; ++i) {
      if (endEpisode[i]) {
        envs_[i].setTerminal(true);
      }

      if (!envs_[i].terminal()) {
        continue;
      }

      scores_[i] = envs_[i].state().Score();
      if (numGame_ > 0) {
        --numGame_;
      }

      if (numGame_ == 0) {
        return;
      }

      auto simHands = sampleHands();
      envs_[i].reset(refState_, simHands);
      for (auto& actor : actors_[i]) {
        actor->reset();
      }
    }
  }
}

}  // namespace search
