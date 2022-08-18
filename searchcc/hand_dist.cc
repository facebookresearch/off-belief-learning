// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <future>

#include "searchcc/game_sim.h"
#include "searchcc/hand_dist.h"

namespace search {

bool handInBelief(
    const hle::HanabiHand& hand, const HandDistribution& handDist) {
  auto cards = hand.CardValues();
  // for (auto& c : cards) {
  //   std::cout << c.ToString() << ", ";
  // }
  // std::cout << std::endl;
  for (int i = 0; i < handDist.size(); ++i) {
    auto belief = handDist.getHand(i);
    if (cards == belief) {
      return true;
    }
  }
  return false;
}

HandDistribution publicToPrivate(
    const HandDistribution& handDist, const std::vector<int> cardCount) {
  auto privHandDist = handDist;
  privHandDist.updateOccurance(cardCount, true);
  privHandDist.computeCdf();
  return privHandDist;
}

void initAllPossibleHands_(
    int numColor,
    int numRank,
    int handSize,
    const std::vector<hle::HanabiCardValue>& partialHand,
    std::vector<int>& cardCount,
    std::vector<std::vector<hle::HanabiCardValue>>& allHands) {
  for (int cardIndex = 0; cardIndex < numColor * numRank; ++cardIndex) {
    if (cardCount[cardIndex] == 0) {
      continue;
    }
    auto newPartialHand = partialHand;
    newPartialHand.push_back(indexToCard(cardIndex, numRank));
    if ((int)newPartialHand.size() == handSize) {
      allHands.push_back(newPartialHand);
    } else {
      --cardCount[cardIndex];
      initAllPossibleHands_(
          numColor, numRank, handSize, newPartialHand, cardCount, allHands);
      ++cardCount[cardIndex];
    }
  }
}

std::vector<std::vector<hle::HanabiCardValue>> initAllPossibleHands(
    const hle::HanabiGame& game, std::vector<int> cardCount) {
  std::vector<std::vector<hle::HanabiCardValue>> allHands;
  initAllPossibleHands_(
      game.NumColors(),
      game.NumRanks(),
      game.HandSize(),
      std::vector<hle::HanabiCardValue>(),
      cardCount,
      allHands);
  return allHands;
}

// assert false if occurance is zero for some hand
std::tuple<std::vector<int>, int64_t> computeNumOccurance(
    const std::vector<std::vector<hle::HanabiCardValue>>& hands,
    const std::vector<int>& cardCount,
    const std::function<int(const hle::HanabiCardValue&)> cardToIndex,
    bool allowZero) {
  std::vector<int> occurance(hands.size(), 0);
  int64_t sum = 0;
  for (size_t i = 0; i < hands.size(); ++i) {
    const auto& hand = hands[i];
    std::vector<int> cardOccurance(cardCount.size(), 0);
    float count = 1.0;
    for (const auto& card : hand) {
      int index = cardToIndex(card);
      assert(cardCount[index] >= 0);
      if (cardCount[index] == 0) {
        count = 0.0;
        break;
      }

      ++cardOccurance[index];
      if (cardOccurance[index] == 1) {
        count *= cardCount[index];
      } else {
        count *= (cardCount[index] - cardOccurance[index] + 1);
      }
    }
    if (count == 0 && !allowZero) {
      for (auto& card : hand) {
        std::cout << card.ToString() << ", ";
      }
      assert(false);
    }
    assert((int)count == count);
    occurance[i] = (int)count;
    sum += (int)count;
  }
  return {occurance, sum};
}

void HandDistribution::init_(
    const hle::HanabiGame& game, const std::vector<int>& cardCount) {
  numColor_ = game.NumColors();
  numRank_ = game.NumRanks();

  allHands_ = initAllPossibleHands(game, cardCount);
  updateOccurance(cardCount, false);
}

void HandDistribution::updateOccurance(
    const std::vector<int>& cardCount, bool filterZero) {
  bool allowZero = filterZero;
  std::tie(numOccurances_, sumOccurance_) = computeNumOccurance(
      allHands_,
      cardCount,
      [rank = numRank_](const hle::HanabiCardValue& card) {
        return cardToIndex(card, rank);
      },
      allowZero);

  if (!filterZero) {
    return;
  }

  std::vector<std::vector<hle::HanabiCardValue>> newHands;
  std::vector<int> newOccurances;

  for (int i = 0; i < (int)allHands_.size(); ++i) {
    if (numOccurances_[i] == 0) {
      continue;
    }

    newHands.push_back(allHands_[i]);
    newOccurances.push_back(numOccurances_[i]);
  }

  allHands_ = std::move(newHands);
  numOccurances_ = std::move(newOccurances);
}

void HandDistribution::filterExact(
    std::vector<hle::HanabiCardValue> hand, size_t extraHand) {
  std::vector<std::vector<hle::HanabiCardValue>> newHands;
  std::vector<int> newOccurances;
  int64_t newSum = 0;

  for (size_t i = 0; i < allHands_.size(); ++i) {
    if (i < extraHand || hand == allHands_[i]) {
      newHands.push_back(allHands_[i]);
      newOccurances.push_back(numOccurances_[i]);
      newSum += numOccurances_[i];
    }
  }
  allHands_ = newHands;
  numOccurances_ = newOccurances;
  sumOccurance_ = newSum;
}

void HandDistribution::filterMePlayDiscard(
    const hle::HanabiHistoryItem& move, const std::vector<int>& cardCount, int handSize) {
  std::vector<std::vector<hle::HanabiCardValue>> newHands;

  int idxPlayed = move.move.CardIndex();
  for (size_t i = 0; i < allHands_.size(); ++i) {
    auto& hand = allHands_[i];
    if (hand[idxPlayed].Color() != move.color || hand[idxPlayed].Rank() != move.rank) {
      continue;
    }

    hand.erase(hand.begin() + idxPlayed);
    if ((int)hand.size() == handSize) {
      // no card to draw
      newHands.push_back(hand);
    } else {
      // draw new card
      auto remainingCardCount = cardCount;
      bool invalid = false;
      for (const auto& card : hand) {
        --remainingCardCount[cardToIndex(card, numRank_)];
        if (remainingCardCount[cardToIndex(card, numRank_)] < 0) {
          invalid = true;
          break;
        }
      }

      if (invalid) {
        continue;
      }

      for (int cidx = 0; cidx < (int)remainingCardCount.size(); ++cidx) {
        if (remainingCardCount[cidx] == 0) {
          continue;
        }
        auto newHand = hand;
        newHand.push_back(indexToCard(cidx, numRank_));
        newHands.push_back(newHand);
      }
    }
  }

  allHands_ = newHands;
  updateOccurance(cardCount, false);
}

void HandDistribution::filterWithCardKnowledge(const hle::HanabiHand& hand) {
  const auto& ck = hand.Knowledge();

  std::vector<std::vector<hle::HanabiCardValue>> newHands;
  std::vector<int> newOccurances;
  int64_t newSum = 0;

  for (int i = 0; i < (int)allHands_.size(); ++i) {
    const auto& cards = allHands_[i];
    assert(cards.size() == ck.size());
    bool plausible = true;
    for (int j = 0; j < (int)cards.size(); ++j) {
      if (!ck[j].IsCardPlausible(cards[j].Color(), cards[j].Rank())) {
        plausible = false;
        break;
      }
    }

    if (plausible) {
      newHands.push_back(cards);
      newOccurances.push_back(numOccurances_[i]);
      newSum += numOccurances_[i];
    }
  }

  allHands_ = std::move(newHands);
  numOccurances_ = std::move(newOccurances);
  sumOccurance_ = newSum;
}

void HandDistribution::filterCounterfactual(
    int myIndex,
    int refAction,
    const HybridModel& partner,
    const hle::HanabiState& state,
    int numThread) {
  std::vector<int> keep(size(), -1);

  auto forwardModel = [&](int begin, int end) {
    std::vector<rela::Future> results;
    results.reserve(end - begin);
    std::vector<HybridModel> partnerCopies(end - begin, partner);
    for (int i = begin; i < end; ++i) {
      const auto& mySimCards = getHand(i);
      // no need for a new seed because we will not simulate/apply move
      GameSimulator simGame(state, {{myIndex, mySimCards}});
      results.push_back(partner.asyncComputeAction(simGame));
    }

    for (int i = 0; i < (int)results.size(); ++i) {
      auto reply = results[i].get();
      auto action = reply.at("a").item<int>();

      if (action != refAction) {
        keep[begin + i] = 0;
      } else {
        keep[begin + i] = 1;
      }
    }
  };

  std::vector<std::future<void>> workloads;
  int chunkSize = keep.size() / numThread + int(bool(keep.size() % numThread));
  for (int i = 0; i < numThread; ++i) {
    int begin = i * chunkSize;
    if (begin >= (int)keep.size()) {
      break;
    }

    int end = std::min((int)keep.size(), (i + 1) * chunkSize);
    workloads.push_back(std::async(std::launch::async, forwardModel, begin, end));
  }
  for (size_t i = 0; i < workloads.size(); ++i) {
    workloads[i].wait();
  }

  // out of parallel code
  std::vector<std::vector<hle::HanabiCardValue>> newHands;
  std::vector<int> newOccurances;
  int64_t newSum = 0;

  for (size_t i = 0; i < keep.size(); ++i) {
    assert(keep[i] != -1);
    if (keep[i] == 0) {
      continue;
    }

    newHands.push_back(allHands_[i]);
    newOccurances.push_back(numOccurances_[i]);
    newSum += numOccurances_[i];
  }

  allHands_ = std::move(newHands);
  numOccurances_ = std::move(newOccurances);
  sumOccurance_ = newSum;
}

std::vector<std::vector<hle::HanabiCardValue>> HandDistribution::sampleHands(
    int num, std::mt19937* rng) const {
  assert((int64_t)cdf_.size() == sumOccurance_);

  std::vector<std::vector<hle::HanabiCardValue>> samples;
  samples.reserve(num);
  for (int i = 0; i < num; ++i) {
    int rand = (*rng)() % sumOccurance_;
    samples.push_back(allHands_[cdf_[rand]]);
  }
  return samples;
}

void updateBelief(
    const std::unique_ptr<hle::HanabiState>& prevState,
    const hle::HanabiGame& game,
    const std::unique_ptr<hle::HanabiHistoryItem>& lastMove,
    const std::vector<int>& cardCount,  // can be either private/public
    const hle::HanabiHand& myHand,
    HybridModel& partner,
    int myIndex,
    HandDistribution& handDist,
    int numThread,
    bool skipCounterfactual) {
  assert((lastMove == nullptr) == (handDist.size() == 0));
  if (lastMove == nullptr) {
    // init belief
    handDist.init(game, cardCount);
    // handDist.filterExact(myHand.CardValues(), 10000);
    assert(handInBelief(myHand, handDist));
    std::cout << "init belief, total possibility: " << handDist.size() << std::endl;
  } else {
    int before = handDist.size();
    std::string info = "no filtering";
    if (lastMove->player == 0) {
      // it was my turn, lastMove->player is ego-centric
      if (lastMove->move.MoveType() == hle::HanabiMove::kPlay ||
          lastMove->move.MoveType() == hle::HanabiMove::kDiscard) {
        info = "filter move after me playing/discarding";
        handDist.filterMePlayDiscard(*lastMove, cardCount, myHand.Cards().size());
      }
    } else {
      if (lastMove->move.MoveType() == hle::HanabiMove::kPlay ||
          lastMove->move.MoveType() == hle::HanabiMove::kDiscard) {
        info = "filter move after partner playing/discarding";
        handDist.updateOccurance(cardCount, true);
      } else {
        // if my partner hints to me
        if (lastMove->player + lastMove->move.TargetOffset() == game.NumPlayers()) {
          info = "filter move after partner hinting";
          handDist.filterWithCardKnowledge(myHand);
        }
      }
    }

    int after = handDist.size();
    std::cout << info << ", size: " << before << "->" << after << std::endl;
    assert(handInBelief(myHand, handDist));

    if (lastMove->player != 0 && !skipCounterfactual) {
      // filter hands that contradict with partner's move
      assert(prevState != nullptr);
      handDist.filterCounterfactual(
          myIndex,
          game.GetMoveUid(lastMove->move),
          partner,
          *prevState,
          numThread);
      assert(handInBelief(myHand, handDist));
      std::cout << "filtering counterfactual, size: " << after << "->" << handDist.size()
                << std::endl;
    }
  }

  handDist.computeCdf();
}
}  // namespace search
