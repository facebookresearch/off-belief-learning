// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "rlcc/r2d2_actor.h"
#include "rlcc/utils.h"

void addHid(rela::TensorDict& to, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    // hid: [num_layer, batch/num_player, dim]
    // -> batched hid: [batchsize, num_layer, batch/num_player, dim]
    auto ret = to.emplace(kv.first, kv.second);
    assert(ret.second);
  }
}

void moveHid(rela::TensorDict& from, rela::TensorDict& hid) {
  for (auto& kv : hid) {
    auto it = from.find(kv.first);
    assert(it != from.end());
    auto newHid = it->second;
    assert(newHid.sizes() == kv.second.sizes());
    hid[kv.first] = newHid;
    from.erase(it);
  }
}

std::vector<hle::HanabiCardValue> sampleCards(
    const std::vector<float>& v0,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand,
    std::mt19937& rng) {
  auto handSize = hand.Cards().size();
  auto cardBelief = extractPerCardBelief(v0, game, handSize);
  auto cardRemain = privCardCount;
  std::vector<hle::HanabiCardValue> cards;

  for (size_t j = 0; j < handSize; ++j) {
    auto& cb = cardBelief[j];
    if (j > 0) {
      // re-mask card belief
      float sum = 0;
      for (size_t k = 0; k < cardRemain.size(); ++k) {
        cb[k] *= int(cardRemain[k] > 0);
        sum += cb[k];
      }

      if (sum <= 1e-6) {
        std::cerr << "Error in sample card, sum = 0" << std::endl;
        assert(false);
      }
    }
    std::discrete_distribution<int> dist(cb.begin(), cb.end());
    int idx = dist(rng);
    --cardRemain[idx];
    assert(cardRemain[idx] >= 0);
    if (invColorPermute.size()) {
      auto fakeColor = indexToCard(idx, game.NumRanks());
      auto realColor =
          hle::HanabiCardValue(invColorPermute[fakeColor.Color()], fakeColor.Rank());
      cards.push_back(realColor);
    } else {
      cards.push_back(indexToCard(idx, game.NumRanks()));
    }
  }

  assert(hand.CanSetCards(cards));
  return cards;
}

std::tuple<std::vector<hle::HanabiCardValue>, bool> filterSample(
    const torch::Tensor& samples,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand) {
  auto sampleAcc = samples.accessor<int64_t, 2>();
  int numSample = sampleAcc.size(0);
  int handSize = hand.Cards().size();

  for (int i = 0; i < numSample; ++i) {
    auto cardRemain = privCardCount;
    std::vector<hle::HanabiCardValue> cards;
    for (int j = 0; j < handSize; ++j) {
      // sampling & v0 belief is done in the color shuffled space
      int idx = sampleAcc[i][j];
      auto card = indexToCard(idx, game.NumRanks());
      // this sample violate card count
      if (cardRemain[idx] == 0) {
        break;
      }
      --cardRemain[idx];

      if (invColorPermute.size()) {
        auto realCard = hle::HanabiCardValue(invColorPermute[card.Color()], card.Rank());
        cards.push_back(realCard);
      } else {
        cards.push_back(card);
      }
    }
    if ((int)cards.size() == handSize && hand.CanSetCards(cards)) {
      return {cards, true};
    }
  }
  return {hand.CardValues(), false};
}

std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b) {
  assert(b.size() == 25);
  std::set<int> colors;
  std::set<int> ranks;
  for (int c = 0; c < 5; ++c) {
    for (int r = 0; r < 5; ++r) {
      if (b[c * 5 + r] > 0) {
        colors.insert(c);
        ranks.insert(r);
      }
    }
  }
  return {colors.size() == 1, ranks.size() == 1};
}

void R2D2Actor::reset(const HanabiEnv& env) {
  hidden_ = getH0(batchsize_, runner_);
  if (beliefRunner_ != nullptr) {
    beliefHidden_ = getH0(batchsize_, beliefRunner_);
  }

  if (r2d2Buffer_ != nullptr) {
    r2d2Buffer_->init(hidden_);
  }

  const auto& game = env.getHleGame();
  int fixColorPlayer = -1;
  if (vdn_ && shuffleColor_) {
    fixColorPlayer = rng_() % game.NumPlayers();
  }

  for (int i = 0; i < batchsize_; ++i) {
    assert(playerEps_.size() > 0 && epsList_.size() > 0);
    playerEps_[i] = epsList_[rng_() % epsList_.size()];
    if (tempList_.size() > 0) {
      assert(playerTemp_.size() > 0);
      playerTemp_[i] = tempList_[rng_() % tempList_.size()];
    }

    // other-play
    if (shuffleColor_) {
      auto& colorPermute = colorPermutes_[i];
      auto& invColorPermute = invColorPermutes_[i];
      colorPermute.clear();
      invColorPermute.clear();
      for (int i = 0; i < game.NumColors(); ++i) {
        colorPermute.push_back(i);
        invColorPermute.push_back(i);
      }
      if (i == fixColorPlayer) {
        continue;
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
}

void R2D2Actor::observeBeforeAct(const HanabiEnv& env) {
  torch::NoGradGuard ng;
  prevHidden_ = hidden_;

  rela::TensorDict input;
  const auto& state = env.getHleState();

  if (vdn_) {
    std::vector<rela::TensorDict> vObs;
    for (int i = 0; i < numPlayer_; ++i) {
      vObs.push_back(observe(
          state,
          i,
          shuffleColor_,
          colorPermutes_[i],
          invColorPermutes_[i],
          hideAction_,
          trinary_,
          sad_));
    }
    input = rela::tensor_dict::stack(vObs, 0);
  } else {
    input = observe(
        state,
        playerIdx_,
        shuffleColor_,
        colorPermutes_[0],
        invColorPermutes_[0],
        hideAction_,
        trinary_,
        sad_);
  }

  // add features such as eps and temperature
  input["eps"] = torch::tensor(playerEps_);
  if (playerTemp_.size() > 0) {
    input["temperature"] = torch::tensor(playerTemp_);
  }

  // push before we add hidden
  if (replayBuffer_ != nullptr) {
    r2d2Buffer_->pushObs(input);
  } else {
    // eval mode, collect some stats
    const auto& game = env.getHleGame();
    auto obs = hle::HanabiObservation(state, state.CurPlayer(), true);
    auto encoder = hle::CanonicalObservationEncoder(&game);
    auto [privV0, cardCount] =
        encoder.EncodeV0Belief(obs, std::vector<int>(), false, std::vector<int>(), false);
    perCardPrivV0_ =
        extractPerCardBelief(privV0, env.getHleGame(), obs.Hands()[0].Cards().size());
  }

  addHid(input, hidden_);

  // no-blocking async call to neural network
  futReply_ = runner_->call("act", input);

  if (!offBelief_) {
    return;
  }

  // forward belief model
  assert(!vdn_);
  auto [beliefInput, privCardCount, v0] = beliefModelObserve(
      state,
      playerIdx_,
      shuffleColor_,
      colorPermutes_[0],
      invColorPermutes_[0],
      hideAction_);
  privCardCount_ = privCardCount;

  if (beliefRunner_ == nullptr) {
    sampledCards_ = sampleCards(
        v0,
        privCardCount_,
        invColorPermutes_[0],
        env.getHleGame(),
        state.Hands()[playerIdx_],
        rng_);
  } else {
    addHid(beliefInput, beliefHidden_);
    futBelief_ = beliefRunner_->call("sample", beliefInput);
  }

  fictState_ = std::make_unique<hle::HanabiState>(state);
}

void R2D2Actor::act(HanabiEnv& env, const int curPlayer) {
  torch::NoGradGuard ng;

  auto& state = env.getHleState();
  auto reply = futReply_.get();
  moveHid(reply, hidden_);

  if (replayBuffer_ != nullptr) {
    r2d2Buffer_->pushAction(reply);
  }

  rela::TensorDict beliefReply;
  if (offBelief_ && beliefRunner_ != nullptr) {
    beliefReply = futBelief_.get();
    moveHid(beliefReply, beliefHidden_);
    // if it is not our turn, then this is all we need for belief
  }

  int action;
  const std::vector<int>* invColorPermute;
  if (vdn_) {
    action = reply.at("a")[curPlayer].item<int64_t>();
    invColorPermute = &(invColorPermutes_[curPlayer]);
  } else {
    action = reply.at("a").item<int64_t>();
    invColorPermute = &(invColorPermutes_[0]);
  }

  if (offBelief_) {
    const auto& hand = fictState_->Hands()[playerIdx_];
    bool success = true;
    if (beliefRunner_ != nullptr) {
      auto sample = beliefReply.at("sample");
      std::tie(sampledCards_, success) = filterSample(
          sample,
          privCardCount_,
          *invColorPermute,
          env.getHleGame(),  // *fictGame_,
          hand);
    }
    if (success) {
      auto& deck = fictState_->Deck();
      deck.PutCardsBack(hand.Cards());
      deck.DealCards(sampledCards_);
      fictState_->Hands()[playerIdx_].SetCards(sampledCards_);
      ++successFict_;
    }
    validFict_ = success;
    ++totalFict_;
  }

  if (!vdn_ && curPlayer != playerIdx_) {
    if (offBelief_) {
      auto partner = partners_[curPlayer];
      assert(partner != nullptr);
      // it is not my turn, I need to re-evaluate my partner on
      // the fictitious transition
      auto partnerInput = observe(
          *fictState_,
          partner->playerIdx_,
          partner->shuffleColor_,
          partner->colorPermutes_[0],
          partner->invColorPermutes_[0],
          partner->hideAction_,
          partner->trinary_,
          partner->sad_);
      // add features such as eps and temperature
      partnerInput["eps"] = torch::tensor(partner->playerEps_);
      if (partner->playerTemp_.size() > 0) {
        partnerInput["temperature"] = torch::tensor(partner->playerTemp_);
      }
      addHid(partnerInput, partner->prevHidden_);
      assert(fictReply_.isNull());
      fictReply_ = partner->runner_->call("act", partnerInput);
    }

    assert(action == env.noOpUid());
    return;
  }

  auto move = state.ParentGame()->GetMove(action);
  if (shuffleColor_ && move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
    int realColor = (*invColorPermute)[move.Color()];
    move.SetColor(realColor);
  }

  if (replayBuffer_ == nullptr) {
    if (move.MoveType() == hle::HanabiMove::kPlay) {
      auto cardBelief = perCardPrivV0_[move.CardIndex()];
      auto [colorKnown, rankKnown] = analyzeCardBelief(cardBelief);

      if (colorKnown && rankKnown) {
        ++bothKnown_;
      } else if (colorKnown) {
        ++colorKnown_;
      } else if (rankKnown) {
        ++rankKnown_;
      } else {
        ++noneKnown_;
      }
    }
  }

  env.step(move);
}

void R2D2Actor::fictAct(const HanabiEnv& env) {
  if (!offBelief_) {
    return;
  }
  torch::NoGradGuard ng;

  hle::HanabiMove move = env.lastMove();
  if (env.lastActivePlayer() != playerIdx_) {
    // it was not our turn, we have computed our partner's fict move
    auto fictReply = fictReply_.get();
    auto action = fictReply.at("a").item<int64_t>();
    move = env.getMove(action);
  }
  auto [fictR, fictTerm] = applyMove(*fictState_, move, false);

  // submit network call to compute value
  auto fictInput = observe(
      *fictState_,
      playerIdx_,
      shuffleColor_,
      colorPermutes_[0],
      invColorPermutes_[0],
      hideAction_,
      trinary_,
      sad_);

  // the hidden is new, so we are good
  addHid(fictInput, hidden_);
  fictInput["reward"] = torch::tensor(fictR);
  fictInput["terminal"] = torch::tensor(float(fictTerm));
  if (playerTemp_.size() > 0) {
    fictInput["temperature"] = torch::tensor(playerTemp_);
  }
  futTarget_ = runner_->call("compute_target", fictInput);
}

void R2D2Actor::observeAfterAct(const HanabiEnv& env) {
  torch::NoGradGuard ng;
  if (replayBuffer_ == nullptr) {
    return;
  }

  if (!futPriority_.isNull()) {
    auto priority = futPriority_.get()["priority"].item<float>();
    replayBuffer_->add(std::move(lastEpisode_), priority);
  }

  float reward = env.stepReward();
  bool terminated = env.terminated();
  r2d2Buffer_->pushReward(reward);
  r2d2Buffer_->pushTerminal(float(terminated));

  if (offBelief_) {
    assert(!futTarget_.isNull());
    auto target = futTarget_.get()["target"];
    auto ret = r2d2Buffer_->obsBack().emplace("target", target);
    assert(ret.second);
    ret = r2d2Buffer_->obsBack().emplace("valid_fict", torch::tensor(float(validFict_)));
    assert(ret.second);
  }

  if (terminated) {
    lastEpisode_ = r2d2Buffer_->popTransition();
    auto input = lastEpisode_.toDict();
    futPriority_ = runner_->call("compute_priority", input);
  }
}
