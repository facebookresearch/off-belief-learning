#pragma once

#include "rela/context.h"

#include "searchcc/game_sim.h"
#include "searchcc/hand_dist.h"
#include "searchcc/hybrid_model.h"


namespace search {

struct JointAction {
  JointAction(int a0, int argmaxA1, std::vector<float> qA1)
      : a0_(a0), argmaxA1_(argmaxA1), qA1_(qA1) {}

  int a0_;
  int argmaxA1_;
  std::vector<float> qA1_;
  std::vector<float> softmaxQA1_;
};

class FinesseActor {
 public:
  FinesseActor(int index, std::vector<std::shared_ptr<rela::BatchRunner>> bps, int seed)
      : index_(index)
      , bp_(bps[0])
      , bpGpus_(bps)
      , prevModel_(index)
      , model_(index)
      , rng_(seed) {
    model_.setBpModel(bp_, getH0(*bp_, 1));
  }

  void setPartners(const std::vector<std::shared_ptr<FinesseActor>>& partners) {
    assert((int)partners.size() > index_);
    partners_ = partners;
    nextActor_ = partners_[(index_ + 1) % partners_.size()];
  }

  void updateBelief(const GameSimulator& env, int numThread) {
    torch::NoGradGuard ng;
    // assert(callOrder_ == 0);
    ++callOrder_;

    const auto& state = env.state();
    int curPlayer = state.CurPlayer();
    int numPlayer = env.game().NumPlayers();
    assert((int)partners_.size() == numPlayer);
    int prevPlayer = (curPlayer - 1 + numPlayer) % numPlayer;

    // always use public belief
    auto [obs, lastMove, cardCount, myHand] =
        observeForSearch(state, index_, false, true);

    search::updateBelief(
        prevState_,
        env.game(),
        lastMove,
        cardCount,
        myHand,
        partners_[prevPlayer]->prevModel_,
        index_,
        publHandDist_,
        numThread,

        skipCounterfactual_);
    if (skipCounterfactual_) {
      skipCounterfactual_ = false;
    }

    auto privCardCount = std::get<2>(observeForSearch(state, index_, false, false));
    privHandDist_ = search::publicToPrivate(publHandDist_, privCardCount);
    std::cout << "from public to private, " << publHandDist_.size()
              << " -> " << privHandDist_.size() << std::endl;
  }

  std::vector<JointAction> finesse_(
      const GameSimulator& env,
      int numOuterHand,
      int numInnerHand,
      int numThread,
      bool argmax,
      float beta,
      bool voteBased) const;

  void finesse(
      const GameSimulator& env,
      int numOuterHand,
      int numInnerHand,
      int numThread,
      float threshold,
      bool argmax,
      float beta,
      bool voteBased,
      bool computeTwice);

  void observe(const GameSimulator& env) {
    // assert(callOrder_ == 1);
    ++callOrder_;

    const auto& state = env.state();
    model_.observeBeforeAct(env, 0);

    if (prevState_ == nullptr) {
      prevState_ = std::make_unique<hle::HanabiState>(state);
    } else {
      *prevState_ = state;
    }
  }

  int decideAction(const GameSimulator& env) {
    // assert(callOrder_ == 2);
    callOrder_ = 0;

    prevModel_ = model_;  // this line can only be in decide action
    int action = model_.decideAction(env, false);
    if (env.state().CurPlayer() == index_ && !finesseMove_.empty()) {
      auto finesseMove = finesseMove_[0];
      finesseMove_.clear();
      if (env.state().MoveIsLegal(finesseMove)) {
        std::cout << "Execute finesse move: " << finesseMove.ToString()
                  << ", original BP move was: " << env.game().GetMove(action).ToString()
                  << std::endl;
        action = env.game().GetMoveUid(finesseMove);

        // set other agent not to do counterfactual filtering
        for (int i = 0; i < (int)partners_.size(); ++i) {
          if (i != index_) {
            partners_[i]->skipCounterfactual_ = true;
          }
        }
      } else {
        std::cout << "Fail to execute finesse move: " << finesseMove.ToString()
                  << std::endl;
      }
    }
    return action;
  }

 private:
  const int index_;
  std::shared_ptr<rela::BatchRunner> const bp_;
  std::vector<std::shared_ptr<rela::BatchRunner>> const bpGpus_;

  HybridModel prevModel_;
  HybridModel model_;
  HandDistribution publHandDist_;
  HandDistribution privHandDist_;
  mutable std::mt19937 rng_;

  std::unique_ptr<hle::HanabiState> prevState_ = nullptr;
  std::vector<std::shared_ptr<FinesseActor>> partners_;
  std::shared_ptr<FinesseActor> nextActor_;
  int callOrder_ = 0;

  std::vector<hle::HanabiMove> finesseMove_;
  bool skipCounterfactual_ = false;
  bool followBp_ = false;
};
}
