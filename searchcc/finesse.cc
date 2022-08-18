#include <future>
#include <iomanip>
#include <queue>
#include <math.h>
#include <assert.h>
#include <limits>

#include "searchcc/finesse.h"

namespace hle = hanabi_learning_env;
using namespace search;

namespace {

class PairsOfHands {
 public:
  PairsOfHands(
      std::vector<hle::HanabiCardValue> first,
      int firstIndex,
      std::vector<std::vector<hle::HanabiCardValue>> seconds,
      int secondIndex)
      : first_(std::move(first))
      , firstIndex_(firstIndex)
      , seconds_(std::move(seconds))
      , secondIndex_(secondIndex) {
  }

  PairsOfHands(
      std::vector<hle::HanabiCardValue> first,
      int firstIndex)
      : first_(std::move(first))
      , firstIndex_(firstIndex) {
  }

  std::vector<SimHand> nextPair() {
    ++index_;
    if (seconds_.empty()) {
      return {SimHand(firstIndex_, first_)};
    } else {
      assert(index_ <= (int)seconds_.size());
      return {SimHand(firstIndex_, first_), SimHand(secondIndex_, seconds_[index_ - 1])};
    }
  }

  bool exhausted() const {
    if (seconds_.empty()) {
      return index_ >= 1;
    } else {
      return index_ >= (int)seconds_.size();
    }
  }

  std::vector<SimHand> getPair(int index) const {
    assert(index_ < (int)seconds_.size());
    return {SimHand(firstIndex_, first_), SimHand(secondIndex_, seconds_[index])};
  }

  int numTrial_ = 0;

 private:
  std::vector<hle::HanabiCardValue> first_;
  int firstIndex_;
  std::vector<std::vector<hle::HanabiCardValue>> seconds_;
  int secondIndex_;

  int index_ = 0;
};

// void printCardCount(std::vector<int> cardCount, const hle::HanabiGame& game) {
//   std::cout << "card count: " << std::endl;
//   for (int i = 0;  i < 25; ++i) {
//     std::cout << game.IndexToCard(i).ToString() << ": " << cardCount[i] << std::endl;
//   }
// }

PairsOfHands regressiveSampleHands(
    const hle::HanabiGame& game,
    const std::vector<int>& refCardCount,
    const std::vector<std::pair<int, const HandDistribution*>>& handDists,
    int numHand,
    int seed) {
  std::mt19937 rng(seed);
  assert(handDists.size() == 2);
  int i = 0;

  while (true) {
    ++i;
    auto cardCount = refCardCount;
    auto firstHandDist = publicToPrivate(*(handDists[0].second), cardCount);
    auto firstHand = firstHandDist.sampleHands(1, &rng)[0];

    for (auto& c : firstHand) {
      int index = game.CardToIndex(c);
      assert(cardCount[index] > 0);
      --cardCount[index];
    }

    auto secondHandDist = publicToPrivate(*(handDists[1].second), cardCount);
    if (secondHandDist.size() > 0) {
      auto secondHands = secondHandDist.sampleHands(numHand, &rng);
      auto ret = PairsOfHands(firstHand, handDists[0].first, secondHands, handDists[1].first);
      ret.numTrial_ = i;
      return ret;
    }
  }
}

std::vector<PairsOfHands> sampleHands(
    const hle::HanabiState& state,
    const std::vector<std::pair<int, const HandDistribution*>>& handDists,
    int numOuterHand,
    int numInnerHand,
    int numThread,
    std::mt19937* rng) {
  std::vector<PairsOfHands> simHands;
  simHands.reserve(numOuterHand);

  auto deck = state.Deck();
  for (auto& kv : handDists) {
    deck.PutCardsBack(state.Hands()[kv.first].Cards());
  }
  auto publCardCount = deck.CardCount();
  const auto& game = *state.ParentGame();

  auto task = [&](int begin, int end, int seed) {
    std::mt19937 rng(seed);
    std::vector<PairsOfHands> ret;
    while (begin < end) {
      auto poh = regressiveSampleHands(
        game,
        publCardCount,
        handDists,
        numInnerHand,
        int(rng()));
      ret.push_back(poh);
      ++begin;
    }
    return ret;
  };

  std::vector<std::future<std::vector<PairsOfHands>>> futures;
  int chunkSize = numOuterHand / numThread + int(bool(numOuterHand % numThread));
  for (int i = 0; i < numThread; ++i) {
    int begin = i * chunkSize;
    if (begin >= numOuterHand) {
      break;
    }
    int end = std::min(numOuterHand, (i + 1) * chunkSize);
    assert(begin < end);
    futures.push_back(std::async(std::launch::async, task, begin, end, int((*rng)())));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    auto ret = futures[i].get();
    simHands.insert(simHands.end(), ret.begin(), ret.end());
  }

  return simHands;
}

std::tuple<std::vector<hle::HanabiMove>, std::vector<float>> getBlueprintMove(
    const hle::HanabiState& refState,
    const std::vector<PairsOfHands>& simHands,
    const HybridModel& model) {
  std::vector<hle::HanabiMove> moves;
  moves.reserve(simHands.size());
  const auto& hleGame = (*refState.ParentGame());
  std::vector<float> actionProb(hleGame.MaxMoves(), -1);

  std::vector<rela::Future> rets;
  rets.reserve(simHands.size());
  for (const auto& pairsOfHands : simHands) {
    // one pair is sufficient since they all produce the same private obs
    GameSimulator simGame(refState, pairsOfHands.getPair(0));
    rets.push_back(model.asyncComputeAction(simGame));
    auto legalMove = simGame.state().LegalMoves(model.index);
    for (auto& move : legalMove) {
      int action = hleGame.GetMoveUid(move);
      if (actionProb[action] < 0) {
        actionProb[action] = 0;
      }
    }
  }

  for (auto& ret : rets) {
    auto reply = ret.get();
    auto action = reply.at("a").item<int>();
    moves.push_back(refState.ParentGame()->GetMove(action));
    assert(actionProb[action] >= 0);
    actionProb[action] += 1;
  }

  for (auto& p : actionProb) {
    if (p > 0) {
      p /= (float)simHands.size();
    }
  }

  return {moves, actionProb};
}

struct SimulationPack {
  SimulationPack(
      const hle::HanabiState& refState,
      std::vector<HybridModel> actors,
      std::vector<hle::HanabiMove> fixedMoves,
      PairsOfHands pairsOfHands,
      int seed)
      : refState_(refState)
      , actors_(std::move(actors))
      , fixedMoves_(std::move(fixedMoves))
      , pairsOfHands_(std::move(pairsOfHands))
      , game_(refState, pairsOfHands_.nextPair(), seed) {
    for (auto& m : fixedMoves_) {
      queueFixedMoves_.push(m);
    }
    for (auto& model : actors_) {
      refBpHids_.push_back(model.getBpHid());
    }
    // std::cout << "******************" << std::endl;
    // std::cout << "moves: ";
    // for (auto& move : fixedMoves_) {
    //   std::cout << move.ToString() << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << game_.state().ToString() << std::endl;
  }

  bool exhausted() const {
    return pairsOfHands_.exhausted();
  }

  void reset() {
    assert(!pairsOfHands_.exhausted());
    game_.reset(refState_, pairsOfHands_.nextPair());
    for (size_t i = 0; i < actors_.size(); ++i) {
      actors_[i].setBpHid(refBpHids_[i]);
    }
    while (!queueFixedMoves_.empty()) {
      queueFixedMoves_.pop();
    }
    for (auto& m : fixedMoves_) {
      queueFixedMoves_.push(m);
    }
  }

  const hle::HanabiState refState_;
  std::vector<rela::TensorDict> refBpHids_;

  std::vector<HybridModel> actors_;
  const std::vector<hle::HanabiMove> fixedMoves_;
  std::queue<hle::HanabiMove> queueFixedMoves_;
  PairsOfHands pairsOfHands_;

  GameSimulator game_;
  std::vector<float> scores_;
};

float mean(std::vector<float> vals) {
  assert(vals.size() > 0);
  float sum = 0;
  for (auto v : vals) {
    sum += v;
  }
  return sum / (float) vals.size();
}

// create sim game for a PairsOfHands
std::vector<SimulationPack> createSimPacks(
    const hle::HanabiState& refState,
    PairsOfHands& pairsOfHands,
    const std::vector<HybridModel>& models,
    hle::HanabiMove bpMove,
    const std::vector<float>& bpDist,
    int seed) {
  std::vector<SimulationPack> games;

  // first game is always the blueprint one
  games.emplace_back(
      refState, models, std::vector<hle::HanabiMove>({bpMove}), pairsOfHands, seed++);

  // get joint actions
  std::vector<std::vector<hle::HanabiMove>> m0m1s;
  const auto& hleGame = *refState.ParentGame();
  for (int i = 0; i < hleGame.MaxMoves(); ++i) {
    if (bpDist[i] > 0) {
      continue;
    }

    auto m0 = hleGame.GetMove(i);
    if (m0.MoveType() == hle::HanabiMove::kPlay || m0.MoveType() == hle::HanabiMove::kDiscard) {
      continue;
    } else if (m0.TargetOffset() != hleGame.NumPlayers() - 1) {
      continue;
    } else if (!refState.MoveIsLegal(m0)) {
      continue;
    }

    // std::vector<hle::HanabiMove> m0m1({m0});
    for (int j = 0; j < hleGame.MaxMoves(); ++j) {
      auto m1 = hleGame.GetMove(j);
      if (m1.MoveType() == hle::HanabiMove::kPlay) {
        m0m1s.push_back({m0, m1});
      }
    }
  }

  for (const auto& m0m1 : m0m1s) {
    games.emplace_back(refState, models, m0m1, pairsOfHands, seed++);
  }
  return games;
}

void runSimPacks(std::vector<SimulationPack>& simPacks, bool strict) {
  std::vector<int> runningSims(simPacks.size());
  std::iota(runningSims.begin(), runningSims.end(), 0);

  while (runningSims.size() > 0) {
    std::vector<int> newRunningSims;
    for (auto simIdx : runningSims) {
      assert(!simPacks[simIdx].game_.state().IsTerminal());
      for (auto& actor : simPacks[simIdx].actors_) {
        actor.observeBeforeAct(simPacks[simIdx].game_, 0);
      }
    }

    for (auto simIdx : runningSims) {
      int action = -1;
      auto& game = simPacks[simIdx].game_;
      for (auto& actor : simPacks[simIdx].actors_) {
        int a = actor.decideAction(game, false);
        if (actor.index == game.state().CurPlayer()) {
          action = a;
        }
      }
      // std::cout << "CCC: " << simIdx << std::endl;

      auto move = game.getMove(action);
      auto& queue = simPacks[simIdx].queueFixedMoves_;
      if (!queue.empty()) {
        if (strict) {
          move = queue.front();
          assert(game.state().MoveIsLegal(move));
        } else if (game.state().MoveIsLegal(queue.front())) {
          // not strict mode, check whether the preset move is legal
          move = queue.front();
        }
        queue.pop();
      }

      game.step(move);

      if (game.terminal()) {
        simPacks[simIdx].scores_.push_back(game.score());
        if(!simPacks[simIdx].exhausted()) {
          simPacks[simIdx].reset();
        }
      }

      if (!game.terminal()) {
        newRunningSims.push_back(simIdx);
      }
    }

    runningSims = newRunningSims;
  }
}

struct DeviationValue {
  DeviationValue() = default;

  DeviationValue(std::pair<int, int> jointAction, float jointQ, float bpQ)
      : jointAction_(jointAction)
      , jointQ_(jointQ)
      , bpQ_(bpQ)
      , devQ_(std::max(jointQ, bpQ)) {
  }

  std::pair<int, int> jointAction_;
  float jointQ_;
  float bpQ_;
  float devQ_;  // max(jointQ, bpQ)
};

// returns (a0, a1, score)
// assumes that simPacks[0] is the bp move
// using JointAction = std::pair<int, int>;
std::vector<DeviationValue> maxQjointQbp(
    const std::vector<SimulationPack>& simPacks) {
  std::vector<DeviationValue> ret;
  assert(simPacks.size() > 1);
  ret.reserve(simPacks.size() - 1);

  assert(simPacks[0].game_.terminal());
  assert(simPacks[0].fixedMoves_.size() == 1);
  float bpScore = mean(simPacks[0].scores_);
  const auto& hleGame = simPacks[0].game_.game();

  for (size_t i = 1; i < simPacks.size(); ++i) {
    assert(simPacks[i].fixedMoves_.size() == 2);
    int a0 = hleGame.GetMoveUid(simPacks[i].fixedMoves_[0]);
    int a1 = hleGame.GetMoveUid(simPacks[i].fixedMoves_[1]);
    assert(simPacks[i].game_.terminal());
    ret.emplace_back(std::make_pair(a0, a1), mean(simPacks[i].scores_), bpScore);
  }
  return ret;
}

// #hand2, #joint_action, max(bp, joint_q)
std::vector<std::vector<DeviationValue>> computeQvaluesFromSimulation(
    const hle::HanabiState& state,
    std::vector<PairsOfHands>& simHands,
    const std::vector<std::vector<HybridModel>>& models,
    const std::vector<hle::HanabiMove>& bpMoves,
    const std::vector<float>& bpDist,
    std::mt19937* rng,
    int numThread) {
  assert(bpMoves.size() == simHands.size());
  std::vector<std::vector<DeviationValue>> ret(bpMoves.size());
  std::vector<int> seeds;
  for (size_t i = 0; i < simHands.size(); ++i) {
    seeds.push_back((int)(*rng)());
  }

  auto task = [&](int begin, int end, int threadIdx) {
    auto& localModels = models[threadIdx % models.size()];
    while (begin < end) {
      auto simPacks = createSimPacks(
          state, simHands[begin], localModels, bpMoves[begin], bpDist, seeds[begin]);
      if (simPacks.size() <= 1) {
        ++begin;
        continue;
      }
      runSimPacks(simPacks, true);
      ret[begin] = maxQjointQbp(simPacks);
      ++begin;
    }
  };

  std::vector<std::future<void>> workloads;
  int chunkSize = simHands.size() / numThread + int(bool(simHands.size() % numThread));
  for (int i = 0; i < numThread; ++i) {
    int begin = i * chunkSize;
    if (begin >= (int)simHands.size()) {
      break;
    }
    int end = std::min((int)simHands.size(), (i + 1) * chunkSize);
    assert(begin < end);
    workloads.push_back(std::async(std::launch::async, task, begin, end, i));
  }
  for (size_t i = 0; i < workloads.size(); ++i) {
    workloads[i].wait();
  }

  if (ret[0].size() == 0) {
    int sum  = 0;
    for (auto& r : ret) {
      sum += (int)r.size();
    }
    assert(sum == 0);
    return {};
  }

  return ret;
}

void printJointQ(
    const std::vector<std::vector<float>>& jointQ,
    const hle::HanabiGame& game) {
  // std::cout << "JointQ Table:" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  for (size_t a0 = 0; a0 < jointQ.size(); ++a0) {
    if ((a0 >= 15 && a0 < 20) || (a0 >= 25 && a0 < 30)) {
      std::cout << game.GetMove(a0).ToString() << ": ";
      for (size_t a1 = 0; a1 < jointQ[a0].size(); ++a1) {
        if (a1 >= 5 && a1 < 10) {
          std::cout << std::setw(5) << jointQ[a0][a1] << ", ";
        }
      }
      std::cout << std::endl;
    }
  }
}

std::tuple<
  std::vector<std::vector<float>>,  // joint max(q(a0, a1), q_bp)
  std::vector<std::vector<float>>,  // joint q(a0*, a1*)
  std::vector<JointAction>
  >
getJointQTable(
    const std::vector<std::vector<DeviationValue>>& stateQvalues,
    int numAction,
    bool voteBased) {
  std::vector<std::vector<int>> count(numAction, std::vector<int>(numAction, 0));
  std::vector<std::vector<float>> jointDevQ(numAction, std::vector<float>(numAction, 0));
  std::vector<std::vector<float>> jointQ(numAction, std::vector<float>(numAction, 0));
  std::vector<JointAction> jointAction;
  float qBp = 0;
  int bpCount = 0;

  for (const auto& qvalues : stateQvalues) {
    qBp += qvalues[0].bpQ_;
    ++bpCount;

    for (const auto& deviation : qvalues) {
      int a0 = deviation.jointAction_.first;
      int a1 = deviation.jointAction_.second;
      assert(a0 < numAction && a1 < numAction);
      if (voteBased) {
        if (deviation.jointQ_ > deviation.bpQ_) {
          jointDevQ[a0][a1] += 1;
        }
      } else {
        jointDevQ[a0][a1] += deviation.devQ_;
      }
      jointQ[a0][a1] += deviation.jointQ_;
      ++count[a0][a1];
    }
  }

  for (int a0 = 0; a0 < numAction; ++a0) {
    float max = 0;
    int argmax = -1;
    for (int a1 = 0; a1 < numAction; ++a1) {
      if (count[a0][a1] == 0) {
        continue;
      }
      jointDevQ[a0][a1] /= count[a0][a1];
      jointQ[a0][a1] /= count[a0][a1];
      if (max < jointDevQ[a0][a1]) {
        max = jointDevQ[a0][a1];
        argmax = a1;
      }
    }
    if (argmax != -1) {
      jointAction.emplace_back(a0, argmax, jointDevQ[a0]);
    }
  }
  std::cout << "***Q(bp)_publ: " << qBp / bpCount << ", averaged over "
            << bpCount << " games***" << std::endl;
  return {jointDevQ, jointQ, jointAction};
}

// void printJointMove(
//     const std::vector<std::pair<int, int>>& jointAction, const hle::HanabiGame& game) {
//   for (const auto& pair : jointAction) {
//     std::cout << game.GetMove(pair.first).ToString() << " -> "
//               << game.GetMove(pair.second).ToString() << std::endl;
//   }
// }

std::tuple<float, std::map<int, float>> searchMove(
    const hle::HanabiState& refState,
    int myIdx,
    const std::vector<std::vector<hle::HanabiCardValue>>& simHands,
    const std::vector<HybridModel>& models,
    const JointAction& jointAction,
    const std::vector<int>& seeds,
    bool argmax) {
  std::vector<SimulationPack> simPacks;
  simPacks.reserve(simHands.size());

  std::vector<std::vector<hle::HanabiMove>> moves;
  std::mt19937 rng(seeds[0]);
  const auto& game = *refState.ParentGame();
  if (argmax || jointAction.argmaxA1_ == -1) {
    std::vector<hle::HanabiMove> presetMove;
    presetMove.push_back(game.GetMove(jointAction.a0_));
    if (jointAction.argmaxA1_ != -1) {
      presetMove.push_back(game.GetMove(jointAction.argmaxA1_));
    }

    for (size_t i = 0; i < simHands.size(); ++i) {
      moves.push_back(presetMove);
    }
  } else {
    auto& weight = jointAction.softmaxQA1_;
    assert(weight.size() > 0);
    std::discrete_distribution<int> dist(weight.begin(), weight.end());
    for (size_t i = 0; i < simHands.size(); ++i) {
      std::vector<hle::HanabiMove> presetMove;
      presetMove.push_back(game.GetMove(jointAction.a0_));
      int a1 = dist(rng);
      presetMove.push_back(game.GetMove(a1));
      moves.push_back(presetMove);
    }
  }

  for (size_t i = 0; i < simHands.size(); ++i) {
    PairsOfHands simHand(simHands[i], myIdx);
    simPacks.emplace_back(refState, models, moves[i], simHand, seeds[i]);
  }

  runSimPacks(simPacks, true);

  std::map<int, std::vector<float>> a1Scores;
  std::vector<float> scores;
  float sum = 0;
  float count = 0;
  for (size_t i = 0; i < simPacks.size(); ++i) {
    sum += simPacks[i].game_.score();
    count += 1;

    if (simPacks[i].fixedMoves_.size() == 2) {
      int a1 = game.GetMoveUid(simPacks[i].fixedMoves_[1]);
      if (a1Scores.find(a1) == a1Scores.end()) {
        a1Scores[a1] = std::vector<float>();
      }
      a1Scores[a1].push_back(simPacks[i].game_.score());
    }
  }

  std::map<int, float> a1MeanScores;
  for (auto& kv : a1Scores) {
    a1MeanScores[kv.first] = mean(kv.second);
  }
  return {sum / count, a1MeanScores};
}

std::vector<hle::HanabiMove> evaluateJointDevAndBp(
    const hle::HanabiState& refState,
    int myIdx,
    const HandDistribution& privHandDist,
    int numHand,
    const std::vector<HybridModel>& models,
    hle::HanabiMove bpMove,
    std::vector<JointAction> jointActions,
    const std::vector<JointAction>& jointActions2,
    std::mt19937* rng,
    float threshold,
    bool argmax) {
  const auto& game = *refState.ParentGame();
  auto simHands = privHandDist.sampleHands(numHand, rng);
  std::vector<int> seeds;
  for (int i = 0; i < numHand; ++i) {
    seeds.push_back((*rng)());
  }

  // add bpMove as a special joint action
  JointAction bpAction(game.GetMoveUid(bpMove), -1, {});
  jointActions.insert(jointActions.begin(), bpAction);

  std::vector<std::future<std::tuple<float, std::map<int, float>>>> workloads;
  for (auto& jointAction : jointActions) {
    workloads.push_back(std::async(
        std::launch::async,
        searchMove,
        refState,
        myIdx,
        simHands,
        models,
        jointAction,
        seeds,
        argmax));
  }

  for (size_t i = 0; i < workloads.size(); ++i) {
    workloads[i].wait();
  }

  float bp = 0;
  float max = 0;
  int bestJointMove = 0;
  std::cout << "bp vs joint deviations" << std::endl;
  for (size_t i = 0; i < jointActions.size(); ++i) {
    auto [score, a1Scores] = workloads[i].get();
    if (i == 0) {
      std::cout << game.GetMove(jointActions[i].a0_).ToString() << " (BP): " << score << std::endl;
    } else {
      if (argmax) {
        std::cout << game.GetMove(jointActions[i].a0_).ToString()
                  << " -> " << game.GetMove(jointActions[i].argmaxA1_).ToString()
                  << ": " << score <<std::endl;
      } else {
        std::cout << game.GetMove(jointActions[i].a0_).ToString()
                  << " -> [";
        auto& weight = jointActions[i].softmaxQA1_;
        for (int i = 0; i < (int)weight.size(); ++i) {
          if (i >= 5 && i < 10) {
            float a1Score = -1;
            if (a1Scores.find(i) != a1Scores.end()) {
              a1Score = a1Scores[i];
            }
            std::cout << game.GetMove(i).ToString() << ": " << weight[i]
                      << " (" << std::setw(5) << a1Score << "), ";
          }
        }
        std::cout << "], score: " << score << std::endl;
      }
    }
    if (i == 0) {
      bp = score;
      max = score;
    } else {
      if (score > max) {
        max = score;
        bestJointMove = i;
      }
    }
  }
  std::cout << "----------------------" << std::endl;

  assert(jointActions.size() == jointActions2.size() + 1);  // jointActions[0] = bp
  if (max - bp >= threshold) {
    std::vector<hle::HanabiMove> jointMove;
    jointMove.push_back(game.GetMove(jointActions[bestJointMove].a0_));
    if (argmax) {
      if (jointActions2[bestJointMove-1].argmaxA1_ == jointActions[bestJointMove].argmaxA1_) {
        std::cout << "compute twice match" << std::endl;
      } else {
        std::cout << "compute twice does not match" << std::endl;
      }
      jointMove.push_back(game.GetMove(jointActions2[bestJointMove-1].argmaxA1_));
    } else {
      // sample an a1 from the distribution
      auto& weight = jointActions2[bestJointMove-1].softmaxQA1_;
      std::discrete_distribution<int> dist(weight.begin(), weight.end());
      int a1 = dist(*rng);
      jointMove.push_back(game.GetMove(a1));
    }
    std::cout << "Found better move: " << jointMove[0].ToString()
              << " -> " << jointMove[1].ToString() << ", sampled? " << !argmax << ", "
              << max << " vs " << bp << " (BP: "  << bpMove.ToString() << ")" << std::endl;
    return jointMove;
  } else {
    return {bpMove};
  }
}

std::vector<float> softmax(std::vector<float> input) {
  std::vector<float> ret(input.size());
  float m = -std::numeric_limits<float>::max();

  for (size_t i = 0; i < input.size(); ++i) {
    if (m < input[i]) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    sum += std::exp(input[i] - m);
  }

  float constant = m + log(sum);
  for (size_t i = 0; i < input.size(); ++i) {
    ret[i] = std::exp(input[i] - constant);
  }
  return ret;
}

}  // namespace

std::vector<JointAction> FinesseActor::finesse_(
    const GameSimulator& env,
    int numOuterHand,
    int numInnerHand,
    int numThread,
    bool argmax,
    float beta,
    bool voteBased) const {
  const auto& state = env.state();
  std::vector<std::pair<int, const HandDistribution*>> handDists;
  // put our partner first since we need to sample it first
  handDists.push_back({nextActor_->index_, &(nextActor_->publHandDist_)});
  handDists.push_back({index_, &publHandDist_});

  auto simHands = sampleHands(state, handDists, numOuterHand, numInnerHand, numThread, &rng_);
  auto [bpMoves, bpDist] = getBlueprintMove(state, simHands, model_);
  // for (size_t i = 0; i < bpDist.size(); ++i) {
  //   std::cout << state.ParentGame()->GetMove(i).ToString() << ": " << bpDist[i]
  //             << std::endl;
  // }

  // get models in correct order
  std::vector<std::vector<HybridModel>> modelGpus;
  for (int gpuIdx = 0; gpuIdx < (int)bpGpus_.size(); ++gpuIdx) {
    std::vector<HybridModel> models;
    for (int i = 0; i < (int)partners_.size(); ++i) {
      int partnerIdx = (i + index_) % (int)partners_.size();
      models.push_back(partners_[partnerIdx]->model_);
      models[i].setBpModel(partners_[partnerIdx]->bpGpus_[gpuIdx]);
    }
    modelGpus.push_back(models);
  }

  auto jointQvalues = computeQvaluesFromSimulation(
      state, simHands, modelGpus, bpMoves, bpDist, &rng_, numThread);
  if (jointQvalues.size() == 0) {
    return {};
  }

  auto [jointDevQs, jointQs, jointActions] = getJointQTable(
      jointQvalues, env.game().MaxMoves(), voteBased);
  std::cout << "***Joint Q***" << std::endl;
  printJointQ(jointQs, env.game());
  std::cout << "***Joint Dev Q***" << std::endl;
  printJointQ(jointDevQs, env.game());

  if (!argmax) {
    // scale with beta and convert to softmax
    for (auto& jointAction : jointActions) {
      for (size_t i = 0; i < jointAction.qA1_.size(); ++i) {
        jointAction.qA1_[i] *= beta;
      }
      jointAction.softmaxQA1_ = softmax(jointAction.qA1_);
      // mask out unwanted values
      for (size_t i = 0; i < jointAction.qA1_.size(); ++i) {
        if (jointAction.qA1_[i] < 1e-5) {
          jointAction.softmaxQA1_[i] = 0;
        }
      }
    }
  }

  return jointActions;
}

void FinesseActor::finesse(
    const GameSimulator& env,
    int numOuterHand,
    int numInnerHand,
    int numThread,
    float threshold,
    bool argmax,
    float beta,
    bool voteBased,
    bool computeTwice) {
  // must be called after update belief and before anythin else
  // assert(callOrder_ == 1);
  if (!finesseMove_.empty()) {
    // finesse has been set by previous player, skip
    return;
  }
  if (followBp_) {
    followBp_ = false;
    return;
  }

  torch::NoGradGuard ng;

  auto jointActions = finesse_(env, numOuterHand, numInnerHand, numThread, argmax, beta, voteBased);
  if (jointActions.empty()) {
    return;
  }

  std::vector<JointAction> jointActions2;
  if (computeTwice) {
    std::cout << "^^^2nd compute^^^" << std::endl;
    jointActions2 = finesse_(env, numOuterHand, numInnerHand, numThread, argmax, beta, voteBased);
    std::cout << jointActions2.size() << " vs " << jointActions.size() << std::endl;
    assert(jointActions2.size() == jointActions.size());
    for (size_t i = 0; i < jointActions.size(); ++i) {
      assert(jointActions[i].a0_ == jointActions2[i].a0_);
    }
  } else {
    jointActions2 = jointActions;
  }

  // get bp move
  int bpAction = model_.asyncComputeAction(env).get().at("a").item<int>();
  auto bpMove = env.game().GetMove(bpAction);

  std::vector<HybridModel> models;
  for (int i = 0; i < (int)partners_.size(); ++i) {
    int partnerIdx = (i + index_) % (int)partners_.size();
    models.push_back(partners_[partnerIdx]->model_);
    models[i].setBpModel(partners_[partnerIdx]->bpGpus_[0]);
  }

  int numHand = 10000 / jointActions.size();
  auto bestMoves = evaluateJointDevAndBp(
      env.state(),
      index_,
      privHandDist_,
      numHand,
      models,
      bpMove,
      jointActions,
      jointActions2,
      &rng_,
      threshold,
      argmax);

  if (bestMoves.size() > 1) {
    // otherwise nothing needs to be done, just follow bp
    finesseMove_.push_back(bestMoves[0]);
    assert(nextActor_->finesseMove_.empty());
    nextActor_->finesseMove_.push_back(bestMoves[1]);
    nextActor_->nextActor_->followBp_ = true;;
  }
}
