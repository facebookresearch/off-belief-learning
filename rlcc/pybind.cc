// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <pybind11/pybind11.h>

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_card.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_hand.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_move.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_observation.h"

#include "rlcc/clone_data_generator.h"
#include "rlcc/hanabi_env.h"
#include "rlcc/thread_loop.h"

namespace py = pybind11;
using namespace hanabi_learning_env;

PYBIND11_MODULE(hanalearn, m) {
  py::class_<HanabiEnv, std::shared_ptr<HanabiEnv>>(m, "HanabiEnv")
      .def(py::init<
           const std::unordered_map<std::string, std::string>&,
           int,  // maxLen
           bool>())
      .def("feature_size", &HanabiEnv::featureSize)
      .def("num_action", &HanabiEnv::numAction)
      .def("reset", &HanabiEnv::reset)
      .def("reset_with_deck", &HanabiEnv::resetWithDeck)
      .def("step", &HanabiEnv::step)
      .def("terminated", &HanabiEnv::terminated)
      .def("get_current_player", &HanabiEnv::getCurrentPlayer)
      .def("last_episode_score", &HanabiEnv::lastEpisodeScore)
      .def("deck_history", &HanabiEnv::deckHistory)
      .def("get_num_players", &HanabiEnv::getNumPlayers)
      .def("get_score", &HanabiEnv::getScore)
      .def("get_life", &HanabiEnv::getLife)
      .def("get_info", &HanabiEnv::getInfo)
      .def("get_fireworks", &HanabiEnv::getFireworks)
      .def("get_hle_state", &HanabiEnv::getHleState)
      .def("get_move", &HanabiEnv::getMove)
      .def("get_obs_show_cards", &HanabiEnv::getObsShowCards)
      .def("get_last_action", &HanabiEnv::getLastAction)
      .def("get_step", &HanabiEnv::numStep)
      .def("set_color_reward", &HanabiEnv::setColorReward);

  py::class_<CloneDataGenerator, std::shared_ptr<CloneDataGenerator>>(
      m, "CloneDataGenerator")
      .def(py::init<
           std::shared_ptr<rela::RNNPrioritizedReplay>,
           int,
           int,
           bool,
           bool,
           int>())
      .def("set_game_params", &CloneDataGenerator::setGameParams)
      .def("add_game", &CloneDataGenerator::addGame)
      .def("start_data_generation", &CloneDataGenerator::startDataGeneration)
      .def("terminate", &CloneDataGenerator::terminate);

  py::class_<R2D2Actor, std::shared_ptr<R2D2Actor>>(m, "R2D2Actor")
      .def(py::init<
           std::shared_ptr<rela::BatchRunner>,  // runner,
           int,                                 // seed,
           int,  // numPlayer,                       // total number os players
           int,  // playerIdx,                       // player idx for this player
           const std::vector<float>&,  // epsList,   // list of eps to sample from
           const std::vector<float>&,  // tempList,  // list of temp to sample from
           bool,                       // vdn,
           bool,                       // sad,
           bool,                       // shuffleColor,
           bool,                       //  hideAction,
           bool,                       // trinary,  // trinary aux task or full aux
           std::shared_ptr<rela::RNNPrioritizedReplay>,  //  replayBuffer,
           // if replay buffer is None, then all params below are not used
           int,       // multiStep,
           int,       // seqLen,
           float>())  // gamma
      .def(py::init<
           std::shared_ptr<rela::BatchRunner>,
           int,      // numPlayer
           int,      // playerIdx
           bool,     // vdn
           bool,     // sad
           bool>())  // hideAction
      .def("set_partners", &R2D2Actor::setPartners)
      .def("set_belief_runner", &R2D2Actor::setBeliefRunner)
      .def("get_success_fict_rate", &R2D2Actor::getSuccessFictRate)
      .def("get_played_card_info", &R2D2Actor::getPlayedCardInfo);

  m.def("observe", py::overload_cast<const hle::HanabiState&, int, bool>(&observe));

  m.def(
      "observe_op",
      py::overload_cast<
          const hle::HanabiState&,
          int,
          bool,
          const std::vector<int>&,
          const std::vector<int>&,
          bool,
          bool,
          bool>(&observe));

  m.def("observe_sad", &observeSAD);

  py::class_<HanabiThreadLoop, rela::ThreadLoop, std::shared_ptr<HanabiThreadLoop>>(
      m, "HanabiThreadLoop")
      .def(py::init<
           std::vector<std::shared_ptr<HanabiEnv>>,
           std::vector<std::vector<std::shared_ptr<R2D2Actor>>>,
           bool>());

  // bind some hanabi util classes
  py::class_<HanabiCard>(m, "HanabiCard")
      .def(py::init<int, int, int>())
      .def("color", &HanabiCard::Color)
      .def("rank", &HanabiCard::Rank)
      .def("id", &HanabiCard::Id)
      .def("is_valid", &HanabiCard::IsValid)
      .def("to_string", &HanabiCard::ToString)
      .def(py::pickle(
          [](const HanabiCard& c) {
            // __getstate__
            return py::make_tuple(c.Color(), c.Rank(), c.Id());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiCard c(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
            return c;
          }));

  // bind some hanabi util classes
  py::class_<HanabiCardValue>(m, "HanabiCardValue")
      .def(py::init<int, int>())
      .def("color", &HanabiCardValue::Color)
      .def("rank", &HanabiCardValue::Rank)
      .def("is_valid", &HanabiCardValue::IsValid)
      .def("to_string", &HanabiCardValue::ToString)
      .def(py::pickle(
          [](const HanabiCardValue& c) {
            // __getstate__
            return py::make_tuple(c.Color(), c.Rank());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiCardValue c(t[0].cast<int>(), t[1].cast<int>());
            return c;
          }));

  py::class_<HanabiHistoryItem>(m, "HanabiHistoryItem")
      .def(py::init<HanabiMove>())
      .def_readwrite("move", &HanabiHistoryItem::move)
      .def_readwrite("scored", &HanabiHistoryItem::scored)
      .def_readwrite("information_token", &HanabiHistoryItem::information_token)
      .def_readwrite("player", &HanabiHistoryItem::player)
      .def_readwrite("color", &HanabiHistoryItem::color)
      .def_readwrite("rank", &HanabiHistoryItem::rank)
      .def_readwrite("reveal_bitmask", &HanabiHistoryItem::reveal_bitmask)
      .def_readwrite(
          "newly_revealed_bitmask", &HanabiHistoryItem::newly_revealed_bitmask);

  py::class_<HanabiHand::CardKnowledge>(m, "CardKnowledge")
      .def(py::init<int, int>())
      .def("num_colors", &HanabiHand::CardKnowledge::NumColors)
      .def("color_hinted", &HanabiHand::CardKnowledge::ColorHinted)
      .def("color", &HanabiHand::CardKnowledge::Color)
      .def("color_plausible", &HanabiHand::CardKnowledge::ColorPlausible)
      .def("apply_is_color_hint", &HanabiHand::CardKnowledge::ApplyIsColorHint)
      .def("apply_is_not_color_hint", &HanabiHand::CardKnowledge::ApplyIsNotColorHint)
      .def("num_ranks", &HanabiHand::CardKnowledge::NumRanks)
      .def("rank_hinted", &HanabiHand::CardKnowledge::RankHinted)
      .def("rank", &HanabiHand::CardKnowledge::Rank)
      .def("rank_plausible", &HanabiHand::CardKnowledge::RankPlausible)
      .def("apply_is_rank_hint", &HanabiHand::CardKnowledge::ApplyIsRankHint)
      .def("apply_is_not_rank_hint", &HanabiHand::CardKnowledge::ApplyIsNotRankHint)
      .def("is_card_plausible", &HanabiHand::CardKnowledge::IsCardPlausible)
      .def("to_string", &HanabiHand::CardKnowledge::ToString);

  py::class_<HanabiHand>(m, "HanabiHand")
      .def(py::init<>())
      .def("cards", &HanabiHand::Cards)
      .def("knowledge_", &HanabiHand::Knowledge_, py::return_value_policy::reference)
      .def("knowledge", &HanabiHand::Knowledge)
      .def("add_card", &HanabiHand::AddCard)
      .def("remove_from_hand", &HanabiHand::RemoveFromHand)
      .def("to_string", &HanabiHand::ToString);

  py::class_<HanabiGame>(m, "HanabiGame")
      .def(py::init<const std::unordered_map<std::string, std::string>&>())
      .def("max_moves", &HanabiGame::MaxMoves)
      .def(
          "get_move_uid",
          (int (HanabiGame::*)(HanabiMove) const) & HanabiGame::GetMoveUid)
      .def("get_move", &HanabiGame::GetMove)
      .def("num_colors", &HanabiGame::NumColors)
      .def("num_ranks", &HanabiGame::NumRanks)
      .def("hand_size", &HanabiGame::HandSize)
      .def("max_information_tokens", &HanabiGame::MaxInformationTokens)
      .def("max_life_tokens", &HanabiGame::MaxLifeTokens)
      .def("max_deck_size", &HanabiGame::MaxDeckSize);

  py::class_<HanabiState>(m, "HanabiState")
      .def(py::init<const HanabiGame*, int>())
      .def("hands", py::overload_cast<>(&HanabiState::Hands, py::const_))
      .def("apply_move", &HanabiState::ApplyMove)
      .def("cur_player", &HanabiState::CurPlayer)
      .def("score", &HanabiState::Score)
      .def("max_possible_score", &HanabiState::MaxPossibleScore)
      .def("info_tokens", &HanabiState::InformationTokens)
      .def("life_tokens", &HanabiState::LifeTokens)
      .def("to_string", &HanabiState::ToString)
      .def("is_terminal", &HanabiState::IsTerminal);

  py::enum_<HanabiMove::Type>(m, "MoveType")
      .value("Invalid", HanabiMove::Type::kInvalid)
      .value("Play", HanabiMove::Type::kPlay)
      .value("Discard", HanabiMove::Type::kDiscard)
      .value("RevealColor", HanabiMove::Type::kRevealColor)
      .value("RevealRank", HanabiMove::Type::kRevealRank)
      .value("Deal", HanabiMove::Type::kDeal);
  // .export_values();

  py::class_<HanabiMove>(m, "HanabiMove")
      .def(py::init<HanabiMove::Type, int8_t, int8_t, int8_t, int8_t>())
      .def("move_type", &HanabiMove::MoveType)
      .def("target_offset", &HanabiMove::TargetOffset)
      .def("card_index", &HanabiMove::CardIndex)
      .def("color", &HanabiMove::Color)
      .def("rank", &HanabiMove::Rank)
      .def("to_string", &HanabiMove::ToString)
      .def("set_color", &HanabiMove::SetColor)
      .def(py::pickle(
          [](const HanabiMove& m) {
            // __getstate__
            return py::make_tuple(
                m.MoveType(), m.CardIndex(), m.TargetOffset(), m.Color(), m.Rank());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 5) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiMove m(
                t[0].cast<HanabiMove::Type>(),
                t[1].cast<int8_t>(),
                t[2].cast<int8_t>(),
                t[3].cast<int8_t>(),
                t[4].cast<int8_t>());
            return m;
          }));

  py::class_<HanabiObservation>(m, "HanabiObservation")
      .def(py::init<const HanabiState&, int, bool>())
      .def(py::init<
           int,
           int,
           const std::vector<HanabiHand>&,
           const std::vector<HanabiCard>&,
           const std::vector<int>&,
           const std::vector<HanabiHistoryItem>&,
           int,
           int,
           int,
           const std::vector<HanabiMove>&,
           const HanabiGame*>())
      .def("legal_moves", &HanabiObservation::LegalMoves)
      .def("last_moves", &HanabiObservation::LastMoves)
      .def("life_tokens", &HanabiObservation::LifeTokens)
      .def("information_tokens", &HanabiObservation::InformationTokens)
      .def("deck_size", &HanabiObservation::DeckSize)
      .def("fireworks", &HanabiObservation::Fireworks)
      .def(
          "card_playable_on_fireworks",
          [](HanabiObservation& obs, int color, int rank) {
            return obs.CardPlayableOnFireworks(color, rank);
          })
      .def("discard_pile", &HanabiObservation::DiscardPile)
      .def("hands", &HanabiObservation::Hands);

  py::class_<CanonicalObservationEncoder>(m, "ObservationEncoder")
      .def(py::init<const HanabiGame*>())
      .def("shape", &CanonicalObservationEncoder::Shape)
      .def("encode", &CanonicalObservationEncoder::Encode);
}
