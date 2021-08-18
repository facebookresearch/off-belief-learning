// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __HANABI_CARD_H__
#define __HANABI_CARD_H__

#include <string>

namespace hanabi_learning_env {

class HanabiCardValue {
 public:
  HanabiCardValue(int color, int rank) : color_(color), rank_(rank) {}
  HanabiCardValue() = default;  // Create an invalid card.
  bool operator==(const HanabiCardValue& other_card) const;
  bool IsValid() const { return color_ >= 0 && rank_ >= 0; }
  std::string ToString() const;
  int Color() const { return color_; }
  int Rank() const { return rank_; }

 private:
  int color_ = -1;  // 0 indexed card color.
  int rank_ = -1;   // 0 indexed card rank.
};

class HanabiCard {
 public:
  HanabiCard(int color, int rank, int id) : color_(color), rank_(rank), id_(id) {}
  HanabiCard(HanabiCardValue value, int id) : color_(value.Color()), rank_(value.Rank()), id_(id) {}
  HanabiCard(int id) : color_(-1), rank_(-1), id_(id) {}
  HanabiCard() = default;  // Create an invalid card.
  bool operator==(const HanabiCard& other_card) const = delete;
  bool IsValid() const { return color_ >= 0 && rank_ >= 0; }
  std::string ToString() const;
  int Color() const { return color_; }
  int Rank() const { return rank_; }
  HanabiCardValue Value() const { return HanabiCardValue(color_, rank_); }
  int Id() const { return id_; }
  HanabiCard HideValue() const { return HanabiCard(-1, -1, id_); }

 private:
  int color_ = -1;  // 0 indexed card color.
  int rank_ = -1;   // 0 indexed card rank.
  int id_ = -1; //Index 0 to N-1 where N is the number of cards in the deck
};

}  // namespace hanabi_learning_env

#endif
