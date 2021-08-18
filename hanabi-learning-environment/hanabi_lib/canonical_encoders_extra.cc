// // encode eps: [0, 1) into normalized log range
// int EncodeEps(const HanabiGame& game,
//               const HanabiObservation& obs,
//               int start_offset,
//               const std::vector<float>* eps,
//               std::vector<float>* encoding) {
//   const int num_players = game.NumPlayers();
//   const float tiny = 1e-6;
//   const float log_tiny = std::log(tiny);

//   int observing_player = obs.ObservingPlayer();
//   int code_offset = start_offset;
//   if (eps != nullptr) {
//     assert(eps->size() == game.NumPlayers());
//   }
//   for (int offset = 1; offset < game.NumPlayers(); ++offset) {
//     float player_eps = 0;
//     if (eps != nullptr) {
//       player_eps = (*eps)[(offset + observing_player) % num_players];
//     }
//     player_eps += tiny;

//     // TODO: magical number 19 to make it [0, 1)
//     float normed = (std::log(player_eps) - log_tiny) / (-log_tiny);
//     assert(normed >= 0 && normed < 1);
//     (*encoding)[code_offset] = normed;
//     ++code_offset;
//   }
//   return code_offset - start_offset;
// }


// int EncodeV1Belief_(const HanabiGame& game,
//                     const HanabiObservation& obs,
//                     int start_offset,
//                     std::vector<float>* encoding) {
//   const int num_iters = 100;
//   const float weight = 0.1;

//   // int bits_per_card = BitsPerCard(game);
//   int num_colors = game.NumColors();
//   int num_ranks = game.NumRanks();
//   int num_players = game.NumPlayers();
//   int hand_size = game.HandSize();
//   const std::vector<HanabiHand>& hands = obs.Hands();

//   std::vector<float> card_knowledge(CardKnowledgeSectionLength(game), 0);
//   int len = EncodeCardKnowledge(game, obs, 0, &card_knowledge);
//   assert(len == card_knowledge.size());

//   std::vector<float> v0_belief(card_knowledge);
//   std::vector<int> card_count;
//   len = EncodeV0Belief_(game, obs, 0, &v0_belief, &card_count);
//   assert(len == card_knowledge.size());

//   const int player_offset = len / num_players;
//   const int per_card_offset = len / hand_size / num_players;
//   assert(per_card_offset == num_colors * num_ranks + num_colors + num_ranks);

//   std::vector<float> v1_belief(v0_belief);
//   std::vector<float> new_v1_belief(v1_belief);
//   std::vector<float> total_cards(card_count.size());

//   assert(total_cards.size() == int(num_colors * num_ranks));
//   for (int step = 0; step < num_iters; ++step) {
//     // first compute total card remaining by excluding info from belief
//     for (int i = 0; i < num_colors * num_ranks; ++i) {
//       total_cards[i] = card_count[i];
//       for (int player_id = 0; player_id < num_players; ++player_id) {
//         int num_cards = hands[player_id].Cards().size();
//         for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
//           int offset = player_offset * player_id + card_idx * per_card_offset + i;
//           assert(offset < (int)v1_belief.size());
//           total_cards[i] -= v1_belief[offset];
//         }
//       }
//     }
//     // for (auto c : total_cards) {
//     //   std::cout << c << ", ";
//     // }
//     // std::cout << std::endl;

//     // compute new belief
//     for (int player_id = 0; player_id < num_players; ++player_id) {
//       int num_cards = hands[player_id].Cards().size();
//       for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
//         // float total = 0;
//         int base_offset = player_offset * player_id + card_idx * per_card_offset;
//         // if (player_id == 0 && card_idx == 0) {
//         //   std::cout << "before norm" << std::endl;
//         // }
//         for (int i = 0; i < num_colors * num_ranks; ++i) {
//           int offset = base_offset + i;
//           assert(offset < (int)v1_belief.size());
//           float p = total_cards[i] + v1_belief[offset];
//           // if (player_id == 0 && card_idx == 0) {
//           //   std::cout << p << ", ";
//           // }
//           p = std::max(p, (float)0.0);
//           new_v1_belief[offset] = p * card_knowledge[offset];
//           // total += new_v1_belief[offset];
//         }
//       }
//     }
//     // interpolate & normalize
//     for (int player_id = 0; player_id < num_players; ++player_id) {
//       int num_cards = hands[player_id].Cards().size();
//       for (int card_idx = 0; card_idx < num_cards; ++card_idx) {
//         float total = 0;
//         int base_offset = player_offset * player_id + card_idx * per_card_offset;
//         for (int i = 0; i < num_colors * num_ranks; ++i) {
//           int offset = i + base_offset;
//           v1_belief[offset] = (1 - weight) * v1_belief[offset] + weight * new_v1_belief[offset];
//           total += v1_belief[offset];
//         }
//         if (total <= 0) {
//           std::cout << "total = 0 " << std::endl;
//           assert(false);
//         }
//         for (int i = 0; i < num_colors * num_ranks; ++i) {
//           int offset = i + base_offset;
//           v1_belief[offset] /= total;
//         }
//       }
//     }
//   }
//
//   for (size_t i = 0; i < v1_belief.size(); ++i) {
//     (*encoding)[i + start_offset] = v1_belief[i];
//   }
//   return v1_belief.size();
// }


// std::vector<float> CanonicalObservationEncoder::EncodeV0Belief(
//     const HanabiObservation& obs,
//     bool all_player) const {
//   std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
//   int len = EncodeV0Belief_(*parent_game_, obs, 0, &encoding);
//   assert(len == (int)encoding.size());
//   auto belief = ExtractBelief(encoding, *parent_game_, all_player);
//   return belief;
// }

// std::vector<float> CanonicalObservationEncoder::EncodeV1Belief(
//     const HanabiObservation& obs,
//     bool all_player) const {
//   std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
//   int len = EncodeV1Belief_(*parent_game_, obs, 0, &encoding);
//   assert(len == (int)encoding.size());
//   auto belief = ExtractBelief(encoding, *parent_game_, all_player);
//   return belief;
// }

// std::vector<float> CanonicalObservationEncoder::EncodeHandMask(
//     const HanabiObservation& obs) const {
//   std::vector<float> encoding(CardKnowledgeSectionLength(*parent_game_), 0);
//   // const int len = EncodeCardKnowledge(game, obs, start_offset, encoding);
//   EncodeCardKnowledge(*parent_game_, obs, 0, &encoding);
//   auto hm = ExtractBelief(encoding, *parent_game_);
//   return hm;
// }

// std::vector<float> CanonicalObservationEncoder::EncodeCardCount(
//     const HanabiObservation& obs) const {
//   std::vector<float> encoding;
//   auto cc = ComputeCardCount(*parent_game_, obs);
//   for (size_t i = 0; i < cc.size(); ++i) {
//     encoding.push_back((float)cc[i]);
//   }
//   return encoding;
// }

// std::vector<float> CanonicalObservationEncoder::EncodeOwnHandTrinary(
//     const HanabiObservation& obs) const {
//   // int len = parent_game_->HandSize() * BitsPerCard(*parent_game_);
//   // hard code 5 cards, empty slot will be all zero
//   int len = 5 * 3;
//   std::vector<float> encoding(len, 0);
//   int bits_per_card = 3; // BitsPerCard(game);
//   int num_ranks = parent_game_->NumRanks();

//   int offset = 0;
//   const std::vector<HanabiHand>& hands = obs.Hands();
//   const int player = 0;
//   const std::vector<HanabiCard>& cards = hands[player].Cards();

//   const std::vector<int>& fireworks = obs.Fireworks();
//   for (const HanabiCard& card : cards) {
//     // Only a player's own cards can be invalid/unobserved.
//     // assert(card.IsValid());
//     assert(card.Color() < parent_game_->NumColors());
//     assert(card.Rank() < num_ranks);
//     assert(card.IsValid());
//     // std::cout << offset << CardIndex(card.Color(), card.Rank(), num_ranks) << std::endl;
//     // std::cout << card.Color() << ", " << card.Rank() << ", " << num_ranks << std::endl;
//     auto firework = fireworks[card.Color()];
//     if (card.Rank() == firework) {
//       encoding[offset] = 1;
//     } else if (card.Rank() < firework) {
//       encoding[offset + 1] = 1;
//     } else {
//       encoding[offset + 2] = 1;
//     }

//     offset += bits_per_card;
//   }

//   assert(offset <= len);
//   return encoding;
// }

// std::vector<float> CanonicalObservationEncoder::EncodeOwnHand(
//     const HanabiObservation& obs) const {
//   int bits_per_card =  BitsPerCard(*parent_game_);
//   int len = parent_game_->HandSize() * bits_per_card;
//   std::vector<float> encoding(len, 0);

//   const std::vector<HanabiCard>& cards = obs.Hands()[0].Cards();
//   const int num_ranks = parent_game_->NumRanks();

//   int offset = 0;
//   for (const HanabiCard& card : cards) {
//     // Only a player's own cards can be invalid/unobserved.
//     assert(card.IsValid());
//     int idx = CardIndex(card.Color(), card.Rank(), num_ranks);
//     encoding[offset + idx] = 1;
//     offset += bits_per_card;
//   }

//   assert(offset == cards.size() * bits_per_card);
//   return encoding;
// }
