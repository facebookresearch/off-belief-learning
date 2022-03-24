# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
from typing import Tuple, Dict
import random
import time
import belief_model

import torch
import numpy as np

# c++ backend
import set_path
set_path.append_sys_path()
import rela
import hanalearn

import utils
import common_utils


class SearchWrapper:
    def __init__(
        self,
        player_idx,
        public_belief,
        weight_file,
        belief_file,
        num_samples,
        explore_eps,
        n_step,
        gamma,
        train_device,
        rl_rollout_device,
        bp_rollout_device,
        belief_device,
        rollout_bsize,
        num_thread,
        num_game_per_thread,
        log_bsize_freq=-1,
    ):
        self.player_idx = player_idx
        self.public_belief = public_belief
        assert not public_belief
        self.num_thread = num_thread
        self.num_game_per_thread = num_game_per_thread
        self.num_samples = num_samples
        self.acceptance_rate = 0.05

        if rl_rollout_device is None:
            self.rl = None
            self.rl_runner = None
        else:
            # NOTE: multi-step is hard-coded to 1
            self.rl, config = utils.load_agent(
                weight_file, {"device": train_device, "off_belief": False}
            )
            assert not config["hide_action"]
            assert not config["boltzmann_act"]
            assert config["method"] == "iql"
            assert self.rl.multi_step == 1

            self.rl_runner = rela.BatchRunner(
                self.rl.clone(rl_rollout_device),
                rl_rollout_device,
                rollout_bsize,
                ["act", "compute_priority"],
            )
            self.rl_runner.start()

        self.bp, config = utils.load_agent(
            weight_file, {"device": bp_rollout_device, "off_belief": False}
        )
        assert not config["hide_action"]
        assert not config["boltzmann_act"]
        assert config["method"] == "iql"
        assert self.bp.multi_step == 1

        self.bp_runner = rela.BatchRunner(
            self.bp,
            bp_rollout_device,
            rollout_bsize,
            ["act", "compute_target"],
        )
        if log_bsize_freq > 0:
            self.bp_runner.set_log_freq(log_bsize_freq)
        self.bp_runner.start()

        if belief_file:
            self.belief_model = belief_model.ARBeliefModel.load(
                belief_file,
                belief_device,
                hand_size=5,
                num_sample=num_samples,
                fc_only=False,
                mode="priv",
            )
            self.blueprint_belief = belief_model.ARBeliefModel.load(
                belief_file,
                belief_device,
                hand_size=5,
                num_sample=num_samples,
                fc_only=False,
                mode="priv",
            )
            self.belief_runner = rela.BatchRunner(
                self.belief_model, belief_device, rollout_bsize, ["observe", "sample"]
            )
            self.belief_runner.start()
        else:
            self.belief_runner = None

        self.explore_eps = explore_eps
        self.gamma = gamma
        self.n_step = n_step
        self.actor = None
        self.reset()

    def reset(self):
        self.actor = hanalearn.RLSearchActor(
            self.player_idx,
            self.bp_runner,
            self.rl_runner,
            self.belief_runner,  # belief runner
            self.num_samples,  # num samples
            self.public_belief,  # public belief
            False,  # joint search
            self.explore_eps,
            self.n_step,
            self.gamma,
            random.randint(1, 999999),
        )
        self.actor.set_compute_config(self.num_thread, self.num_game_per_thread)

    def update_rl_model(self, model):
        self.rl_runner.update_model(model)

    def reset_rl_to_bp(self):
        self.rl.online_net.load_state_dict(self.bp.online_net.state_dict())
        self.rl.target_net.load_state_dict(self.bp.online_net.state_dict())
        self.update_rl_model(self.rl)
        self.actor.reset_rl_rnn()


def train(game, search_actor, replay_buffer, args, eval_seed):
    if args.search_exact_belief:
        sim_hands = [[[]]]
        use_sim_hands = False
    else:
        sim_hands = search_actor.actor.sample_hands(game.state(), args.num_samples)
        use_sim_hands = True
        if len(sim_hands[0]) < search_actor.num_samples * search_actor.acceptance_rate:
            print(
                f"Belief acceptance rate is less than {search_actor.acceptance_rate}; "
                f"falling back to blueprint"
            )
            return None, None

    max_possible_score = game.state().max_possible_score()
    bp_scores = search_actor.actor.run_sim_games(
        game, args.num_eval_game, 0, eval_seed, sim_hands, use_sim_hands
    )
    assert np.mean(bp_scores) <= max_possible_score + 1e-5
    if max_possible_score - np.mean(bp_scores) < args.threshold:
        return np.mean(bp_scores), 0

    search_actor.actor.start_data_generation(
        game, replay_buffer, args.num_rl_step, sim_hands, use_sim_hands, False
    )

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("Done: replay buffer size:", replay_buffer.size())

    optim = torch.optim.Adam(
        search_actor.rl.online_net.parameters(), lr=args.lr, eps=args.eps
    )

    if args.final_only:
        for p in search_actor.rl.online_net.parameters():
            p.requires_grad = False
        for p in search_actor.rl.online_net.fc_v.parameters():
            p.requires_grad = True
        for p in search_actor.rl.online_net.fc_a.parameters():
            p.requires_grad = True
        for p in search_actor.rl.online_net.pred_1st.parameters():
            p.requires_grad = True

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    saver = common_utils.TopkSaver(args.save_dir, 5)

    for epoch in range(args.num_epoch):
        tachometer.start()
        stat.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                search_actor.rl.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                search_actor.update_rl_model(search_actor.rl)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")
            loss, priority, _ = search_actor.rl.loss(batch, args.aux, stat)
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                search_actor.rl.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            replay_buffer.update_priority(priority)
            stopwatch.time("other")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        print("EPOCH: %d" % epoch)
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, 1)
        stat.summary(epoch)
        stopwatch.summary()

    search_actor.actor.stop_data_generation()
    search_actor.update_rl_model(search_actor.rl)
    rl_scores = search_actor.actor.run_sim_games(
        game, args.num_eval_game, args.num_rl_step, eval_seed, sim_hands, use_sim_hands
    )

    rl_mean = np.mean(rl_scores)
    rl_sem = np.std(rl_scores) / np.sqrt(len(rl_scores))
    bp_mean = np.mean(bp_scores)
    bp_sem = np.std(bp_scores) / np.sqrt(len(bp_scores))
    print(f">>>>>bp score: {bp_mean:.3f} +/- {bp_sem:.3f}")
    print(f">>>>>rl score: {rl_mean:.3f} +/- {rl_sem:.3f}")
    print(f"mean diff: {rl_mean - bp_mean}")
    print(f"mean-sem diff: {rl_mean - rl_sem - bp_mean}")
    print(f"mean-(sem+sem) diff: {rl_mean - rl_sem - bp_mean - bp_sem}")

    return np.mean(bp_scores), np.mean(rl_scores)


def run(seed, actors, search_actor, args):
    params = {
        "players": str(len(actors)),
        "seed": str(seed),
        "bomb": str(0),
        "hand_size": str(5),
        "max_information_tokens": str(args.num_hint),  # global variable
        "random_start_player": str(0),  # do not randomize start_player
    }
    game = hanalearn.GameSimulator(params)
    step = 0

    # created once, reused for the rest of training
    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        random.randint(1, 999999),
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    while not game.terminal():
        print("================STEP %d================" % step)
        print(f"{game.state().to_string()}\n")

        using_rl = search_actor.actor.using_rl()
        assert using_rl >= 0
        cur_player = game.state().cur_player()

        for i, actor in enumerate(actors):
            if i != search_actor.player_idx:
                continue
            if args.maintain_exact_belief:
                print(f"---Actor {i} update belief---")
                actor.update_belief(game)

        # if already in rl mode, then no more training
        if (
            args.num_rl_step > 0
            and using_rl == 0
            and cur_player == search_actor.player_idx
        ):
            replay_buffer.clear()
            # always restart from bp
            search_actor.reset_rl_to_bp()
            t = time.time()
            bp_score, rl_score = train(
                game, search_actor, replay_buffer, args, random.randint(1, 999999)
            )

            print(
                "rl - bp:  %.4f" % (rl_score - bp_score),
                ", time taken: %ds" % (time.time() - t),
            )
            if rl_score - bp_score >= args.threshold:
                print("set use rl")
                search_actor.actor.set_use_rl(args.num_rl_step)

        for i, actor in enumerate(actors):
            print(f"---Actor {i} observe---")
            actor.observe(game)

        if search_actor.belief_runner is not None:
            for i, actor in enumerate(actors):
                if i != search_actor.player_idx:
                    continue
                print(f"---Actor {i} update belief hid---")
                actor.update_belief_hid(game)

        actions = []
        for i, actor in enumerate(actors):
            print(f"---Actor {i} decide action---")
            action = actor.decide_action(game)
            actions.append(action)

        move = game.get_move(actions[cur_player])
        if args.sparta and cur_player == search_actor.player_idx:
            move = actor.sparta_search(
                game, move, args.sparta_num_search, args.sparta_threshold
            )

        print(f"Acitve Player {cur_player} pick action: {move.to_string()}")
        game.step(move)
        step += 1

    print(f"Final Score: {game.get_score()}, Seed: {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/rl_search1")

    parser.add_argument("--public_belief", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--game_seed", type=int, default=1)
    parser.add_argument("--n_step", type=int, default=1, help="n_step return")
    parser.add_argument("--num_eval_game", type=int, default=5000)
    parser.add_argument("--final_only", type=int, default=0)
    parser.add_argument("--sparta", type=int, default=0)
    parser.add_argument("--sparta_num_search", type=int, default=10000)
    parser.add_argument("--sparta_threshold", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--num_rl_step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")

    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--burn_in_frames", type=int, default=5000)
    parser.add_argument("--priority_exponent", type=float, default=0.9, help="alpha")
    parser.add_argument("--priority_weight", type=float, default=0.6, help="beta")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--rl_rollout_device", type=str, default="cuda:1")
    parser.add_argument("--bp_rollout_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--rollout_batchsize", type=int, default=8000)

    parser.add_argument("--num_thread", type=int, default=1, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    parser.add_argument("--aux", type=float, default=0.25)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--epoch_len", type=int, default=5000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    parser.add_argument("--num_hint", type=int, required=True)
    parser.add_argument("--weight_file", type=str, required=True)

    parser.add_argument("--belief_file", type=str, default="")
    parser.add_argument("--belief_device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--maintain_exact_belief", type=int, default=1)
    parser.add_argument("--search_exact_belief", type=int, default=1)

    args = parser.parse_args()
    if args.debug:
        args.num_epoch = 1
        args.epoch_len = 200
        args.num_eval_game = 500

    return args


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    args = parse_args()

    common_utils.set_all_seeds(args.seed)
    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    print("explore eps:", explore_eps)
    print("mean eps:", np.mean(explore_eps))
    search_wrapper0 = SearchWrapper(
        0,
        args.public_belief,
        args.weight_file,
        args.belief_file,
        args.num_samples,
        explore_eps,
        args.n_step,
        args.gamma,
        args.train_device,
        None,  # no rl model args.rl_rollout_device,
        args.bp_rollout_device,
        args.belief_device,
        args.rollout_batchsize,
        args.num_thread,
        args.num_game_per_thread,
    )
    search_wrapper1 = SearchWrapper(
        1,
        args.public_belief,
        args.weight_file,
        args.belief_file,
        args.num_samples,
        explore_eps,
        args.n_step,
        args.gamma,
        args.train_device,
        args.rl_rollout_device,
        args.bp_rollout_device,
        args.belief_device,
        args.rollout_batchsize,
        args.num_thread,
        args.num_game_per_thread,
    )
    search_wrapper0.actor.set_partner(search_wrapper1.actor)
    search_wrapper1.actor.set_partner(search_wrapper0.actor)

    run(
        args.game_seed,
        [search_wrapper0.actor, search_wrapper1.actor],
        search_wrapper1,
        args,
    )
