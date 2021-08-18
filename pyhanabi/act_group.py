# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import set_path

set_path.append_sys_path()

import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


class ActGroup:
    def __init__(
        self,
        devices,
        agent,
        seed,
        num_thread,
        num_game_per_thread,
        num_player,
        explore_eps,
        boltzmann_t,
        method,
        sad,
        shuffle_color,
        hide_action,
        trinary,
        replay_buffer,
        multi_step,
        max_len,
        gamma,
        off_belief,
        belief_model,
    ):
        self.devices = devices.split(",")

        self.model_runners = []
        for dev in self.devices:
            runner = rela.BatchRunner(agent.clone(dev), dev)
            runner.add_method("act", 5000)
            runner.add_method("compute_priority", 100)
            if off_belief:
                runner.add_method("compute_target", 5000)
            self.model_runners.append(runner)
        self.num_runners = len(self.model_runners)

        self.off_belief = off_belief
        self.belief_model = belief_model
        self.belief_runner = None
        if belief_model is not None:
            self.belief_runner = []
            for bm in belief_model:
                print("add belief model to: ", bm.device)
                self.belief_runner.append(
                    rela.BatchRunner(bm, bm.device, 5000, ["sample"])
                )

        self.actors = []
        if method == "vdn":
            for i in range(num_thread):
                thread_actors = []
                for j in range(num_game_per_thread):
                    actor = hanalearn.R2D2Actor(
                        self.model_runners[i % self.num_runners],
                        seed,
                        num_player,
                        0,
                        explore_eps,
                        boltzmann_t,
                        True,
                        sad,
                        shuffle_color,
                        hide_action,
                        trinary,
                        replay_buffer,
                        multi_step,
                        max_len,
                        gamma,
                    )
                    seed += 1
                    thread_actors.append([actor])
                self.actors.append(thread_actors)
        elif method == "iql":
            for i in range(num_thread):
                thread_actors = []
                for j in range(num_game_per_thread):
                    game_actors = []
                    for k in range(num_player):
                        actor = hanalearn.R2D2Actor(
                            self.model_runners[i % self.num_runners],
                            seed,
                            num_player,
                            k,
                            explore_eps,
                            boltzmann_t,
                            False,
                            sad,
                            shuffle_color,
                            hide_action,
                            trinary,
                            replay_buffer,
                            multi_step,
                            max_len,
                            gamma,
                        )
                        if self.off_belief:
                            if self.belief_runner is None:
                                actor.set_belief_runner(None)
                            else:
                                actor.set_belief_runner(
                                    self.belief_runner[i % len(self.belief_runner)]
                                )
                        seed += 1
                        game_actors.append(actor)
                    for k in range(num_player):
                        partners = game_actors[:]
                        partners[k] = None
                        game_actors[k].set_partners(partners)
                    thread_actors.append(game_actors)
                self.actors.append(thread_actors)
        print("ActGroup created")

    def start(self):
        for runner in self.model_runners:
            runner.start()

        if self.belief_runner is not None:
            for runner in self.belief_runner:
                runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner.update_model(agent)
