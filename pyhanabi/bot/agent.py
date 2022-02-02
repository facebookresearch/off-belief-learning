import abc
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'pyhanabi'))
from set_path import append_sys_path
append_sys_path()

import torch
import rela
import hanalearn

import r2d2
from belief_model import ARBeliefModel
from utils import load_agent, load_supervised_agent
from game_state import HleGameState


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_h0(self):
        pass

    @abc.abstractmethod
    def observe_and_maybe_act(self, state: HleGameState, hid):
        pass


class RLAgent(Agent):
    def __init__(self, model_path, override):
        self.model_path = model_path
        default_override = {"device": "cpu", "vdn": False}
        default_override.update(override)
        self.agent, self.cfgs = load_agent(model_path, default_override)
        self.agent.train(False)

    def get_h0(self):
        h0 = self.agent.get_h0(1)
        for k, v in h0.items():
            h0[k] = v.unsqueeze(0)
        return h0

    def observe_and_maybe_act(self, state: HleGameState, hid):
        priv_s, publ_s, legal_move = state.observe()
        adv, new_hid = self.agent.online_net.act(
            priv_s, publ_s, hid,
        )
        move = None
        if state.is_my_turn():
            # assert self.next_moves[table_id] is None
            legal_adv = (1 + adv - adv.min()) * legal_move
            action = legal_adv.argmax(1).detach()
            action_uid = int(action.item())
            move = state.hle_game.get_move(action_uid)

            # legal_adv = adv - (1 - legal_move) * 1e9
            # prob = torch.nn.functional.softmax(legal_adv * 5, 1)
            # logp = torch.nn.functional.log_softmax(legal_adv * 5, 1)
            # xent = -(prob * logp).sum().item()
        return move, new_hid


class SLAgent(Agent):
    def __init__(self, model_path):
        self.model_path = model_path
        self.agent = load_supervised_agent(model_path, "cpu")
        self.agent.train(False)

    def get_h0(self):
        h0 = self.agent.get_h0(1)
        return h0

    def observe_and_maybe_act(self, state: HleGameState, hid):
        priv_s, publ_s, legal_move = state.observe()
        logit, new_hid = self.agent.forward(
            priv_s.unsqueeze(0), publ_s.unsqueeze(0), hid,
        )
        move = None
        if state.is_my_turn():
            logit = logit.squeeze()
            action = (logit - (1 - legal_move) * 1e9).argmax(0)
            action_uid = int(action.item())
            move = state.hle_game.get_move(action_uid)
        return move, new_hid


class QreSpartaAgent(Agent):
    def __init__(self, bp_path, belief_path, search_per_step, qre_lambda, seed):
        self.bp_path = bp_path
        self.belief_path = belief_path
        self.search_per_step = search_per_step
        self.qre_lambda = qre_lambda
        self.seed = seed

        device = "cuda:0"
        self.bp_model = load_supervised_agent(bp_path, device)
        self.belief_model = ARBeliefModel.load(
            belief_path,
            device,
            5,
            2 * search_per_step // 10,
            False, # fc_only, now deprecated
        )
        self.bp_model.train(False)
        self.belief_model.train(False)
        self.bp_runner = rela.BatchRunner(self.bp_model, device, 2000, ["act"])
        self.belief_runner = rela.BatchRunner(self.belief_model, device, 2000, ["observe", "sample"])
        self.bp_runner.start()
        self.belief_runner.start()

        # defer the creation to later as we need to know the player idx
        self.sparta_agent = None
        print("======creation done======")

    def get_h0(self):
        self.sparta_agent = None
        return None #  the h0 for sparta agent is tracked by the c++ object

    def observe_and_maybe_act(self, state: HleGameState, hid):
        if self.sparta_agent is None:
            partners = [
                hanalearn.SpartaActor(idx, self.bp_runner, self.seed)
                for idx in range(state.num_player)
            ]
            self.sparta_agent = partners[state.my_index]
            self.sparta_agent.set_belief_runner(self.belief_runner)
            self.sparta_agent.set_partners(partners)
            print("sparta agent created")

        game_sim = hanalearn.GameSimulator(state.hle_game, state.hle_state)
        self.sparta_agent.observe(game_sim)
        # decide action has to be called to make it work
        bp_move = self.sparta_agent.decide_action(game_sim)

        move = None
        if state.is_my_turn():
            bp_move = game_sim.get_move(bp_move)
            move, _, _, _ = self.sparta_agent.sparta_search(
                game_sim, bp_move, self.search_per_step, 0.05, self.qre_lambda
            )

        total_step, fail_step = self.sparta_agent.get_stats()
        print(f">>>sparta stats {fail_step}/{total_step} belief failure")
        return move, hid
