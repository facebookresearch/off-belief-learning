# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Main DQN agent."""
import os
import sys
from typing import Dict, Tuple
import pprint
import argparse

import torch
import torch.nn as nn

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import utils


class LSTMNet(torch.jit.ScriptModule):
    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        hide_action,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = num_lstm_layer
        self.hide_action = hide_action
        assert not hide_action

        self.priv_in_dim = in_dim - 25 * hand_size
        self.publ_in_dim = in_dim - 2 * 25 * hand_size

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h0 = obs["h0"].transpose(0, 1).contiguous()
        c0 = obs["c0"].transpose(0, 1).contiguous()

        s = obs["s"].unsqueeze(0)
        assert s.size(2) == self.in_dim
        # get private/public feature
        priv_s = s[:, :, -self.priv_in_dim :]
        publ_s = s[:, :, -self.publ_in_dim :]

        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (h0, c0))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o

        a = self.fc_a(o).squeeze(0)
        # v = self.fc_v(o).squeeze(0)

        # legal_move = obs["legal_move"]
        # legal_a = legal_move * a
        # q = v + legal_a - legal_a.mean(1, keepdim=True)
        # assert legal_move.size() == a.size()

        return {
            "a": a,
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }


class ARBeliefNet(torch.jit.ScriptModule):
    __constants__ = ["in_dim", "hid_dim", "hand_size"]

    def __init__(self, device, in_dim, hid_dim, hand_size, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.hand_size = hand_size
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.emb = nn.Linear(25, self.hid_dim // 8, bias=False)
        self.auto_regress = nn.LSTM(
            self.hid_dim + self.hid_dim // 8,
            self.hid_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self.auto_regress.flatten_parameters()

        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print(">>>>> called: with", obs["num_samples"])
        # print(">>>s: ", batch.obs["s"].size())
        # print(">>>t: ", batch.obs["own_hand"].size())
        h0 = obs["h0"].transpose(0, 1).contiguous()
        c0 = obs["c0"].transpose(0, 1).contiguous()

        s = obs["s"].unsqueeze(0)
        # assert s.size(2) == 838
        # remove last_action feature
        # s[:, :, 378:433] = 0
        # remove redundant input & sad feature
        s = s[:, :, self.hand_size * 25 :]
        # s = s[:, :, :self.in_dim]
        # print('>>>>>', s.size())

        x = self.net(s)
        o, (h, c) = self.lstm(x, (h0, c0))

        if obs["num_samples"].sum() == 0:
            print("<<<forward>>> belief model, bsize: ", obs["num_samples"].size(0))
            # forward the model, do not sample
            return {
                "h0": h.transpose(0, 1).contiguous(),
                "c0": c.transpose(0, 1).contiguous(),
            }

        print("<<<sample>>> belief model, bsize: ", obs["num_samples"].size(0))
        num_samples = int(obs["num_samples"].item())
        # o: [seq_len(1), batch, dim]
        seq, bsize, hid_dim = o.size()
        assert seq == 1, "seqlen should be 1"
        assert bsize == 1, "batchsize for BeliefModel.sample should be 1"
        o = o.view(bsize, hid_dim)
        o = o.unsqueeze(1).expand(bsize, num_samples, hid_dim)

        in_t = torch.zeros(bsize, num_samples, hid_dim // 8, device=o.device)
        shape = (1, bsize * num_samples, self.hid_dim)
        ar_hid = (
            torch.zeros(*shape, device=o.device),
            torch.zeros(*shape, device=o.device),
        )
        sample_list = []
        for i in range(self.hand_size):
            ar_in = torch.cat([in_t, o], 2).view(bsize * num_samples, 1, -1)
            # print(ar_in.sum(2).squeeze(0))
            # print('...')
            # print('ar_in:', ar_in[:, :10])
            ar_out, ar_hid = self.auto_regress(ar_in, ar_hid)
            logit = self.fc(ar_out.squeeze(1))
            # print(">>>>>logit:", logit.size())
            prob = nn.functional.softmax(logit, 1)
            # print(prob[0])
            sample_t = prob.multinomial(1)
            # print(prob[:, :5])
            # print('...')
            # sample_t = prob.multinomial(1)
            # print(sample_t[0].item(), prob[0][sample_t[0]].item())
            # print('===============')
            # print(sample_t[1].item(), prob[1][sample_t[1]].item())
            sample_t = sample_t.view(bsize, num_samples)
            onehot_sample_t = torch.zeros(
                bsize, num_samples, 25, device=sample_t.device
            )
            onehot_sample_t.scatter_(2, sample_t.unsqueeze(2), 1)
            in_t = self.emb(onehot_sample_t)
            sample_list.append(sample_t)

        sample = torch.stack(sample_list, 2)

        return {
            "sample": sample,
            # "h0": h.transpose(0, 1).contiguous(),
            # "c0": c.transpose(0, 1).contiguous(),
        }


class V0BeliefNet(torch.jit.ScriptModule):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.hand_size = 5
        self.bit_per_card = 25

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        """dummy function"""
        shape = (1, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    # @torch.jit.script_method
    # def forward(self, obs : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     # print("observe")
    #     return obs

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if obs["num_samples"].sum() == 0:
            print("<<<forward>>> belief model, bsize: ", obs["num_samples"].size(0))
            # forward the model, do not sample
            return obs

        print("<<<sample>>> belief model, bsize: ", obs["num_samples"].size(0))
        num_samples = int(obs["num_samples"].item())
        # print("sample")
        # for k, v in obs.items():
        #     print(k, v.size())

        v0 = obs["s"][:, -350:-175]
        # v0 = obs["v0"]
        bsize = v0.size(0)
        v0 = v0.view(bsize, self.hand_size, -1)[:, :, : self.bit_per_card]

        v0 = v0.view(-1, self.bit_per_card).clamp(min=1e-5)
        sample = v0.multinomial(num_samples, replacement=True)
        # smaple: [bsize * handsize, num_sample]
        sample = sample.view(bsize, self.hand_size, num_samples)
        sample = sample.transpose(1, 2)
        return {"sample": sample}


## main program ##
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--clone_model", type=str, default=None)
parser.add_argument("--belief_model", type=str, default=None)
parser.add_argument("--v0_belief_model", type=int, default=0)
args = parser.parse_args()

if args.model is not None:
    device = "cuda"
    agent, cfg = utils.load_agent(args.model, {"device": device})
    print("after loading model")
    search_model = LSTMNet(
        device,
        agent.online_net.in_dim[0],
        agent.online_net.hid_dim,
        agent.online_net.out_dim,
        agent.online_net.num_lstm_layer,
        5,
        cfg["hide_action"],
    )
    utils.load_weight(
        search_model, None, device, state_dict=agent.online_net.state_dict()
    )
    save_path = args.model.rsplit(".", 1)[0] + ".sparta"
    print("saving model to:", save_path)
    torch.jit.save(search_model, save_path)

if args.belief_model is not None:
    state_dict = torch.load(args.belief_model)
    hid_dim, in_dim = state_dict["net.0.weight"].size()
    out_dim = state_dict["fc.weight"].size(0)
    hand_size = 5

    device = "cuda"
    model = ARBeliefNet(device, in_dim, hid_dim, hand_size, out_dim)
    model.load_state_dict(state_dict)
    save_path = args.belief_model.rsplit(".", 1)[0] + ".sparta"
    print("saving model to:", save_path)
    torch.jit.save(model, save_path)

if args.clone_model is not None:
    device = "cuda"
    agent = utils.load_supervised_agent(args.clone_model, device)
    search_model = LSTMNet(
        device,
        agent.priv_in_dim + 125,
        agent.hid_dim,
        agent.out_dim,
        agent.num_lstm_layer,
        5,  # cfg["hand_size"],
        False,  # cfg["hide_action"]
    )
    # utils.load_weight(search_model, None, device, state_dict=agent.online_net.state_dict())
    search_model.priv_net.load_state_dict(agent.priv_net.state_dict())
    search_model.publ_net.load_state_dict(agent.publ_net.state_dict())
    search_model.lstm.load_state_dict(agent.lstm.state_dict())
    search_model.fc_a.load_state_dict(agent.out_layer.state_dict())
    # save_path = args.clone_model.rsplit(".", 1)[0] + ".sparta"
    save_path = "/private/home/hengyuan/HanabiModels/clone_bot.sparta"
    print("saving model to:", save_path)
    torch.jit.save(search_model, save_path)


if args.v0_belief_model:
    device = "cuda"
    model = V0BeliefNet(device)
    save_path = "/private/home/hengyuan/HanabiModels/v0_belief.sparta"
    print("saving model to:", save_path)
    torch.jit.save(model, save_path)
