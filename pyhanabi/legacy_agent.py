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
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple, Dict
import common_utils
import utils


class LegacyNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_ff_layer, skip_connect):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = num_ff_layer
        self.num_lstm_layer = 2
        self.skip_connect = skip_connect

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred = nn.Linear(hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = s.unsqueeze(0)
        x = self.net(s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = x + o
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}


class LegacyAgent(torch.jit.ScriptModule):
    __constants__ = []

    def __init__(self, device, in_dim, hid_dim, out_dim, num_ff_layer, skip_connect):
        super().__init__()
        self.online_net = LegacyNet(
            device, in_dim, hid_dim, out_dim, num_ff_layer, skip_connect
        ).to(device)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.online_net.num_ff_layer,
            self.onlone_net.skip_connect,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid  # , pred_t

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        s = obs["s"]
        legal_move = obs["legal_move"]
        eps = obs["eps"].flatten(0, 1)

        vdn = s.dim() == 3

        if vdn:
            bsize, num_player = obs["s"].size()[:2]
            s = obs["s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
        else:
            bsize, num_player = obs["s"].size()[0], 1

        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": obs["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": obs["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        greedy_action, new_hid = self.greedy_act(s, legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if vdn:
            action = action.view(bsize, num_player)
            greedy_action = greedy_action.view(bsize, num_player)
            rand = rand.view(bsize, num_player)

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.online_net.num_lstm_layer,
            bsize,
            num_player,
            self.online_net.hid_dim,
        )
        h0 = new_hid["h0"].view(*interim_hid_shape).transpose(0, 1)
        c0 = new_hid["c0"].view(*interim_hid_shape).transpose(0, 1)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "rand": rand.detach().cpu(),
            "h0": h0.detach().contiguous().cpu(),
            "c0": c0.detach().contiguous().cpu(),
        }
        return reply

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {"priority": torch.ones_like(input_["reward"].sum(1))}


def load_legacy_agent(weight_file):
    config = utils.get_train_config(weight_file)
    state_dict = torch.load(weight_file, map_location="cuda:0")
    in_dim = 838
    hid_dim = 512
    out_dim = 21
    agent = LegacyAgent(
        "cuda:0",
        in_dim,
        hid_dim,
        out_dim,
        config["num_ff_layer"],
        config["skip_connect"],
    )
    agent.online_net.load_state_dict(state_dict)
    return (
        agent,
        {
            "sad": True,
            "hide_action": False,
            "num_player": 2,
            "train_bomb": 0,
            "max_len": 80,
        },
    )
