# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class LSTMNet(torch.jit.ScriptModule):
    def __init__(self, device, priv_in_dim, hid_dim, out_dim, num_lstm_layer, dropout):
        super().__init__()
        self.priv_in_dim = priv_in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()
        self.out_layer = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.net(priv_s)

        if hid is not None:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
            new_hid = {"h0": h, "c0": c}
        else:
            o, _ = self.lstm(x)
            new_hid = {}

        logit = self.out_layer(self.dropout(o))
        return logit, new_hid


class PublicLSTMNet(torch.jit.ScriptModule):
    def __init__(
        self,
        device,
        priv_in_dim,
        publ_in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        dropout,
    ):
        super().__init__()
        self.priv_in_dim = priv_in_dim
        self.publ_in_dim = publ_in_dim

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer
        self.dropout = nn.Dropout(dropout)

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.out_layer = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.publ_net(publ_s)

        if hid is not None:
            publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
            new_hid = {"h0": h, "c0": c}
        else:
            publ_o, _ = self.lstm(x)
            new_hid = {}
        priv_o = self.priv_net(priv_s)

        o = priv_o * publ_o
        logit = self.out_layer(self.dropout(o))
        return logit, new_hid


class SupervisedAgent(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(
        self,
        device,
        priv_in_dim,
        publ_in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        net,
        dropout,
    ):
        super().__init__()
        assert net in ["lstm", "publ-lstm"]
        if net == "lstm":
            self.net = LSTMNet(
                device,
                priv_in_dim,
                hid_dim,
                out_dim,
                num_lstm_layer,
                dropout,
            )
        elif net == "publ-lstm":
            self.net = PublicLSTMNet(
                device,
                priv_in_dim,
                publ_in_dim,
                hid_dim,
                out_dim,
                num_lstm_layer,
                dropout,
            )

        self.net_type = net
        self.priv_in_dim = priv_in_dim
        self.publ_in_dim = publ_in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = num_lstm_layer
        self.dropout = dropout
        self.to(device)

    def clone(self, device):
        cloned = SupervisedAgent(
            device,
            self.priv_in_dim,
            self.publ_in_dim,
            self.hid_dim,
            self.out_dim,
            self.num_lstm_layer,
            self.net_type,
            self.dropout,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.net(priv_s, publ_s, hid)

    def greedy_act(
        self,
        priv_s: torch.Tensor,  # [batchsize, dim]
        publ_s: torch.Tensor,  # [batchsize, dim]
        legal_move: torch.Tensor,  # batchsize, dim]
        hid: Dict[str, torch.Tensor],  # [num_layer, batchsize, dim]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """greedy act for 1 timestep"""
        priv_s = priv_s.unsqueeze(0)  # add time dim
        publ_s = publ_s.unsqueeze(0)
        logit, new_hid = self.forward(priv_s, publ_s, hid)
        logit = logit.squeeze(0)  # remove time dim
        assert logit.size() == legal_move.size()
        legal_logit = logit - (1 - legal_move) * 1e6
        action = legal_logit.max(1)[1]
        return action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsize, dim = obs["priv_s"].size()

        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]

        hid = {
            "h0": obs["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": obs["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        logit, new_hid = self.forward(priv_s.unsqueeze(0), publ_s.unsqueeze(0), hid)
        logit = logit.squeeze(0)
        legal_logit = logit - (1 - legal_move) * 1e6
        action = legal_logit.max(1)[1]
        # sample with temp=1
        # action = nn.functional.softmax(legal_logit, 1)
        # action = action.multinomial(1)

        hid_shape = (
            self.num_lstm_layer,
            bsize,
            1,
            self.hid_dim,
        )
        h0 = new_hid["h0"].view(*hid_shape).transpose(0, 1)
        c0 = new_hid["c0"].view(*hid_shape).transpose(0, 1)

        return {
            "a": action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """for use in belief training"""
        return {"priority": torch.ones_like(input_["reward"].sum(1))}

    @torch.jit.script_method
    def get_h0(self, batchsize: int):
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {
            "h0": torch.zeros(*shape),
            "c0": torch.zeros(*shape),
        }
        return hid
