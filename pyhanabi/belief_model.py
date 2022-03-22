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
import torch
from torch import nn
from typing import Tuple, Dict
import numpy as np


class V0BeliefModel(torch.jit.ScriptModule):
    def __init__(self, device, num_sample):
        super().__init__()
        self.device = device
        self.hand_size = 5
        self.bit_per_card = 25
        self.num_sample = num_sample

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        """dummy function"""
        shape = (1, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def observe(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("observe")
        return obs

    @torch.jit.script_method
    def sample(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("sample")
        v0 = obs["v0"]
        bsize = v0.size(0)
        v0 = v0.view(bsize, self.hand_size, -1)[:, :, : self.bit_per_card]

        v0 = v0.view(-1, self.bit_per_card).clamp(min=1e-5)
        sample = v0.multinomial(self.num_sample, replacement=True)
        # smaple: [bsize * handsize, num_sample]
        sample = sample.view(bsize, self.hand_size, self.num_sample)
        sample = sample.transpose(1, 2)
        return {"sample": sample, "h0": obs["h0"], "c0": obs["c0"]}


def pred_loss(logp, gtruth, seq_len):
    """
    logit: [seq_len, batch, hand_size, bits_per_card]
    gtruth: [seq_len, batch, hand_size, bits_per_card]
        one-hot, can be all zero if no card for that position
    """
    assert logp.size() == gtruth.size()
    logp = (logp * gtruth).sum(3)
    hand_size = gtruth.sum(3).sum(2).clamp(min=1e-5)
    logp_per_card = logp.sum(2) / hand_size
    xent = -logp_per_card.sum(0)
    # print(seq_len.size(), xent.size())
    assert seq_len.size() == xent.size()
    avg_xent = xent / seq_len
    nll_per_card = -logp_per_card
    return xent, avg_xent, nll_per_card


class ARBeliefModel(torch.jit.ScriptModule):
    def __init__(
        self, device, in_dim, hid_dim, hand_size, out_dim, num_sample, fc_only
    ):
        """
        mode: priv: private belief prediction
              publ: public/common belief prediction
        """
        super().__init__()
        self.device = device
        self.input_key = "priv_s"
        self.ar_input_key = "own_hand_ar_in"
        self.ar_target_key = "own_hand"

        self.in_dim = in_dim
        self.hand_size = hand_size
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.num_sample = num_sample
        self.fc_only = fc_only

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
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @classmethod
    def load(cls, weight_file, device, hand_size, num_sample, fc_only):
        state_dict = torch.load(weight_file)
        hid_dim, in_dim = state_dict["net.0.weight"].size()
        out_dim = state_dict["fc.weight"].size(0)
        model = cls(device, in_dim, hid_dim, hand_size, out_dim, num_sample, fc_only)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model

    @torch.jit.script_method
    def forward(self, x, ar_card_in):
        # x = batch.obs[self.input_key]
        x = self.net(x)
        if self.fc_only:
            o = x
        else:
            o, (h, c) = self.lstm(x)

        # ar_card_in  = batch.obs[self.ar_input_key]
        seq, bsize, _ = ar_card_in.size()
        ar_card_in = ar_card_in.view(seq * bsize, self.hand_size, 25)

        ar_emb_in = self.emb(ar_card_in)
        # ar_card_in: [seq * batch, 5, 64]
        # o: [seq, batch, 512]
        o = o.view(seq * bsize, self.hid_dim)
        o = o.unsqueeze(1).expand(seq * bsize, self.hand_size, self.hid_dim)
        ar_in = torch.cat([ar_emb_in, o], 2)
        ar_out, _ = self.auto_regress(ar_in)

        logit = self.fc(ar_out)
        logit = logit.view(seq, bsize, self.hand_size, -1)
        return logit

    def loss(self, batch, beta=1):
        logit = self.forward(batch.obs[self.input_key], batch.obs[self.ar_input_key])
        logit = logit * beta
        logp = nn.functional.log_softmax(logit, 3)
        gtruth = batch.obs[self.ar_target_key]
        gtruth = gtruth.view(logp.size())
        seq_len = batch.seq_len
        xent, avg_xent, nll_per_card = pred_loss(logp, gtruth, seq_len)

        # v0: [seq, batch, hand_size, bit_per_card]
        v0 = batch.obs["priv_ar_v0"]
        v0 = v0.view(v0.size(0), v0.size(1), self.hand_size, 35)[:, :, :, :25]
        logv0 = v0.clamp(min=1e-6).log()
        _, avg_xent_v0, _ = pred_loss(logv0, gtruth, seq_len)
        return xent, avg_xent, avg_xent_v0, nll_per_card

    @torch.jit.script_method
    def observe(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsize, num_lstm_layer, num_player, dim = obs["h0"].size()
        h0 = obs["h0"].transpose(0, 1).flatten(1, 2).contiguous()
        c0 = obs["c0"].transpose(0, 1).flatten(1, 2).contiguous()

        s = obs[self.input_key].unsqueeze(0)
        x = self.net(s)
        if self.fc_only:
            o, (h, c) = x, (h0, c0)
        else:
            o, (h, c) = self.lstm(x, (h0, c0))

        h = h.view(num_lstm_layer, bsize, num_player, dim)
        c = c.view(num_lstm_layer, bsize, num_player, dim)
        return {
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }

    @torch.jit.script_method
    def sample(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bsize, num_lstm_layer, num_player, dim = obs["h0"].size()
        h0 = obs["h0"].transpose(0, 1).flatten(1, 2).contiguous()
        c0 = obs["c0"].transpose(0, 1).flatten(1, 2).contiguous()

        s = obs[self.input_key].unsqueeze(0)
        x = self.net(s)
        if self.fc_only:
            o, (h, c) = x, (h0, c0)
        else:
            o, (h, c) = self.lstm(x, (h0, c0))
        # o: [seq_len(1), batch, dim]
        seq, bsize, hid_dim = o.size()

        assert seq == 1, "seqlen should be 1"
        # assert bsize == 1, "batchsize for BeliefModel.sample should be 1"
        o = o.view(bsize, hid_dim)
        o = o.unsqueeze(1).expand(bsize, self.num_sample, hid_dim)

        in_t = torch.zeros(bsize, self.num_sample, hid_dim // 8, device=o.device)
        shape = (1, bsize * self.num_sample, self.hid_dim)
        ar_hid = (
            torch.zeros(*shape, device=o.device),
            torch.zeros(*shape, device=o.device),
        )
        sample_list = []
        for i in range(self.hand_size):
            ar_in = torch.cat([in_t, o], 2).view(bsize * self.num_sample, 1, -1)
            ar_out, ar_hid = self.auto_regress(ar_in, ar_hid)
            logit = self.fc(ar_out.squeeze(1))
            prob = nn.functional.softmax(logit, 1)
            sample_t = prob.multinomial(1)
            sample_t = sample_t.view(bsize, self.num_sample)
            onehot_sample_t = torch.zeros(
                bsize, self.num_sample, 25, device=sample_t.device
            )
            onehot_sample_t.scatter_(2, sample_t.unsqueeze(2), 1)
            in_t = self.emb(onehot_sample_t)
            sample_list.append(sample_t)

        sample = torch.stack(sample_list, 2)

        h = h.view(num_lstm_layer, bsize, num_player, dim)
        c = c.view(num_lstm_layer, bsize, num_player, dim)
        return {
            "sample": sample,
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }
