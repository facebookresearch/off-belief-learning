# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from agent import RLAgent, SLAgent

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_root = os.path.join(root, "models")

BotFactory = {
    "Bot-OBL1": lambda : RLAgent(os.path.join(model_root, "icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw"), {}),
    "Bot-OBL2": lambda : RLAgent(os.path.join(model_root, "icml_OBL2/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw"), {}),
    "Bot-OBL3": lambda : RLAgent(os.path.join(model_root, "icml_OBL3/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw"), {}),
    "Bot-OBL4": lambda : RLAgent(os.path.join(model_root, "icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw"), {}),
}
