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

import glob
import torch
import os
import time


class ModelCkpt:
    def __init__(self, prefix, models_to_keep=10, model_tag="newest"):
        self.prefix = prefix
        if os.path.isdir(prefix):
            self.prefix = os.path.join(self.prefix, model_tag)
        self.models_to_keep = models_to_keep

        self._last_loaded_model_meta = {}

    def get_all_versions(self):
        versions = []
        for path in glob.glob(f"{self.prefix}_*"):
            if path.endswith(".tmp"):
                continue
            try:
                idx = int(path.split("_")[-1])
            except ValueError:
                print(f"Bad file: {path}")
                continue
            versions.append((idx, path))
        return sorted(versions)

    def save(self, obj):
        versions = self.get_all_versions()
        if versions:
            new_id = versions[-1][0] + 1
        else:
            new_id = 0
        path = f"{self.prefix}_{new_id:08d}"
        torch.save(obj, path + ".tmp")
        os.rename(path + ".tmp", path)
        models_to_delete = (len(versions) + 1) - self.models_to_keep
        if models_to_delete > 0:
            for _, path in versions[:models_to_delete]:
                os.remove(path)

    def get_last_version(self):
        # Small hack that deals with fixed models
        if ".pthw" in self.prefix:
            return self.prefix

        while True:
            versions = self.get_all_versions()
            if not versions:
                print("Waiting for checkpoints to appear", self.prefix)
                time.sleep(5)
                continue
            return versions[-1][1]

    def maybe_load_state_dict(self, last_version):
        version, path = self.get_last_version()
