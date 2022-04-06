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
import torch._C

cmake_cxx_flags = []
for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
    val = getattr(torch._C, f"_PYBIND11_{name}")
    # print(val, getattr(torch._C, f"_PYBIND11_{name}"))
    if val is not None:
        # print([f'-DPYBIND11_{name}=\\"{val}\\"'])
        cmake_cxx_flags += [fr'-DPYBIND11_{name}=\"{val}\"']
print(" ".join(cmake_cxx_flags), end="")
