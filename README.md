# Off-Belief Learning

## Introduction

This repo contains the implementation of the algorithm proposed
in [Off-Belief Learning, ICML 2021] (https://arxiv.org/pdf/2103.04000.pdf).

## Environment Setup

We have been using `pytorch-1.5.1`, `cuda-10.1`, and `cudnn-v7.6.5` in
our development environment. Other settings may also work but we have
not tested it extensively under different configurations. We also use
conda/miniconda to manage environments.

There are known issues when using this repo with newer versions
of pytorch, such as this [illegal move issue] (https://github.com/facebookresearch/hanabi_SAD/issues/20#issue-848431066).

```shell
conda create -n hanabi python=3.7
conda activate hanabi

# install pytorch 1.5.1
# note that newer versions may cause compilation issues
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install other dependencies
pip install psutil

# install a newer cmake if the current version is < 3.15
conda install -c conda-forge cmake
```

To help cmake find the proper libraries (e.g. libtorch), please either
add the following lines to your `.bashrc`, or add it to a separate file
and `source` it before you start working on the project.

```shell
# activate the conda environment
conda activate hanabi

# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

Finally, to compile this repo:

```shell
# under project root
mkdir build
cd build
cmake ..
make -j10
```

## Code Structure

For an overview of how the training infrastructure, please refer to Figure 5 of the
[Off-Belief Learning] (https://arxiv.org/pdf/2103.04000.pdf) paper.

`hanabi-learning-environment` is a modified version of the original
[HLE from Deepmind] (https://github.com/deepmind/hanabi-learning-environment).

Notable modifications includes:

1) Card knowledge part of the observation encoding is changed to
v0-belief, i.e.  card knowledge normalized by the remaining public
card count.

2) Functions to reset the game state with sampled hands.

`rela` (REinforcement Learning Assemly) is a set of tools for
efficient batched neural network inference written in C++ with
multi-threading.

`rlcc` implements the core of various algorithms. For example, the
logic of fictitious transitions are implemented in `r2d2_actor.cc`.
It also contains implementations of baselines such as other-play, VDN
and IQL.

`pyhanabi` is the main entry point of the repo. It contains implementations for
Q-network, recurrent DQN training, belief network and training, as well as some tools
to analyze trained models.

## Run the code

Please refer to the README in pyhanabi for detailed instruction on how to train a model.

## Copyright
Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

This source code is licensed under the license found in the LICENSE
file in the root directory of this source tree.
