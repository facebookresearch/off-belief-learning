# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay.py \
       --save_dir exps/iql \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 2000 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \
