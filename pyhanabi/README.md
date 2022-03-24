### Training policy with RL

Run command:
```shell
sh scripts/iql.sh
```
This command uses the simplest IQL method to train a policy. This
script will create 80 threads (`--num_thread`) and run 80 games in
each thread (`--num_game_per_thread`).  These two parameters should be
adjusted based on the hardware. `num_thread` should be roughly equal
to the number of hyper-threads/cores in the machine. A higher
`num_game_per_thread` often leads to higher CPU utlization and thus
faster data generation.

After 1 epoch, it should print something like this:
```
EPOCH: 0
Speed: train: 1728.9, buffer_add: 1853.3, buffer_size: 100108
Total Time: 0H 01M 14S, 74s
Total Sample: train: 128K, buffer: 137.214K
@@@Time
        sync and updating : 2 MS, 3.15%
        sample data       : 0 MS, 1.32%
        forward & backward: 60 MS, 81.96%
        update model      : 9 MS, 13.51%
        updating priority : 0 MS, 0.06%
@@@total time per iter: 73.99 ms
[0] Time spent = 74.04 s
0:boltzmann_t  [1000]: avg:   0.0000, min:   0.0000[   0], max:   0.0000[   0]
0:grad_norm    [1000]: avg:   2.1818, min:   0.3475[ 133], max:  10.6211[ 247]
0:loss         [1000]: avg:   1.5138, min:   0.8786[ 136], max:   2.8483[ 917]
0:rl_loss      [1000]: avg:   0.2704, min:   0.0968[ 970], max:   0.6993[  14]
epoch 0, eval score: 0.7300, perfect: 0.00, model saved: True
```
The important numbers to check are the speed of train and buffer
add. The `train/buffer_add` ratio affects how many times each trajectory
is used for training on average. The ideal value for this ratio is 1,
and anything less than 4 works fine in our experience. If the
ratio is too large, i.e. >10, the model may overfit and the final
performance may be significantly worse.


### Train belief model given policy

The following command will train a belief model on the IQL policy we
just trained with `sh scripts/iql.sh`. It assumes that the model
weights are saved at `exps/iql`. Change the `--policy` argument to
train the belief for a different policy.
```shell
sh scripts/belief.sh
```


### Train OBL policies

First we need to train the grounded belief model, i.e. the belief
model when playing with a random policy.

```shell
sh scripts/belief_obl0.sh
```

Notice the additional `--rand 1` flag, which turns up the exploration
epsilon to 1 so that the policy behave like a random policy.

Once the belief for random policy is trained (referred to as
belief-0), we can start training OBL level 1, the optimal grounded
policy. The following command can be used to train OBL level 1
assuming that the belief-0 is trained with `sh scripts/belief_obl0.sh`
and stored at `exps/belief_obl0/model0.pthw`.

```shell
sh scripts/obl1.sh
```

Then we can proceed to training OBL level 2, starting from first
training a belief model on OBL level 1 policy. Note that different
from belief-0, here we set `--load_model 1` so that the belief model
is initialized with previous level belief model to reduce training
time.

```shell
sh scripts/belief_obl1.sh
```

Similarly, when training the policy of OBL level 2, we also initialize
the network with the previous level, OBL level 1.

```shell
sh scripts/obl2.sh
```

Subsequent levels can be trained the same way.


### Evaluation

Run the following command to evaluate a trained model, such as those provided with the repo.
```
python tools/eval_model.py --weight1 ../models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw
```

To compute the cross play score of independely trained models with different seeds:
```shell
python tools/cross_play.py --root ../models/icml_OBL1/ --include BZA0 --num_player 2
```

The final lines of the output are:
```
#model: 1, #groups 5, score: 20.92 +/- 0.08
#model: 2, #groups 10, score: 20.90 +/- 0.04
```
The first line means there are 5 groups with 1 model (i.e. selfplay)
and the second line means that there are 10 (5 choose 2) groups of 2
model (crossplay).

### Test time policy improvement with Sparta or RL Search

To run sparta on one of the provided OBL agents:
```
python sparta.py --weight_file ../models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw
```

To run RL search with exact belief (This simplified command may use different hyperparameters from the orignal paper):
```
python rl_search.py --weight_file ../models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw --num_hint 8
```
