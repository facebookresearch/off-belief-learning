### Evaluate a trained RL model
```bash
# evaluate pretrained monster_bot, or color_bot
python tools/eval_model.py --weight1 monster_bot

# evaluate other pretrained
python tools/eval_model.py --weight1 absolute_path_to_bot

# evaluate 2 bots (cross play)
python tools/eval_model.py --weight1 absolute_path_to_bot1 --weight2 absolute_path_to_bot2
```

Use optinal`--num_game X` and `--num_run Y` to specify total num of games `Y * X`

### Evaluate clone bot & against clone bot
#### using the default clone-bot in the model zoo
```bash
# selfplay
python tools/eval_clone_bot.py --bomb 0

# crossplay
python tools/eval_clone_bot.py --other_bot /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_2/heir_8/model_epoch1540.pthw --bomb 0

# crossplay with monster_bot/color_bot
python tools/eval_clone_bot.py --other_bot monster_bot/color_bot
```

#### using other clone-bots
```bash
# selfplay
python tools/eval_clone_bot.py --bomb 0 --clone_bot /checkpoint/lep/hanabi/supervised/supervised_50split/LR0.001_TR_SPLITtrain1/checkpoint-20-20.578.p

# crossplay
python tools/eval_clone_bot.py --clone_bot /checkpoint/lep/hanabi/supervised/supervised_50split/LR0.001_TR_SPLITtrain1/checkpoint-20-20.578.p --other_bot /checkpoint/lep/hanabi/supervised/br_2p/train2/RNN_HID_DIM768_BZA_OTHER0_SEEDa/model0.pthw --bomb 0

# crossplay with monster_bot/color_bot
python tools/eval_clone_bot.py --other_bot monster_bot/color_bot
```

### Evaluate CH models

```bash
# evaluate every level in CH with selfplay
python tools/eval_ch.py --root /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_1

# evaluate every level with monster bot/color_bot
python tools/eval_ch.py --root /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_1 --other_bot monster_bot/color_bot

# cross play between same level from 2 runs
python tools/eval_ch.py --root /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_1 --xp_root /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_2

# evaluate against clone bot 
TODO
```

### Render Condition Action Matrix
```bash
# render action matrix for a specific model
python tools/action_matrix.py --model /private/home/hengyuan/HanabiModels/rl2_lstm1024/HIDE_ACTION1_RNN_HID_DIM1024_LSTM_LAYER2_SEEDc/model4.pthw --output exps/monster_am.png

# to look at an image in iterm (installed by clicking scroll down menu of "iTerm2", select "Install Shell Integration"
imgcat exps/monster_am.png

# render action matrix for each level in CH
python tools/action_matrix_ch.py --root /private/home/bcui/OneHanabi/rl/heirarchy_br/10_agents_boltzmann_random_2 --output exps/ch_run2_action_matrix
```
