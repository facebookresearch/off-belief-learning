# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
def run_game(agents, seed, verbose, op):
    params = {
        "players": str(len(agents)),
        "seed": str(seed),
        "bomb": str(0),
        "random_start_player": str(0),
    }
    game = hanalearn.HanabiEnv(params, -1, False)  # max_len  # verbose
    game.reset()

    hids = [agent.get_h0(1) for agent in agents]
    for h in hids:
        for k, v in h.items():
            if isinstance(agents[0], r2d2.R2D2Agent):
                h[k] = v.cuda().unsqueeze(0)  # add batch dim
            else:
                h[k] = v.cuda()

    if op:
        color_permutes = []
        inv_color_permutes = []
        for _ in range(len(agents)):
            perm = [0, 1, 2, 3, 4]
            random.shuffle(perm)
            inv_perm = np.argsort(perm).tolist()
            for i in range(5):
                assert inv_perm[perm[i]] == i
            color_permutes.append(perm)
            inv_color_permutes.append(inv_perm)

    step = 0
    moves = []
    while not game.terminated():
        # print(game.get_hle_state().to_string())
        actions = []
        new_hids = []

        for i, (agent, hid) in enumerate(zip(agents, hids)):
            # Note: argument here is (game_state, player_idx, hide_action)
            # make sure to specify the correct hide_action value
            if op:
                obs = hanalearn.observe_op(
                    game.get_hle_state(),
                    i,
                    True,
                    color_permutes[i],
                    inv_color_permutes[i],
                    False,
                    True,
                    False,
                )
            else:
                obs = hanalearn.observe(game.get_hle_state(), i, False)

            priv_s = obs["priv_s"].cuda().unsqueeze(0)
            publ_s = obs["publ_s"].cuda().unsqueeze(0)
            legal_move = obs["legal_move"].cuda().unsqueeze(0)

            action, new_hid = agent.greedy_act(priv_s, publ_s, legal_move, hid)
            if i == 0:
                actions.append([action.item()])
            else:
                actions[-1].append(action.item())
            new_hids.append(new_hid)

        hids = new_hids
        cur_player = game.get_current_player()
        move = game.get_move(actions[-1][cur_player])
        if op and move.move_type() == hanalearn.MoveType.RevealColor:
            move.set_color(inv_color_permutes[cur_player][move.color()])
        moves.append(move)

        game.step(move)
        step += 1

    return game.get_score(), moves
