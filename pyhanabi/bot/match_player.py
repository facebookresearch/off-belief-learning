# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import requests
import time


# Authenticate, login to the Hanabi Live WebSocket server, and run forever
def main(player1, player2):
    protocol = 'http'
    ws_protocol = 'ws'
    host = 'localhost:3999'

    path = f'/pairTest?player1={player1}&player2={player2}'
    ws_path = '/ws'
    url = protocol + '://' + host + path
    ws_url = ws_protocol + '://' + host + ws_path
    resp = requests.post(url)

    print(resp)
    print(resp.url)
    print(resp.text)

    # Handle failed authentication and other errors
    if resp.status_code != 200:
        print('command failed:')
        print(resp.text)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--player1", type=str, default=None, required=True)
    parser.add_argument("--player2", type=str, default=None, required=True)
    parser.add_argument("--num_game", type=int, default=1)
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()

    for i in range(args.num_game):
        main(args.player1, args.player2)
