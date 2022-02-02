# hanabi-live-bot

The code here allows to host bot on a customized version of hanabi.live
so that researchers can play with their AI bots for evaluation/fun!

The [customized platform](https://hanabi.marl-human-ai.com/) has
hacks to allow the bot to reconstruct the full game state using the game
engine from Hanabi-Learning-Environment (HLE), the simulator widely
used for human-AI/multi-agent research on the Hanabi benchmark.
Please kindly *only* use this for human-AI evaluation and use the
original [hanabi.live](https://hanab.live/) to other leisure games.

This is based on an example [reference
bot](https://github.com/Zamiell/hanabi-live-bot) from one of the main
maintainer/contributor of hanabi.live online platform.

### Setup Instructions

* Install the dependencies:
  * `pip install -r requirements.txt`
* Run it (use obl-level4 bot as example):
  * `python main.py --name Bot-OBL4 --login_name NAME --password PASSWORD`
  * Replace the NAME and PASSWORD with any name/password you like.
  * Don't use any password you would use for other website!
* In a browser, log on to https://hanabi.marl-human-ai.com/
  * Again, don't use any password you would use for other website!
* Start a new table.
* In the pre-game chat window, send a private message to the bot in order to get it to join you:
  * `/msg username /join`
* Then, start the game and play!
