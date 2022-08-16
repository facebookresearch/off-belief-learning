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

# launcher for sync obl training
import argparse
import os
import sys
import numpy as np
import subprocess
import signal
import time
import logging

srun_base = """\
mkdir -p {SAVE_DIR}
sbatch --job-name {JOB_NAME} \\
       --mem 350G \\
       --partition learnlab \\
       -C volta16gb \\
       --gres gpu:{N_GPU} \\
       --nodes 1 \\
       --ntasks-per-node 1 \\
       --cpus-per-task 45 \\
       --output {SAVE_DIR}/std.out \\
       --error {SAVE_DIR}/std.err \\
       --time 2160 \\
       --wrap "
#!/bin/bash
"""

default_policy_config = {
    "save_dir": None,
    "coop_agents": None,
    "coop_sync_freq": 50,
    "seed": None,
    # most of the following parameters should not be changed
    "num_thread": 80,
    "num_game_per_thread": 80,
    "method": "iql",
    "sad": 0,
    "act_base_eps": 0.1,
    "act_eps_alpha": 7,
    "lr": 6.25e-5,
    "eps": 1.5e-5,
    "grad_clip": 5,
    "gamma": 0.999,
    "batchsize": 128,
    "burn_in_frames": 10000,
    "replay_buffer_size": 100000,
    "epoch_len": 1000,
    "num_epoch": 3000,
    "priority_exponent": 0.9,
    "priority_weight": 0.6,
    "train_bomb": 0,
    "eval_bomb": 0,
    "num_player": 2,
    "rnn_hid_dim": 512,
    "multi_step": 3,
    "train_device": "cuda:0",
    "act_device": "cuda:1",
    "shuffle_color": 0,
    "aux_weight": 0.25,
    "num_lstm_layer": 2,
    "hide_action": 0,
    "boltzmann_act": 0,
    "min_t": 0.01,
    "max_t": 0.1,
    "num_fict_sample": 10,
    "load_model": "None",
    "net": "publ-lstm",
    "mode": "klr",
}


def get_savedir_for_levels(save_dir, num_level):
    policy_dirs = []
    for i in range(num_level):
        policy_dirs.append(os.path.join(save_dir, f"klr_{i+1}"))
    return policy_dirs


def config_to_flags(config):
    flags = []
    for k, v in config.items():
        flags.append(f"--{k}")
        flags.append(v)
    return flags


def write_command_to_file(filename, command):
    first_line = " ".join(command[:3])
    first_line += " \\"
    command = command[3:]
    lines = [first_line]
    assert len(command) % 2 == 0
    for i in range(0, len(command), 2):
        print(i, command[i])
        if command[i] == "--coop_agents" and (
            command[i + 1] is None or len(command[i + 1]) == 0
        ):
            continue
        line = f"\t{command[i]} {command[i+1]} \\"
        lines.append(line)

    content = "\n".join(lines)
    content += "\n"
    print(filename)
    print(content)
    with open(filename, "w") as f:
        f.write(content)

    with open(filename + "_dev", "w") as f:
        f.write(f"echo '{filename}'\n")


parser = argparse.ArgumentParser(description="train klr on hanabi")
parser.add_argument("--local", type=int, default=1)
parser.add_argument("--gpu8", type=int, default=0)
parser.add_argument("--dry", type=int, default=1)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--seed", type=int, required=1)
parser.add_argument("--num_level", type=int, default=4)
parser.add_argument("--partition", type=str, default="learnlab", required=False)
parser.add_argument("--with_br", type=int, default=0)
parser.add_argument("--method", type=str, default="iql")

args = parser.parse_args()
assert args.method in ["vdn", "iql", "ppo"]
default_policy_config["method"] = args.method
args.save_dir = os.path.abspath(args.save_dir)

pyhanabi_path = os.path.dirname(os.path.abspath(__file__))
obl_main = os.path.join(pyhanabi_path, "best_response.py")

obl_command_base = ["python", "-u", obl_main]
policy_dirs = get_savedir_for_levels(args.save_dir, args.num_level)

# create folders
for d in policy_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

max_num_device = 2

for i in range(args.num_level):
    num_device = 1
    level = i + 1
    config = default_policy_config.copy()
    job_name = f"obl_policy{level}"
    config["seed"] = args.seed
    config["save_dir"] = policy_dirs[i]

    # cheaper config for local
    if args.local:
        config["num_thread"] = 2
        config["num_game_per_thread"] = 5
        config["batchsize"] = 8
        config["burn_in_frames"] = 50
        config["epoch_len"] = 20
        config["coop_sync_freq"] = 10

    coop_agents = []
    if i > 0:
        coop_agents = [policy_dirs[i - 1]]
        config["coop_agents"] = " ".join(coop_agents)

    if (not args.local or args.gpu8) and (coop_agents):
        coop_devices = ["cuda:1"]
        #        coop_devices += [f"cuda:{2+i}" for i in range(len(coop_agents))]
        config["act_device"] = ",".join(coop_devices)
        num_device += len(coop_devices)

    max_num_device = max(max_num_device, num_device)

    # dump to file a a human readable way for individual debugging
    policy_command = obl_command_base + config_to_flags(config)
    write_command_to_file(os.path.join(policy_dirs[i], "train.sh"), policy_command)

task_folders = policy_dirs

if args.local:
    # launch on local machine for dev
    processes = []
    for task_dir in task_folders:
        print("launching: ", task_dir)
        script = os.path.join(task_dir, "train.sh")
        stdout = os.path.join(task_dir, "stdout.txt")
        p = subprocess.Popen(
            f"sh {script}",
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=open(stdout, "w"),
            preexec_fn=os.setsid,
        )
        processes.append(p)

    while True:
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("Stopping.")
            break

    for p in processes:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

else:
    import submitit

    print("Total jobs:", len(policy_dirs))
    [print(p.split("/")[-1]) for p in policy_dirs]
    print("max num gpu:", max_num_device)
    assert max_num_device <= 8

    def job_func(task_folders):
        job_env = submitit.JobEnvironment()
        job_id = job_env.global_rank
        task_script = os.path.join(task_folders[job_id], "train.sh")
        subprocess.run(["sh", task_script])

    if not args.dry:
        executor = submitit.AutoExecutor(folder=args.save_dir)
        executor.update_parameters(
            tasks_per_node=1,
            nodes=len(task_folders),
            timeout_min=4320,
            slurm_partition=args.partition,
            mem_gb=400,
            cpus_per_task=min(40, max_num_device * 10),
            slurm_gpus_per_task=max_num_device,
            slurm_constraint="volta16gb",
        )
        job = executor.submit(job_func, task_folders)
