#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

# start 3 randomized experiments at once (Sacred saves logfiles individually)
# TODO: buggy, can't end all 3 with CTRL-C
# (can use pkill python3 as hack)
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0" &
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0" &
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0" &

# ignore
# logfile_root_name="scripts/experiment1"
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
# env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0" \
# 2>&1 | tee $logfile_root_name"_run1.log" 
