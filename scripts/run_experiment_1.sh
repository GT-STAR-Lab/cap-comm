#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0"
