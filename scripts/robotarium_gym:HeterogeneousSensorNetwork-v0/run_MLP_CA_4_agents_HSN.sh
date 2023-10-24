#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!!!!
echo Enter to continue
read temp

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000 which_seed="seed_1" 

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000 which_seed="seed_2" 

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000  which_seed="seed_3" 
