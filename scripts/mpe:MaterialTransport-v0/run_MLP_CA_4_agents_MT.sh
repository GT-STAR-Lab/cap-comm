#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

# python3 scripts/set_correct_environment_config.py material_transport config_capability_awareness.yaml

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=64 \
env_args.time_limit=100 env_args.key="mpe:MaterialTransport-v0" t_max=40000000 which_seed="seed_1" 

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=64 \
env_args.time_limit=100 env_args.key="mpe:MaterialTransport-v0" t_max=40000000 which_seed="seed_2" 

python3 src/main.py with alg_yaml=mappo_gnn agent="mlp" env_yaml=gymma batch_size_run=64 \
env_args.time_limit=100 env_args.key="mpe:MaterialTransport-v0" t_max=40000000 which_seed="seed_3" 
