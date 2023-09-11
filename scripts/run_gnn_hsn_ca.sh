#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!
echo Enter to continue
read temp


python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=100 env_args.key="mpe:HeterogeneousSensorNetworkCA-v0" seed=9976 t_max=10000000

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=100 env_args.key="mpe:HeterogeneousSensorNetworkCA-v0" seed=6047 t_max=10000000

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=100 env_args.key="mpe:HeterogeneousSensorNetworkCA-v0" seed=4126 t_max=10000000