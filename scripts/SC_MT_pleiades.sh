#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!
echo Enter to continue
read temp

# Single-channel (run 3 times, sacred generates new seeds for each run)
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=24 batch_size=32 buffer_size=32 \
env_args.time_limit=50 env_args.key="mpe:MaterialTransport-v0" t_max=10000000 which_seed="seed_1" &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=24 batch_size=32 buffer_size=32 \
env_args.time_limit=50 env_args.key="mpe:MaterialTransport-v0" t_max=10000000 which_seed="seed_2" &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=24 batch_size=32 buffer_size=32 \
env_args.time_limit=50 env_args.key="mpe:MaterialTransport-v0" t_max=10000000  which_seed="seed_3" 