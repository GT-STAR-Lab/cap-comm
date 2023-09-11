#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

# Single-channel (run 3 times, sacred generates new seeds for each run)
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=72 batch_size=32 buffer_size=32 \
env_args.time_limit=50 env_args.key="mpe:MaterialTransport-v0" t_max=10000000

# python3 src/main.py with alg_yaml=qmix env_yaml=gymma agent="gnn" \
# env_args.time_limit=100 env_args.key="mpe:MaterialTransport-v0" t_max=10000000 