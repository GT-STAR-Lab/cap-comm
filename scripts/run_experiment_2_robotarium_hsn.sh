#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!
echo Enter to continue
read temp

# Single-channel GNN
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=16 buffer_size=16 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976 &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=16 buffer_size=16 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047 &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=16 buffer_size=16 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126 &

# Single-channel GAT
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126

# Dual-channel GNN
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126

# Dual-channel GAT
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126

# # Capability-awareness no communication
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126
