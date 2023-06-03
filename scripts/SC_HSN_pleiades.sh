#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!
echo Enter to continue
read temp

# Single-channel (run 3 times, sacred generates new seeds for each run)
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000 which_seed="seed_1" &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000 which_seed="seed_2" &

python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=16 batch_size=32 buffer_size=32 \
env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000  which_seed="seed_3" 

# ############################################################################################################################333
# # Dual-channel GNN
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000 which_seed="seed_1" &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000  which_seed="seed_2" &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma agent="dual_channel_gnn" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000 which_seed="seed_3" 

# # Capability-awareness no communication
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976 t_max=5000000 &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047 t_max=5000000 &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma num_layers=0 batch_size_run=16 batch_size=16 buffer_size=16 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126 t_max=5000000


# Single-channel GAT
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000 which_seed="seed_1" &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000 which_seed="seed_2" &

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="gat" batch_size_run=16 batch_size=32 buffer_size=32 \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" capability_shape=1 t_max=20500000 which_seed="seed_3"

# Dual-channel GAT
# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=9976

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=6047

# python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma n_heads=2 agent="dual_channel_gat" \
# env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" seed=4126