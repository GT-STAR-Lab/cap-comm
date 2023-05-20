#!/bin/bash

proj_dir=$(find ~ -name "ca-gnn-marl")
echo $proj_dir
cd $proj_dir

echo Please ensure the config settings for agent-id/capability-awareness in \
./Heterogeneous-MARL-CA/env-name/config.yaml are correct!
echo Enter to continue
read temp


for exp_num in 1 .. 3
do
        echo -----------------------------------------------------------------
        echo Running experiment 1, seed $exp_num \(see Sacred for true seed\)
        
        ### mappo_gnn robotarium
        # python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
        # env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0"

        ### mappo_gat robotarium
        python3 src/main.py with alg_yaml=mappo_gat env_yaml=gymma \
        env_args.time_limit=1000 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0"
        
        ### mappo_gnn simplespread
        # python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
        # env_args.time_limit=100 env_args.key="mpe:SimpleSpread-v0"
        
        ### ippo_gnn (GPPO) simplespread
        #python3 src/main.py with alg_yaml=ippo_gnn env_yaml=gymma \
        #env_args.time_limit=100 env_args.key="mpe:SimpleSpread-v0"
done
