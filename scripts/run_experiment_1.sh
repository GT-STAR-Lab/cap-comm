#!/bin/bash

if [ $# -eq 0 ]; then
    >&2 echo "ERROR: No arguments provided"
    exit 1
fi

scenario=$1
echo Executing Experiment 1: agent-id vs capability-aware on scenario $scenario

proj_dir=$(find ~ -name "ca-gnn-marl")
cd $proj_dir

echo Please ensure that ca-gnn-marl/Heterogeneous-MARL-CA/$scenario/config.yaml \
has the right fields set for agent-id/capability awareness

# echo Hit enter to start
# read any

# run experiments, w/ same settings, in sequence
python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma \
env_args.time_limit=100 env_args.key=$scenario
