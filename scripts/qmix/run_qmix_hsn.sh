user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn-marl"
echo $proj_dir
cd $proj_dir
python3 src/main.py --config=qmix --env-config=gymma \
    with env_args.time_limit=50 \
    env_args.key="mpe:HeterogeneousSensorNetwork-v0" \
    env_args.config_path="src/config/hets/hsn/cap_aware_partial_obs.yaml"