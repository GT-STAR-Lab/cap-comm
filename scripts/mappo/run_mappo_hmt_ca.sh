user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn"
echo $proj_dir
cd $proj_dir
python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="mpe:HeterogeneousMaterialTransportCA-v0"