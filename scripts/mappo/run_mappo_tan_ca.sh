user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn-marl"
echo $proj_dir
cd $proj_dir
python3 src/main.py with 'env_yaml=gymma' 'alg_yaml=mappo_debug' env_args.time_limit=50 env_args.key="mpe:TerrainAwareNavigationCA-v0"