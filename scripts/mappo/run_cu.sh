user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn"
echo $proj_dir
bash $proj_dir/scripts/mappo/run_mappo_hmt_cu.sh
bash $proj_dir/scripts/mappo/run_mappo_hsn_cu.sh
bash $proj_dir/scripts/mappo/run_mappo_tan_cu.sh