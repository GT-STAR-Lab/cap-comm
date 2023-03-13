user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn-marl"
echo $proj_dir
bash $proj_dir/scripts/mappo/run_mappo_hmt_ca.sh
bash $proj_dir/scripts/mappo/run_mappo_hsn_ca.sh
bash $proj_dir/scripts/mappo/run_mappo_tan_ca.sh