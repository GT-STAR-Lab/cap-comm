user=$USER
proj_dir="/home/${user}/star_lab/ca-gnn-marl"
echo $proj_dir
bash $proj_dir/scripts/qmix/run_qmix_hmt_cu.sh
bash $proj_dir/scripts/qmix/run_qmix_hsn_cu.sh
bash $proj_dir/scripts/qmix/run_qmix_tan_cu.sh