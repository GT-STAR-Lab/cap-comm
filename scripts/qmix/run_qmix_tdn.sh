# cd ~/documents/hetmarl/mpe
# pip install -e .
cd /home/mrudolph/documents/hetmarl/
python3 src/main.py --config=qmix --env-config=gymma \
    with env_args.time_limit=50 \
    env_args.key="mpe:TerrainDependantNavigation-v0" \
    env_args.config_path="src/config/hets/tdn/cap_aware_partial_obs.yaml"