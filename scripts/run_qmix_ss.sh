# cd ~/documents/hetmarl/mpe
# pip install -e .

cd ../
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="mpe:SimpleSpread-v0" \
env_args.obs_trait=1
