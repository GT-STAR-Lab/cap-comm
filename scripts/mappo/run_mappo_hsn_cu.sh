cd ~/documents/hetmarl/mpe
pip install -e .
cd /home/mrudolph/documents/hetmarl/
python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="mpe:HeterogeneousSensorNetwork-v0"
