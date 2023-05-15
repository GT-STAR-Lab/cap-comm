import subprocess
import os

proj_dir = subprocess.check_output(['find', os.path.expanduser('~'), '-name', "ca-gnn-marl"]).decode().strip()
print(proj_dir)
subprocess.call(f'cd {proj_dir}')
cmd = subprocess.call('python3 src/main.py with alg_yaml=mappo_gnn env_yaml=gymma env_args.time_limit=100 env_args.key="robotarium_gym:PredatorCapturePreyGNN-v0"'.split(' '))
print(cmd)

