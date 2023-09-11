#!/bin/bash
#SBATCH -Jsc_gnn_hsn               # Job name
#SBATCH --account=gts-star-lab                  # charge account
#SBATCH -N1 --ntasks-per-node=24               # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH -t2-00:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oslurm_out/sc_gnn_hsn_3-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=pierce.howell@gatech.edu        # E-mail address for notifications
cd /storage/home/hcoda1/5/dwalkerhowell3/star_lab/ca-gnn-marl      # Change to working directory

module load anaconda3                   # Load module dependencies
srun conda init && conda activate ca-gnn-marl && python3 -u src/main.py with alg_yaml=mappo_gnn env_yaml=gymma batch_size_run=10 batch_size=32 buffer_size=32 env_args.time_limit=1000 env_args.key="robotarium_gym:HeterogeneousSensorNetwork-v0" t_max=20500000 local_results_path=/storage/home/hcoda1/5/dwalkerhowell3/scratch
