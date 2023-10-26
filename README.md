# CAP-COMM
This codebase was used for the expeirments in [Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities](https://openreview.net/forum?id=N3VbFUpwaa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Drobot-learning.org%2FCoRL%2F2023%2FConference%2FAuthors%23your-submissions))

*Pierce Howell, Max Rudolph, Reza Torbati, Kevin Fu, & Harish Ravichandar. Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities, 7th Annual Conference on Robot Learning (CoRL), 2023*

In BibTex format:
```tex
@inproceedings{
  howell2023generalization,
  title={Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities},
  author={Pierce Howell and Max Rudolph and Reza Joseph Torbati and Kevin Fu and Harish Ravichandar},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023},
  url={https://openreview.net/forum?id=N3VbFUpwaa}
}
```


# Table of Contents
- [CAP-COMM Citation](#cap-comm)
- [Table of Contents](#table-of-contents)
- [Installation Instructions](#installation-instructions)
- [Citing MARBLER and EPyMARL](#citing-marbler-and-epymarl)
- [License](#license)

# Installation Instructions
## Download the Repository
```bash
git clone ....
cd ...
git submodule init # Initialize the MARBLER submodule
```


## Python Environment
### Anaconda

Create the anaconda environment for python 3.8
```bash
conda create -n cap-comm python=3.8 pip
conda activate cap-comm
```

Install pytorch for the specifications of your system. See [Pytorch installation instructions](https://pytorch.org/).

Install requirements
```
pip install -r requirements.txt
```

## Multi-Particle Environment
Now the multi-agent particle environment must be installed. `cd` into the `mpe` directory and run
```
pip install -e .
```

## MARBLER
To install the dependencies, follow the installation instructions in the MARBLER repo [README](https://github.com/GT-STAR-Lab/MARBLER).


## Download Pre-trained Models

Download and extract the models used for the results in [paper](https://openreview.net/forum?id=N3VbFUpwaa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Drobot-learning.org%2FCoRL%2F2023%2FConference%2FAuthors%23your-submissions)) 

```bash
cd [repo_path]/cap-comm/
mkdir pretrained_models && cd pretrained_models

wget -O mpe-MaterialTransport-v0.zip https://www.dropbox.com/scl/fi/7q9yxveugls2udligm453/mpe-MaterialTransport-v0.zip?rlkey=6xfl9s3wyiyw58meu92w5yylh&dl=0

wget -O "robotarium_gym-HeterogeneousSensorNetwork.zip" https://www.dropbox.com/scl/fi/eosst3qs2artsfo1sxlwx/robotarium_gym-HeterogeneousSensorNetwork-v0.zip?rlkey=fsok69570xir1c49sccqfetm6&dl=0

unzip mpe-MaterialTransport-v0.zip
unzip robotarium_gym-HeterogeneousSensorNetwork-v0.zip

```

The trained models need to be moved to the `eval` folder under the correct experiment. From within `pretrained_models`, run the following:

```bash
cp -r "mpe:MaterialTransport-v0/experiments/*" "../eval/eval_experiments_and_configs/mpe:MaterialTransport-v0/experiments/"
cp -r "robotarium_gym:HeterogeneousSensorNetwork-v0/*" "../eval/eval_experiments_and_configs/robotarium_gym:HeterogeneousSensorNetwork-v0/experiments/"
```

# Evaluations
The evaluations can be reproduced using the installed pretrained models. The evaluation process is comprised of two stages: i) data collection and ii) reporting. During data collection,
the trained models are deployed on the target environment and evaluation metrics are recorded. The data collection script will load models (at each seed) from the `eval/eval_experiments_and_configs/[ENVIRONMENT_NAME]/experiments` directory and the evaluation configuration from `eval/eval_experiments_and_configs[ENVIRONMENT_NAME]/eval_configs`. For reporting, the collected data is plotted and the corresponding plots are saved as figures in the directory `eval/eval_experiments_and_configs/mpe:MaterialTransport-v0/eval_figures`.

## Heterogeneous Matieral Transport Environment (HMT)
Data Collection
```bash
cd [REPO_PATH]/cap-comm/eval
python run_all_mpe_material_transport_evals.py 
```
All the models will be ran for each config. Please see the scripts `eval/eval_mpe_material_transport.py` and `eval/run_all_mpe_material_transport_evals.py` for more details on how the evaluations are performed. Note, `eval/run_all_mpe_material_transport_evals.py` handles running `eval/eval_mpe_material_transport.py` using multi-processing for faster evaluation.


Reporting
```bash
cd [REPO_PATH]/cap-comm/eval

# begin the jupyter notebook
jupyter notebook
```
Open jupyter notebook in a local browser, then open the file `mpe_material_transport_evaluation_reporting.ipynb`. Go ahead and run all the cells to produced the evaluation figures. The figures will be saved in the directory `eval/eval_experiments_and_configs/mpe:MaterialTransport-v0/eval_figures`.


## Heterogeneous Sensor Network Environment (HSN)
Data Collection
```bash
cd [REPO_PATH]/cap-comm/eval
python run_all_marbler_hsn_evals.py
```

Reporting
```bash
cd [REPO_PATH]/cap-comm/eval

# begin the jupyter notebook
jupyter notebook
```
Within the jupyter notebook file directory, open the file `marbler_hsn_evaluation_reporting.ipynb`. Run all the cells to produce the evaluation figures. The figures will be saved in the directory `eval/eval_experiments_and_configs/robotarium_gym:HeterogeneousSensorNetwork-v0/eval_figures`.


# Training New Models
The models are trained using the [EPyMarl](https://github.com/uoe-agents/epymarl) training framework. This section demonstrates the commands that were run to generate the policies used in the paper.

## Heterogeneous Material Transport Environment (HMT)
Before executing training, it is important to verify that the configuration of the environment is correct. At this time, command line arguments DO NOT override the environment-specific configurations. These must be configured in the `mpe/mpe/scenarios/configs/material_transport/base_config.yaml`. The main configuration parameters to change (depending on the experiment) are the following:
- `n_agents`: Choose the number of agents for training. Typically kept at 4 agents.
- `capability_aware / agent_id`: Set with a boolean of `True` or `False` depending if the experiment requires capability aware agents or ID-based agents
- `load_from_predefined_agents`: If set to `True`, then use the predefined agents from the predefined_coalition file. If `False`, then agents with new capabilities are sampled from a distribution (see `traits` in the config file).

The commands for the experiments are provided in the `/scripts/mpe:MaterialTransport-v0` directory:
```bash
# Run the GNN with capability awareness (i.e. CA+CC (GNN))
# set capability_aware = True and agent_id = False
bash run_GNN_CA_4_agents_MT.sh

# Run the GNN with no communication of capabilities, but the agent's action network is conditioned on capabilities (i.e. CA (GNN))
# set capability_aware = True and agent_id = False
bash run_GNN_CA_SKIP_4_agents_MT.sh

# Run the GNN with agent ID (i.e. ID (GNN))
# set capability_aware = False and agent_id = True
bash run_GNN_ID_4_agents_MT.sh

# Run the MLP with capability-aware agents (i.e. CA (MLP))
# set capability_aware = True and agent_id = False
bash run_MLP_CA_4_agents_MT.sh

# Run the MLP with agent IDs (i.e. ID (MLP))
# set capability_aware = False and agent_id = True
bash run_MLP_ID_4_agents_MT.sh
```
### Saving the Models
Each model will save 3 seeds (although extra seeds can be added). Checkpoint models and training metrics will be saved in the `/results` directory. 

TODO: Finish this section!

## Heterogeneous Sensor Network Environment (HSN)

# Citing MARBLER and EPyMarl
The experiments relied on the [MARBLER-CA](https://github.com/GT-STAR-Lab/MARBLER-CA) codebase, which is a fork of the [original MARBLER](https://github.com/GT-STAR-Lab/MARBLER) repository. This fork introduced the heterogeneous sensor network environment with capability aware robots. The MARBLER framework is presented in [MARBLER: An Open Platform for Standardized Evaluation of Multi-Robot Reinforcement Learning Algorithms](https://arxiv.org/abs/2307.03891).

*Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Shashwat Nigam, Harish Ravichandar. MARBLER: An Open Platform for Standardized Evaluation of Multi-Robot Reinforcement Learning Algorithms, 2023*

```tex
@misc{torbati2023marbler,
      title={MARBLER: An Open Platform for Standardized Evaluation of Multi-Robot Reinforcement Learning Algorithms}, 
      author={Reza Torbati and Shubham Lohiya and Shivika Singh and Meher Shashwat Nigam and Harish Ravichandar},
      year={2023},
      eprint={2307.03891},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
