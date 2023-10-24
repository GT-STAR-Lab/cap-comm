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
To install the depedencies, followed the installation instructions in the MARBLER [README]()


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
the trained models are deployed on the target environment and evaluation metrics are recorded. For reporting, the collected data is plotted and the corresponding plots are saved as figures.  

## Heterogeneous Matieral Transport Environment (HMT)
Data Collection
```bash
cd [REPO_PATH]/cap-comm/eval
python run_all_mpe_material_transport_evals.py 
```
The data collection script will load models (at each seed) from the `eval/eval_experiments_and_configs/[ENVIRONMENT_NAME]/experiments` directory and the evaluation configuration from `eval/eval_experiments_and_configs[ENVIRONMENT_NAME]/eval_configs`. All the models will be ran for each config. Please see the scripts `eval/eval_mpe_material_transport.py` and `eval/run_all_mpe_material_transport_evals.py` for more details on how the evaluations are performed.


## Heterogeneous Sensor Network Environment (HSN)

# Training New Models

## Heterogeneous Matieral Transport Environment (HMT)

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
