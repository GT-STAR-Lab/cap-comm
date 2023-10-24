# Extended Python MARL framework - EPyMARL


# Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)

- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions
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


For MPE, our fork is needed. Essentially all it does (other than fixing some gym compatibility issues) is i) registering the environments with the gym interface when imported as a package and ii) correctly seeding the environments iii) makes the action space compatible with Gym (I think MPE originally does a weird one-hot encoding of the actions).

The environments names in MPE are:
```
...
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
...
```
Therefore, after installing them you can run it using:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpeakerListener-v0"
```

The pretrained agents are included in this repo [here](https://github.com/uoe-agents/epymarl/tree/main/src/pretrained). You can use them with:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleAdversary-v0" env_args.pretrained_wrapper="PretrainedAdversary"
```
and
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleTag-v0" env_args.pretrained_wrapper="PretrainedTag"
```


# Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.


# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

# Citing CAP-COMM, EPyMARL and PyMARL

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
