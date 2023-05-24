#!/usr/bin/env python3

import datetime
import numpy as np
import os
import random
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger, get_unique_dirname
import yaml
from yaml import Loader
import time
from run import run
import argparse

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d:%I-%M-%S-%p')}"
# logger.info(f"Saving to FileStorageObserver in results/sacred/{unique_token}.")
# file_obs_path = os.path.join(results_path, f"sacred/{unique_token}")
# ex.observers.append(FileStorageObserver.create(file_obs_path))

# saving to disk because MongoObserver requires the Docker image to be set
# up, which is hard
# https://sacred.readthedocs.io/en/stable/examples.html#docker-setup
# however, this is necessary to use Omniboard/other frontends
# Tensorboard is configured separately, doesn't provide full Sacred
# frontend but useful to see metrics/compare two runs
# - Kevin
#
# ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
# ex.observers.append(MongoObserver())

# Give run.py's run() the Sacred runner, config, and logging modules to run
# the experiment

# results_path = "/home/ubuntu/data"

"""
@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    # run the framework
    run(_run, config, _log)
"""

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def get_config_filepath(filename, subfolder=""):
    if subfolder:
        return os.path.join(os.path.dirname(__file__), "config", subfolder, f"{filename}.yaml")
    else:
        return os.path.join(os.path.dirname(__file__), "config", f"{filename}.yaml")


"""
Configure experiment, Sacred's CLI interface allows us to change any of the
local variables below, e.g.

>> ./src/main.py print_config with 'env_yaml=gymma' 'alg_yaml=qmix'
"""
@ex.config
def config():
    base_yaml = "default"
    env_yaml = "gymma"
    alg_yaml = "qmix"

    # Based on the config filenmaes given above, inject each {filename}.yaml
    # into Sacred's config
    ex.add_config(get_config_filepath(base_yaml))
    ex.add_config(get_config_filepath(env_yaml, "envs"))
    ex.add_config(get_config_filepath(alg_yaml, "algs"))

    # Inject map_name into env_args
    # map_name = "mpe:SimpleSpeakerListener-v0"
    # ex.add_config({"env_args": {"key": map_name}})

@ex.main
def main(_run, _config, _log, seed):
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # Standardize the random seed throughout the modules to match Sacred's
    # auto-generated seed
    np.random.seed(seed)
    th.manual_seed(seed)

    """
    new_het_config = None
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

        if param.startswith("env_args.config_path"):
            new_het_config = param.split("=")[1]
            
    env_config['env_args']['config_path'] = env_config['env_args']['config_path'] if new_het_config is None else new_het_config
    env_config['env_args']['key'] = map_name
    print((env_config['env_args']['config_path']))
    
    if env_config['env_args']['config_path'] is not None:
        with open(env_config['env_args']['config_path'], "r") as f:
            het_config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        het_config = {'name': 'no het_config used'}
    """

    # detect anomlies in backward() for training torch networks (only do this for debuggin, can signficiantly
    # slow down training)
    if(_config["detect_autograd_anomaly"]):
        th.autograd.set_detect_anomaly(True)

    # Give run.py's run() the Sacred runner, config, and logging modules to run
    # the experiment

    run(_run, _config, _log)

    """
    subfolders = [ f.path for f in os.scandir(file_obs_path) if f.is_dir() ]
    # print(subfolders)
    subfolders = [int(subfolder.split('/')[-1]) for subfolder in subfolders if not subfolder.split('/')[-1] == '_sources']
    last_folder = str(max(subfolders))
    het_config_path = os.path.join(file_obs_path, last_folder + "/het_agents.yaml")
    with open(het_config_path , 'w') as file:
        documents = yaml.dump(het_config, file)
    """

if __name__ == '__main__':
    
    print(sys.argv)
    
    for arg in sys.argv:
        if("env_args.key" in arg):
            map_name = arg.split(":")[-1]
    results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
    unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d:%I-%M-%S-%p')}"
    ex.add_config(unique_token=unique_token)
    
    sys.argv.append(f"-i {unique_token}") # set the id of the experiment to be the time it was ran. This matches with tensorboard
    
    logger.info(f"Saving to FileStorageObserver in results/sacred/{unique_token}.")
    
    ex.observers.append(FileStorageObserver(os.path.join(results_path, "sacred_runs", map_name))) # save experiments based on the environment
    
    # main()
    ex.run_commandline()
