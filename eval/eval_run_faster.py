import torch
import sys
import os
from make_env import make_env
sys.path.append('../src')
sys.path.append('../mpe')
from modules.agents import *
import argparse
import json
import numpy as np
import time
from PIL import Image
import yaml
import re
import importlib
import logging
from tqdm import tqdm
import pandas as pd

class DictView(object):
        def __init__(self, d):
            self.__dict__ = d
        def __str__(self):
             
             return(str(self.__dict__))

def load_experiment(experiment_dir, run_index, env, results_rel_dir="results"):
    """Load the sacred config and find the model path"""
    sacred_dir = os.path.join(experiment_dir, results_rel_dir, "sacred_runs", env.split(":")[-1], str(run_index))
    with open(os.path.join(sacred_dir, "config.json"), 'r') as config_file:
        config = json.load(config_file)
        
        config = DictView(config)
    # find the models path and tensor board path
    unique_token=config.unique_token
    models_dir = os.path.join(experiment_dir, results_rel_dir, "models", env, unique_token, str(run_index))
    tb_dir = os.path.join(experiment_dir, "results", "tb_logs", env, unique_token, str(run_index))
    
    return config, models_dir, tb_dir

def load_model(models_dir, config):

    # find the last checkpoint
    ckts = [int(re.sub("[^0-9]", "", ckt) if len(re.sub("[^0-9]", "", ckt)) > 0 else str(-1)) for ckt in os.listdir(models_dir)]
    print("Loading model checkpoint: ", str(max(ckts)))
    ckt = max(ckts)

    model_file = os.path.join(models_dir, str(ckt), 'agent.th')
    model_weights = torch.load(model_file, map_location=torch.device('cpu'))
    input_dim = model_weights[list(model_weights.keys())[0]].shape[1]

    if(hasattr(config, "capabilities_skip_gnn")):
        if(config.capabilities_skip_gnn):
            input_dim += 1

    if config.agent=='mlp':
        model = MLPAgent(input_dim, config)
    elif config.agent=='rnn':
        model = RNNAgent(input_dim, config)
    elif config.agent == 'gnn':
        model = GNNAgent(input_dim, config)
    model.load_state_dict(model_weights)
    # model.eval()
    return(model)

def run_eval(env_name, model, config, env_config):
    # env = make_env(env_name)
    if(env_name == "robotarium_gym:HeterogeneousSensorNetwork-v0"):
        env_module = importlib.import_module(f'robotarium_gym.scenarios.HeterogeneousSensorNetwork.HeterogeneousSensorNetwork')
        env_class = getattr(env_module, "HeterogeneousSensorNetwork")
        env_config = DictView(env_config)
        env = env_class(env_config)

    obs = np.array(env.reset())
    n_agents = len(obs)
    
    totalReturn = []
    totalConnectivity = []
    totalSteps = []
    totalViolations = []
    totalOverlap = []
    
    max_edges = n_agents * (n_agents - 1) / 2.0
    
    for i in tqdm(range(env_config.episodes)):
        episodeReturn = 0
        episodeSteps = 0
        episodeViolations = 0
        episodeConnectivity = [0 for _ in range(int(max_edges+1))]
        episodeOverlap = []
        hs = np.array([np.zeros((config.hidden_dim, )) for i in range(n_agents)])
        
        for j in range(env_config.max_episode_steps+1):      
            
            if config.agent == "gnn":
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(env.adj_matrix))
            else:
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
              
            actions = np.argmax(q_values.detach().numpy(), axis=1)

            obs, reward, done, info = env.step(actions)
            
            # log data
            episodeViolations += 1.0 if info["violation_occurred"] else 0.0
            episodeConnectivity[info["connectivity"]] += 1
            episodeOverlap.append(info["total_overlap"])

            if env_config.shared_reward:
                episodeReturn += reward[0]
            else:
                episodeReturn += sum(reward)
            if done[0]:
                episodeSteps = j+1
                break
        
        if episodeSteps == 0:
            episodeSteps = env_config.max_episode_steps
        
        obs = np.array(env.reset())
        totalReturn.append(episodeReturn)
        totalSteps.append(episodeSteps)
        totalConnectivity.append(list(np.array(episodeConnectivity)/episodeSteps))
        totalViolations.append(episodeViolations)
        totalOverlap.append(np.mean(episodeOverlap))
    

    eval_data_dict = {
        "returns": totalReturn,
        "steps": totalSteps,
        "violations": totalViolations,
        "connectivity": totalConnectivity,
        "overlap": totalOverlap
    }
    return(eval_data_dict)

if __name__ == "__main__":

    environment = "robotarium_gym:HeterogeneousSensorNetwork-v0"
    experiment_path = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl"
    env_config_dir = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl/eval_env_configs" # this is the where "config.yamls" for the robotarium environment are located
    save_eval_result_dir = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl/eval_saves"


    ##################
    env_config_filename = "eval_4_agents_ID_seen_bc_default.yaml"
    sacred_run = 3; 
    experiment_name = "SC_ID_4_agents_REDO"
    results_rel_dir="results"
    ################

    save_filename = env_config_filename.split(".yaml")[0] + "_" + experiment_name + "_sacred_run_" + str(sacred_run) + ".json"
    print("Evaluation Name:", save_filename)
    experiment_dir = os.path.join(experiment_path, experiment_name)
    env_config_file = os.path.join(env_config_dir, env_config_filename)
    config, model_dir, tb_dir = load_experiment(experiment_dir, sacred_run, environment, results_rel_dir=results_rel_dir)
    print("Model Dir:", model_dir)
    config.n_actions = 5
    model = load_model(model_dir, config)

    # load the environment config
    with open(env_config_file, 'r') as f:
        env_config = yaml.load(f, Loader=yaml.SafeLoader)

    config.n_agents = env_config["n_agents"]

    eval_output_dict = run_eval(environment, model, config, env_config)

    print("Evaluation output file path:", save_eval_result_dir)
    print("\t Evaluatin ouput file Name:", save_filename)
    with open(os.path.join(save_eval_result_dir, save_filename), 'w') as f:
        json.dump(eval_output_dict, f)