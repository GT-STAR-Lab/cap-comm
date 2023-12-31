import torch
import sys
import os
from make_env import make_env
sys.path.append('../src')
sys.path.append('../mpe')
from modules.agents import *
from mpe.environment import MultiAgentEnv
import mpe.scenarios as scenarios
import argparse
import json
import numpy as np
import time
from PIL import Image
import PIL.ImageDraw as ImageDraw
import yaml
import re
import importlib
import logging
from tqdm import tqdm
import pandas as pd
import argparse
import imageio


class DictView(object):
        def __init__(self, d):
            self.__dict__ = d
        def __str__(self):
             
             return(str(self.__dict__))

def parse_args():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Argument Parser Example')

    # The name of the environment
    parser.add_argument('--environment', type=str, help='Environment argument', default="mpe:MaterialTransport-v0")
        # the directory of the evaluation configuration files
    parser.add_argument('--eval_config_dir', type=str, help='Env config directory argument', default=os.path.join(current_dir, "eval_experiments_and_configs", "mpe:MaterialTransport-v0", "eval_configs"))
    # the specific evaluation configuration you want to use for this evaluation
    parser.add_argument('--eval_config_filename', type=str, help='Env config filename argument')
    # The sacred run of the model you want to evaluate.
    parser.add_argument('--sacred_run', type=str, help='Sacred run argument')
    parser.add_argument('--experiment_results_dir', type=str, help='Experiment path argument', default=os.path.join(current_dir, "eval_experiments_and_configs", "mpe:MaterialTransport-v0", "experiments"))
    # The name of the experiment 
    parser.add_argument('--experiment_name', type=str, help='Experiment name argument')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--num_episodes', default=1, type=int, help="Number of episodes for evaluation")
    parser.add_argument('--max_num_steps', default=100, type=int, help="Maximum number of steps for an episode")
    parser.add_argument('--render_freq', default=10, type=int, help="Frequency of episodes to display/save render")
    parser.add_argument('--save_renders', default=False, action='store_true')
    # Where evaluation results are saved.
    parser.add_argument('--save_eval_result_dir', type=str, help='Save evaluation result directory argument', default=os.path.join(current_dir, "eval_experiments_and_configs", "mpe:MaterialTransport-v0", "eval_outputs"))
    

    args = parser.parse_args()
    return args

def _label_with_data(frame, info):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)
    
    text_color = (0,0,0)

    i = 0
    string = ""
    for key, val in info.items():
        if isinstance(val, list):
            text_color = val[-1]
            val = val[0]
        string += '\n'
        # string += f'{key}: {val}'
        drawer.text((im.size[0]/20,im.size[1]/18), string+f'{key}: {val}', fill=text_color)
        

    return im

def load_experiment(experiment_dir, run_index, env, results_rel_dir="results"):
    """Load the sacred config and find the model path"""
    sacred_dir = os.path.join(experiment_dir, results_rel_dir, "sacred_runs", env.split(":")[-1], str(run_index))
    
    # loads the configuration from the sacred experiment (primarly config related to the model)
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
    print("Input dim", input_dim)

    if(hasattr(config, "capabilities_skip_gnn")):
        if(config.capabilities_skip_gnn):
            input_dim += 2

    if config.agent=='mlp':
        model = MLPAgent(input_dim, config)
    elif config.agent=='rnn':
        model = RNNAgent(input_dim, config)
    elif config.agent == 'gnn':
        model = GNNAgent(input_dim, config)
    model.load_state_dict(model_weights)
    model.eval()
    return(model)

def _make_env(env_config):
    """
    Build the environment with the appropriate configuration.
    """
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # with open(os.path.join(current_dir, '../scenarios', 'configs', "material_transport", 'config.yaml'), 'r') as f:
    #         config = DictView(yaml.load(f, Loader=yaml.SafeLoader))
    
    scenario = scenarios.load("material_transport.py").Scenario(config=env_config)
    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.info, scenario.done)
    return env

def run_eval(env_name, model, config, env_config):
    
        
    env = _make_env(env_config)
    obs = np.array(env.reset())
    n_agents = env.n_agents
    
    totalReturn = []
    totalSteps = []
    lumber_quota_filled, concrete_quota_filled = [], []
    total_lumber_quota_remaining_perc, total_concrete_quota_remaining_perc = [], []
    toal_toal_quota_filled_per_step = []
    total_quota_filled = []
    totalInfo = [] # all the info returned from each episode (at end of episode)

    for i in tqdm(range(num_episodes)):
        episodeReturn = 0
        episodeSteps = 0
        lumber_quota_reamining_perc = []
        concrete_quota_remaining_perc = []
        total_quota_filled_per_step = []
        rewards = []
        hs = np.array([np.zeros((config.hidden_dim, )) for i in range(n_agents)])
        frames = []
        info_list = []
        for j in range(max_num_steps):      
            
            if config.agent == "gnn":
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(env.adj_matrix))
            else:
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
              
            actions = np.argmax(q_values.detach().numpy(), axis=1)

            obs, reward, done, info = env.step(actions)
            rewards.append(reward[0])
            info = info['n'][0]
            info_list.append(info)
            lumber_quota_reamining_perc.append(info_list[-1]['lumber_remaining (%)'])
            concrete_quota_remaining_perc.append(info_list[-1]['concrete_remaining (%)'])
            lumber_delivered, lumber_quota = info["lumber_delivered"], info["lumber_quota"]
            concrete_delivered, concrete_quota = info["concrete_delivered"], info["concrete_quota"]
            total_quota_filled_per_step.append(float(info["total_quota_filled"]))
            # print(info)
            if env_config.shared_reward:
                episodeReturn += reward[0]
            else:
                episodeReturn += sum(reward)
            if(render and (i % args.render_freq == 0)):
                frame = env.render(mode="rgb_array")
                draw_data={'Episode': i, 
                           'Lumber Delivered / Lumber Quota': [f'{info["lumber_delivered"]}/{info["lumber_quota"]}', (255, 0, 0)],
                           'Concrete Delivered / Concrete Quota': [f'{info["concrete_delivered"]}/{info["concrete_quota"]}', (0, 0, 255)]}
                if(args.save_renders):
                    frames.append(_label_with_data(frame, draw_data))

            if done[0]:
                episodeSteps = j+1
                break
        if episodeSteps == 0:
            episodeSteps = max_num_steps
        
        if(args.save_renders and (i % args.render_freq == 0)):
            imageio.mimsave(os.path.join(save_renders_dir, "%d.mp4"%i), frames, fps=10)
            print(info)

        obs = np.array(env.reset())
        # rewards = np.array(rewards)
        # episodeReturn = np.sum(rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
        totalReturn.append(episodeReturn)
        totalSteps.append(episodeSteps)
        concrete_quota_filled.append(info_list[-1]['concrete_quota_filled'])
        lumber_quota_filled.append(info_list[-1]['lumber_quota_filled'])
        total_quota_filled.append(info_list[-1]['total_quota_filled'])
        total_lumber_quota_remaining_perc.append(lumber_quota_reamining_perc)
        total_concrete_quota_remaining_perc.append(concrete_quota_remaining_perc)
        toal_toal_quota_filled_per_step.append(total_quota_filled_per_step)
        # print("Episode Return:", episodeReturn)
  
    # totalReturn = np.array(totalReturn)
    # totalReturn = (totalReturn - np.mean(totalReturn)) / (np.std(totalReturn) + 1e-15)
    # totalReturn = totalReturn.tolist()
    eval_data_dict = {
        "returns": totalReturn,
        "steps": totalSteps,
        "lumber_quota_filled": lumber_quota_filled,
        "concrete_quota_filled": concrete_quota_filled,
        "total_quota_filled": total_quota_filled,
        "total_lumber_quota_remaining_perc": total_lumber_quota_remaining_perc,
        "total_concrete_quota_remaining_perc": total_concrete_quota_remaining_perc,
        "total_quota_filled_per_step": toal_toal_quota_filled_per_step
    }
    return(eval_data_dict)

if __name__ == "__main__":

    args = parse_args()
    environment = args.environment
    experiment_name = args.experiment_name
    experiment_results_dir = args.experiment_results_dir
    eval_config_dir = args.eval_config_dir
    save_eval_result_dir = args.save_eval_result_dir
    eval_config_filename = args.eval_config_filename
    sacred_run = int(args.sacred_run)
    experiment_name = args.experiment_name
    render = args.render
    num_episodes = args.num_episodes
    max_num_steps = args.max_num_steps
    np.random.seed(42)

    # environment = "robotarium_gym:HeterogeneousSensorNetwork-v0"
    # experiment_results_dir = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl"
    # eval_config_dir = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl/eval_env_configs" # this is the where "config.yamls" for the robotarium environment are located
    # save_eval_result_dir = "/home/dwalkerhowell3/star_lab/experiments_ca-gnn-marl/eval_saves"


    ##################
    # eval_config_filename = "eval_4_agents_ID_seen_bc_default.yaml"
    # sacred_run = 3; 
    # experiment_name = "SC_ID_4_agents_REDO"
    results_rel_dir="results"
    ################

    save_filename = eval_config_filename.split(".yaml")[0] + "_" + experiment_name + "_sacred_run_" + str(sacred_run) + ".json"
    print("Evaluation Name:", save_filename)
    experiment_dir = os.path.join(experiment_results_dir, experiment_name)
    env_config_file = os.path.join(eval_config_dir, eval_config_filename)
    save_renders_dir = os.path.join(save_eval_result_dir, 'renderings', save_filename.split(".json")[0]+"_renderings")
    if(not os.path.exists(save_renders_dir) and args.save_renders):
        os.mkdir(save_renders_dir)

    config, model_dir, tb_dir = load_experiment(experiment_dir, sacred_run, environment, results_rel_dir=results_rel_dir)
    config.n_actions = 5
    model = load_model(model_dir, config)

    # load the environment config
    with open(env_config_file, 'r') as f:
        env_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # only use the following for debugging
    # env_config["episodes"] = 1
    config.n_agents = env_config["n_agents"]
    
    eval_output_dict = run_eval(environment, model, config, DictView(env_config))

    print("Evaluation output file path:", save_eval_result_dir)
    print("\t Evaluatin ouput file Name:", save_filename)
    with open(os.path.join(save_eval_result_dir, save_filename), 'w') as f:
        json.dump(eval_output_dict, f)