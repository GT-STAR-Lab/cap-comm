import torch
import sys
sys.path.append('../src')
from modules.agents import *
sys.path.append('../mpe')
import os
from make_env import make_env
import argparse
import json
import numpy as np
import time
from PIL import Image
import yaml
import re

def load_files(args):


    class DictView(object):
        def __init__(self, d):
            self.__dict__ = d

    config = open(args.run + '/config.json')

    config = DictView(json.load(config))
    print(config.env_args)

    try:
        with open('../' + config.env_args['config_path'], 'r') as outfile:
            task_config = yaml.load(outfile, Loader=yaml.SafeLoader)
    except TypeError as e:
        print("Warning: cannot open the task config file, running without one")
        task_config = {}

    config.n_actions = 5
    cout = open(args.run + '/cout.txt')
    cout = cout.readlines()
    model_save = ''
    for line in cout:
        if 'Saving models to' in line:
            model_save = line
            break

    model_path = model_save[(model_save.find('to ') + 3):]
    model_path = model_path[:model_path.rfind("/")+1]

    ckts = [int(re.sub("[^0-9]", "", ckt) if len(re.sub("[^0-9]", "", ckt)) > 0 else str(-1)) for ckt in os.listdir('../' + model_path)]
    print('../' + model_path + str(max(ckts)) + '/agent.th')

    params = torch.load('../' + model_path + str(max(ckts)) + '/agent.th',map_location=torch.device('cpu'))
    print(params.keys())
    input_dim = params[list(params.keys())[0]].shape[1]


    if config.agent=='mlp':
        model = MLPAgent(input_dim, config)
    elif config.agent=='rnn':
        model = RNNAgent(input_dim, config)
    elif config.agent == 'gnn':
        model = GNNAgent(input_dim, config)

    model.load_state_dict(params)
    model.eval()

    config.sacred_path = model_path
    task_config['train'] = False
    return model, config, task_config


def visualize(args):

    model, config, task_config = load_files(args)

    if 'Transport' in config.env_args['key']:
        env_name = 'heterogeneous_material_transport'
    elif 'Network' in config.env_args['key']:
        env_name = 'heterogeneous_sensor_network'
    elif 'AwareNavigation' in config.env_args['key']:
        env_name = 'terrain_aware_navigation'
    elif 'DependantNavigation' in config.env_args['key']:
        env_name = 'terrain_dependant_navigation'
    elif 'Search' in config.env_args['key']:
        env_name = 'search_and_capture'
    
    env = make_env(env_name)
    if len(task_config) > 1:
        env.set_config(config=task_config)

    obs = env.reset()

    # env.render()
    n_agents = len(obs)
    model.args.n_agents = n_agents
    steps = 50
    num_eps = args.num_eps
    #I've heard rumors that the 700,700 below may need to be 1400,1400 depending on the version of gym being used
    imgs = np.zeros((num_eps * steps, 700, 700, 3), dtype=np.uint8)
    
    hs = [np.zeros((config.hidden_dim, )) for i in range(n_agents)]

    eval_rews = np.zeros((num_eps, ))
    
    for j in range(num_eps):
        for k in range(steps):
            if config.agent == 'gnn':
                adj_matrix = env.get_adj_matrix()
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(adj_matrix))
            else:
                q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
            actions = np.argmax(q_values.detach().numpy(), axis=1)
            print(actions)

            obs, reward, done, _ = env.step(actions)

            eval_rews[j] += np.sum(reward) / n_agents
            
            if args.render or args.save_gif:
                img = env.render(mode='rgb_array')
                imgs[j * steps + k, :, : :] = img
                time.sleep(.05)
        
        obs = env.reset()

    dictionary = {
        "rew": list(eval_rews),
        "het_config": task_config,
        "sacred_path": config.sacred_path
    }
            
            # Serializing json
    json_object = json.dumps(dictionary, indent=4)
            
    # Writing to sample.json
    with open(args.run + '/eval.json', "w") as outfile:
        outfile.write(json_object)

    
    # duration is the number of milliseconds between frames; this is 40 frames per second
    if args.save_gif:
        imgs = [Image.fromarray(img) for img in imgs]
        imgs[0].save('gifs/' + env_name + '_' + config.name + '.gif', save_all=True, append_images=imgs[1:], duration=50, loop=0)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run', default=None, help="The sacred folder for the experiment you wish to run")
    parser.add_argument('--num-eps', default=8, type=int)
    parser.add_argument('--save-gif', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--save-eval', default=False, action='store_true')


    args = parser.parse_args()


    visualize(args)    