import torch
import sys
sys.path.append('../src')
from modules.agents import *
sys.path.append('../mpe')
from make_env import make_env
import argparse
import json
import numpy as np
import time
from PIL import Image

def visualize(args):

    class DictView(object):
        def __init__(self, d):
            self.__dict__ = d

    config = open(args.run + '/config.json')
    config = DictView(json.load(config))
    config.n_actions = 5
    cout = open(args.run + '/cout.txt')
    cout = cout.readlines()
    model_save = ''
    for line in cout:
        if 'Saving models to' in line:
            model_save = line
            break
    model_path = model_save[(model_save.find('to ') + 3):]
    model_path = model_path[:-3]
    print(model_path)
    ckt = args.ckt
    params = torch.load('../' + model_path + str(ckt) + '/agent.th',map_location=torch.device('cpu'))
    input_dim = params['fc1.weight'].shape[1]
    if config.agent=='mlp':
        model = MLPAgent(input_dim, config)
    elif config.agent=='rnn':
        model = RNNAgent(input_dim, config)

    model.load_state_dict(params)
    model.eval()

    if 'TransportCA' in config.env_args['key']:
        env_name = 'heterogeneous_material_transport_ca'
    elif 'NetworkCA' in config.env_args['key']:
        env_name = 'heterogeneous_sensor_network_ca'
    elif 'NavigationCA' in config.env_args['key']:
        env_name = 'terrain_aware_navigation_ca'
    elif 'Transport' in config.env_args['key']:
        env_name = 'heterogeneous_material_transport'
    elif 'Network' in config.env_args['key']:
        env_name = 'heterogeneous_sensor_network'
    elif 'Navigation' in config.env_args['key']:
        env_name = 'terrain_aware_navigation'
    
    env = make_env(env_name)

    obs = env.reset()
    
    env.render()
    n_agents = len(obs)
    steps = 50
    num_eps = args.num_eps
    imgs = np.zeros((num_eps * steps, 1400, 1400, 3), dtype=np.uint8)
    
    hs = [np.zeros((config.hidden_dim, )) for i in range(n_agents)]
    for j in range(num_eps):
        for k in range(steps):
            actions = []
            
            for i in range(n_agents):
                ind = np.zeros((n_agents,))
                one_hot = np.zeros((5,))
                ind[i] = 1
                obs_n = obs[i]
                if config.obs_agent_id:
                    q_values, hs[i] = model(torch.Tensor(np.concatenate((obs_n, ind))), hs[i])
                else:
                    q_values, hs[i] = model(torch.Tensor(obs_n), hs[i])
                act = np.argmax(q_values.detach().numpy())
                actions.append(act)

            obs, reward, done, _ = env.step(actions)
            img = env.render(mode='rgb_array')
            imgs[j * steps + k, :, : :] = img
            time.sleep(.05)
        
        obs = env.reset()

    imgs = [Image.fromarray(img) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save('gifs/' + env_name + '_' + config.name + '.gif', save_all=True, append_images=imgs[1:], duration=50, loop=0)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run', default=None)
    parser.add_argument('--ckt', default=18800500)
    parser.add_argument('--num-eps', default=8, type=int)
    parser.add_argument('--save-gif', default=False, action='store_true')


    args = parser.parse_args()


    visualize(args)    