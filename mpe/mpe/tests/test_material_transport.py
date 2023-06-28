import torch
import sys
sys.path.append('../mpe')
import os
from make_env import make_env
#sys.path.append("../../../src")
#from envs import REGISTRY as env_REGISTRY
import argparse
import json
import numpy as np
import time
from PIL import Image
import yaml
from gym import spaces
import gym
from mpe.environment import MultiAgentEnv
import mpe.scenarios as scenarios


class DictView(object):
        def __init__(self, d):
            self.__dict__ = d
        def __str__(self):
             
             return(str(self.__dict__))


LUMBER_DEPOT = np.array([0.5, 0.5])
CONCRETE_DEPOT = np.array([-0.5, 0.5])
CONSTRUCTION_SITE = np.array([0.0, -0.5])

def _get_step_toward_goal_pos(curr_pos, goal_pos):
    x1, y1 = curr_pos
    x2, y2 = goal_pos
    theta = np.arctan2(y2 - y1, x2 - x1)
    # print("(x1, y1)=(%0.2f, %0.2f) -- (x2, y2)=(%0.2f, %0.2f)" % (x1, y1, x2, y1))
    # print("\t Theta = %0.4f" % (theta * 180/(np.pi)))
    if(theta <= np.pi / 4.0 and theta > -1*np.pi / 4.0):
        action = [0, 1, 0, 0, 0] # right
    elif((theta <= 3 * np.pi / 4.0) and (theta > np.pi / 4.0)):
        action = [0, 0, 0, 1, 0] # up
    elif(theta > -3 * np.pi / 4.0 and theta <= -1*np.pi/4.0):
        action = [0, 0, 0, 0, 1] # down
    else:
        action = [0, 0, 1, 0, 0] # left
    # print("\tAction:", action)
    return(action)

def greedy_cap_controller_step(obs, env):
    """Each agent greedily takes what it is best at carrying."""
    agents = env.agents
    actions = []

    for i, agent in enumerate(agents):
        
        lumber_quota_filled = False if (obs[i][11] - obs[i][10]) < 0 else True
        concrete_quota_filled = False if (obs[i][13] - obs[i][12]) < 0 else True
        
        # if agent is empty, it greedily goes to the depot it can carry the most from
        if(agent.empty()):
            if(agent.lumber_cap > agent.concrete_cap and not lumber_quota_filled):
                action = _get_step_toward_goal_pos(agent.state.p_pos, LUMBER_DEPOT)
            elif(agent.lumber_cap <= agent.concrete_cap and not concrete_quota_filled):
                action = _get_step_toward_goal_pos(agent.state.p_pos, CONCRETE_DEPOT)
            else:
                action = _get_step_toward_goal_pos(agent.state.p_pos, CONSTRUCTION_SITE)
        # if the agent is loaded, it goes to the construction site
        else:
            action = _get_step_toward_goal_pos(agent.state.p_pos, CONSTRUCTION_SITE)
        actions.append(action)
    return(actions)

def _make_env():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, '../scenarios', 'configs', "material_transport", 'config.yaml'), 'r') as f:
            config = DictView(yaml.load(f, Loader=yaml.SafeLoader))
    
    scenario = scenarios.load("material_transport.py").Scenario(config=config)
    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.info, scenario.done)
    return env

def _print_observation(obs, index):
    """
    Helper function to print observation
    """
    _obs = obs[index]
    agent_pos = _obs[0:2]
    agent_vel = _obs[2:4]
    dist_lumber_depot = _obs[4:6]
    dist_concrete_depot = _obs[6:8]
    dist_construction_site = _obs[8:10]
    resource_status = _obs[10:14]
    agent_current_load = _obs[14:16]
    print("Observation:")
    print("\tPosition:", agent_pos)
    print("\tVelocity:", agent_vel)
    print("\tDist. Lumber Depot:", dist_lumber_depot)
    print("\tDist. Concrete_Depto:", dist_concrete_depot)
    print("\tDist. Construction Site:", dist_construction_site)
    print("\tResource Status:", resource_status)
    print("\tCurrent Load:", agent_current_load)

def main(args):
    # env = make_env("material_transport", config="1.yaml")
    # env = gym.make("mpe:MaterialTransport-v0")
    # print(env_REGISTRY["gymma"])
    # env = env_REGISTRY["gymma"](key="mpe:MaterialTransport-v0", time_limit=80, pretrained_wrapper=False)
    env = _make_env()
    obs = env.reset()
    
    episodes = 1
    steps = 100
    for episode in range(episodes):
        eps_return = 0
        for step in range(steps):
            actions = greedy_cap_controller_step(obs, env)
            
            actions = np.argmax(actions, axis=1)
            obs, reward, done, info = env.step(actions)
            _print_observation(obs, 0)
            shared_reward = np.sum(reward)
            eps_return += shared_reward
            # print("Info", info)
            print("Return", eps_return)
            # if render, show render
            if args.render:
                img = env.render(mode="rgb_array")
                time.sleep(0.1)

            if(all(done)):
                print("Done!")
                print("\tInfo:", info)
                break
            
        obs = env.reset()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Material Transport')
    parser.add_argument('--render', default=False, action='store_true')

    args = parser.parse_args()

    main(args)