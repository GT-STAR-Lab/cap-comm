import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import sys
import yaml
np.set_printoptions(threshold=sys.maxsize)

class Scenario(BaseScenario):
    def __init__(self, num_agents=4, arena_size=1, config=None):
        self.num_agents = num_agents
        self.target_radius = 0.5 # fixing the target radius for now  
        self.arena_size = arena_size
        self.dist_thres = 0.05
        self.succ_thres = 0.1



        self.config = config
        self.traits = np.random.random((self.num_agents, ))#self.config['net']
        max_rad = 30 #np.max(self.traits)
        min_rad = 0 #np.min(self.traits)

        des_max_rad = 0.5
        des_min_rad = 0.2
        trait_dist = np.random.normal(loc = 2, scale = 0.5, size=(self.num_agents, ))
        self.traits = np.clip(0,np.max(trait_dist), trait_dist)




    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = self.num_agents
        num_landmarks = num_agents
        world.collaborative = False

        world.agents = [Agent() for i in range(num_agents)]
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
            agent.color = np.array([0, 0, 0])

        self.reset_world(world)
        return world

    def reset_world(self, world):

        trait_dist = np.random.normal(loc = 0.25, scale = 0.1, size=(self.num_agents, ))
        self.traits = np.clip(0,np.max(trait_dist), trait_dist)

        for i, agent in enumerate(world.agents):
            # agent.trait_dict = {}
            # agent.trait_dict['sensing_rad'] = self.traits[i]
            agent.idx = i
            agent.color = np.array([0, 0, 0])
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.size = self.traits[i]
            agent.is_success = False


        world.steps = 0
        self.rads = self.traits

    def reward(self, agent, world):
        if agent.idx == 0:
            pos = np.reshape(np.array([agent.state.p_pos for agent in world.agents]), (self.num_agents, 1, 2))
            rew = 0
            for af in world.agents:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - af.state.p_pos))) for a in world.agents]
                comm_dists = np.abs(np.array(dists)  - (self.rads + self.rads[af.idx] ))
                af.is_success = True if np.any(np.abs(comm_dists - self.succ_thres) < self.succ_thres) else False 
                rew -= np.sum(comm_dists) 
                
            rew -= min([np.sqrt(np.sum(np.square(a.state.p_pos))) for a in world.agents])
            self.reward = rew

        return self.reward


    def observation(self, agent, world):
        pos = np.array(agent.state.p_pos)
        vel = np.array(agent.state.p_vel)
        sensing_rad = np.array([self.traits[agent.idx]])

        # Retrieve sensing radii and positions of other n closest agents 
        n = 2
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        impt_agent_idx = np.argsort(dists).astype(int)
        other_sensing_rad = np.array([self.traits[impt_agent_idx[1:(1 + n)]]]).flatten()
        relevant_agents = [world.agents[i] for i in impt_agent_idx[1:(1+n)]]
        other_pos = np.array([ag.state.p_pos for ag in relevant_agents]).flatten()

        ret_obs = np.concatenate((pos,vel,other_pos, sensing_rad, other_sensing_rad))

        return ret_obs


    def done(self, agent, world):
        if world.steps >= 50:
            return True
        else:
            return False


    def info(self, agent, world):
        info = {'is_success': agent.is_success, 'world_steps': world.steps,
                'reward': self.reward, 'dists': 0}
        
        return info