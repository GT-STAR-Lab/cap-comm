import numpy as np
from  mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import yaml

class Scenario(BaseScenario):
    def __init__(self, num_agents=3, config=None):
        self.num_agents = num_agents
        self.ideal_theta_separation = (2*np.pi)/self.num_agents # ideal theta difference between two agents 

        self.alt_thresh = 0.5
        self.succ_thres = 0.1
        self.damage_factor = 1.0

        self.terrain_damage = np.random.random((self.num_agents, ))
        self.parsed_config = False
        self.agent_id = False
        self.fully_obsrved = False
        self.num_relevant_agents = self.num_agents

  


    def make_world(self):
        world = World()


        self.goal_locs = np.random.random((1, self.num_agents, 2))
        world.agents = [Agent() for i in range(self.num_agents)]
        world.landmarks = [Landmark() for i in range(self.num_agents)]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark   %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False

        self.reset_world(world)
        return world

    def parse_config(self, world):
        self.parsed_config = True
        self.num_train_candidates = world.config['num_train_candidates']
        self.num_test_candidates = world.config['num_test_candidates']
        self.num_relevant_agents = world.config['num_relevant_agents']
        self.num_relevant_agents = self.num_agents if self.num_relevant_agents == 'all' else self.num_relevant_agents
        self.agent_id = world.config['use_agent_id']
        self.config = world.config
        self.reset_world(world)
        

    def reset_world(self, world):
        if not self.parsed_config:
            print('Reseting without config')
        elif self.parsed_config:
            if world.config['train']:
                selected_agents = np.random.choice(self.num_train_candidates, size=(self.num_agents, ), replace=False)
            else:
                test_agent_ids = np.arange(self.num_train_candidates, self.num_test_candidates+self.num_train_candidates)
                selected_agents = np.random.choice(test_agent_ids, size=(self.num_agents, ), replace=False)
            self.terrain_damage = np.array([self.config['agents'][k]['terrain_damage'] for k in selected_agents])
            
            self.ids = [self.config['agents'][k]['id'] for k in selected_agents]
            for i, agent in enumerate(world.agents):
                agent.size = 0.2
                agent.bin_id = self.ids[i]
                agent.bin_id_vec = [1 if ch == '1' else 0 for ch in agent.bin_id]
                agent.dec_id = selected_agents[i]


        for i, agent in enumerate(world.agents):
            agent.trait_dict = {}
            agent.state.p_pos = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=0)]) 
            agent.color = np.array([1.0, self.terrain_damage[i], 0]) 
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.silent = True
            agent.size = 0.075 
            agent.is_success = False

            agent.idx = i
            
        for i in range(self.num_agents):
            world.landmarks[i].color = np.array([.45, 0.1, .98])
            world.landmarks[i].state.p_pos =  np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)])
            world.landmarks[i].state.p_vel = np.array([0,0])
            world.landmarks[i].collide = False
            world.landmarks[i].size = 0.1

        world.steps = 0


    def reward(self, agent, world):

        rew = 0
        for i, l in enumerate(world.landmarks[0:self.num_agents]):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            world.agents[np.argmin(dists)].is_success = True if np.min(dists) < self.succ_thres else False
            rew -= min(dists)
        
        for l in world.agents:
            rew -= self.damage_factor * self.terrain_damage[i] * (l.state.p_pos[1] > 0)
            

        self.reward = rew
        return rew


    def observation(self, agent, world):
        if not self.parsed_config:
            key_list = dir(world)
            if 'config' in key_list:
                self.parse_config(world)
                print('\nConfig has been parsed.\n')
            else:
                print('Warning: config has not been parsed.\n')
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
        agent_pos = np.array(agent.state.p_pos)
        agent_vel = np.array(agent.state.p_vel)
        agent_terrain_damage = np.array([self.terrain_damage[agent.idx]])
        goals = np.array([entity.state.p_pos for entity in world.landmarks[:self.num_agents]]).flatten()

        # Obtain locations of n closest agents
        n = self.num_relevant_agents
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        impt_agent_idx = np.argsort(dists).astype(int)
        relevant_agents = [world.agents[i] for i in impt_agent_idx[1:(1+n)]]
        other_pos = np.array([ag.state.p_pos for ag in relevant_agents]).flatten()
        other_terrain_damage = self.terrain_damage[impt_agent_idx[1:(1+n)]]

     
        obs = np.concatenate((agent_pos, agent_vel, goals))
        if self.agent_id:
            obs = np.concatenate((obs, agent.bin_id_vec))
        else:
            obs = np.concatenate((obs, agent_terrain_damage))

        if self.fully_obsrved and self.agent_id:
            obs = np.concatenate((obs, np.array([ag.bin_id_vec for ag in relevant_agnets]).flatten()))
        elif self.fully_obsrved and not self.agent_id:
            obs = np.concatenate((obs, other_pos, other_terrain_damage))


        return obs
  
    def done(self, agent, world):
        if world.steps >= 50:
            return True
        else:
            return False

    def info(self, agent, world):
        info = {'is_success': agent.is_success, 'world_steps': world.steps,
                'reward': self.reward, 'dists': 0}

        return info