import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import time
# random.seed(0)

# 0 is water
# 1 is lumber

class Scenario(BaseScenario):
    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.nec_dist = 0.2
        self.parsed_config = False
        self.agent_id = False
        self.fully_obsrved = False
        self.num_relevant_agents = self.num_agents


    def make_world(self):

        world = World()
        world.landmarks = 4
        self.agent_caps = np.zeros((2, self.num_agents))
        self.agent_caps[0,:] = np.zeros((self.num_agents,))
        self.agent_caps[1,:] = np.zeros((self.num_agents, ))
        

        world.agents = [Agent() for i in range(self.num_agents)]
        world.landmarks = [Landmark(), Landmark(), Landmark(), Landmark()]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark   %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15

        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
            agent.idx = i
            agent.size = 0.1
            agent.color = np.array([self.agent_caps[0,i], 0, self.agent_caps[1,i]])

        self.reset_world(world)
        self.world_made = True
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
        
        world.landmarks[0].color = np.array([1, 0, 0])
        world.landmarks[1].color = np.array([1, 0.5, 0.5])
        world.landmarks[2].color = np.array([0, 0, 1])
        world.landmarks[3].color = np.array([0.5, 0.5, 1])

        max_cap = 10
        self.agent_caps[0,:] = np.random.random((self.num_agents, )) * max_cap
        self.agent_caps[1,:] = max_cap - self.agent_caps[0, :]
        self.agent_cur_payload = np.copy(self.agent_caps)
    
        for i, agent in enumerate(world.agents):
            agent.trait_dict = {}
            agent.color = np.array([self.agent_caps[0,i], 0, self.agent_caps[1,i]])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.idx = i

        world.landmarks[0].state.p_pos = np.array([0.5, 0.5]) #np.random.uniform(-1, 1, world.dim_p)
        world.landmarks[0].state.p_vel = np.array([0, 0])
        world.landmarks[1].state.p_pos = np.array([-0.5, 0.5]) #np.random.uniform(-1, 1, world.dim_p)
        world.landmarks[1].state.p_vel = np.array([0, 0])
        world.landmarks[2].state.p_pos = np.array([0.5, -0.5]) #np.random.uniform(-1, 1, world.dim_p)
        world.landmarks[2].state.p_vel = np.array([0, 0])
        world.landmarks[3].state.p_pos = np.array([-0.5, -0.5]) #np.random.uniform(-1, 1, world.dim_p)
        world.landmarks[3].state.p_vel = np.array([0, 0])
        
        world.steps = 0
        self.rew = 0

        if self.parsed_config:
            if world.config['train']:
                selected_agents = np.random.choice(self.num_train_candidates, size=(self.num_agents, ), replace=False)
            else:
                test_agent_ids = np.arange(self.num_train_candidates, self.num_test_candidates+self.num_train_candidates)
                selected_agents = np.random.choice(test_agent_ids, size=(self.num_agents, ), replace=False)

            self.agent_caps[0,:] = np.array([self.config['agents'][k]['lumber'] for k in selected_agents])
            self.agent_caps[1,:] = np.array([self.config['agents'][k]['water'] for k in selected_agents])
            self.agent_cur_payload = np.copy(self.agent_caps)
            self.ids = [self.config['agents'][k]['id'] for k in selected_agents]
            for i, agent in enumerate(world.agents):
                agent.color = np.array([self.agent_caps[0,i], 0, self.agent_caps[1,i]])
                agent.bin_id = self.ids[i]
                agent.bin_id_vec = [1 if ch == '1' else 0 for ch in agent.bin_id]
                agent.dec_id = selected_agents[i]


    def reward(self, agent, world):
        
        rew = 0
        if agent.idx == 0:
            pos = np.reshape(np.array([agent.state.p_pos for agent in world.agents]), (self.num_agents, 1, 2))
            for i in range(self.num_agents):

                dist_to_goal_water = np.linalg.norm(pos[i, 0, :] - world.landmarks[1].state.p_pos )
                dist_to_refuel_water = np.linalg.norm(pos[i, 0, :] - world.landmarks[0].state.p_pos )
            
                dist_to_goal_lumber = np.linalg.norm(pos[i, 0, :] - world.landmarks[3].state.p_pos )
                dist_to_refuel_lumber = np.linalg.norm(pos[i, 0, :] - world.landmarks[2].state.p_pos )
                
                if (self.agent_cur_payload[0,i] > 0) and (dist_to_goal_water < self.nec_dist):    
                    rew += np.copy(self.agent_caps[0,i]) 
                    self.agent_cur_payload[0,i] = 0
                
                if (np.sum(self.agent_cur_payload[:,i]) < 1) and (dist_to_refuel_water < self.nec_dist):    
                    self.agent_cur_payload[0,i] = np.copy(self.agent_caps[0,i])
                    rew += np.copy(self.agent_caps[0,i]) 

                if (self.agent_cur_payload[1,i] > 0) and (dist_to_goal_lumber < self.nec_dist):    
                    rew += np.copy(self.agent_caps[1,i])
                    self.agent_cur_payload[1,i] = 0
                
                if (np.sum(self.agent_cur_payload[:,i]) < 1) and (dist_to_refuel_lumber < self.nec_dist):    
                    self.agent_cur_payload[1,i] = np.copy(self.agent_caps[1,i])
                    rew += np.copy(self.agent_caps[1,i]) 

            self.rew = rew
        return self.rew


    def observation(self, agent, world):
        if not self.parsed_config:
            key_list = dir(world)
            if 'config' in key_list:
                self.parse_config(world)
                print('\nConfig has been parsed.\n')
            else:
                print('Warning: config has not been parsed.\n')

        # get positions of all entities in this agent's reference frame
        agent_pos = np.array(agent.state.p_pos).flatten()
        agent_vel = np.array(agent.state.p_vel).flatten()
        agent_pay_water = np.copy(self.agent_cur_payload[0,agent.idx] > 0).flatten()
        agent_pay_lumber = np.copy(self.agent_cur_payload[1,agent.idx] > 0).flatten()
        agent_cap_water = np.copy(self.agent_caps[0,agent.idx]).flatten()
        agent_cap_lumber = np.copy(self.agent_caps[1,agent.idx]).flatten()

        # Retrieve sensing radii and positions of other n closest agents 
        n = self.num_relevant_agents
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        sorted_arg_dist = np.argsort(dists)
        relevant_agnets = [world.agents[i] for i in sorted_arg_dist[1:(1 + n)]]


        # base observation
        obs = np.concatenate((agent_pos, agent_vel, agent_pay_water, agent_pay_lumber)) 

        if self.agent_id:
            obs = np.concatenate((obs, agent.bin_id_vec))
        else:
            obs = np.concatenate((obs, agent_cap_water, agent_cap_lumber))

        if self.fully_obsrved and self.agent_id:
            obs = np.concatenate((obs, np.array([ag.bin_id_vec for ag in relevant_agnets]).flatten()))
        elif self.fully_obsrved and not self.agent_id:
            other_pay_water = np.copy(self.agent_cur_payload[0,sorted_arg_dist[1:(1 + n)]] > 0).flatten()
            other_pay_lumber = np.copy(self.agent_cur_payload[1,sorted_arg_dist[1:(1 + n)]] > 0).flatten()
            other_cap_water = np.copy(self.agent_caps[0,sorted_arg_dist[1:(1 + n)]]).flatten()
            other_cap_lumber = np.copy(self.agent_caps[1,sorted_arg_dist[1:(1 + n)]]).flatten()
            obs = np.concatenate((obs, other_pay_water, other_pay_lumber, other_cap_water, other_cap_lumber))
            

        return obs



    def done(self, agent, world):
        if world.steps >= 50:
            return True
        else:
            return False

    def info(self, agent, world):
        info = {'is_success': self.done(agent, world), 'world_steps': world.steps,
                'reward': 0, 'dists': 0}
        return info