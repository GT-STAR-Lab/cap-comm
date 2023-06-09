import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import random
import yaml
import copy
import os

DEPOT_SIZE=0.2
CONSTRUCTION_SITE_SIZE = 0.2
DIST_THRESHOLD = 0.2

class TransportAgent(Agent):
    def __init__(self):
        super(TransportAgent, self).__init__()
        self.id = None
        self.index = None
        self.lumber_cap = 5
        self.concrete_cap = 5
        self.lumber_loaded = 0
        self.concrete_loaded = 0
        self.done = False
        
    def set_done(self):
        self.done = True

    def unload(self):
        """Resets the ability to load and concrete"""
        l, c = self.lumber_loaded+0, self.concrete_loaded+0
        self.lumber_loaded = 0; self.concrete_loaded = 0
        return l, c
    
    def load(self, choice):
        """Load either lumber or concrete"""
        if(choice=="lumber" and self.lumber_loaded == 0):
            self.lumber_loaded = self.lumber_cap
        elif(choice=="concrete" and self.concrete_loaded == 0):
            self.concrete_loaded = self.concrete_cap
        else:
            print("Cannot load if agent is alread loaded")
            raise RuntimeError
    def empty(self):
        return (self.lumber_loaded == 0 and self.concrete_loaded == 0)
    

class Scenario(BaseScenario):
    def __init__(self, config=None):
        """Initialize the environment"""

        self.load_from_debug_agents = False
        self.config = config
        self.n_agents = self.config.n_agents
        self.local_ratio = self.config.local_ratio
        self.time_penalty = self.config.time_penalty
        self.delta_dist_to_quota_multiplier = self.config.delta_dist_to_quota_multiplier
        self.episode_number = 0
        self.team = []
        self.max_cap = self.config.traits["lumber"]["high"] - 1

        # load coalitions to train on
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{current_dir}/configs/material_transport/{self.config.coalition_file}", "r") as f:
            self.predefined_coalitions = yaml.safe_load(f)

    def _build_agent(self, world, name, index, id, inital_pos=np.array([0., 0.]), lumber_cap=5, concrete_cap=5):
        """Helper function to build an agent"""
        agent = TransportAgent()
        agent.name = name
        agent.collide = False
        agent.silent = True
        agent.id = id
        agent.index = index

        max_cap = lumber_cap + concrete_cap
        agent.lumber_cap = lumber_cap; agent.concrete_cap = concrete_cap
        agent.color = np.array([lumber_cap, 0, concrete_cap]) / max_cap
        agent.state.p_pos = inital_pos
        agent.state.p_vel = np.zeros(2)
        agent.state.c = np.zeros(world.dim_c)
        return agent
    
    def load_debug_agents(self, world):
        """
        Loades a set of homogeneou agents. This is used for debugging purposes
        """
        def one_hot_id(idx, N=4):
            vector = [0] * N
            vector[idx] = 1
            return ''.join(str(num) for num in vector)
        agents = []
        for i in range(self.n_agents):

            initial_pos = np.array([0.0, -0.5 + np.random.uniform(-0.4, 0.4)]) # start agents at construction site
            if(i%2):
                lumber_cap=1
                concrete_cap=0
            else:
                lumber_cap=0
                concrete_cap=1
            agent = self._build_agent(world, str(i), i, one_hot_id(i), 
                                      inital_pos=initial_pos, lumber_cap=lumber_cap, concrete_cap=concrete_cap)
            agents.append(agent)
        return(agents)

    def reinitialize_positions(self, team):
        team_ = []
        for agent in team:
            agent.state.p_pos = np.array([0.0, -0.5 + np.random.uniform(-0.2, 0.2)]) # start agents at construction site
            team_.append(agent)
        return(team_)

    def load_agents_from_trait_distribution(self, world):
        """
        Resample new agents given the trait distribution specification
        """
        agents = []
        func_args = copy.deepcopy(self.config.traits['lumber'])
        del func_args['distribution']
        for idx in range(self.n_agents):
            lumber_cap = int(getattr(np.random, self.config.traits["lumber"]["distribution"])(**func_args))
            concrete_cap = self.max_cap - lumber_cap
            default_id = ['0'] * (self.n_agents * self.config.n_coalitions)
            agent = self._build_agent(world, str(idx), idx, default_id, 
                                       lumber_cap=lumber_cap, concrete_cap=concrete_cap)
            agents.append(agent)
        return(agents)

    def load_agents_from_predefined_coalitions(self, world):
        "Load a set of agents (as a coalition) from predefined coalitions"
        t = "train"
        agents = []
        coalition_idx = np.random.randint(self.config.n_coalitions)
        s = str(self.n_agents) + "_agents"
        coalition = self.predefined_coalitions[t]["coalitions"][s][coalition_idx]
        idx = 0
        for _, agent in coalition.items():
            lumber_cap, concrete_cap = agent["lumber_cap"], agent["concrete_cap"]
            id = agent["id"]
            a = self._build_agent(world, str(idx), idx, id, 
                                       lumber_cap=lumber_cap, concrete_cap=concrete_cap)
            idx += 1
            agents.append(a)
        return(agents)

    def load_agents_from_predefined_agents(self, world):
        "Load a new coalition that is composed of predefined agents"
        t = "train"
        agent_pool = []
        agents = []

        s = "4_agents"
        for coalition_idx in range(self.config.n_coalitions):
            coalition = self.predefined_coalitions[t]["coalitions"][s][coalition_idx]

            idx = 0
            for _, agent in coalition.items():
                lumber_cap, concrete_cap = agent["lumber_cap"], agent["concrete_cap"]
                id = agent["id"]
                a = self._build_agent(world, str(idx), idx, id, 
                                        lumber_cap=lumber_cap, concrete_cap=concrete_cap)
                idx += 1
                agent_pool.append(a)
        index = 0
        for i in range(self.n_agents):
            agent = random.choice(agent_pool)
            agent.index = index
            agents.append(agent)
            index += 1
        return(agents)
        
    def load_agents(self, world):
        """Load agents according to the loading scheme"""
        # load the agents according to the loading scheme
        if(self.load_from_debug_agents):
            agents = self.load_debug_agents(world)

        elif self.config.load_from_predefined_coalitions:
            agents = self.load_agents_from_predefined_coalitions(world)
        
        elif(self.config.load_from_predefined_agents):
            agents = self.load_agents_from_predefined_agents(world)
        
        else:
            agents = self.load_agents_from_trait_distribution(world)

        # now with the agents loaded, lets build a random possible quota
        lumber_quota, concrete_quota = 1, 1
        for agent in agents:
            which_resource = np.random.randint(0, 2)
            if(which_resource == 0):
                lumber_quota += agent.lumber_cap
            else:
                concrete_quota += agent.concrete_cap

        self.lumber_quota = lumber_quota; self.concrete_quota = concrete_quota 

        
        return(agents)

    def make_world(self):

        world = World()
        lumber_landmark = Landmark()
        lumber_landmark.name = "lumber depot"
        lumber_landmark.collide = False; lumber_landmark.movable = False
        lumber_landmark.size = DEPOT_SIZE
        lumber_landmark.color = np.array([255, 205, 153]) / 255 # yellowish brown
        lumber_landmark.state.p_pos = np.array([0.5, 0.5]); lumber_landmark.state.p_vel = np.array([0., 0.])

        concrete_landmark = Landmark()
        concrete_landmark.name = "concrete depot"
        concrete_landmark.collide = False; concrete_landmark.movable = False
        concrete_landmark.size = DEPOT_SIZE
        concrete_landmark.color = np.array([128, 128, 128]) / 255 # gray
        concrete_landmark.state.p_pos = np.array([-0.5, 0.5]); concrete_landmark.state.p_vel = np.array([0., 0.])

        construction_site_landmark = Landmark()
        construction_site_landmark.name = "construction site"
        construction_site_landmark.collide = False; construction_site_landmark.movable = False
        construction_site_landmark.size = CONSTRUCTION_SITE_SIZE
        construction_site_landmark.color = np.array([153, 204, 0]) / 255
        construction_site_landmark.state.p_pos = np.array([0.0, -0.5]); construction_site_landmark.state.p_vel = np.array([0., 0.])

        world.landmarks = [lumber_landmark, concrete_landmark, construction_site_landmark]
        self.team = self.load_agents(world)
        world.agents = self.team

        self.reset_world(world)
        return(world)
    
    def reset_world(self, world):
        """reset the world / episode"""
        self.episode_number += 1

        if self.config.resample and (self.episode_number % self.config.resample_frequency == 0):
            self.team = self.load_agents(world)
        
        
        self.team = self.reinitialize_positions(self.team)
        
        world.agents = self.team

        world.steps = 0
        self.lumber_delivered = 0
        self.concrete_delivered = 0
        self.quota_filled = False

    def reward(self, agent, world):
        """
        reward function for agents. The reward for each agent is a combination of local reward
        and the global reward
        """
        rew = 0
        pos = agent.state.p_pos
        agent_index = agent.index

        # compute distance to lumber and concrete site
        dist_to_lumber_landmark = np.linalg.norm(pos - world.landmarks[0].state.p_pos)
        dist_to_concrete_landmark = np.linalg.norm(pos - world.landmarks[1].state.p_pos)
        dist_to_construction_site_landmark = np.linalg.norm(pos - world.landmarks[2].state.p_pos)

        local_reward = 0

        # agents don't get to load or unload lumber after they finish
        
        if(self.team[agent_index].empty()):
            # local_reward -= min(dist_to_lumber_landmark, dist_to_concrete_landmark)
            if(dist_to_lumber_landmark < DIST_THRESHOLD): # pick up lumber
                self.team[agent_index].load("lumber")
                local_reward += 0.1 if self.team[agent_index].lumber_cap > 0.0 else 0.0
                
            elif(dist_to_concrete_landmark < DIST_THRESHOLD): # pick up concrete
                self.team[agent_index].load("concrete")
                local_reward += 0.1 if self.team[agent_index].concrete_cap > 0.0 else 0.0
                
        else:
            # reward agents based on how much they contribute towards meeting the quota
            # if they help a lot, then they get more reward. If they overshoot, they
            # get penalized.
            if(dist_to_construction_site_landmark < DIST_THRESHOLD):
                
                lumber, concrete = self.team[agent_index].unload()
                
                l_rew = ((lumber + self.lumber_delivered) / self.lumber_quota)
                c_rew = ((concrete + self.concrete_delivered) / self.concrete_quota)
                if(l_rew > 1.0):
                    l_rew = 0.25*(1 - l_rew)
                    
                if(c_rew > 1.0):
                    c_rew = 0.25*(1 - c_rew)

                local_reward += l_rew + c_rew

                self.lumber_delivered += lumber
                self.concrete_delivered += concrete
        
        # big reward if you meet the total quota
        if((self.lumber_quota - self.lumber_delivered) <= 0.01 and  (self.concrete_quota - self.concrete_delivered) <= 0.01):
            local_reward += 5.0

        rew = local_reward + self.time_penalty
        return rew

    def observation(self, agent, world):
        """Get the observation for each robot"""
        observation = []
        agent_pos = (agent.state.p_pos).tolist(); agent_vel = (agent.state.p_vel).tolist()
        agent_capabilities = [agent.lumber_cap, agent.concrete_cap]

        lumber_remaining_percentage = (self.lumber_quota - self.lumber_delivered) / self.lumber_quota
        concrete_remaining_percentage = (self.concrete_quota - self.concrete_delivered) / self.concrete_quota

        dists_to_resource_quota = [lumber_remaining_percentage, 
                                    concrete_remaining_percentage]
        agents_current_load = [agent.lumber_loaded, agent.concrete_loaded]
        
        base_observation = [*agent_pos, *agent_vel, *dists_to_resource_quota, *agents_current_load, float(agent.done)]
        
        if(self.config.capability_aware): # append capability    
            observation = base_observation + [*agent_capabilities]
        elif(self.config.agent_id): # append agent id
            agent_id = [int(bit) for bit in agent.id]
            observation = base_observation + [*agent_id]
        else:
            observation = base_observation

        return np.array(observation)

    def done(self, agent, world):

        if((self.lumber_quota - self.lumber_delivered) <= 0.01 and  (self.concrete_quota - self.concrete_delivered) <= 0.01):
            self.quota_filled = True
            return True
        
        # elif(agent.done):
        #     self.quota_filled = False
        #     return True
        else:

            self.quota_filled = False
            return False
        
    def info(self, agent, world):
        lumber_remaining_percentage = (self.lumber_quota - self.lumber_delivered) / self.lumber_quota
        concrete_remaining_percentage = (self.concrete_quota - self.concrete_delivered) / self.concrete_quota

        info = {'quota_filled': self.quota_filled, 'lumber_remaining (%)': lumber_remaining_percentage,
                    'concrete_remaining (%)': concrete_remaining_percentage}
        return info

    

