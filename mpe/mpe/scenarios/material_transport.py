import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import random
import yaml
import copy
import os

LUMBER_DEPOT = np.array([0.5, 0.5])
CONCRETE_DEPOT = np.array([-0.5, 0.5])
CONSTRUCTION_SITE = np.array([0.0, -0.5])
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

        self.load_from_debug_agents = config.load_from_debug_agent_func # Set true if you are trying to debug the environment. Simplist agent to resource quota.
        self.config = config
        self.n_agents = self.config.n_agents
        
        self.lumber_pickup_reward, self.concrete_pickup_reward = self.config.lumber_pickup_reward, self.config.concrete_pickup_reward
        self.dropoff_reward = self.config.dropoff_reward
        self.quota_filled_reward_scalar = self.config.quota_filled_reward_scalar
        self.time_penalty = self.config.time_penalty
        self.surplus_penalty_scalar = self.config.surplus_penalty_scalar

        # lower and upper quotas (absolute) considered for a single agent
        # these values will be multiplied by n_agents to scale to the number of agents
        # better
        self.lower_quota_limit =0.5 * self.n_agents
        self.upper_quota_limit = 2.0 * self.n_agents

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
            lumber_cap = getattr(np.random, self.config.traits["lumber"]["distribution"])(**func_args)
            concrete_cap = 1.0 - lumber_cap
            # concrete_cap = getattr(np.random, self.config.traits["concrete"]["distribution"])(**func_args)
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
            
        return(agents)
    
    def _initialize_quota(self, agents, type="random"):
        """
        This function should be called after agents are initialized, and is used to determine
        a quota that is possible.
        """
        # at least one exact solution that requires only a single trip
        if(type=="single_exact"):
            lumber_quota, concrete_quota = 0, 0
            for agent in agents:
                which_resource = np.random.randint(0, 2)
                if(which_resource == 0):
                    lumber_quota += agent.lumber_cap
                else:
                    concrete_quota += agent.concrete_cap

            self.lumber_quota = max(1, lumber_quota); self.concrete_quota = max(1, concrete_quota) 
        
        elif(type=="debug"):
            self.lumber_quota=2
            self.concrete_quota=1

        elif(type=="random"):
            self.lumber_quota = np.random.randint(self.lower_quota_limit, self.upper_quota_limit)
            self.concrete_quota = np.random.randint(self.lower_quota_limit, self.upper_quota_limit)
        elif(type=="fixed"):
            self.lumber_quota = self.upper_quota_limit
            self.concrete_quota = self.upper_quota_limit

    def make_world(self):

        world = World()
        lumber_landmark = Landmark()
        lumber_landmark.name = "agent_lumber depot"
        lumber_landmark.collide = False; lumber_landmark.movable = False
        lumber_landmark.size = DEPOT_SIZE
        lumber_landmark.color = np.array([255, 0, 0]) / 255 # yellowish brown
        lumber_landmark.state.p_pos = np.array([0.5, 0.5]); lumber_landmark.state.p_vel = np.array([0., 0.])

        concrete_landmark = Landmark()
        concrete_landmark.name = "agent_concrete depot"
        concrete_landmark.collide = False; concrete_landmark.movable = False
        concrete_landmark.size = DEPOT_SIZE
        concrete_landmark.color = np.array([0, 0, 255]) / 255 # gray
        concrete_landmark.state.p_pos = np.array([-0.5, 0.5]); concrete_landmark.state.p_vel = np.array([0., 0.])

        construction_site_landmark = Landmark()
        construction_site_landmark.name = "agent_construction site"
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
        
        
        self._initialize_quota(self.team, type="random")
        self.team = self.reinitialize_positions(self.team)
        
        world.agents = self.team

        world.steps = 0
        # keep track of amount of lumber and concrete
        self.lumber_delivered = 0
        self.concrete_delivered = 0

        # quota met booleans
        self.total_quota_filled = False
        self.lumber_quota_filled = False
        self.concrete_quota_filled = False

    def reward(self, agent, world):
        """
        reward function for agents. The reward for each agent is a combination of local reward
        and the global reward
        """
        rew = 0
        pos = agent.state.p_pos
        agent_index = agent.index

        # compute distance to lumber, concrete site, and construction site
        dist_to_lumber_landmark = np.linalg.norm(pos - world.landmarks[0].state.p_pos)
        dist_to_concrete_landmark = np.linalg.norm(pos - world.landmarks[1].state.p_pos)
        dist_to_construction_site_landmark = np.linalg.norm(pos - world.landmarks[2].state.p_pos)
        local_reward = 0

        # Agents load up on lumber and concrete at depot. Rewarded only if quotas haven't been filled. 
        if(self.team[agent_index].empty()):
            
            if(dist_to_lumber_landmark < DIST_THRESHOLD): # pick up lumber
                self.team[agent_index].load("lumber")
                local_reward += self.lumber_pickup_reward if (self.team[agent_index].lumber_cap > 0.0 and not self.lumber_quota_filled) else 0.0

            elif(dist_to_concrete_landmark < DIST_THRESHOLD): # pick up concrete
                self.team[agent_index].load("concrete")
                local_reward += self.concrete_pickup_reward if (self.team[agent_index].concrete_cap > 0.0 and not self.concrete_quota_filled) else 0.0

        else:
            # reward agents positively only if they contribute to quota before it is filled
            if(dist_to_construction_site_landmark < DIST_THRESHOLD):
                
                lumber, concrete = self.team[agent_index].unload()
                lumber_required = self.lumber_quota - self.lumber_delivered
                concrete_required = self.concrete_quota - self.concrete_delivered
                pos_lumber_addition = min(0, lumber - lumber_required)
                pos_concrete_addition = min(0, concrete - concrete_required)

                self.lumber_delivered += lumber
                self.concrete_delivered += concrete
                lumber_surplus = self.lumber_delivered - self.lumber_quota
                concrete_surplus = self.concrete_delivered - self.concrete_quota
                
                # only reward/penalize if the agent drops off lumber or concrete
                if(lumber > 0):
                    if(lumber_surplus > 0.01):
                        local_reward += (-lumber + pos_lumber_addition) * self.surplus_penalty_scalar
                        self.lumber_quota_filled = True
                    if(pos_lumber_addition > 0):
                        local_reward += self.dropoff_reward
                if(concrete > 0):
                    if(concrete_surplus > 0.01):
                        local_reward += (-concrete + pos_concrete_addition) * self.surplus_penalty_scalar
                        self.concrete_quota_filled = True
                    if(pos_concrete_addition > 0):
                        local_reward += self.dropoff_reward
                
                # finished meeting quota
                if(self.concrete_quota_filled and self.lumber_quota_filled):
                    self.total_quota_filled = True
                    local_reward += self.quota_filled_reward_scalar * (self.concrete_quota + self.lumber_quota)

        if(not self.total_quota_filled):
            rew = local_reward + self.time_penalty
        else:
            rew = local_reward
        return rew

    def observation(self, agent, world):
        """Get the observation for each robot. The observation space
        is approximately normalized.
        """
        observation = []
        agent_pos = (agent.state.p_pos).tolist(); agent_vel = (agent.state.p_vel).tolist()
        agent_capabilities = [agent.lumber_cap, agent.concrete_cap]

        # state of the quota and how much it is filled
        resource_status = [self.lumber_quota / self.upper_quota_limit,
                           self.lumber_delivered / self.upper_quota_limit,
                           self.concrete_quota / self.upper_quota_limit,
                           self.concrete_delivered / self.upper_quota_limit]

        # state for how much the agent is holding.
        agents_current_load = [agent.lumber_loaded, agent.concrete_loaded]

        # distance to lumber, concrete depot, and construction site
        dist_lumber_depot = list(agent_pos - LUMBER_DEPOT); dist_concrete_depot = list(agent_pos - CONCRETE_DEPOT); dist_construction_site = list(agent_pos - CONSTRUCTION_SITE)
        
        # build the base observations (i.e. without capabilities or ID)
        base_observation = [*agent_pos, *agent_vel, *dist_lumber_depot, *dist_concrete_depot, *dist_construction_site,
                            *resource_status, *agents_current_load]
        
        if(self.config.capability_aware): # append capability    
            observation = base_observation + [*agent_capabilities]
        elif(self.config.agent_id): # append agent id
            agent_id = [int(bit) for bit in agent.id]
            observation = base_observation + [*agent_id]
        else:
            observation = base_observation

        return np.array(observation)

    def done(self, agent, world):
        
        if(self.total_quota_filled):
            return True
        else:
            return False
        
    def info(self, agent, world):
        lumber_remaining_percentage = (self.lumber_quota - self.lumber_delivered) / self.lumber_quota
        concrete_remaining_percentage = (self.concrete_quota - self.concrete_delivered) / self.concrete_quota
        lumber_surplus = self.lumber_delivered - self.lumber_quota
        concrete_surplus = self.concrete_delivered - self.concrete_quota

        info = {'total_quota_filled': self.total_quota_filled, 'lumber_remaining (%)': lumber_remaining_percentage,
                    'concrete_remaining (%)': concrete_remaining_percentage,
                    'lumber_surplus': lumber_surplus, 'concrete_surplus': concrete_surplus,
                    'lumber_quota_filled': self.lumber_quota_filled, 'concrete_quota_filled': self.concrete_quota_filled,
                    'lumber_quota': self.lumber_quota, 'concrete_quota': self.concrete_quota,
                    'lumber_delivered': self.lumber_delivered, 'concrete_delivered': self.concrete_delivered}
        return info

    

