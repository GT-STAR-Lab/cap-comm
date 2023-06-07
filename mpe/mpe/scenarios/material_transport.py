import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import random
import yaml

DEPOT_SIZE=0.2
CONSTRUCTION_SITE_SIZE = 0.3
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

        self.load_from_debug_agents = True
        self.config = config
        self.n_agents = self.config.n_agents
        self.local_ratio = self.config.local_ratio
        self.time_penalty = self.config.time_penalty
        self.delta_dist_to_quota_multiplier = self.config.delta_dist_to_quota_multiplier
        self.episode_number = 0

    def _build_agent(self, name, index, id, lumber_cap=5, concrete_cap=5):
        """Helper function to build an agent"""
        agent = TransportAgent()
        agent.name = name
        agent.collide = False
        agent.silent = True
        agent.id = id
        agent.index = index
        return agent
    
    def load_debug_agents(self):
        """
        Loades a set of homogeneou agents. This is used for debugging purposes
        """
        def one_hot_id(idx, N=4):
            vector = [0] * N
            vector[idx] = 1
            return ''.join(str(num) for num in vector)
        agents = []
        for i in range(self.n_agents):
            agent = self._build_agent(str(i), i, one_hot_id(i))
            agents.append(agent)
        return(agents)

    def load_agents_from_trait_distribution(self):
        """
        Resample new agents given the trait distribution specification
        """
        pass

    def load_agents_from_predefined_coalitions(self):
        "Load a set of agents (as a coalition) from predefined coalitions"
        pass
    
    def load_agents_from_predefined_agents(self):
        "Load a new coalition that is composed of predefined agents"
        pass
    
    def load_agents(self):
        """Load agents according to the loading scheme"""
        # load the agents according to the loading scheme
        if(self.load_from_debug_agents):
            agents = self.load_debug_agents()

        elif self.config.load_from_predefined_coalitions:
            agents = self.load_agents_from_predefined_coalitions()
        
        elif(self.config.load_from_predefined_agents):
            agents = self.load_agents_from_predefined_agents()
        
        else:
            agents = self.load_agents_from_trait_distribution()
        
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

        world.agents = self.load_agents()
        self.lumber_quota = 30
        self.concrete_quota = 10

        self.reset_world()
        return(world)
    
def reset_world(self, world):
    """reset the world / episode"""
    self.episode_number += 1

    if self.config.resample and (self.episode_number % self.config.resample_frequency == 0):
        world.agents = self.load_agents()

    world.steps = 0
    self.lumber_delivered = 0
    self.concrete_delivered = 0
    self.prev_dist_to_quota = np.linalg.norm(np.array([self.lumber_quota, self.concrete_quota]))
    self.quota_filled = False

def reward(self, agent, world):
    """
    reward function for agents. The reward for each agent is a combination of local reward
    and the global reward
    """
    rew = 0
    pos = agent.state.p_pos

    # compute distance to lumber and concrete site
    dist_to_lumber_landmark = np.linalg.norm(pos - world.landmarks[0].state.p_pos) - DIST_THRESHOLD
    dist_to_concrete_landmark = np.linalg.norm(pos - world.landmarks[1].state.p_pos) - DIST_THRESHOLD 
    dist_to_construction_site_landmark = np.linalg(pos - world.landmarks[2].state.p_pos)

    local_reward = 0
    
    if(agent.empty()):
        local_reward -= min(dist_to_lumber_landmark, dist_to_concrete_landmark)
        if(dist_to_lumber_landmark < DIST_THRESHOLD): # pick up lumber
            agent.load("lumber")
        elif(dist_to_concrete_landmark < DIST_THRESHOLD): # pick up concrete
            agent.load("concrete")
    else:
        local_reward -= dist_to_construction_site_landmark
        if(dist_to_construction_site_landmark < DIST_THRESHOLD):
            lumber, concrete = agent.unload()
            self.lumber_delivered += lumber
            self.concrete_delivered += concrete

    
    # global reward is added if the distance to quota changes
    actual_dist_to_quota = np.linalg.norm(np.array([self.lumber_delivered, self.concrete_delivered]) - 
                                            np.array([self.lumber_quota, self.concrete_quota]))
    change_in_dist_to_quota = (self.prev_dist_to_quota - actual_dist_to_quota)
    global_reward = (change_in_dist_to_quota*self.dist_to_quota_multipler) + self.time_penalty
    self.prev_dist_to_quota = actual_dist_to_quota

    rew = self.local_ratio * local_reward + (1 - self.local_ratio)*global_reward # / self.n_agents
    return rew

def observation(self, agent, world):
    """Get the observation for each robot"""
    observation = []
    agent_pos = (agent.state.p_pos).tolist(); agent_vel = (agent.state.p_vel).tolist()
    agent_capabilities = [agent.lumber_capacity, agent.concrete_capacity]
    dists_to_resource_quota = [(self.lumber_quota - self.lumber_delivered) / self.lumber_quota, 
                               (self.concrete_quota - self.concrete_delivered) / self.concrete_quota]
    agents_current_load = [agent.lumber_loaded, agent.concrete_loaded]
    
    base_observation = [*agent_pos, *agent_vel, *dists_to_resource_quota, *agents_current_load]
    
    if(self.config.capability_aware): # append capability    
        observation = base_observation + [*agent_capabilities]
    elif(self.config.agent_id): # append agent id
        agent_id = [int(bit) for bit in agent.id]
        observation = base_observation + [*agent_id]
    else:
        observation = base_observation

    return observation

def done(self, agent, world):
    if(world.steps >= self.config.max_steps):
        self.quota_filled = False
        return True
    
    # if the quota is met
    elif((self.lumber_quota - self.lumber_delivered) < 0 and  (self.concrete_quota - self.concrete_delivered) < 0):
        self.quota_filled = True
        return True
    else:
        self.quota_filled = False
        return False
    
def info(self, agent, world):
    info = {'is_success': self.done(agent, world), 'world_steps': world.steps,
                'reward': 0, 'quota_filled': self.quota_filled, 'lumber_deliverd': self.lumber_deliverd,
                'concrete_delivered': self.concrete_deliverd}
    return info

    

