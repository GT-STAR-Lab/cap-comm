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

        if (config is None) or (config == 'none'):
            self.altitudes = np.random.random((self.num_agents, ))

  


    def make_world(self):
        world = World()

        self.boundaries = 30
        self.goal_locs = np.random.random((1, self.num_agents, 2))
        world.agents = [Agent() for i in range(self.num_agents)]
        world.landmarks = [Landmark() for i in range(self.num_agents + self.boundaries)]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark   %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False

        self.reset_world(world)
        return world

    def reset_world(self, world):

        self.agent_caps = np.zeros((2, self.num_agents))
        for i, agent in enumerate(world.agents):
            agent.trait_dict = {}
            agent.state.p_pos = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)]) 
            agent.trait_dict['can_fly'] =  True if self.agent_caps[0,i] == 1 else False
            agent.trait_dict['can_ground'] = True if self.agent_caps[1,i] == 1 else False
            agent.trait_dict['altitude'] = self.altitudes[i]
            agent.collide = True if self.altitudes[i] < self.alt_thresh else False
            agent.color = np.array([1.0, agent.trait_dict['can_fly'], 0]) 
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.silent = True
            agent.size = 0.075 
            agent.is_success = False

            agent.idx = i
            
        for i in range(self.num_agents):
            world.landmarks[i].color = np.array([.45, 0.1, .98])
            # if i < self.num_flyers:
            #     world.landmarks[i].state.p_pos = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=0.25, high=1)])
            # else:              
            #     world.landmarks[i].state.p_pos = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=-.25)])

            world.landmarks[i].state.p_pos =  np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)])

            world.landmarks[i].state.p_vel = np.array([0,0])
            world.landmarks[i].collide = False
            world.landmarks[i].size = 0.1

        t = np.linspace(-1.25, 1.25, self.boundaries)
        for i in range(self.num_agents,len(world.landmarks)):
            world.landmarks[i].color = np.array([0,0,1])
            world.landmarks[i].state.p_pos = np.array([ t[i - self.num_agents],0])
            world.landmarks[i].state.p_vel = np.array([0,0])
            world.landmarks[i].collide = True
            world.landmarks[i].size = 0.075

        world.steps = 0

    def reward(self, agent, world):

        rew = 0
        for l in world.landmarks[0:self.num_agents]:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            world.agents[np.argmin(dists)].is_success = True if np.min(dists) < self.succ_thres else False
            rew -= min(dists)

        self.reward = rew
        return rew


    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
        pos = np.array(agent.state.p_pos)
        trait = np.array([int(agent.trait_dict['can_fly'])])
        goals = np.array([entity.state.p_pos for entity in world.landmarks[:self.num_agents]]).flatten()
        ret_obs = np.concatenate((pos, trait, goals))
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