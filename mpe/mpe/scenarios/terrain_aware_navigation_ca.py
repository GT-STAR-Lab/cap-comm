import numpy as np
from  mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import yaml

class Scenario(BaseScenario):
    def __init__(self, num_agents=3, config=None):
        self.num_agents = num_agents
        # Not used 
        # self.ideal_theta_separation = (2*np.pi)/self.num_agents # ideal theta difference between two agents 

        self.alt_thresh = 0.5
        self.succ_thres = 0.1

        # no config yet for altitudes of agents; this is randomly set here
        if (config is None) or (config == 'none'):
            self.altitudes = np.random.random((self.num_agents, ))

    def make_world(self):
        """
        Build the world with agents, goals, and landmarks
        """
        world = World()

        self.boundaries = 30 #TODO: What is a boundary?
        self.goal_locs = np.random.random((1, self.num_agents, 2)) # Not used
        world.agents = [Agent() for i in range(self.num_agents)]
        world.landmarks = [Landmark() for i in range(self.num_agents + self.boundaries)] #TODO: What is a landmark? Is it a terrain feature that constrains agent's paths?

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark   %d' % i
            landmark.collide = False # agents can't collide with landmarks.
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False # TODO: Is this used? Doesn't look like it. Note in core.py

        self.reset_world(world)
        return world

    def reset_world(self, world):

        self.agent_caps = np.zeros((2, self.num_agents)) # Not used
        for i, agent in enumerate(world.agents):
            agent.trait_dict = {}
            agent.state.p_pos = np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)]) 
            agent.trait_dict['altitude'] = self.altitudes[i] # single trait?
            agent.collide = True if self.altitudes[i] < self.alt_thresh else False
            agent.color = np.array([1.0, agent.collide * 1.0, 0]) 
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c) # Not used
            agent.silent = True # agents can't communicate their actions as part of their state
            agent.size = 0.075 
            agent.is_success = False

            agent.idx = i # agent id
        
        # A set of randomly placed landmarks which do not have a collision; I believe these are agent goal location. Thus why they have no
        # collisions.
        for i in range(self.num_agents):
            world.landmarks[i].color = np.array([.45, 0.1, .98])
            world.landmarks[i].state.p_pos =  np.array([np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)])
            world.landmarks[i].state.p_vel = np.array([0,0])
            world.landmarks[i].collide = False
            world.landmarks[i].size = 0.1

        t = np.linspace(-1.25, 1.25, self.boundaries)
        for i in range(self.num_agents,len(world.landmarks)):
            world.landmarks[i].color = np.array([0,0,1])
            # The middle of the map is segmented by this collision landmark. Only agents with altitude higher than alt_thresh can pass over 
            world.landmarks[i].state.p_pos = np.array([ t[i - self.num_agents],0])
            world.landmarks[i].state.p_vel = np.array([0,0])
            world.landmarks[i].collide = True
            world.landmarks[i].size = 0.075

        world.steps = 0

    def reward(self, agent, world):
        
        rew = 0
        for l in world.landmarks[0:self.num_agents]:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]

            # TODO: Make agents go into idle if they reach a location. This is to get success rate.
            # TODO: Make an action where agents get to choose to lock their position (they can do this once); right now, there is no measure of success,
            # since the environment performance is only measured by agent's distance to goal, not that they decide to make it and stay on goals.
            world.agents[np.argmin(dists)].is_success = True if np.min(dists) < self.succ_thres else False
            rew -= min(dists)

        self.reward = rew
        return rew


    def observation(self, agent, world):
        # positions of all entities in this agent's reference frame, because no other way to bring the landmark information
        pos = np.array(agent.state.p_pos)
        vel = np.array(agent.state.p_vel)
        trait = np.array([int(agent.trait_dict['altitude'])])
        # TODO: Change observation of goals to not be a vector; This is not scalable if the number of agents change; Unless we do a mask
        # TODO: Could agents communicate a goal if they see it in their FOV? We could do closest goals? Or a GNN to aggregate agent's relationship to goals.
        goals = np.array([entity.state.p_pos for entity in world.landmarks[:self.num_agents]]).flatten() 

        # Obtain locations of n closest agents
        # TODO: Make this a configurable parameter.
        n = 2
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        impt_agent_idx = np.argsort(dists).astype(int)
        relevant_agents = [world.agents[i] for i in impt_agent_idx[1:(1+n)]]
        other_pos = np.array([ag.state.p_pos for ag in relevant_agents]).flatten()


        ret_obs = np.concatenate((pos, vel, goals, other_pos, trait)) # agent observations: position, velocity, goals, other agent positions, trait
        return ret_obs
  
    def done(self, agent, world):
        # TODO: Make environment end after set number of steps, or all agent reach each location
        if world.steps >= 50:
            return True
        else:
            return False

    def info(self, agent, world):
        info = {'is_success': agent.is_success, 'world_steps': world.steps,
                'reward': self.reward, 'dists': 0}

        return info