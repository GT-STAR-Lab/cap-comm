import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import random
import yaml
random.seed(0)

# 0 is water
# 1 is lumber

class Scenario(BaseScenario):
    def __init__(self, num_agents=4, config=None):
        self.num_agents = num_agents
        self.nec_dist = 0.2
        if (config is None) or (config == 'none'):
            self.num_water = self.num_agents // 2
            self.num_lumber = self.num_agents - self.num_water
            self.num_water_lumber = 0 
            self.num_none = 0
            self.num_agents_trained = self.num_agents
            self.lumber_caps = np.random.random((self.num_agents))
            self.water_caps = np.random.random((self.num_agents))


    def make_world(self):

        world = World()
        world.landmarks = 4
        self.agent_caps = np.zeros((2, self.num_agents))
        self.agent_caps[0,:] = np.array(self.lumber_caps)
        self.agent_caps[1,:] = np.array(self.water_caps)
        

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
        return world

    def reset_world(self, world):

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
                

                if (self.agent_cur_payload[0,i] < self.agent_caps[0,i]) and (dist_to_refuel_water < self.nec_dist):    
                    self.agent_cur_payload[0,i] = np.copy(self.agent_caps[0,i])
                    rew += np.copy(self.agent_caps[0,i]) 

                if (self.agent_cur_payload[1,i] > 0) and (dist_to_goal_lumber < self.nec_dist):    
                    rew += np.copy(self.agent_caps[1,i])
                    self.agent_cur_payload[1,i] = 0
                

                if (self.agent_cur_payload[1,i] < self.agent_caps[1,i]) and (dist_to_refuel_lumber < self.nec_dist):    
                    self.agent_cur_payload[1,i] = np.copy(self.agent_caps[1,i])
                    rew += np.copy(self.agent_caps[1,i]) 

            self.rew = rew

        return self.rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        agent_pos = np.array(agent.state.p_pos).flatten()
        agent_pay_water = np.copy(self.agent_cur_payload[0,agent.i] > 0).flatten()
        agent_pay_lumber = np.copy(self.agent_cur_payload[1,agent.i] > 0).flatten()
        agent_cap_water = np.copy(self.agent_caps[0,agent.idx] > 0).flatten()
        agent_cap_lumber = np.copy(self.agent_caps[1,agent.idx] > 0).flatten()

        # Retrieve sensing radii and positions of other n closest agents 
        n = 2
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        sorted_arg_dist = np.argsort(dists)
        relevant_agnets = [world.agents[i] for i in sorted_arg_dist[1:(1 + n)]]

        ret_obs = np.concatenate((agent_pos, agent_pay_water, agent_pay_lumber, agent_cap_water, agent_cap_lumber))

        return ret_obs


    def done(self, agent, world):
        if world.steps >= 50:
            return True
        else:
            return False

    def info(self, agent, world):
        info = {'is_success': self.done(agent, world), 'world_steps': world.steps,
                'reward': 0, 'dists': 0}
        return info