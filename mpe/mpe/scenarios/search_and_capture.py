import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
import time

class Scenario(BaseScenario):
    def __init__(self, num_sensing = 2, num_capture = 2, num_both = 1, num_landmarks = 4, ca = True):
        self.sensing = num_sensing
        self.capture = num_capture
        self.both = num_both #agents with both sensing capture capability
        self.num_agents = self.sensing + self.capture + self.both
        self.L = num_landmarks #Number of landmarks
        self.ca = ca #ca stands for capability aware

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0 #IDK what this is
        world.collaborative = False  # Force them to be collaborative in how reward is structured

        self.gray = np.array([.55,.55,.55])
        self.green = np.array([.25,.75,.25])
        self.red = np.array([.75,.25,.25])

        self.blue = np.array([.25,.25,.75])
        self.yellow = np.array([.75,.75,.05])
        self.orange = np.array([.75,.5,.05])

        self.iterations = 0
        self.mid = .2 #Sensing radius will be greater than this, capture radius will be less
        self.scale = .1 #sensing r will be random*scale+mid, capture r will be mid-random*scale
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.trait_dict = {}
            agent.name = 'agent %d' % i
            agent.collide = False
            if i < self.sensing: #sensing
                agent.color = self.blue
            elif i < self.sensing + self.capture: #capture
                agent.color = self.yellow
            else: #both
                agent.color = self.orange
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.L)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):                             
        # set initial states
        for i, agent in enumerate(world.agents):
            agent.idx = i
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.sensed_prey = np.array([-9999, -9999])
            agent.silent = True
            
            if i < self.sensing:
                agent.trait_dict['capture_radius'] = 0
                agent.trait_dict['sensing_radius'] = self.mid + np.random.random() * self.scale
            elif i < self.sensing + self.capture:
                agent.trait_dict['capture_radius'] = self.mid - np.random.random() * self.scale
                agent.trait_dict['sensing_radius'] = 0
            else:
                agent.trait_dict['capture_radius'] = self.mid - np.random.random() * self.scale
                agent.trait_dict['sensing_radius'] = self.mid + np.random.random() * self.scale

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = self.gray
            landmark.size = 0.1

    def reward(self, agent, world):
        '''
        The way the reward currently works
        For each landmark:
            If the landmark has already been captured (is green), reward of 0
            If the landmark has been sensed (is red) then:
                If at least one agent is within its capture radius -> reward += 10
                If no agent is within its capture radius:
                    For each agent reward += max(-.1 * (dist/agent.sensing_radius), -.1)
            If the landmark is gray:
                For each agent reward += max(-.1 * (dist/agent.sensing_radius), -.1)
        Reward is shared across all agents
        '''
        if agent.idx == 0:
            reward = 0
            for l in world.landmarks:
                if (l.color == self.green).all(): #prey has been fully caught
                    continue
                new_color = l.color
                landmark_reward = 0
                for a in world.agents:
                    dist = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                    if (l.color == self.red).all(): #prey has already been sensed so can be captured
                        if dist < a.trait_dict['capture_radius']:
                            landmark_reward = 10 #capture gives a large reward
                            new_color = self.green
                            break
                        else:
                            if dist < a.trait_dict['sensing_radius']:
                                landmark_reward -= .1 * (dist/a.trait_dict['sensing_radius']) #The closer the sensing agent is to the landmark, the lower its punishment
                            else:
                                landmark_reward -= .1
                    else: #color must be gray
                        if dist < a.trait_dict['sensing_radius']:
                            landmark_reward -= .1 * (dist/a.trait_dict['sensing_radius'])
                            new_color = self.red #prey has now been sensed so can be captured
                        else:
                            landmark_reward -= .1
                reward += landmark_reward
                l.color = new_color
            self.reward = reward
            print(self.reward)
        return self.reward

    def observation(self, agent, world):
        '''
        Current observations
        If no capability awareness:
            (agent id, agent position, agent velocity, nearest prey's location if within sensing radius)
            concatenated with n nearest neighbor's (id, position, nearest prey's location if within sensing radius)
        If capability aware:
            (agent position, agent velocity, agent capture radius, agent sensing radius, nearest prey's location if within sensing radius)
            concatenated with n nearest neighbor's (position, capture radius, sensing radius, nearest prey's location if within sensing radius)

        If there is no prey within the agent's sensing radius, this field defaults to [-9999, -9999]. TODO: does this mess with the gradients too much?
        '''
        agent_pos = np.array(agent.state.p_pos)
        agent_vel = np.array(agent.state.p_vel)
        
        #Find the location of the nearest prey if within sensing radius
        sensed_prey = np.array([0, 0])
        if agent.trait_dict['sensing_radius'] > 0:
            dists = np.array([np.sqrt(np.sum(np.square(l.state.p_pos - agent.state.p_pos))) for l in world.landmarks])
            poses = np.array([ l.state.p_pos.flatten() for l in world.landmarks])
            if np.amin(dists) <= agent.trait_dict['sensing_radius']:
                sensed_prey = poses[np.argmin(dists)]
        agent.sensed_prey = sensed_prey

        #gather some information from n nearest neighbors.
        n = 2
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.agents]
        impt_agent_idx = np.argsort(dists).astype(int)
        relevant_agents = [world.agents[i] for i in impt_agent_idx[1:(1+n)]]
        other_pos = np.array([ag.state.p_pos for ag in relevant_agents]).flatten()
        other_sensed_prey = np.array([ag.sensed_prey for ag in relevant_agents]).flatten()

        #Combine the data into a single array. Includes the capabilities or the agent's id based on self.ca
        if not self.ca:
            #agent_idx = np.array([agent.idx])#.flatten()
            #other_idx =  np.array([ag.idx for ag in relevant_agents]).flatten()
            obs = np.concatenate((agent_idx, agent_pos, agent_vel, sensed_prey, other_idx, other_pos, other_sensed_prey))
        else:
            capture_radius = np.array([(agent.trait_dict['capture_radius'])])
            sensing_radius = np.array([(agent.trait_dict['sensing_radius'])])
            other_sensing_radius = np.array([np.array([(agent.trait_dict['sensing_radius'])]) for ag in relevant_agents]).flatten()
            other_capture_radius = np.array([np.array([(agent.trait_dict['capture_radius'])]) for ag in relevant_agents]).flatten()
            obs = np.concatenate((agent_pos, agent_vel, capture_radius, sensing_radius, sensed_prey, other_pos, other_capture_radius, other_sensing_radius, other_sensed_prey))

        return obs
    
    def done(self, agent, world):
        if world.time >= 50:
            return True
        for l in world.landmarks:
            if not (l.color == self.green).all():
                return False
        return True
            
