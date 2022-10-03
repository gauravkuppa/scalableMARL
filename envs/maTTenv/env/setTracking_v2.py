import os, copy, pdb
import math
import random

import numpy as np
from numpy import linalg as LA
import torch
from gym import spaces, logger
from envs.maTTenv.maps import map_utils
import envs.maTTenv.util as util 
from envs.maTTenv.agent_models import *
from envs.maTTenv.belief_tracker import KFbelief
from envs.maTTenv.metadata import METADATA
from envs.maTTenv.env.maTracking_Base import maTrackingBase
from copy import deepcopy
from tryalgo import union_rectangles
"""
Target Tracking Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target

[Environment Description]
Varying number of agents, varying number of randomly moving targets
No obstacles

setTrackingEnv0 : Double Integrator Target model with KF belief tracker
    obs state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] *nb_targets
            where nb_targets and nb_agents vary between a range
            num_targets describes the upperbound on possible number of targets in env
            num_agents describes the upperbound on possible number of agents in env
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

"""

class setTrackingEnv2(maTrackingBase):

    def __init__(self, num_agents=1, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets,
                        map_name=map_name, is_training=is_training)
        self.steps = 0
        self.id = 'setTracking-v2'
        self.metadata = self.id
        self.scaled = kwargs["scaled"]
        self.reward_type = kwargs["reward_type"]
        self.nb_agents = num_agents #only for init, will change with reset()
        self.nb_targets = num_targets #only for init, will change with reset()
        self.agent_dim = 3
        self.target_dim = 4
        self.target_init_vel = METADATA['target_init_vel']*np.ones((2,))
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
                                np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed

        # normalize
        self.limit['state'] = [np.array(([0.0, 0.0, 0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0])),
                               np.array(([1.0, 1.0, 600.0, np.pi, rel_vel_limit, 10*np.pi, 50.0, 2.0]))]

        self.communication_range = METADATA['comms_r']

        observation_space = spaces.Box(np.tile(self.limit['state'][0], (self.num_targets, 1)), np.tile(self.limit['state'][1], (self.num_targets, 1)), dtype=np.float32)
        self.observation_space = {f"agent-{i}":observation_space for i in range(self.nb_agents)}
        self.observation_space = spaces.Dict(self.observation_space)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1), 
                                       [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
                        np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        self.coverage_reward_factor = METADATA['coverage_reward']
        self.global_coverage_map = np.zeros(shape=self.MAP.mapmax)
        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()
        # Use custom reward
        #self.get_reward()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [KFbelief(agent_id = 'target-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]
        # Initialise individual target beliefs for each agent

        for agent in self.agents:
            agent.setupBelief(deepcopy(self.belief_targets))


    def get_reward(self, observed=None, is_training=True):
        return self.reward_fun(self.agents, self.nb_targets,self.belief_targets,observed,is_training,c_mean=0.1,scaled = self.scaled)
    
    def reset(self,**kwargs):
        """
        Random initialization a number of agents and targets at the reset of the env epsiode.
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return an observation state dict with agent ids (keys) that refer to their observation
        """
        self.rng = np.random.default_rng()
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
        # Initialize targets and beliefs
        for i in range(self.nb_targets):
            # reset target
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))


            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            for j,agent in enumerate(self.agents):
                agent.belief[i].reset(np.concatenate((init_pose['targets'][i][:2],np.zeros(2))), self.target_init_cov)
                

        # For nb agents calculate belief of targets assigned

        for kk in range(self.nb_agents):
            obs_dict[self.agents[kk].agent_id] = self.observe_single(kk)

        return obs_dict

    def observe_single(self,agentID,action_vw = None,isObserved = None):
        observation = []
        for jj in range(self.nb_targets):
            r, alpha = util.relative_distance_polar(self.agents[agentID].belief[jj].state[:2],
                                                    xy_base=self.agents[agentID].state[:2],
                                                    theta_base=self.agents[agentID].state[2])
            if action_vw is None:
                r_dot_b,alpha_dot_b = 0.0,0.0
            else:
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                    self.agents[agentID].belief[jj].state[:2],
                    self.agents[agentID].belief[jj].state[2:],
                    self.agents[agentID].state[:2], self.agents[agentID].state[-1],
                    action_vw[0], action_vw[1])

            logdetcov = np.log(LA.det(self.agents[agentID].belief[jj].cov))
            if action_vw is None:
                observed = 0.0
            else:
                observed = float(isObserved[jj])

            observation.append([self.agents[agentID].state[0]/self.MAP.mapmax[0], self.agents[agentID].state[1]/self.MAP.mapmax[0], r, alpha, r_dot_b, alpha_dot_b, logdetcov, observed])

        return torch.tensor(observation,dtype=torch.float32)
    
    def communicate_graph(self):
        '''
        Returns a dictionary of agents that are within communication range
        '''
        agent_comms_dict = {}

        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                r, _ = util.relative_distance_polar(agent_j.state[:2],
                                                        xy_base=agent_i.state[:2],
                                                        theta_base=agent_i.state[2])

                if i != j and (r <= self.communication_range):
                    if agent_i.agent_id not in agent_comms_dict.keys():
                        agent_comms_dict[i] = [j]
                    else:
                        agent_comms_dict[i].append(j)

        return agent_comms_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        # Targets move (t -> t+1)
        for i in range(self.nb_targets):
            # update target
            self.targets[i].update() # self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
            for j in range(self.nb_agents):

                self.agents[j].belief[i].predict()

            self.belief_targets[i].predict() # Belief state at t+1
        # Target and map observations
        observed = np.zeros((self.nb_agents, self.nb_targets), dtype=bool)

        # Communication
        agent_graph = self.communicate_graph()

        update_comm_beliefs = []

        for id in agent_graph.keys():
            comm_recv_beliefs = [self.agents[ID].belief for ID in agent_graph[id]]
            update_comm_beliefs.append(self.agents[id].updateCommBelief(comm_recv_beliefs))

        for agentid, updatedCommBelief in enumerate(update_comm_beliefs):
            self.agents[agentid].setupBelief(updatedCommBelief)

        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]

            # Locations of all targets and agents in order to maintain a margin between them
            margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            for p, ids in enumerate(action_dict):
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_vw, margin_pos)


            # Update beliefs of targets
            for jj in range(self.nb_targets):
                # Observe
                obs, z_t = self.observation(self.targets[jj], self.agents[ii])
                observed[ii][jj] = obs
                if obs: # if observed, update the target belief.
                    #Update agents indivuudla be,liefs based on observation
                    self.agents[ii].updateBelief(targetID=jj,z_t = z_t)


                    # Update global belief
                    # TODO: Gaurav says: how to update global belief_target
                    #TODO: Global belief updates with all agent observations as usual
                    self.belief_targets[jj].update(z_t, self.agents[ii].state)


            obs_dict[self.agents[ii].agent_id] = self.observe_single(ii,action_vw=action_vw,isObserved = observed[ii])

            # shuffle obs to promote permutation invariance
            #self.rng.shuffle(obs_dict[agent_id])



        # Get all rewards after all agents and targets move (t -> t+1)
        reward, reward_dict, done, mean_nlogdetcov = self.get_reward(observed, self.is_training)
        done_dict['__all__'], info_dict['mean_nlogdetcov'] = done, mean_nlogdetcov

        info_dict['reward_all'] = reward_dict
        info_dict['metrics'] = [self.calculate_total_uncertainity(), self.calculate_max_uncertainity()]
        self.steps += 1
        return obs_dict, reward, done, info_dict


    def calculate_total_uncertainity(self):
        """
        Calculating metric
        - sum(tr(cov) over all beliefs)
        
        """
        total_uncertainity = 0


        for agent in self.agents:
            total_uncertainity += sum([np.sum(np.diag(b_target.cov)[:2]) for b_target in agent.belief])
        
        return total_uncertainity

    def calculate_max_uncertainity(self):
        """
        Calculating metric
        - sum(max(tr(cov)) over targets)
        
        """
        max_uncertainity = [0 for _ in range(self.nb_targets)]
        for agent in self.agents:
            for i, b_target in enumerate(agent.belief):
                uncertainity = np.sum(np.diag(b_target.cov)[:2])
                if max_uncertainity[i] < uncertainity:
                    max_uncertainity[i] = uncertainity
                
        return sum(max_uncertainity)


    def draw_circle(self, grid, x0, y0, radius):
        x0 = math.ceil(x0)
        y0 = math.ceil(y0)
        radius = int(radius)
        for y in range(-radius, radius): # (y = -radius; y <= radius; y++)
            for x in range(-radius, radius): # for (x = -radius; x <= radius; x++)
                #print(x0, x, y0, y)
                if ((x ** 2) + (y ** 2) <= (radius ** 2)) and x >= self.MAP.mapmin[0] and x < self.MAP.mapmax[0] and y >= self.MAP.mapmin[1] and y < self.MAP.mapmax[1]:
                    if x0 + x - 1 < self.MAP.mapmax[0] and y0 + y - 1 < self.MAP.mapmax[1]:
                        grid[x0 + x - 1, y0 + y - 1] = 1
                    if x0 - x - 1 >= self.MAP.mapmin[0] and y0 - y - 1 >= self.MAP.mapmin[1]:
                        grid[x0 - x - 1, y0 - y - 1] = 1
                    if x0 + x - 1 < self.MAP.mapmax[0] and y0 - y - 1 >= self.MAP.mapmin[1]:
                        grid[x0 + x - 1, y0 - y - 1] = 1
                    if x0 - x - 1 >= self.MAP.mapmin[0] and y0 + y - 1 < self.MAP.mapmax[1]:
                        grid[x0 - x - 1, y0 + y - 1] = 1

    def update_global_coverage_map(self, rectangle_map, decay):
        if self.global_coverage_map is not None:
            coverage_map2= deepcopy(self.global_coverage_map)
            coverage_map2[rectangle_map] = 1.0
            coverage_reward = np.sum(coverage_map2) - np.sum(self.global_coverage_map)
            self.global_coverage_map = deepcopy(coverage_map2)
        else:
            self.global_coverage_map = deepcopy(rectangle_map.to(np.float))
            coverage_reward = np.sum(self.coverage_map)

        return coverage_reward

    def reward_fun(self, agents, nb_targets, belief_targets, observed, is_training=True, c_mean=0.1,scaled = False):
        #TODO: reward should be per agent
        """centralized_belief = {f"target-{i}": LA.det(b.cov[:2, :2]) for i, b in enumerate(belief_targets)}
        best_decentralized_belief = {f"target-{i}": float('inf') for i in range(nb_targets)}
        for id,agent in enumerate(self.agents):
            detcov = [LA.det(b.cov[:2, :2]) for b in agent.belief]
            for target_id, cov in enumerate(detcov):
                key = f"target-{target_id}"
                if key in best_decentralized_belief.keys():
                    if cov < best_decentralized_belief[key]:
                        best_decentralized_belief[key] = cov
                else:
                    print(best_decentralized_belief.keys())
        assert np.isclose(np.array(list(centralized_belief.values())), np.array(list(best_decentralized_belief.values()))).all(), f"centralized belief: {centralized_belief} best decentralized belief {best_decentralized_belief}"""
        #import pdb; pdb.set_trace()
        global_weight = 1
        local_weight = 0.2
        globaldetcov = [LA.det(b_target.cov[:2, :2]) for b_target in belief_targets]
        globaldetcov = np.ravel(globaldetcov)
        global_max_reward = -np.log(np.max(globaldetcov))
        global_reward = global_weight * c_mean * global_max_reward

        r_detcov_sum = - np.sum(np.log(globaldetcov))
        reward = c_mean * r_detcov_sum

        ## discretize grid
        #grid = torch.zeros(self.MAP.mapmax[0], self.MAP.mapmax[1])
        ## find occupied cells by all agent's sensor radius
        square_side_divided_by_2 = METADATA['sensor_r']/2 * np.sqrt(np.pi)
        rectangles = []
        coverage_rew_dict = np.zeros(shape=len(self.agents))

        # randomising the coverage so that no agent is given priority
        agent_list_id = np.arange(len(self.agents))
        random.shuffle(agent_list_id)
        decay = np.exp(np.array([-1/40]))
        self.global_coverage_map = np.copy(self.global_coverage_map) * decay

        for j in agent_list_id:
            #import pdb; pdb.set_trace()
            x, y = self.agents[j].state[0], self.agents[j].state[1]

            # TODO: https://colab.research.google.com/drive/15LiJpRjjNOGBWzlJUNAu8e5RpWIUa2SV?usp=sharing
            r1 = int(x - square_side_divided_by_2)
            c1 = int(y - square_side_divided_by_2)
            r2 = int(x + square_side_divided_by_2)
            c2 = int(y + square_side_divided_by_2)
            r1 = r1 if r1 >0 else 0
            c1 = c1 if c1 > 0 else 0
            r2 = r2 if r2 < self.MAP.mapmax[0] else self.MAP.mapmax[0]
            c2 = c2 if c2 < self.MAP.mapmax[1] else self.MAP.mapmax[1]

            rectangles_x,rectangles_y = np.meshgrid(np.arange(self.MAP.mapmax[0]),np.arange(self.MAP.mapmax[1]))
            rectangles = np.logical_and(np.logical_and(rectangles_x>=r1,rectangles_x<=r2)
                                        ,np.logical_and(rectangles_y>=c1,rectangles_y<=c2))
            coverage_reward = self.update_global_coverage_map(rectangles,np.exp(np.array([-1/40])))
            coverage_rew_dict[j] = coverage_reward/(np.prod(self.MAP.mapmax))
            #self.draw_circle(grid, agent.state[0], agent.state[1], METADATA['sensor_r'])
        #sensor_footprint = union_rectangles(rectangles)

        #import pdb; pdb.set_trace()
        # if not self.coverage_reward_factor:
        #     self.coverage_reward_factor = sensor_footprint
        # else:
        #     # TODO: coverage shouldnt be dependent on number of agents; decay; how to pick number of steps to decay? size of the
        #     self.coverage_reward_factor = self.coverage_reward_factor/np.prod(self.MAP.mapmax) * torch.exp(torch.Tensor([-(self.steps)/50])) + sensor_footprint
        
        reward_dict = []
        for id,agent in enumerate(self.agents):
            reward_dict.append(global_reward)
            """observed_detcov = [LA.det(belief.cov) for target_id, belief in enumerate(agent.belief) if observed[id][target_id]]
            detcov = [LA.det(b.cov) for b in agent.belief]
            detcov = np.ravel(detcov)
            if is_training:
                if observed_detcov:
                    #print(len(observed_detcov))
                    detcov = np.ravel(observed_detcov)
                    local_detcov = - np.mean(np.log(detcov))
                else:
                    local_detcov = - np.log(np.min(detcov)) # penalize
            else:
                local_detcov = - np.log(np.max(detcov))
                #print("individual")
            reward_dict.append(local_weight*c_mean*local_detcov)"""
        mean_nlogdetcov = None
        if not(is_training):
            logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets[:nb_targets]]
            mean_nlogdetcov = -np.mean(logdetcov)
        return global_reward, np.array(reward_dict), False, mean_nlogdetcov


def reward_fun(scaled, agents, nb_targets, belief_targets, is_training=True, c_mean=0.1):

    # detcov =
    detcov = [[LA.det(belief.cov) for belief in agents_beliefs] for agents_beliefs in belief_targets]
    """
    [ [{'agent-0_target-0': 810029.250273439},
    {'agent-0_target-1': 810029.250273439}],
    [{'agent-1_target-0': 810029.250273439},
    {'agent-1_target-1': 810029.250273439}] ]
    """
    reward = [c_mean * -np.mean(np.log(agent_detcov)) for agent_detcov in detcov]
    
    # reward = np.sum(np.where(observed, reward, -1))
    
    #
    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [[np.log(LA.det(belief.cov)) for belief in agents_beliefs] for agents_beliefs in belief_targets]
        mean_nlogdetcov = [-np.mean(_logdetcov) for _logdetcov in logdetcov]
        mean_nlogdetcov = np.stack(mean_nlogdetcov)
        # assert r_detcov_mean == mean_nlogdetcov
    return np.array(reward), False, mean_nlogdetcov
    