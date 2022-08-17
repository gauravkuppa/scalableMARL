import numpy as np
import torch
import torch.nn as nn
from algos.maTT.modules import *
from torch.distributions.categorical import Categorical

__author__ = 'Christopher D Hsu'
__copyright__ = ''
__credits__ = ['Christopher D Hsu', 'SpinningUp']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Christopher D Hsu'
__email__ = 'chsu8@seas.upenn.edu'
__status__ = 'Dev'

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftActionSelector(nn.Module):
    '''
    Soft parameterization of q value logits,
    pi_log = (1/Z)*(e^(v(x)) - min(v(x))
    If determinstic take max value as action,
    Else (stochastic),
    Sample from multinomial of the soft logits.
    '''
    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, q, deterministic=False, with_logprob=True):
        q_soft = q - torch.min(q)

        # Convert q values to log probability space
        try:
            pi_log = self.logsoftmax(q_soft)
        except:
            q_soft = q_soft.unsqueeze(0)
            pi_log = self.logsoftmax(q_soft)

        # Select action
        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            q_log_dist = torch.distributions.multinomial.Multinomial(1, logits=pi_log)
            action = q_log_dist.sample()
            pi_action = torch.argmax(action, dim=1, keepdim=True)

        # Calculate log probability if training
        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi_action)
        else:
            logp_pi = None
        
        return pi_action, logp_pi

class DeepSetAttention(nn.Module):
    """ Written by Christopher Hsu:

    """
    def __init__(self, dim_input, dim_output, num_outputs=1,
                        dim_hidden=128, num_heads=4, ln=True):
        super().__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    # v(x)
    def values(self, obs):
        v = self.enc(obs)
        v = v.sum(dim=1, keepdim=True)  #pooling mechanism: sum, mean, max
        v = self.dec(v).squeeze()
        return v

    # q(x,a)
    def forward(self, obs, act):
        v = self.enc(obs)
        v = v.sum(dim=1, keepdim=True)  #pooling mechanism: sum, mean, max
        v = self.dec(v).squeeze()
        q = torch.gather(v, 1, act)
        return q

class DeepSetmodel(nn.Module):

    def __init__(self, observation_space, action_space, dim_hidden=128):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = SoftActionSelector(act_dim)
        self.q1 = DeepSetAttention(dim_input=obs_dim, dim_output=act_dim, dim_hidden=dim_hidden)
        self.q2 = DeepSetAttention(dim_input=obs_dim, dim_output=act_dim, dim_hidden=dim_hidden)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            v1 = self.q1.values(obs)
            v2 = self.q2.values(obs)

            a, _ = self.pi(v1+v2, deterministic, False)
            # Tensor to int
            return int(a)

class PPO(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.envs = envs
        self.args = args
        
        self.key = list(envs.single_observation_space.keys())[0]
        self.keys_agent = list(envs.full_observation_space[self.key].keys())

        obs_dim = {}
        self.act_dim = envs.single_action_space.n
        for k in self.keys_agent:
            obs_dim[k] = np.array(envs.full_observation_space[self.key][k].shape)

        self.obs_dim = obs_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim['target'].prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim['target'].prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    def decode(self,observation):
        split_size = [self.obs_dim[k].prod() for k in self.obs_dim.keys()]
        observation_list = torch.split(observation,split_size,dim=-1)
        observation_list_reshaped = {}
        for obs,key in zip(observation_list,list(self.obs_dim.keys())):
            obs = obs.view([-1] + self.obs_dim[key].tolist())
            observation_list_reshaped[key] = obs
        return observation_list_reshaped

    def get_value(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)

        x_list = self.decode(x)
        x = x_list['target']
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)
        x_list = self.decode(x)
        x = x_list['target']
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPOAR(PPO):
    def __init__(self, envs, args):
        super(PPOAR, self).__init__(envs,args)
        obs_dim = {}
        self.backend = {}
        for k in self.keys_agent:
            obs_dim[k] = np.array(envs.full_observation_space[self.key][k].shape)
            self.backend[k] = nn.Sequential(
                layer_init(nn.Linear(obs_dim[k].prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            ).to(self.device)
        self.obs_dim = obs_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01)
        )

    def get_value(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)

        x_list = self.decode(x)
        out_list = []
        for _,k in enumerate(self.backend.keys()):
            x_list[k] = x_list[k].view(-1, x_list[k].shape[-2] * x_list[k].shape[-1])
            out_list.append(self.backend[k](x_list[k]))
        x = torch.cat(out_list,dim=-1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)
        x_list = self.decode(x)
        out_list = []
        for _,k in enumerate(self.backend.keys()):
            x_list[k] = x_list[k].view(-1, x_list[k].shape[-2] * x_list[k].shape[-1])
            out_list.append(self.backend[k](x_list[k]))

        x = torch.cat(out_list,dim=-1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPOAttention(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.envs = envs
        self.args = args
        self.key = list(envs.single_observation_space.keys())[0]
        self.keys_agent = list(envs.full_observation_space[self.key].keys())

        obs_dim = {}
        self.act_dim = envs.single_action_space.n
        for k in self.keys_agent:
            obs_dim[k] = np.array(envs.full_observation_space[self.key][k].shape)

        self.obs_dim = obs_dim

        # 5 agent states and 6 target states
        self.target_state_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim['target'][-1]-5, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
        )

        self.agent_state_embedder = nn.Sequential(
            layer_init(nn.Linear(5*self.obs_dim['target'][0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim['target'][0]*64+64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1))
        )

        self.actor_attention = MAB(dim_Q=64, dim_K=64, dim_V=64, num_heads=1, ln=True)
        self.target_attention = SAB(dim_in=64,dim_out=64, num_heads=1, ln=True)
        self.pos_enc1 = nn.Linear(1,64)
        self.pos_enc2 = nn.Linear(1,64)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64*3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # decodes the observation tensor
    def decode(self,observation):
        split_size = [self.obs_dim[k].prod() for k in self.obs_dim.keys()]
        observation_list = torch.split(observation,split_size,dim=-1)
        observation_list_reshaped = {}
        for obs,key in zip(observation_list,list(self.obs_dim.keys())):
            obs = obs.view([-1] + self.obs_dim[key].tolist())
            observation_list_reshaped[key] = obs
        return observation_list_reshaped

    def decode_target(self,observation):
        split_size = [5, self.obs_dim['target'][1]-5]
        # splits to (batch,n_targ,5)  and  (batch,n_targ,obs_dim['target'][1]-5)
        state_obs,target_obs = torch.split(observation, split_size, dim=-1)
        # reshapes state observation to (batch,n_targ*5)
        state_obs = state_obs.reshape(state_obs.shape[0],-1)
        return state_obs,target_obs

    def get_value(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)
        x_list = self.decode(x)
        x_state,x_target = self.decode_target(x_list['target'])

        x_state = self.agent_state_embedder(x_state)
        x_target = self.target_state_encoder(x_target)
        x_concat = torch.cat([x_state,x_target.view(-1,x_target.shape[-2]*x_target.shape[-1])],dim =-1)
        return self.critic(x_concat)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not x.is_cuda:
            x = x.to(self.device)
        x_list = self.decode(x)
        x_state, x_target = self.decode_target(x_list['target'])

        x_state = self.agent_state_embedder(x_state)
        x_target = self.target_state_encoder(x_target)
        x_concat = torch.cat([x_state, x_target.reshape(-1, x_target.shape[-2] * x_target.shape[-1])], dim=-1)

        value = self.critic(x_concat)

        # target permutations matter
        pos_enc_shape = np.ones(len(x_target.shape),dtype=int)[:-2].tolist()+[self.obs_dim['target'][0],1]
        pos_enc = torch.tensor(np.arange(0,self.obs_dim['target'][0])/
                               self.obs_dim['target'][0],dtype=torch.float32).reshape(tuple(pos_enc_shape))
        pos_enc = pos_enc.repeat(x_target.shape[0],1,1).to(self.device)

        target_enc,attention_targ = self.target_attention(x_target+self.pos_enc1(pos_enc))
        actor_en,attention = self.actor_attention(x_state.unsqueeze(dim=-2),target_enc+self.pos_enc2(pos_enc))

        # compounded attention weights
        attention_comp = torch.bmm(attention,attention_targ)

        x_actor = torch.cat([x_state,actor_en.squeeze(),target_enc.sum(dim=-2)],dim=-1)
        logits = self.actor(x_actor)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, attention_comp.squeeze()

