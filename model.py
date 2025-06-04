import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

#def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#    torch.nn.init.orthogonal_(layer.weight, std)
#    torch.nn.init.constant_(layer.bias, bias_const)
#    return layer

class Agent(nn.Module):
    # from clearn RL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py
    def __init__(self, obs_space, action_space, num_agents=4, num_envs=1):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.network = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),  # Adjust for 1D input
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, action_space.n)
        self.critic = nn.Linear(256, 1)  # decentralized critic for each agent

        self.centralised_critics = nn.Sequential(
            nn.Linear(obs_space.shape[0]* num_agents*num_envs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    """

    args:
        x: (batch_size, num_agent, obs_dim)
        joint_obs: (batch_size, num_agent * obs_dim)
    
    """
    def get_value(self, x, joint_obs=None):
        if joint_obs is not None:
            assert joint_obs.shape[1] == self.obs_space.shape[0] * self.num_agents * self.num_envs
            return self.centralised_critics(joint_obs)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, joint_obs=None):
        # x dim (num_agent*num_envs, concat obs shape)
        hidden = self.network(x)
        logits = self.actor(hidden)  # dim (num_agent*num_env, action_size per agent)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        if joint_obs is not None:
            #assert joint_obs.shape[1] == self.obs_space.shape[0] * self.num_agents
            v = self.centralised_critics(joint_obs)
        else:
            v = self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), v
    
    def get_prob(self, x):
        # x dim (num_agent, concat obs shape)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(
            (nn.Linear(num_agents * obs_dim, 512)),
            nn.ReLU(),
            (nn.Linear(512, 1))
        )

    def forward(self, states):
        """States should be concatenated observations from all agents."""
        return self.critic(states)