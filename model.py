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
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], 512),  # Adjust for 1D input
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, env.action_spaces[0].n)
        self.critic = nn.Linear(256, 1)  # remove

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        # x dim (num_agent, concat obs shape)
        hidden = self.network(x)
        logits = self.actor(hidden)  # dim (num_agent, action_size per agent)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_agents * obs_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1))
        )

    def forward(self, states):
        """States should be concatenated observations from all agents."""
        return self.critic(states)