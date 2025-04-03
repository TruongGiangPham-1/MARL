import numpy as np
import torch
class Buffer(object):
    def __init__(self, state_dim, num_agents, max_size=128):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_buff = torch.zeros((max_size, num_agents, state_dim)).to(self.device)
        self.actions_buff = torch.zeros((max_size, num_agents)).to(self.device)
        self.rewards_buff = torch.zeros((max_size, num_agents)).to(self.device)
        self.dones_buff = torch.zeros((max_size, num_agents)).to(self.device)
        self.logprobs_buff = torch.zeros((max_size, num_agents)).to(self.device)
        self.values_buff = torch.zeros((max_size, num_agents)).to(self.device)


    def add(self, obs, actions, rewards, dones, logprobs, values):
        self.obs_buff[self.ptr] = obs
        self.actions_buff[self.ptr] = actions
        self.rewards_buff[self.ptr] = rewards
        self.dones_buff[self.ptr] = dones
        self.logprobs_buff[self.ptr] = logprobs
        self.values_buff[self.ptr] = values
        # Update the pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

