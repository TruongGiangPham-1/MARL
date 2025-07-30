import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    """Individual Q-network for each agent"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim))
        )
    
    def forward(self, obs):
        return self.network(obs)

class MixingNetwork(nn.Module):
    """Mixing network that combines individual Q-values"""
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        # Hypernetworks for generating weights and biases
        self.hyper_w_1 = nn.Linear(state_dim, self.embed_dim * num_agents)
        self.hyper_w_final = nn.Linear(state_dim, self.embed_dim)
        
        # Hypernetwork for bias (ensure non-negative mixing)
        self.hyper_b_1 = nn.Linear(state_dim, self.embed_dim)
        
        # V(s) - state value function
        self.V = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: [batch_size, num_agents] - individual Q-values
            states: [batch_size, state_dim] - global state
        Returns:
            q_tot: [batch_size, 1] - mixed Q-value
        """
        batch_size = agent_qs.size(0)
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w_1(states))  # [batch_size, embed_dim * num_agents]
        b1 = self.hyper_b_1(states)  # [batch_size, embed_dim]
        
        w1 = w1.view(-1, self.num_agents, self.embed_dim)  # [batch_size, num_agents, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)  # [batch_size, 1, embed_dim]
        
        # First layer
        agent_qs = agent_qs.view(-1, 1, self.num_agents)  # [batch_size, 1, num_agents]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch_size, 1, embed_dim]
        
        # Final layer
        w_final = torch.abs(self.hyper_w_final(states))  # [batch_size, embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [batch_size, embed_dim, 1]
        
        # State value
        v = self.V(states).view(-1, 1, 1)  # [batch_size, 1, 1]
        
        # Final Q_tot
        q_tot = torch.bmm(hidden, w_final) + v  # [batch_size, 1, 1]
        
        return q_tot.view(batch_size, 1)

class QMixBuffer:
    """Experience replay buffer for QMIX"""
    def __init__(self, capacity, num_agents, obs_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, actions, rewards, next_obs, dones):
        """
        Args:
            obs: [num_agents, obs_dim]
            actions: [num_agents]
            rewards: [num_agents]
            next_obs: [num_agents, obs_dim]
            dones: [num_agents]
        """
        experience = (obs, actions, rewards, next_obs, dones)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        # Convert to tensors
        obs = torch.stack(obs)  # [batch_size, num_agents, obs_dim]
        actions = torch.stack(actions)  # [batch_size, num_agents]
        rewards = torch.stack(rewards)  # [batch_size, num_agents]
        next_obs = torch.stack(next_obs)  # [batch_size, num_agents, obs_dim]
        dones = torch.stack(dones)  # [batch_size, num_agents]
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)

class QMIX:
    """QMIX Algorithm Implementation"""
    def __init__(self, env, num_agents, obs_dim, action_dim, state_dim,
                 lr=0.0005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, target_update_freq=200, buffer_size=5000,
                 batch_size=32, mixing_embed_dim=32, hidden_dim=256,
                 save_path=None, log_dir=None, log=False, args=None):
        
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Shared Q-network for all agents
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Mixing networks
        self.mixing_network = MixingNetwork(num_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixing_network = MixingNetwork(num_agents, state_dim, mixing_embed_dim).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Optimizers (now just one for shared network + mixing network)
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixing_network.parameters()), 
            lr=lr
        )
        
        # Experience replay buffer
        self.buffer = QMixBuffer(buffer_size, num_agents, obs_dim)
        
        # Logging
        self.save_path = save_path
        self.log_dir = log_dir
        self.log = log
        self.args = args
        
        if log:
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.summary_writer = None
            
        self.update_count = 0
        self.episode_count = 0
    
    def act(self, obs, state=None, training=True):
        """
        Select actions using epsilon-greedy policy
        Args:
            obs: [num_agents, obs_dim] - assuming num_envs = 1
            state: global state (optional, for mixing network)
            training: whether in training mode
        Returns:
            actions: [num_agents]
        """
        # Assuming num_envs = 1, so obs shape is [num_agents, obs_dim]
        if obs.dim() == 2 and obs.shape[0] == self.num_agents:
            actions = self._get_actions_single_env(obs, training)
            return actions, None, None, None  # Match MAPPO interface
        else:
            raise ValueError(f"Expected obs shape [{self.num_agents}, obs_dim], got {obs.shape}")
    
    def _get_actions_single_env(self, obs, training=True):
        """Get actions for a single environment using shared network"""
        with torch.no_grad():
            # Process all agents at once with shared network
            obs_batch = obs.to(self.device)  # [num_agents, obs_dim]
            q_values = self.q_network(obs_batch)  # [num_agents, action_dim]
            
            actions = []
            for i in range(self.num_agents):
                if training and random.random() < self.epsilon:
                    action = torch.randint(0, self.action_dim, (1,)).to(self.device)
                else:
                    action = q_values[i].argmax(dim=0)  # Get action for agent i
                
                actions.append(action.item())
            
            return torch.tensor(actions, dtype=torch.long)
    
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs=None, values=None):
        """Add experience to buffer (compatibility with existing interface)"""
        # Assuming num_envs = 1, so we store the experience directly
        self.current_obs = obs  # [num_agents, obs_dim]
        self.current_actions = actions  # [num_agents]
        self.current_rewards = rewards  # [num_agents]
        self.current_dones = dones  # [num_agents]
    
    def update(self, next_obs):
        """Update the QMIX networks"""
        if hasattr(self, 'current_obs'):
            # Add experience to buffer - assuming num_envs = 1
            obs = self.current_obs  # [num_agents, obs_dim]
            actions = self.current_actions  # [num_agents]
            rewards = self.current_rewards  # [num_agents]
            dones = self.current_dones  # [num_agents]
            next_obs_reshaped = next_obs  # [num_agents, obs_dim]
            
            # Add experience to buffer (state is computed from obs in buffer.sample())
            self.buffer.add(
                obs.cpu(),
                actions.cpu(),
                rewards.cpu(),
                next_obs_reshaped.cpu(),
                dones.cpu()
            )
        
        # Update networks if we have enough samples
        if len(self.buffer) > self.batch_size:
            self._update_networks()
            
        # Update target networks
        if self.update_count % self.target_update_freq == 0:
            self._update_target_networks()
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.update_count += 1
    
    def _update_networks(self):
        """Update Q-networks and mixing network"""
        # Sample batch from buffer
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        # Compute global states from observations (flattened obs)
        states = obs.view(obs.size(0), -1)  # [batch_size, num_agents * obs_dim]
        next_states = next_obs.view(next_obs.size(0), -1)  # [batch_size, num_agents * obs_dim]
        
        # Compute current Q-values using shared network
        obs_flat = obs.view(-1, self.obs_dim)  # [batch_size * num_agents, obs_dim]
        q_vals_flat = self.q_network(obs_flat)  # [batch_size * num_agents, action_dim]
        q_vals = q_vals_flat.view(obs.size(0), self.num_agents, -1)  # [batch_size, num_agents, action_dim]
        
        # Get Q-values for chosen actions
        actions_expanded = actions.unsqueeze(-1)  # [batch_size, num_agents, 1]
        current_q_values = q_vals.gather(2, actions_expanded).squeeze(-1)  # [batch_size, num_agents]
        
        # Compute target Q-values using shared target network
        with torch.no_grad():
            next_obs_flat = next_obs.view(-1, self.obs_dim)  # [batch_size * num_agents, obs_dim]
            next_q_vals_flat = self.target_q_network(next_obs_flat)  # [batch_size * num_agents, action_dim]
            next_q_vals = next_q_vals_flat.view(next_obs.size(0), self.num_agents, -1)  # [batch_size, num_agents, action_dim]
            
            # Get max Q-values for next state
            next_q_values = next_q_vals.max(2)[0]  # [batch_size, num_agents]
            
            # Mix target Q-values
            target_q_tot = self.target_mixing_network(next_q_values, next_states)  # [batch_size, 1]
            
            # Compute target values
            # Use team reward (sum of individual rewards)
            team_rewards = rewards.sum(dim=1, keepdim=True)  # [batch_size, 1]
            team_dones = dones.any(dim=1, keepdim=True).float()  # [batch_size, 1]
            
            targets = team_rewards + self.gamma * (1 - team_dones) * target_q_tot  # [batch_size, 1]
        
        # Mix current Q-values
        current_q_tot = self.mixing_network(current_q_values, states)  # [batch_size, 1]
        
        # Compute loss
        loss = F.mse_loss(current_q_tot, targets)
        
        # Update shared network and mixing network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Logging
        if self.log and self.summary_writer:
            self.summary_writer.add_scalar("losses/qmix_loss", loss.item(), self.update_count)
            self.summary_writer.add_scalar("charts/epsilon", self.epsilon, self.update_count)
            self.summary_writer.add_scalar("charts/q_values_mean", current_q_tot.mean().item(), self.update_count)
    
    def _update_target_networks(self):
        """Update target networks"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
    
    def save_model(self):
        """Save the model"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Save shared Q-network
            torch.save(self.q_network.state_dict(), f"{self.save_path}_q_net.pth")
            
            # Save mixing network
            torch.save(self.mixing_network.state_dict(), f"{self.save_path}_mixing_net.pth")
            
            print(f"QMIX model saved to {self.save_path}")
    
    def load_model(self, path):
        """Load the model"""
        self.q_network.load_state_dict(torch.load(f"{path}_q_net.pth"))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.mixing_network.load_state_dict(torch.load(f"{path}_mixing_net.pth"))
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        print(f"QMIX model loaded from {path}")