
from MAPPO import MAPPO
import torch


# inherit from MAPPO
class CMAPPO(MAPPO):
    def __init__(self, env, optimzer, policy, buffer,
                 single_agent_obs, single_agent_action,
                 batch_size=128,
                 num_mini_batches=4,
                 num_agents=4,
                ppo_epoch=10,
                clip_param=0.2,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                gamma=0.99,
                lam=0.95,
                 save_path=None, log_dir=None, log=False, args=None):
        super().__init__(env, optimzer, policy, buffer,
                         single_agent_obs, single_agent_action,
                         batch_size=batch_size,
                         num_mini_batches=num_mini_batches,
                         num_agents=num_agents,
                            ppo_epoch=ppo_epoch,
                            clip_param=clip_param,
                            value_loss_coef=value_loss_coef,
                            entropy_coef=entropy_coef,
                            max_grad_norm=max_grad_norm,
                            gamma=gamma,
                            lam=lam,
                         save_path=save_path, log_dir=log_dir, log=log, args=args)
        
    
    def compute_value_loss(self, target, new_values):
        """
        
        Args:
            target (torch.Tensor): Mini-batch value target. size (mini_batch_size, num_agents)
            new_values (torch.Tensor): New values from the critic.    # size (mini_batch_size, 1)
        """
        centralized_adv = target.mean(dim=0, keepdim=True)  # dim (mini_batch_size, 1)

        value_loss = 0.5 * ((new_values - centralized_adv)**2).mean()
        return value_loss

    def compute_gae(self, rewards, dones, values, next_values):
        """
        Compute Generalized Advantage Estimation (GAE) using centralised critic.
        adapt from clearn rl https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py

        only difference is that I use mean of the rewards with computing TD error delta

        Args:
            rewards (torch.Tensor): shape (num_steps, num_agents)
            dones (torch.Tensor): shape (num_steps, num_agents)
            values [V(S_i)]: shape (num_steps, 1)
            next_values (torch.Tensor): shape (1, 1) for CMAPPO
            gamma (float): Discount factor.
            lam (float): Lambda for GAE.

        Returns:
            advantages (torch.Tensor): Computed advantages.
        """
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)  # shape (num_steps, num_agents)
            lastgaelam  = torch.zeros(self.num_agents).to(self.device)
            for t in reversed(range(self.buffer.max_size)):
                if t ==  self.buffer.max_size - 1:
                    mask = 1.0 - dones[-1]  # shape (num_agents,)
                    nextvalues = next_values           # shape (1, num_agents) or (1, 1) for CMAPPO
                else:
                    mask = 1.0 - dones[t + 1]    # shape (num_agents,)
                    nextvalues = values[t + 1]             # shape (num_agents,) or (1, 1) for CMAPPO

                rewards_t = rewards[t].mean()  # scalar, the only difference from MAPPO

                delta = rewards_t + self.gamma * nextvalues * mask - values[t]  # A: r_t + \gamma*V(s_t+1) - V(s_t)    shape (num_agents,) or (1, 1) for CMAPPO

                advantages[t] = lastgaelam = delta + self.gamma * self.lam * mask * lastgaelam  # shape (num_agents,)
        return advantages 

    