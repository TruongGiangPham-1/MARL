import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class MAPPO:

    def __init__(self, env, optimzer, policy, buffer,
            single_agent_obs, single_agent_action,
            collect_steps=128,
            num_agents=4, 
            save_path=None, log_dir=None, log=False):
        self.env = env
        self.optimizer = optimzer
        self.policy = policy

        self.collect_steps = 128
        self.num_agents = num_agents
        self.single_agent_obs = single_agent_obs  # tuple
        self.single_agent_action = single_agent_action
        self.buffer = buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.collect_steps
        self.mini_batch_size = self.batch_size // 4
        self.ppo_epoch = 10
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        self.gamma = 0.99
        self.lam = 0.95

        self.save_path = save_path
        self.log_dir = log_dir

        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.log = log  # whether to log or not

        self.num_gradient_steps = 0



    def act(self, obs):
        """
        Returns the action for the given observation.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            action (torch.Tensor): Action tensor.
        """
        with torch.no_grad():
            action, logprob, entropy, values = self.policy.get_action_and_value(obs)
        return action, logprob, entropy, values
    
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs, values):
        self.buffer.add(obs, actions, rewards, dones, logprobs, values)

    def compute_gae(self, rewards, dones, values, next_values):
        """
        Compute Generalized Advantage Estimation (GAE).
        adapt from clearn rl https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py

        Args:
            rewards (torch.Tensor): shape (num_steps, num_agents)
            dones (torch.Tensor): shape (num_steps, num_agents)
            values (torch.Tensor): shape (num_steps, num_agents)
            next_values (torch.Tensor): shape (1, num_agents)
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
                    nextnonterminal = 1.0 - dones[-1]  # shape (num_agents,)
                    nextvalues = next_values           # shape (1, num_agents)
                else:
                    nextnonterminal = 1.0 - dones[t + 1]    # shape (num_agents,)
                    nextvalues = values[t + 1]             # shape (num_agents,)
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]  # shape (num_agents,)

                advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam  # shape (num_agents,)

            returns = advantages + values  # shape (num_steps, num_agents)
        return returns

    def update(self, next_obs):
        """
        Update the policy using the collected data.

        Args:
            obs (torch.Tensor): Observation tensor.
            actions (torch.Tensor): Action tensor.
            rewards (torch.Tensor): Reward tensor.
            dones (torch.Tensor): Done tensor.
            logprobs (torch.Tensor): Log probability tensor.
            values (torch.Tensor): Value tensor.
        """
        # Implement the update logic here
        print("Updating policy with collected data...")

        # compute GAE
        with torch.no_grad():
            next_values = self.policy.get_value(next_obs).reshape(1, self.num_agents).to(self.device)
        advantages = self.compute_gae(
            self.buffer.rewards_buff,
            self.buffer.dones_buff,
            self.buffer.values_buff,
            next_values=next_values
        )  # dim (num_steps, num_agents)
       
      
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.ppo_epoch):
            # Sample a batch of data from the buffer
            np.random.shuffle(b_inds)  # shuffle in place

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > self.batch_size:
                    end = self.batch_size

                mb_inds = b_inds[start:end]             # dim (mini_batch_size,)
                mb_obs = self.buffer.obs_buff[mb_inds]  # dim (minibatch, num_agents, obs_dim)
                mb_actions = self.buffer.actions_buff.long()[mb_inds]
                

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(mb_obs, mb_actions)
                """
                newlogprob shape (minibatch_size, num_agent)
                entropy shape (minibatch_size, num_agent)
                newvalue shape (minibatch_size, num_agent, 1)
                """
                logratio = newlogprob - self.buffer.logprobs_buff[mb_inds]

                ratio = logratio.exp()  # dim (mini_batch_size, num_agents)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    )

                mb_advantages = advantages[mb_inds]  # dim (mini_batch_size, num_agents)


                # policy loss
                pg_loss1 = -mb_advantages * ratio  # dim (minibatch, num_agent)
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (no clipping TODO: clip)
                newvalue = newvalue.squeeze()  # dim (minibatch_size, num_agents)

                v_loss = 0.5 * (newvalue - mb_advantages **2).mean()  # dim 
                entropy_loss = entropy.mean()  # dim (1)

                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.log:
                    self.summary_writer.add_scalar('policy_')
                    self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.num_gradient_steps)
                    self.writer.add_scalar("losses/value_loss", v_loss.item(), self.num_gradient_steps)
                    self.writer.add_scalar("losses/policy_loss", pg_loss.item(), )
                    self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.num_gradient_steps)
                    self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_gradient_steps)
                    self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_gradient_steps)
                    self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.num_gradient_steps)

        self.num_gradient_steps += 1

        if self.save_path is not None:
            torch.save(self.policy.state_dict(), os.path.join(self.save_path, "policy.pth"))
            print(f'saved model at {self.save_path}')
        # Reset the buffer
        self.buffer.reset()
