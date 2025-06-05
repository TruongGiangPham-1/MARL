import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class MAPPO:

    def __init__(self, env, optimzer, policy, buffer,
            single_agent_obs, single_agent_action,
            batch_size=1,
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
        self.env = env
        self.optimizer = optimzer
        self.policy = policy

        self.num_agents = num_agents
        self.single_agent_obs = single_agent_obs  # tuple
        self.single_agent_action = single_agent_action
        self.buffer = buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.mini_batch_size = self.batch_size // num_mini_batches
        self.ppo_epoch = ppo_epoch  
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.lam = lam

        self.save_path = save_path
        self.log_dir = log_dir
        self.args = args

        self.log = log  # whether to log or not
        if log:
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.summary_writer = None

        self.num_gradient_steps = 0



    def act(self, obs):
        """
        Returns the action for the given observation.

        Args:
            obs (torch.Tensor): Observation tensor. size (num_agents, obs_dim)

        Returns:
            action (torch.Tensor): Action tensor.
        """
        with torch.no_grad():
            obs = obs.to(self.device)  # dim (num_agents, obs_dim)
            if type(self).__name__ == "CMAPPO":
                # convert obs to joint_obs. obs dim (num_agents, obs_dim) to (1, num_agents * obs_dim)
                joint_obs = obs.view(1, -1).to(self.device)
                action, logprob, entropy, values = self.policy.get_action_and_value(obs, joint_obs=joint_obs)
            elif type(self).__name__ == "MAPPO":
                action, logprob, entropy, values = self.policy.get_action_and_value(obs)
            else: raise ValueError("Unknown class name. Cannot determine if CMAPPO or MAPPO.")
        return action, logprob, entropy, values
    
    def add_to_buffer(self, obs, actions, rewards, dones, logprobs, values):
        self.buffer.add(obs, actions, rewards, dones, logprobs, values)

    def compute_gae(self, rewards, dones, values, next_values):
        """
        Compute Generalized Advantage Estimation (GAE).
        adapt from clearn rl https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py

        Args:
            rewards (torch.Tensor): shape (num_steps, num_env*num_agents)
            dones (torch.Tensor): shape (num_steps, num_env*num_agents)
            values [V(S_i)]: shape (num_steps, num_env*num_agents)
            next_values (torch.Tensor): shape (1, num_env*num_agents)
            gamma (float): Discount factor.
            lam (float): Lambda for GAE.

        Returns:
            advantages (torch.Tensor): Computed advantages.
        """
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)  # shape (num_steps, num_env*num_agents)
            lastgaelam  = torch.zeros(self.args.num_envs*self.num_agents).to(self.device)
            for t in reversed(range(self.buffer.max_size)):
                if t ==  self.buffer.max_size - 1:
                    mask = 1.0 - dones[-1]  # shape (num_env*num_agents,)
                    nextvalues = next_values           # shape (1, num_env*num_agents)
                else:
                    mask = 1.0 - dones[t + 1]    # shape (num_env*num_agents,)
                    nextvalues = values[t + 1]             # shape (num_env*num_agents,)
                delta = rewards[t] + self.gamma * nextvalues * mask - values[t]  # A: r_t + \gamma*V(s_t+1) - V(s_t)    shape (num_agents,)

                advantages[t] = lastgaelam = delta + self.gamma * self.lam * mask * lastgaelam  # shape (num_agents,)

            #returns = advantages + values  #  A + V = r_t + \gamma*V(s_t+1) - V(s_t) + V(s_t) = r_t + \gamma*V(s_t+1) = TD target estimate for V(s_t)
            # TODO: clearn rl uses this. but I am not sure if it is correct.
        return advantages 

    def update(self, next_obs):
        """
        Update the policy using the collected data.

        Args:
            next_obs (torch.Tensor): Observation tensor. Size (num_agents*num_env, obs_dim)
        """
        # Implement the update logic here

        # compute GAE
        with torch.no_grad():
            if type(self).__name__ == "CMAPPO":
                # convert next_obs to joint_obs. next_obs dim (num_agents, obs_dim) to (1, num_agents * obs_dim)
                joint_obs = next_obs.view(1, -1)  # dim (1, num*env*num_agents * obs_dim)
                next_values = self.policy.get_value(next_obs, joint_obs=joint_obs).to(self.device)
                assert next_values.shape == (1, 1)
            elif type(self).__name__ == "MAPPO":
                next_values = self.policy.get_value(next_obs).reshape(1, self.num_agents*self.args.num_envs).to(self.device)
            else:
                raise ValueError("Unknown class name. Cannot determine if CMAPPO or MAPPO.")
        advantages = self.compute_gae(
            self.buffer.rewards_buff,
            self.buffer.dones_buff,
            self.buffer.values_buff,
            next_values=next_values
        )  # dim (num_steps, num_env*num_agents)

        # A + V = r_t + \gamma*V(s_t+1) - V(s_t) + V(s_t) = r_t + \gamma*V(s_t+1) = TD target estimate for V(s_t)
        value_target = advantages + self.buffer.values_buff  # dim (num_steps, num_env*num_agents)

        # flatten the buffer
        b_obs = self.buffer.obs_buff.reshape((-1, ) + self.env.single_observation_space.shape)  # (num_steps * num_env * num_agents, obs_dim)
        b_logprobs = self.buffer.logprobs_buff.reshape(-1)                                      # dim (num_steps * num_env * num_agents, )
        b_actions = self.buffer.actions_buff.reshape((-1, )  + self.env.single_action_space.shape)  # dim (num_steps * num_env * num_agents, action_dim)
        b_advantages = advantages.reshape(-1)  # dim (num_steps * num_env * num_agents, )        
        b_value_target = value_target.reshape(-1)  # dim (num_steps * num_env * num_agents, )
        b_values = self.buffer.values_buff.reshape(-1)  # dim (num_steps * num_env * num_agents, )

        assert self.batch_size == self.args.num_envs * self.num_agents * self.args.num_steps  # total number of samples
        b_inds = np.arange(self.batch_size)  # batch size is num_steps * num_env * num_agents
        clipfracs = []
        for _ in range(self.ppo_epoch):
            # Sample a batch of data from the buffer
            np.random.shuffle(b_inds)  # shuffle in place

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > self.batch_size:
                    end = self.batch_size

                mb_inds = b_inds[start:end]             # dim (mini_batch_size,)
                mb_obs = b_obs[mb_inds]  # dim (mini)
                mb_actions = b_actions.long()[mb_inds]
                
                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(mb_obs, mb_actions)
                """
                newlogprob shape (minibatch_size, num_agent)
                entropy shape (minibatch_size, num_agent)
                newvalue shape (minibatch_size, num_agent, 1)
                """
                logratio = newlogprob - b_logprobs[mb_inds]

                ratio = logratio.exp()  # dim (mini_batch_size, num_env*num_agents*num_steps)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()    # k1 approxiatino
                    approx_kl = ((ratio - 1) - logratio).mean()  # k3: lower variance than k1 estimator
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]  # dim (mini_batchLower variance than plain_size, num_agents)
                mb_value_targets = b_value_target[mb_inds]  # dim (mini_batch_size, num_env*num_agents*num_steps)


                # policy loss
                pg_loss1 = -mb_advantages * ratio  # dim (minibatch, num_env*num_agent*num_steps)
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (no clipping TODO: clip)
                newvalue = newvalue.squeeze()  # dim (minibatch_size, num_env*num_agents*num_steps)

                v_loss = self.compute_value_loss(mb_value_targets, newvalue)  # dim (mini_batch_size, num_env*num_agents*num_steps)


                entropy_loss = entropy.mean()  # dim (1)

                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.log:
                    self.summary_writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.num_gradient_steps)
                    self.summary_writer.add_scalar("losses/value_loss", v_loss.item(), self.num_gradient_steps)
                    self.summary_writer.add_scalar("losses/policy_loss", pg_loss.item(), self.num_gradient_steps)
                    self.summary_writer.add_scalar("losses/entropy", entropy_loss.item(), self.num_gradient_steps)
                    self.summary_writer.add_scalar("NN/policy_grad_norm", self.get_grad_norm(self.policy), self.num_gradient_steps)
                    self.summary_writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_gradient_steps)
                    self.summary_writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_gradient_steps)

                    #self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_gradient_steps)
                    #self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_gradient_steps)
                    #self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.num_gradient_steps)

        self.num_gradient_steps += 1

        if self.save_path is not None and self.num_gradient_steps % 100 == 0:
            bool_to_str = lambda x: "centralised" if x else "decentralised"
            final_path = os.path.join(PROJECT_ROOT, self.save_path, f"{bool_to_str(self.args.centralised)}_policy_{self.args.num_agents}_agents_{self.args.layout}_seed_{self.args.seed}.pth")

            torch.save(self.policy.state_dict(), final_path)
            print(f'saved model at {self.save_path}')
        # Reset the buffer
        self.buffer.reset()
    
    def get_grad_norm(self, nn):
        total_norm = 0
        for p in nn.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (0.5)

    def compute_value_loss(self, target, newvalue):
        """
        Compute the value loss.
        """
        v_loss = 0.5 * ((newvalue - target)**2).mean()
        return v_loss
