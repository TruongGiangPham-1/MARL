# loop
import os
import torch
import numpy as np
import imageio

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import evaluate_state

def agent_environment_loop(agent, env, device, num_update=1000, log_dir=None, args=None):
    """
    agent: mappo agent
    """
    summary_writer = SummaryWriter(log_dir=log_dir)
    collect_steps = agent.batch_size
    num_updates = 5    

    # TODO: move this to replay buffer
    episodes_reward = []
    episode_reward = 0  # undiscount reward
    num_episdes = 0
    frequency_delivery_per_episode = 0
    frequency_plated_per_episode = 0  # how many times agent plated the food
    frequency_ingredient_in_pot_per_episode = 0  # how many times agent put ingredient in pot

    frequency_delivery_per_episode_list = []
    frequency_plated_per_episode_list = []
    frequency_ingredient_in_pot_per_episode_list = []

    # action prob frames
    action_prob_frames = []

    obs, info = env.reset()  # obs is a dict of obs for each agent
    obs = torch.stack([     torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
    dones = torch.zeros((agent.num_agents,)).to(device)
    global_step = 0
    
    for _ in tqdm(range(num_update)):
        for step in range(collect_steps):
            actions, logprobs, _, values = agent.act(obs)  # with no grad action dim (num_agents,)
            """
            actions dim (num_agents,)
            logprobs dim (num_agents,)
            values dim (num_agents, 1)
            """
            env_action = {i: action for i, action in enumerate(actions)}


            next_obs, rewards, terminated, truncated, info = env.step(env_action)
            """
            rewards = {agent_id: R for agent_id in N}
            terminated = {agent_id: terminated for agent_id in N}
            truncated = {agent_id: truncated for agent_id in N}
            """
            

            rewards = torch.tensor([rewards[i] for i in range(agent.num_agents)]).to(device)  # dim (num_agents,)

            # if there is 1 in rewards tensor, print hello
            if torch.any(rewards >= 1):
                frequency_delivery_per_episode += 1
                print(f'global_step {global_step} agent sucessfully delievered. rewards {rewards}')
            if torch.any(rewards == 0.3):
                frequency_plated_per_episode += 1
            if torch.any(rewards == 0.1):
                frequency_ingredient_in_pot_per_episode += 1
            episode_reward += rewards.float().mean().item()  # undiscounted reward

            #if rewards.float().mean().item() > 0:
            #    print(f'global_step {global_step} rewards {rewards.float().mean().item()} non 0')


            agent.add_to_buffer(obs, actions, rewards, dones, logprobs, values.squeeze(1))

            if torch.all(dones):
                # handle reset 
                next_obs, info = env.reset()
                episodes_reward.append(episode_reward)
                if log_dir is not None:
                    summary_writer.add_scalar('episode_rewards', episode_reward, num_episdes)
                    summary_writer.add_scalar('freq/frequency_delivery_per_episode', frequency_delivery_per_episode, num_episdes)
                    summary_writer.add_scalar('freq/frequency_plated_per_episode', frequency_plated_per_episode, num_episdes)
                    summary_writer.add_scalar('freq/frequency_ingredient_in_pot_per_episode', frequency_ingredient_in_pot_per_episode, num_episdes)
                
                frequency_delivery_per_episode_list.append(frequency_delivery_per_episode)
                frequency_plated_per_episode_list.append(frequency_plated_per_episode)
                frequency_ingredient_in_pot_per_episode_list.append(frequency_ingredient_in_pot_per_episode)

                episode_reward = 0
                frequency_delivery_per_episode = 0
                frequency_plated_per_episode = 0
                frequency_ingredient_in_pot_per_episode = 0
                num_episdes += 1

            obs = torch.stack([   torch.FloatTensor(next_obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
            dones = torch.tensor([terminated[i] or truncated[i] for i in range(agent.num_agents)]).to(device)

            global_step += 1

        # Update the agent with the collected data
        agent.update(obs)

        image = evaluate_state(agent, env, device, global_step=global_step)
        image = imageio.imread(image)
        action_prob_frames.append(image)
    
    freq_dict = {
        'frequency_delivery_per_episode': frequency_delivery_per_episode_list,
        'frequency_plated_per_episode': frequency_plated_per_episode_list,
        'frequency_ingredient_in_pot_per_episode': frequency_ingredient_in_pot_per_episode_list
    }

    # save gif
    imageio.mimsave(f"data/{args.num_agents}_{args.layout}_seed_{args.seed}_action_prob_frames.gif", action_prob_frames)
    return episodes_reward, freq_dict




def mpe_environment_loop(agent, env, device, num_episodes=1000, log_dir=None):
    """
    agent: mappo agent
    """
    collect_steps = agent.collect_steps
    num_updates = 5
    obs, info = env.reset()

    obs = torch.stack([   torch.FloatTensor(obs[a]) for a in (env.possible_agents)], dim=0).to(device)  # (num_agents, 18)
    dones = torch.zeros((env.num_agents,)).to(device)
    for episode in range(num_episodes):
        for step in range(collect_steps):
            actions, logprobs, _, values = agent.act(obs)  # with no grad action dim (num_agents,)
            """
            actions dim (num_agents,)
            logprobs dim (num_agents,)
            values dim (num_agents, 1)
            """
            env_action = {a: action.cpu().numpy() for a, action in zip(env.possible_agents, actions)}
            print(f'env_action {env_action}')


            next_obs, rewards, terminated, truncated, info = env.step(env_action)
            """
            next_obs = {agent_id: obs for agent_id in N}          {} if torch.all(dones)
            rewards = {agent_id: R for agent_id in N}             {} if torch.all(dones)
            terminated = {agent_id: terminated for agent_id in N}
            truncated = {agent_id: truncated for agent_id in N}   {} if torch.all(dones)

            TODO: how can I handle truncated and terminated. when rewards is empty.
            Simple_spread reward is sum of all agents distance to the target. So making terminated reward 0 is a problem.
            """
            full_rewards = torch.zeros((env.num_agents)).to(device)
            for aval_agent_str in env.agents:
                agent_id = int(aval_agent_str.split('_')[1])
                full_rewards[agent_id] = rewards[aval_agent_str]
                

            print(f'rewards before {rewards} env.agents {env.possible_agents} done {dones} truncated {truncated}')
            rewards = full_rewards

            #print(f'size of stuff adding to buffer {obs.shape}, {actions.shape}, {rewards.shape}, {dones.shape}, {logprobs.shape}, {values.squeeze(1).shape}')
            agent.add_to_buffer(obs, actions, rewards, dones, logprobs, values.squeeze(1))


            if torch.all(dones):
                # handle reset 
                next_obs, info = env.reset()
                obs = torch.stack([   torch.FloatTensor(next_obs[a]) for a in (env.possible_agents)], dim=0).to(device)
            obs = torch.stack([   torch.FloatTensor(next_obs[a]) for a in (env.possible_agents)], dim=0).to(device)
            dones = torch.tensor([terminated[a] or truncated[a] for a in (env.possible_agents)]).to(device)


        # Update the agent with the collected data
        agent.update(obs)
    return []



"""
Problem: in MPE, 

"""