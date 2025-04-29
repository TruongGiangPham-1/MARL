# loop
import torch
import numpy as np


def agent_environment_loop(agent, env, device, num_episodes=1000, log_dir=None):
    """
    agent: mappo agent
    """
    collect_steps = agent.collect_steps
    num_updates = 5    

    # TODO: move this to replay buffer

    obs, info = env.reset()  # obs is a dict of obs for each agent
    obs = torch.stack([     torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
    dones = torch.zeros((agent.num_agents,)).to(device)

    for episode in range(num_episodes):
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

            agent.add_to_buffer(obs, actions, rewards, dones, logprobs, values.squeeze(1))


            obs = torch.stack([   torch.FloatTensor(next_obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
            dones = torch.tensor([terminated[i] or truncated[i] for i in range(agent.num_agents)]).to(device)

        # Update the agent with the collected data
        agent.update(obs)
    return []

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