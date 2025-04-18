# loop
import torch


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


