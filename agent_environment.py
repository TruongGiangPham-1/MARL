# loop
import torch


def agent_environment_loop(agent, env, device, num_episodes=1000):
    """
    agent: mappo agent
    """
    collect_steps = agent.collect_steps
    num_updates = 5    

    # TODO: move this to replay buffer
    obs_buff = torch.zeros((collect_steps, agent.num_agents, agent.single_agent_obs[0])).to(device)
    actions_buff = torch.zeros((collect_steps, agent.num_agents)).to(device)
    rewards_buff = torch.zeros((collect_steps, agent.num_agents)).to(device)
    dones_buff = torch.zeros((collect_steps, agent.num_agents)).to(device)
    logprobs_buff = torch.zeros((collect_steps, agent.num_agents)).to(device)
    values_buff = torch.zeros((collect_steps, agent.num_agents)).to(device)

    obs, info = env.reset()  # obs is a dict of obs for each agent
    next_obs = torch.stack([     torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
    next_dones = torch.zeros((agent.num_agents,)).to(device)
    for episode in range(num_episodes):
        for step in range(collect_steps):
            obs_buff[step] = next_obs
            dones_buff[step] = next_dones

            actions, logprobs, _, values = agent.act(next_obs)  # with no grad action dim (num_agents,)
            """
            actions dim (num_agents,)
            logprobs dim (num_agents,)
            values dim (num_agents, 1)
            """
            env_action = {i: action for i, action in enumerate(actions)}

            # Store the data
            actions_buff[step] = actions  # dim (num_agents,)
            logprobs_buff[step] = logprobs # dim (num_agents,)
            next_obs, rewards, terminated, truncated, info = env.step(env_action)
            """
            rewards = {agent_id: R for agent_id in N}
            terminated = {agent_id: terminated for agent_id in N}
            truncated = {agent_id: truncated for agent_id in N}
            """

            rewards_buff[step] = torch.tensor([rewards[i] for i in range(agent.num_agents)]).to(device)  # dim (num_agents,)
            next_dones = torch.tensor([terminated[i] or truncated[i] for i in range(agent.num_agents)]).to(device)
            dones_buff[step] = next_dones  # dim (num_agents,)
            values_buff[step] = values.squeeze(1)  # dim (num_agents,)
            
            next_obs = torch.stack([   torch.FloatTensor(next_obs[i]['n_agent_overcooked_features']) for i in range(agent.num_agents)], dim=0).to(device)
        # compute GAE
        _, _, _, next_values = agent.act(next_obs)  # with no grad dim (num_agents,)
        # advantage

        # Update the agent with the collected data
        for update in range(num_updates):
            agent.update(obs_buff, actions_buff, rewards_buff, dones_buff, logprobs_buff, values_buff)
    return []


def collect(next_obs, next_dones):
    """
    Collects trajectories from the environment.

    Args:
        obs (torch.Tensor): Observation tensor.

    Returns:
        trajectory (dict): Collected trajectory data (obs, actions, rewards, etc.).
    """


    for step in range(self.collect_steps): 
        self.obs[step] = next_obs
        self.dones[step] = next_dones
        with torch.no_grad():
            actions, logprobs, _, values = self.policy.get_action_and_value(next_obs)
        self.actions[step] = actions
        assert actions.shape == (self.env.config['num_agents'],)
        self.logprobs[step] = logprobs
        self.values[step] = values


        # Take a step in the environment
        env_action = {i: action for i, action in enumerate(actions)}
        next_obs, rewards, terminated, truncated, info = self.env.step(env_action)