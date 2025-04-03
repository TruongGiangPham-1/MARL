# loop
import torch


def agent_environment_loop(agent, env, device, num_episodes=1000):
    """
    agent: mappo agent
    """


    return []

def collect(agent, env, device):
    """
    Collects trajectories from a multi-agent environment for PPO.

    Args:
        agent (nn.Module): Shared policy network. TODO: dsitribute later
        env (PettingZoo Environment): Multi-agent environment.
        device (torch.device): CPU/GPU device.

    Returns:
        trajectory (dict): Collected trajectory data (obs, actions, rewards, etc.).
    """
    # Initialize trajectory storage
    trajectory = {
        "obs": [], "actions": [], "log_probs": [], "rewards": [], "values": [], "dones": []
    }

    obs, info = env.reset()
    done = {agent_id: False for agent_id in env.config['num_agents']}  # Track done per agent

    while not all(done.values()):  # Continue until all agents are done
        obs_tensor = torch.tensor(
            [obs[agent_id] for agent_id in env.agents], dtype=torch.float32
        ).to(device)  # Shape: (n_agents, obs_dim)

        with torch.no_grad():
            actions, log_probs, _, values = agent.get_action_and_value(obs_tensor)  # (n_agents,)

        # Convert to PettingZoo's action dict format
        env_actions = {agent_id: actions[i].item() for i, agent_id in enumerate(env.agents)}

        # Step environment
        next_obs, rewards, terminated, truncated, _ = env.step(env_actions)

        # Store trajectory data
        trajectory["obs"].append(obs_tensor)
        trajectory["actions"].append(actions)
        trajectory["log_probs"].append(log_probs)
        trajectory["values"].append(values)
        trajectory["rewards"].append(torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float32).to(device))
        trajectory["dones"].append(torch.tensor([done[agent] for agent in env.agents], dtype=torch.float32).to(device))

        # Move to the next observation
        obs = next_obs

    # Convert lists to tensors
    for key in trajectory:
        trajectory[key] = torch.stack(trajectory[key])  # (timesteps, n_agents, ...)

    return trajectory