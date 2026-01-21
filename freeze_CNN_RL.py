
from model import Agent
import gymnasium as gym
import argparse
import torch
import numpy as np
from buffer import Buffer
from main import make_env
from MAPPO import MAPPO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/policy.pth",
        help="Path to the trained model",
    )
    num_agents = 2
    args = parser.parse_args()
    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(202,), dtype=np.float32)
    action_space = gym.spaces.Discrete(7)

    env = make_env(num_agents=2, layout="overcooked_cramped_room_v0", feature="global_obs")
    agent = Agent(obs_space, action_space, num_agents=2, num_envs=16).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4, betas=(0.9, 0.95))
    buffer = Buffer(obs_space.shape[0], 2, 1, max_size=256)
    mappo = MAPPO(env, None, agent, None, None, None, num_agents=num_agents)  # THE RL AGENT

    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(checkpoint)  # load the model

    # 1. Freeze the shared representation network
    for param in agent.network.parameters():
        param.requires_grad = False

    #2. (Optional) If you also want to freeze the centralized critic
    for param in agent.centralised_critics.parameters():
        param.requires_grad = False

    # Verify what is trainable
    for name, param in agent.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")    

    obs, info = env.reset()
    while True:
        obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)
        actions, _, _, _ = mappo.act(obs)  # actions is a tensor of shape (num_agents,)
        env_action = {i: action for i, action in enumerate(actions)}
        obs, rewards, terminated, truncated, info = env.step(env_action)
        env.render()
        done = torch.tensor([terminated[i] or truncated[i] for i in range(num_agents)]).to(device)
        if torch.all(done):
            obs, info = env.reset()  # obs is a dict of obs for each agentj
            break
        
    return

if __name__ == '__main__':
    main()