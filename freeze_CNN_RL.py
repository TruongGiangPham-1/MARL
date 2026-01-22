import time
from model import Agent
import gymnasium as gym
import argparse
import torch
import numpy as np
from buffer import Buffer
from main import make_env
from MAPPO import MAPPO
from semi_gradient_sarsa import SemiGradientSARSA_NN, NNEpsilonGreedyExplorer
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrozenNeuralExtractor:
    def __init__(self, pytorch_model):
        self.model = pytorch_model
        self.model.eval() # Set to evaluation mode

    def __call__(self, state):
        # 1. Convert state to torch tensor
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # 2. Forward pass through the FROZEN part only
        with torch.no_grad():
            features = self.model.network(state_t) # Using the 'network' block from your Agent
            
        # 3. Return as a flat numpy array for your self.w math
        return features.cpu().numpy().flatten()
    
# Assuming 'agent' is your instance
def reinitialize_head(layer):
    if isinstance(layer, nn.Linear):
        # Orthogonal initialization is standard for RL
        nn.init.orthogonal_(layer.weight, gain=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/policy.pth",
        help="Path to the trained model",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--layout", type=str, default="overcooked_cramped_room_v0", help="Layout of the Overcooked environment")

    num_agents = 2
    args = parser.parse_args()
    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(202,), dtype=np.float32)
    action_space = gym.spaces.Discrete(7)

    env = make_env(num_agents=2, layout="overcooked_cramped_room_v0", feature="global_obs", render_mode=None)
    nn = Agent(obs_space, action_space, num_agents=2, num_envs=16).to(device)
    buffer = Buffer(obs_space.shape[0], 2, 1, max_size=256)
    mappo = MAPPO(env, None, nn, None, None, None, num_agents=num_agents)  # THE RL AGENT

    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    nn.load_state_dict(checkpoint)  # load the model

    # 1. Freeze the shared representation network
    for param in nn.network.parameters():
        param.requires_grad = False

    #2. (Optional) If you also want to freeze the centralized critic
    for param in nn.centralised_critics.parameters():
        param.requires_grad = False
    
    # 3. Reinitialize the actor and critic heads
    reinitialize_head(nn.actor)
    reinitialize_head(nn.critic)

    # Only pass parameters that have requires_grad=True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, nn.parameters()), 
        lr=1e-4
    )

    # Verify what is trainable
    for name, param in nn.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")    

    obs, info = env.reset()
    num_agents = 1
    log_dir = "logs"
    summaries_writer = SummaryWriter(log_dir)
    feature_extractor = FrozenNeuralExtractor(nn)
    num_state_action_features = 256
    explorer = NNEpsilonGreedyExplorer(num_actions=action_space.n, epsilon_start=0.1, epsilon_end=0.1, decay_steps=10000)
    agent = SemiGradientSARSA_NN(
        agent=nn,
        explorer=explorer,
        step_size=0.01,
        discount=0.99,
        n=3,
        log_dir=log_dir
    )
    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)  # (1, 202)
    # TD loop
    episodes_rewards = []
    for episode in tqdm(range(args.episodes)):
        done = False
        obs, info = env.reset()
        obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)  # (1, 202)
        state = obs
        action = agent.act(state)
        episode_reward = 0
        while not done:
            env_action = {0: action, 1: 6}  # other agent does no-op
            next_obs, rewards, terminated, truncated, info = env.step(env_action)
            #env.render()
            next_state = torch.stack([   torch.FloatTensor(next_obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)
            reward = rewards[0]
            episode_reward += reward
            done = terminated[0] or truncated[0]
            next_action = agent.act(next_state) if not done else None

            # Update the agent
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            #print(f"Episode {episode+1}, Reward: {reward}, Done: {done} action {action}")
            #time.sleep(5)
        print(f"episode {episode} reward {episode_reward}")
        episodes_rewards.append(episode_reward)
    
    # plot the rewards
    import matplotlib.pyplot as plt
    plt.plot(episodes_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards over Time")
    plt.savefig("episode_rewards.png")
    plt.close()



    #while True:
    #    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)
    #    actions, _, _, _ = mappo.act(obs)  # actions is a tensor of shape (num_agents,)
    #    env_action = {i: action for i, action in enumerate(actions)}
    #    obs, rewards, terminated, truncated, info = env.step(env_action)
    #    env.render()
    #    done = torch.tensor([terminated[i] or truncated[i] for i in range(num_agents)]).to(device)
    #    if torch.all(done):
    #        obs, info = env.reset()  # obs is a dict of obs for each agentj
    #        break
    return

if __name__ == '__main__':
    main()