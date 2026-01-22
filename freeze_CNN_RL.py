
from model import Agent
import gymnasium as gym
import argparse
import torch
import numpy as np
from buffer import Buffer
from main import make_env
from MAPPO import MAPPO
from semi_gradient_sarsa import SemiGradientSARSA_NN
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

    feature_extractor = FrozenNeuralExtractor(nn)
    num_state_action_features = 256
    agent = SemiGradientSARSA_NN(
        agent=nn,
        step_size=0.01,
        discount=0.99,
        n=3
    )
    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(num_agents)], dim=0).to(device)  # (1, 202)
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