import numpy as np
from model import Agent
from overcooked_config import four_agent_overcooked_config
import functools
from cogrid.envs.overcooked import overcooked
from cogrid.envs import registry
import argparse
import torch


from MAPPO import MAPPO
#env

registry.register(
    "FourAgentOvercooked-V0",
    functools.partial(
        overcooked.Overcooked, config=four_agent_overcooked_config
    ),
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/policy.pth",
        help="Path to the trained model",
    )
    args = parser.parse_args()

    env = registry.make("FourAgentOvercooked-V0")  # the overcooked game engine
    nn = Agent(env).to(device)  # neural network
    nn.load_state_dict(torch.load(args.model_path))

    mappo = MAPPO(env, None, nn, None, None, None)  # THE RL AGENT
    obs, info = env.reset() 
    """
    observvation for each agents, each agent's state vector is sized 404

    obs: {
        0: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        1: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        2: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        3: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32))
    }
    """
    # combine the observations of all agents into a single tensor. shape (num_agents, 404)
    state = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)


    # actions for each agents. Each agent action is in the range of 0 to 6
    action, _, _, _ = mappo.act(state)  # action is a vector with dimention (num_agents,)

    action = {  # 
        "agent_0": action[0].item(),   # Discrete(7)
        "agent_1": action[1].item(),   # Discrete(7)
        "agent_2": action[2].item(),   # Discrete(7)
        "agent_3": action[3].item()    # Discrete(7)
    }

    print(f'action: {action}')
    return


if __name__ == "__main__":
    main()