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

    env = registry.make("FourAgentOvercooked-V0")
    nn = Agent(env).to(device)  # neural network
    nn.load_state_dict(torch.load(args.model_path))

    mappo = MAPPO(env, None, nn, None, None, None)
    obs, info = env.reset() 
    """
    obs: {
        0: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        1: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        2: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        3: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32))
    }
    """
    state = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)


    action = mappo.act(state)
    """
    action: {
        0: Discrete(7), 
        1: Discrete(7), 
        2: Discrete(7), 
        3: Discrete(7)
    }
    """
    return


if __name__ == "__main__":
    main()