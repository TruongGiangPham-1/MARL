import functools

from cogrid.feature_space import feature_space
from cogrid.envs.overcooked import overcooked
from cogrid.core import layouts
from cogrid.envs import registry
from overcooked_config import four_agent_overcooked_config

# import supersuit
import supersuit as ss
from model import Agent 
import torch
import argparse
from MAPPO import MAPPO
from agent_environment import agent_environment_loop
from buffer import Buffer

# Finally, we register the environment with CoGrid. This makes it convenient
# to instantiate the environment from the registry as we do below, but you could
# also just pass the config to the Overcooked constructor directly.
registry.register(
    "FourAgentOvercooked-V0",
    functools.partial(
        overcooked.Overcooked, config=four_agent_overcooked_config
    ),
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU for training.",
    )
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the model')
    parser.add_argument('--save', action='store_true', default=False, help='Save the model')
    args = parser.parse_args()

    env = registry.make("FourAgentOvercooked-V0", render_mode="human")
    env.reset()
    #print(f'env obs {env.observation_spaces[0]["n_agent_overcooked_features"].shape}')  # (404)
    obs, R, terminated, truncated, info = env.step(
        {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.agents
        }
    )  
    print(f'{R}, {terminated}, {truncated}')
    
    """
    obs spaces:
    {0: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    1: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    2: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    3: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32))}

    action spaces:
    {0: Discrete(7), 1: Discrete(7), 2: Discrete(7), 3: Discrete(7)}

    o, r, t, t, i = env.step()

    r = {agent_id: R for agent_id in N}
    t = {agent_id: terminated for agent_id in N}
    t = {agent_id: truncated for agent_id in N}
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    net = Agent(env).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    buffer = Buffer(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], env.config["num_agents"], max_size=128)


    collect_steps = 5

    single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape  # 
    sigle_agent_action_dim = env.action_spaces[0].n  # int
    ppo_agent = MAPPO(env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, collect_steps=collect_steps, 
                       save_path=args.save_path)

    reward = agent_environment_loop(ppo_agent, env, device, num_episodes=1000)

    return
    
if __name__ == "__main__":
    main()