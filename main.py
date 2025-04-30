import functools
import time

from cogrid.feature_space import feature_space
from cogrid.envs.overcooked import overcooked
from cogrid.core import layouts
from cogrid.envs import registry
from overcooked_config import N_agent_overcooked_config

# import supersuit
import supersuit as ss
from model import Agent 
import torch
import argparse
from MAPPO import MAPPO
from CentralizedMAPPO import CMAPPO
from agent_environment import agent_environment_loop
from buffer import Buffer


def make_env(num_agents=4, layout="large_overcooked_layout", render_mode="human"):
    config = N_agent_overcooked_config.copy()  # get config obj
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout

    # Finally, we register the environment with CoGrid. This makes it convenient
    # to instantiate the environment from the registry as we do below, but you could
    # also just pass the config to the Overcooked constructor directly.
    registry.register(
        "NAgentOvercooked-V0",
        functools.partial(
            overcooked.Overcooked, config=config
        ),
    )
    return registry.make(
        "NAgentOvercooked-V0",
        render_mode=render_mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU for training.",
    )
    parser.add_argument('--num-agents', type=int, default=4, help='number of agents')
    parser.add_argument('--layout', type=str, default='large_overcooked_layout', help='layout')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the model')
    parser.add_argument('--save', action='store_true', default=False, help='Save the model')
    parser.add_argument('--total-steps', type=int, default=1000, help='total env steps')
    parser.add_argument('--batch-size', type=int, default=5, help='number of sample to collect before update')
    parser.add_argument('--num-minibatches', type=int, default=4, help='')

    parser.add_argument('--centralised', action='store_true', default=False, help='False is decentralised, True is centralised')
    args = parser.parse_args()
    print(f'num_agents: {args.num_agents}, layout: {args.layout}, save_path: {args.save_path}, batch_size: {args.batch_size}')
    env = make_env(args.num_agents, layout=args.layout)
    env.reset()

    obs_space = env.observation_spaces[0]['n_agent_overcooked_features']  # box (-inf, inf, (404,), float32)
    action_space = env.action_spaces[0]  # Discrete(7)
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
    net = Agent(obs_space, action_space, num_agents=args.num_agents).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    buffer = Buffer(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], env.config["num_agents"], max_size=128)

    import os
    os.makedirs("logs", exist_ok=True)
    log_dir = f"logs/run__{int(time.time())}"

    single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape  # 
    sigle_agent_action_dim = env.action_spaces[0].n  # int
    if not args.centralised:
        print(f'Using decentralised critic')
        ppo_agent = MAPPO(env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=args.batch_size, 
                          num_mini_batches=args.num_minibatches,
                        save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents)
    else:
        print(f'Using centralised critic')
        ppo_agent = CMAPPO(env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=args.batch_size, 
                           num_mini_batches=args.num_minibatches,
                        save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents)
    reward = agent_environment_loop(ppo_agent, env, device, num_update=args.total_steps // args.batch_size)

    return
    
if __name__ == "__main__":
    main()