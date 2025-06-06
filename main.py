import functools
import time

from cogrid.feature_space import feature_space
from cogrid.envs.overcooked import overcooked
from cogrid.core import layouts
from cogrid.envs import registry
from overcooked_config import N_agent_overcooked_config
import random
import numpy as np

# import supersuit
import supersuit as ss
from model import Agent 
import torch
import argparse
from MAPPO import MAPPO
from CentralizedMAPPO import CMAPPO
from agent_environment import agent_environment_loop
from buffer import Buffer
from plot import plot_alg_results
import pandas as pd

from utils import concat_vec_envs_v1


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

def make_vector_env(num_envs, overcooked_env):
    """
    Create a vectorized environment with the specified number of environments.
    """
    overcooked_env = ss.pettingzoo_env_to_vec_env_v1(overcooked_env)  # Convert the Overcooked environment to a vectorized environment
    print(f'env.observation_spaces: {overcooked_env.observation_space}')  # Check observation spaces
    print(f'env.action_space: {overcooked_env.action_space}')  # Check action spaces
    envs = concat_vec_envs_v1(
        overcooked_env,
        num_vec_envs=num_envs // 2,  # if num_envs is 8 actual number of envs is 4
        num_cpus=num_envs,  # Use a single CPU for vectorized environments
        base_class="gymnasium",  # Use gymnasium as the base class
    )
    envs.single_observation_space = overcooked_env.observation_space['n_agent_overcooked_features']  # Set the single observation space
    envs.single_action_space = overcooked_env.action_space  # Set the single action space

    out = envs.reset()
    print(f'Observation after reset: {out[0][
        'n_agent_overcooked_features'
    ].shape}')  #  (num_envs, obs_shape)

    next_obs, R, terminated, truncated, info = envs.step(

    )
    return envs


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
    parser.add_argument('--log', action='store_true', default=False, help='log the training to tensorboard')
    parser.add_argument('--render', action='store_true', default=False, help='render the env')
    parser.add_argument('--seed', type=int, default=1,  help='seed')
    
    # ppo args
    """
            ppo_epoch=10,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            gamma=0.99,
            lam=0.95,
    
    """
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ppo-epoch', type=int, default=10, help='number of ppo epochs')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max gradient norm')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='lambda for GAE')
    


    parser.add_argument('--centralised', action='store_true', default=False, help='False is decentralised, True is centralised')
    args = parser.parse_args()
    print(f'num_agents: {args.num_agents}, layout: {args.layout}, save_path: {args.save_path}, batch_size: {args.batch_size}')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    render_mode = "human" if args.render else None
    env = make_env(args.num_agents, layout=args.layout, render_mode=render_mode)
    #vec_env = make_vector_env(num_envs=8, overcooked_env=env)  # create vectorized environment with 2 envs
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Agent(obs_space, action_space, num_agents=args.num_agents).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = Buffer(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], env.config["num_agents"], max_size=args.batch_size)

    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)  # contain .csv files of returns
    log_dir = f"logs/run__{int(time.time())}"

    single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape  # 
    sigle_agent_action_dim = env.action_spaces[0].n  # int
    if not args.centralised:
        print(f'Using decentralised critic')
        ppo_agent = MAPPO(env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=args.batch_size, 
                          num_mini_batches=args.num_minibatches, ppo_epoch=args.ppo_epoch, clip_param=args.clip_param,
                        value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        gamma=args.gamma, lam=args.lam,
                        save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents, log=args.log, args=args)
    else:
        print(f'Using centralised critic')
        ppo_agent = CMAPPO(env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=args.batch_size, 
                           num_mini_batches=args.num_minibatches, ppo_epoch=args.ppo_epoch, clip_param=args.clip_param,
                        value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                        gamma=args.gamma, lam=args.lam,
                        save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents, log=args.log, args=args)
    episode_returns, freq_dict = agent_environment_loop(ppo_agent, env, device, num_update=args.total_steps // args.batch_size, log_dir=log_dir,
                                                        args=args)
    print(f'episode returns {episode_returns}')
    #plot_alg_results(episode_returns, f"results/{args.num_agents}_{args.layout}.png", label="PPO", ylabel="Return")
    df = pd.DataFrame(episode_returns)
    df.to_csv(f'data/{args.num_agents}_{args.layout}_returns_seed_{args.seed}.csv', index=False)

    df = pd.DataFrame(freq_dict["frequency_delivery_per_episode"])
    df.to_csv(f'data/{args.num_agents}_{args.layout}_frequency_delivery_per_episode_seed_{args.seed}.csv', index=False)
    df = pd.DataFrame(freq_dict["frequency_plated_per_episode"])
    df.to_csv(f'data/{args.num_agents}_{args.layout}_frequency_plated_per_episode_seed_{args.seed}.csv', index=False)
    df = pd.DataFrame(freq_dict["frequency_ingredient_in_pot_per_episode"])
    df.to_csv(f'data/{args.num_agents}_{args.layout}_frequency_ingredient_in_pot_per_episode_seed_{args.seed}.csv', index=False)

    # save args to file
    with open(f'data/{args.num_agents}_{args.layout}_args_seed_{args.seed}.txt', 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    return
    
if __name__ == "__main__":
    main()