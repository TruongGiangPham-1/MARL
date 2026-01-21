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
from QMIX import QMIX
from semi_gradient_sarsa import SemiGradientSARSA, ConstantEpsilonGreedyExploration, SarsaFeatureExtractor
from agent_environment import agent_environment_loop, qmix_environment_loop, agent_environment_sarsa_loop
from buffer import Buffer
from plot import plot_alg_results
import pandas as pd

from utils import concat_vec_envs_v1
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f'PROJECT_ROOT: {PROJECT_ROOT}')

def make_env(num_agents=4, layout="large_overcooked_layout", feature="global_obs", render_mode="human"):
    """
    Normal overcooked envs
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
    config = N_agent_overcooked_config.copy()  # get config obj
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout
    config["features"] = feature  # set the feature to use for the environment

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

    vec_obs: {'n_agent_overcooked_features': Box(-inf, inf, (num_agent*num_env, obshape), float32)}
    
    step(actions \in (num_envs*num_agents,)), corresponding

    Note:
    pettingzoo_env_to_vec_env_v1(overcooked_env)  # sets num_envs to num_agents
    concat_vec_envs_v1()                          # sets num_envs to num_agents * num_envs
    num_envs is super misleading variable so dont trust. The source code for ProcConcatVec properly initializes actual number of env

    
    """

    overcooked_env = ss.pettingzoo_env_to_vec_env_v1(overcooked_env)  # Convert the Overcooked environment to a vectorized environment
    print(f'env.observation_spaces: {overcooked_env.observation_space}')  # Check observation spaces
    print(f'env.action_space: {overcooked_env.action_space}')  # Check action spaces
    print(f'env_to_vec_env.num_envs: {overcooked_env.num_envs}')  # == num_agents
    envs = concat_vec_envs_v1(
        overcooked_env,
        num_vec_envs=num_envs,  # if num_envs is 8 actual number of envs is 4
        num_cpus=num_envs,  # Use a single CPU for vectorized environments
        base_class="gymnasium",  # Use gymnasium as the base class
    )
    envs.single_observation_space = overcooked_env.observation_space['n_agent_overcooked_features']  # Set the single observation space
    envs.single_action_space = overcooked_env.action_space  # Set the single action space

    out = envs.reset()
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
    parser.add_argument('--num-envs', type=int, default=8, help='number of env')
    parser.add_argument('--layout', type=str, default='large_overcooked_layout', help='layout')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the model')
    parser.add_argument('--data-path', type=str, default='data', help='Path to save the csv files for plotting')
    parser.add_argument('--save', action='store_true', default=False, help='Save the model')
    parser.add_argument('--total-steps', type=int, default=1000, help='total env steps')
    #parser.add_argument('--batch-size', type=int, default=5, help='number of sample to collect before update')
    parser.add_argument('--num-steps' , type=int, default=128, help='number of steps per environment before update')
    parser.add_argument('--num-minibatches', type=int, default=4, help='')
    parser.add_argument('--log', action='store_true', default=False, help='log the training to tensorboard')
    parser.add_argument('--render', action='store_true', default=False, help='render the env')
    parser.add_argument('--seed', type=int, default=1,  help='seed')
    parser.add_argument("--feature", type=str, default="global_obs", help="feature to use for the environment")
    
    # ppo args
    """
            ppo_epoch=10,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            gamma=0.99,
            lam=0.95,
    
    num_steps: number of steps per environment before update
    buffer (num_steps=128, num_envs=8, num_agents=4, obs_shape=(404,), action_shape=(7,), device='cpu')
    batch_size=num_step*num_env
    minibatch_size=batch_size/num_minibatches
    num_update=1,000,000 // batch_size
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
    parser.add_argument('--algorithm', type=str, default='mappo', choices=['mappo', 'cmappo', 'qmix', 'sarsa'], help='Algorithm to use')
    
    # QMIX specific arguments
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Final epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target-update-freq', type=int, default=200, help='Target network update frequency')
    parser.add_argument('--buffer-size', type=int, default=5000, help='Experience replay buffer size')
    parser.add_argument('--batch-size-qmix', type=int, default=32, help='Batch size for QMIX')
    parser.add_argument('--mixing-embed-dim', type=int, default=32, help='Mixing network embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes for QMIX training')

    # Semi-Gradient SARSA specific arguments
    parser.add_argument('--step-size', type=float, default=0.01, help='Step size for Semi-Gradient SARSA')
    
    args = parser.parse_args()
    print(f'num_agents: {args.num_agents}, layout: {args.layout}, save_path: {args.save_path}, algorithm: {args.algorithm}')

    batch_size = args.num_envs * args.num_agents * args.num_steps  # number of samples to collect before update

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # ENV stuff ----------
    render_mode = "human" if args.render else None
    env = make_env(args.num_agents, layout=args.layout, feature=args.feature, render_mode=render_mode)
    vec_env = make_vector_env(num_envs=args.num_envs, overcooked_env=env)  # create vectorized environments.
    assert vec_env.num_envs == args.num_envs*args.num_agents, f"Number of environments {vec_env.num_envs} does not match num_envs*args.num_agents {args.num_envs*args.num_agents}"
    vec_env.reset()

    #obs, R, terminated, truncated, info = vec_env.step(
    #    np.random.randint(0, vec_env.single_action_space.n, size=(args.num_agents*args.num_envs,))  # random actions for each env
    #)  
    #print(f"reward shape {R.shape}, terminated shape {terminated.shape}, truncated shape {truncated.shape} next_obs shape {obs['n_agent_overcooked_features'].shape}")
    #print(f"obs {obs}, shape is {obs['n_agent_overcooked_features'].shape}")

    obs_space = vec_env.single_observation_space  # get the observation space
    action_space = vec_env.single_action_space  # get the action space
    # ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, args.data_path), exist_ok=True)  # contain .csv files of returns
    log_dir = f"logs/run__{int(time.time())}"

    single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape  # 
    sigle_agent_action_dim = env.action_spaces[0].n  # int
    
    # Select algorithm
    if args.algorithm == 'qmix':
        print('Using QMIX algorithm')
        # For QMIX, we assume num_envs = 1 for simplicity
        if args.num_envs != 1:
            print(f"Warning: QMIX implementation assumes num_envs=1, but got num_envs={args.num_envs}. Setting num_envs=1.")
            args.num_envs = 1
            # Recreate vectorized environment with num_envs=1
            vec_env = make_vector_env(num_envs=1, overcooked_env=env)
            vec_env.reset()
        
        obs_dim = single_agent_obs_dim[0]
        action_dim = sigle_agent_action_dim
        state_dim = args.num_agents * obs_dim  # Use concatenated observations as global state
        
        # For QMIX, we don't need the batch_size calculation from PPO
        save_path_qmix = None
        if args.save_path:
            save_path_qmix = os.path.join(PROJECT_ROOT, args.save_path, f"qmix_{args.num_agents}_agents_{args.layout}_seed_{args.seed}")
        
        agent = QMIX(
            env=vec_env,
            num_agents=args.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size_qmix,
            mixing_embed_dim=args.mixing_embed_dim,
            hidden_dim=args.hidden_dim,
            save_path=save_path_qmix,
            log_dir=log_dir,
            log=args.log,
            args=args
        )
    elif args.algorithm == 'mappo' or (args.algorithm == 'mappo' and not args.centralised):
        print('Using decentralised MAPPO')
        net = Agent(obs_space, action_space, num_agents=args.num_agents, num_envs=args.num_envs).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.95))
        buffer = Buffer(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], env.config["num_agents"], args.num_envs, max_size=args.num_steps)
        
        agent = MAPPO(vec_env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=batch_size,
                      num_mini_batches=args.num_minibatches, ppo_epoch=args.ppo_epoch, clip_param=args.clip_param,
                    value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                    gamma=args.gamma, lam=args.lam,
                    save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents, log=args.log, args=args)
        num_updates = args.total_steps // batch_size
    elif args.algorithm == 'cmappo' or (args.algorithm == 'mappo' and args.centralised):
        print('Using centralised MAPPO')
        net = Agent(obs_space, action_space, num_agents=args.num_agents, num_envs=args.num_envs).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.95))
        buffer = Buffer(env.observation_spaces[0]['n_agent_overcooked_features'].shape[0], env.config["num_agents"], args.num_envs, max_size=args.num_steps)
        
        agent = CMAPPO(vec_env, optimizer, net, buffer, single_agent_obs_dim, sigle_agent_action_dim, batch_size=batch_size,
                       num_mini_batches=args.num_minibatches, ppo_epoch=args.ppo_epoch, clip_param=args.clip_param,
                    value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                    gamma=args.gamma, lam=args.lam,
                    save_path=args.save_path, log_dir=log_dir, num_agents=args.num_agents, log=args.log, args=args)
        num_updates = args.total_steps // batch_size
    elif args.algorithm == 'sarsa':
        print('Using Semi-Gradient SARSA algorithm')
        if args.num_agents != 1:
            raise ValueError("Semi-Gradient SARSA implementation only supports single agent (num_agents=1).")
        explorer = ConstantEpsilonGreedyExploration(0.01, sigle_agent_action_dim)
        step_size = args.step_size
        discount = args.gamma
        feature_extractor = SarsaFeatureExtractor(single_agent_obs_dim[0], sigle_agent_action_dim)
        num_state_action_features = feature_extractor.size
        agent = SemiGradientSARSA(num_state_action_features,
                                  sigle_agent_action_dim,
                                  feature_extractor,
                                  step_size,
                                  explorer,
                                  discount,
                                  initial_weight_value=0.0, n=5)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Use appropriate environment loop based on algorithm
    if args.algorithm == 'qmix':
        # QMIX uses episode-based learning
        episode_returns, freq_dict = qmix_environment_loop(agent, vec_env, device, num_episodes=args.num_episodes, log_dir=log_dir, args=args)
    elif args.algorithm == 'sarsa':
        # Semi-Gradient SARSA uses step-based learning
        num_episode = args.total_steps // 1000  # assuming average episode length of 1000 steps
        episode_returns, freq_dict = agent_environment_sarsa_loop(agent, vec_env, num_episodes=num_episode, log_dir=log_dir)
    else:
        # MAPPO/CMAPPO use step-based learning
        episode_returns, freq_dict = agent_environment_loop(agent, vec_env, device, num_update=num_updates, log_dir=log_dir, args=args)
        
    print(f'episode returns {episode_returns}')

    def get_algorithm_name(args):
        if args.algorithm == 'qmix':
            return 'qmix'
        elif args.algorithm == 'mappo' or args.algorithm == 'cmappo':
            return "centralised" if args.centralised else "decentralised"
        else:
            return args.algorithm
    
    alg_name = get_algorithm_name(args)
    folder_path = os.path.join(PROJECT_ROOT, args.data_path)

    df = pd.DataFrame(episode_returns)
    df.to_csv(os.path.join(folder_path, f'{alg_name}_{args.num_agents}_{args.layout}_returns_seed_{args.seed}.csv'), index=False)

    df = pd.DataFrame(freq_dict["frequency_delivery_per_episode"])
    df.to_csv(os.path.join(folder_path, f'{alg_name}_{args.num_agents}_{args.layout}_frequency_delivery_per_episode_seed_{args.seed}.csv'), index=False)

    df = pd.DataFrame(freq_dict["frequency_plated_per_episode"])
    df.to_csv(os.path.join(folder_path, f'{alg_name}_{args.num_agents}_{args.layout}_frequency_plated_per_episode_seed_{args.seed}.csv'), index=False)

    df = pd.DataFrame(freq_dict["frequency_ingredient_in_pot_per_episode"])
    df.to_csv(os.path.join(folder_path, f'{alg_name}_{args.num_agents}_{args.layout}_frequency_ingredient_in_pot_per_episode_seed_{args.seed}.csv'), index=False)

    # save args to file
    with open(os.path.join(folder_path, f'{alg_name}_{args.num_agents}_{args.layout}_args_seed_{args.seed}.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Save model if requested
    if args.save and hasattr(agent, 'save_model'):
        agent.save_model()
        
    return
    
if __name__ == "__main__":
    main()