"""
Train petting zoo MPE
"""


from pettingzoo.mpe import simple_spread_v3
from MAPPO import MAPPO
from agent_environment import mpe_environment_loop 
import argparse
import torch
from model import Agent
from buffer import Buffer
import time


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
    parser.add_argument('--batch-size', type=int, default=5, help='number of sample to collect before update')
    args = parser.parse_args()

    env = simple_spread_v3.parallel_env()  # the overcooked game engine
    env.reset()


    print(f'obs {env.observation_spaces["agent_0"].shape}')  # (18 ,)
    obs_space = env.observation_spaces['agent_0']  # (18,)
    action_space = env.action_spaces['agent_0']  # Discrete(5)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    net = Agent(obs_space, action_space).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    buffer = Buffer(obs_space.shape[0], env.num_agents, max_size=128)


    collect_steps = args.batch_size

    import os
    # make logs folder if not exist
    os.makedirs("logs", exist_ok=True)

    log_dir = f"logs/run__{int(time.time())}"

    
    ppo_agent = MAPPO(env, optimizer, net, buffer, obs_space.shape, action_space.n, collect_steps=collect_steps, 
                       save_path=args.save_path, log_dir=log_dir)
    reward = mpe_environment_loop(ppo_agent, env, device, num_episodes=1000)
    return


if __name__ == "__main__":
    main()

"""
    env = simple_spread_v3.parallel_env()


    env.reset()
    print(f'num agents {env.num_agents}')  # 4

    o, info = env.reset()

    print(f'obs {o}')  # {"agent_0": obs}

    for agent in env.agents:
        assert type(agent) == str  # agent_i str
        print(f'agent {agent} obs {o[agent]}')



"""