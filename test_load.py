import numpy as np
from model import Agent
from overcooked_config import N_agent_overcooked_config
import functools
from cogrid.envs.overcooked import overcooked
from cogrid.envs import registry
import argparse
import torch
import time
from cogrid.core.directions import Directions
from cogrid.core.actions import Actions


from MAPPO import MAPPO
#env

def make_env(num_agents=4, layout="large_overcooked_layout", feature="global_obs", render_mode="human"):
    config = N_agent_overcooked_config.copy()  # get config obj
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout
    config["feature"] = feature  # set feature to global_obs or local_obs

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

def pick_up_drop_onion(env):
    obs, _, _, _, _ = env.step({0: Directions.Left, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Actions.PickupDrop, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Actions.MoveRight, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Actions.MoveUp, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Actions.PickupDrop, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Directions.Left, 1: 6})  # counter shoult be on the right
    obs, _, _, _, _ = env.step({0: Directions.Left, 1: 6})  # counter shoult be on the right



def main():
    num_agents = 2
    #layout = "overcooked_coordination_ring_v0"
    #layout = "overcooked_forced_coordination_v0"
    #layout = "overcooked_counter_circuit_v0"
    layout = "overcooked_cramped_room_v0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/policy.pth",
        help="Path to the trained model",
    )
    parser.add_argument("--layout", type=str, default=layout, help="Layout of the Overcooked environment")
    args = parser.parse_args()

    env = make_env(num_agents=num_agents, layout=layout, render_mode="human")
    obs_space = env.observation_spaces[0]['n_agent_overcooked_features']  # box (-inf, inf, (404,), float32)
    action_space = env.action_spaces[0]  # Discrete(7)
    nn = Agent(obs_space, action_space, num_agents=num_agents, num_envs=1).to(device)  # neural network
    nn.load_state_dict(torch.load(args.model_path, map_location=device))  # load the model

    mappo = MAPPO(env, None, nn, None, None, None, num_agents=num_agents)  # THE RL AGENT
    obs, info = env.reset() 
    #env.render()
    #time.sleep(1)
    #pick_up_drop_onion(env)
    #time.sleep(2)
    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)
    #with open("test_load.txt", "w") as f:
    #    f.write(f"{obs[0]}\n")
    print(f'obs shape is {obs.shape},')  # obs is a tensor of shape (num_agents, 404)
    run_inference(mappo, env, device)  # run inference
    #run_single_agent_inference(mappo, env, 1, device)  # run single agent inference
    return

def run_single_agent_inference(mappo, env, agent_id, device):
    obs, info = env.reset()  # obs is a dict of obs for each agent
    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)
    num_steps = 0
    while True:
        num_steps += 1
        sing_agent_obs = obs[agent_id]
        actions, _, _, _ = mappo.act(sing_agent_obs)  # action is single action (1,)

        agent_0_action = actions if agent_id == 0 else 6
        agent_1_action = actions if agent_id == 1 else 6
        env_action = {
            0: agent_0_action,  # Discrete(7)
            1: agent_1_action,  # Discrete(7)
        }


        obs, rewards, terminated, truncated, info = env.step(
            env_action
        )  
        done = torch.tensor([terminated[i] or truncated[i] for i in range(mappo.num_agents)]).to(device)
        if torch.all(done):
            print(f'termianteed after {num_steps} steps')
            obs, info = env.reset()  # obs is a dict of obs for each agentj
            break

        obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)

def run_inference(mappo, env, device):
    obs, info = env.reset()  # obs is a dict of obs for each agent
    obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)
    num_steps = 0
    while True:
        num_steps += 1
        #agent_0_obs = obs[0]
        #agent_1_obs = obs[1]
        actions, _, _, _ = mappo.act(obs)  # action is a vector with dimention (num_agents,)
        env_action = {i: action for i, action in enumerate(actions)}
        obs, rewards, terminated, truncated, info = env.step(
            env_action
        )  
        done = torch.tensor([terminated[i] or truncated[i] for i in range(mappo.num_agents)]).to(device)
        if torch.all(done):
            print(f'termianteed after {num_steps} steps')
            obs, info = env.reset()  # obs is a dict of obs for each agentj
            break

        obs = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)


"""
    INPUT ----------
    is an observvation for each agents, each agent's state vector is sized 404
    obs: {
        0: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        1: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        2: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
        3: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32))
    }

    # combine the observations of all agents into a single matrix. shape (num_agents, 404)
    state = torch.stack([   torch.FloatTensor(obs[i]['n_agent_overcooked_features']) for i in range(mappo.num_agents)], dim=0).to(device)


    # actions for each agents. Each agent action is in the range of 0 to 6
    action, _, _, _ = mappo.act(state)  # action is a vector with dimention (num_agents,)

    print(f'action: {action}')
    action = {  # 
        "agent_0": action[0].item(),   # Discrete(7)
        "agent_1": action[1].item(),   # Discrete(7)
        "agent_2": action[2].item(),   # Discrete(7)
        "agent_3": action[3].item()    # Discrete(7)
    }

    print(f'action: {action}')


    # ----------------------------------------------------------
    # constructing custom input 
    print(f'Inputting custom input to the model. Must be a tensor of shape (num_agents, 404)')
    # (num_agents, 404) of random numbers
    custom_input = torch.randn((num_agents, 404)).to(device)

    action, _, _, _ = mappo.act(custom_input)  # action is a vector with dimention (num_agents,)

    action = {  # 
        "agent_0": action[0].item(),   # Discrete(7)
        "agent_1": action[1].item(),   # Discrete(7)
        "agent_2": action[2].item(),   # Discrete(7)
        "agent_3": action[3].item()    # Discrete(7)
    }
    print(f'Resulting action is : {action}')


    # ----------------------- One agent ---------------------------
    action, _, _, _ = mappo.act(custom_input[0].unsqueeze(0))  # custom_input[0] is a tensor of shape (1, 404)

    print(f'Resulting action for agent 0 is : {action[0].item()}')  # action is a vector with dimention (num_agents,)


"""


if __name__ == "__main__":
    main()