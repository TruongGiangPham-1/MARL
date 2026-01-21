# loop
import os
import torch
import numpy as np
import imageio

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import evaluate_state

def agent_environment_loop(agent, env, device, num_update=1000, log_dir=None, args=None):
    """
    agent: mappo agent
    """
    summary_writer = SummaryWriter(log_dir=log_dir)
    collect_steps = args.num_steps

    # TODO: move this to replay buffer
    episodes_reward = []
    episode_reward = 0  # undiscount reward
    num_episdes = 0
    frequency_delivery_per_episode = 0
    frequency_plated_per_episode = 0  # how many times agent plated the food
    frequency_ingredient_in_pot_per_episode = 0  # how many times agent put ingredient in pot

    frequency_delivery_per_episode_list = []
    frequency_plated_per_episode_list = []
    frequency_ingredient_in_pot_per_episode_list = []

    # action prob frames
    action_prob_frames = []

    obs, info = env.reset()  # obs is a dict of obs for each agent
    #print(f'obs shape {obs['n_agent_overcooked_features'].shape}')  # (num_env*num_agents, obs_dim)
    obs = torch.FloatTensor(obs['n_agent_overcooked_features']).to(device)  # (num_env*num_agents, obs_dim)
    dones = torch.zeros((args.num_envs*agent.num_agents,)).to(device)
    global_step = 0
    
    for _ in tqdm(range(num_update)):
        for step in range(collect_steps):
            actions, logprobs, _, values = agent.act(obs)  # with no grad action dim (num_agents,)
            assert actions.shape == (args.num_agents*args.num_envs,)
            """
            actions dim (num_agents,)
            logprobs dim (num_agents,)
            values dim (num_agents, 1)
            """
            #env_action = {i: action for i, action in enumerate(actions)}
            next_obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
            assert rewards.shape == (args.num_envs*agent.num_agents,)
            assert terminated.shape == (args.num_envs*agent.num_agents,)
            assert truncated.shape == (args.num_envs*agent.num_agents,)
            #assert next_obs['n_agent_overcooked_features'].shape == (args.num_envs*agent.num_agents, ) + env.single_obs_space.shape

            """
            rewards = {agent_id: R for agent_id in N}
            terminated = {agent_id: terminated for agent_id in N}
            truncated = {agent_id: truncated for agent_id in N}
            """
            
            #rewards = torch.tensor([rewards[i] for i in range(agent.num_agents)]).to(device)  # dim (num_agents,)
            rewards = torch.FloatTensor(rewards).to(device)

            # if there is 1 in rewards tensor, print hello
            if torch.any(rewards[0:2] >= 1):
                frequency_delivery_per_episode += 1
                print(f'global_step {global_step} agent sucessfully delievered. rewards {rewards}')
            if torch.any(rewards[0:2] == 0.3):
                frequency_plated_per_episode += 1
            if torch.any(rewards[0:2] == 0.1):
                frequency_ingredient_in_pot_per_episode += 1

            
            # sum of reward
            episode_reward +=  rewards[0:2].float().mean().item()  # mean reward for the episode
     

            #if rewards.float().mean().item() > 0:
            #    print(f'global_step {global_step} rewards {rewards.float().mean().item()} non 0')

            if values is not None:
                values = values.squeeze(1)
            agent.add_to_buffer(obs, actions, rewards, dones, logprobs, values)

            if torch.all(dones[0:2]):
                # handle reset 
                next_obs, info = env.reset()
                episodes_reward.append(episode_reward)
                if args.log:
                    summary_writer.add_scalar('episode_rewards', episode_reward, num_episdes)
                    summary_writer.add_scalar('freq/frequency_delivery_per_episode', frequency_delivery_per_episode, num_episdes)
                    summary_writer.add_scalar('freq/frequency_plated_per_episode', frequency_plated_per_episode, num_episdes)
                    summary_writer.add_scalar('freq/frequency_ingredient_in_pot_per_episode', frequency_ingredient_in_pot_per_episode, num_episdes)
                
                frequency_delivery_per_episode_list.append(frequency_delivery_per_episode)
                frequency_plated_per_episode_list.append(frequency_plated_per_episode)
                frequency_ingredient_in_pot_per_episode_list.append(frequency_ingredient_in_pot_per_episode)

                episode_reward = 0
                frequency_delivery_per_episode = 0
                frequency_plated_per_episode = 0
                frequency_ingredient_in_pot_per_episode = 0
                num_episdes += 1

            obs = torch.FloatTensor(next_obs['n_agent_overcooked_features']).to(device)  # (num_env*num_agents, obs_dim)
            dones = torch.tensor([terminated[i] or truncated[i] for i in range(args.num_envs*agent.num_agents)]).to(device)

            global_step += 1

        # Update the agent with the collected data
        agent.update(obs)

        if args.log:
            #image = evaluate_state(agent, env, device, global_step=global_step)
            #image = imageio.imread(image)
            #action_prob_frames.append(image)
            pass
    
    freq_dict = {
        'frequency_delivery_per_episode': frequency_delivery_per_episode_list,
        'frequency_plated_per_episode': frequency_plated_per_episode_list,
        'frequency_ingredient_in_pot_per_episode': frequency_ingredient_in_pot_per_episode_list
    }

    # save gif
    if args.log:
        #imageio.mimsave(f"data/{args.num_agents}_{args.layout}_seed_{args.seed}_action_prob_frames.gif", action_prob_frames)
        pass
    return episodes_reward, freq_dict


def qmix_environment_loop(agent, env, device, num_episodes=1000, log_dir=None, args=None):
    """
    Episode-based environment loop specifically for QMIX algorithm
    """
    summary_writer = SummaryWriter(log_dir=log_dir)

    # Episode tracking
    episodes_reward = []
    episode_reward = 0  # undiscounted reward
    num_completed_episodes = 0
    
    # Frequency tracking
    frequency_delivery_per_episode = 0
    frequency_plated_per_episode = 0
    frequency_ingredient_in_pot_per_episode = 0
    
    frequency_delivery_per_episode_list = []
    frequency_plated_per_episode_list = []
    frequency_ingredient_in_pot_per_episode_list = []

    global_step = 0
    
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        obs = torch.FloatTensor(obs['n_agent_overcooked_features']).to(device)  # (num_agents, obs_dim)
        obs = torch.clamp(obs, min=-5.0, max=5.0)  # Clip observations
        
        episode_reward = 0
        frequency_delivery_per_episode = 0
        frequency_plated_per_episode = 0
        frequency_ingredient_in_pot_per_episode = 0
        
        terminated_ = False
        truncate_ = False
        step_count = 0
        #max_steps_per_episode = args.num_steps if hasattr(args, 'num_steps') else 200

        while not (terminated_ or truncate_):
            # Select actions using QMIX
            actions, _, _, _ = agent.act(obs, state=None, training=True)
            
            # Environment step
            next_obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs['n_agent_overcooked_features']).to(device)
            next_obs = torch.clamp(next_obs, min=-5.0, max=5.0)  # Clip observations
            rewards = torch.FloatTensor(rewards).to(device)
            
            # Check for episode termination
            dones = torch.tensor([terminated[i] or truncated[i] for i in range(args.num_agents)]).to(device)
            
            # Track specific rewards
            if torch.any(rewards >= 1):
                frequency_delivery_per_episode += 1
                if args.log:
                    print(f'Episode {episode}, Step {step_count}: Delivery! Rewards: {rewards}')
            if torch.any(rewards == 0.3):
                frequency_plated_per_episode += 1
            if torch.any(rewards == 0.1):
                frequency_ingredient_in_pot_per_episode += 1
            
            # Accumulate episode reward
            episode_reward += rewards.mean().item()
            terminated_ = np.any(terminated)  # Check if any agent is terminated
            truncate_ = np.any(truncated)
            
            # Store experience in QMIX buffer (values are None for QMIX)
            agent.add_to_buffer(obs, actions, rewards, dones, None, None)
            
            # Move to next state
            obs = next_obs
            step_count += 1
            global_step += 1
            
            # Update QMIX networks periodically
            if global_step % 1 == 0:  # Update every step for QMIX
                agent.update(next_obs)
        
        # Episode completed
        episodes_reward.append(episode_reward)
        frequency_delivery_per_episode_list.append(frequency_delivery_per_episode)
        frequency_plated_per_episode_list.append(frequency_plated_per_episode)
        frequency_ingredient_in_pot_per_episode_list.append(frequency_ingredient_in_pot_per_episode)
        
        num_completed_episodes += 1
        
        # Logging
        if args.log and summary_writer:
            summary_writer.add_scalar('episode_rewards', episode_reward, num_completed_episodes)
            summary_writer.add_scalar('episode_length', step_count, num_completed_episodes)
            summary_writer.add_scalar('freq/frequency_delivery_per_episode', frequency_delivery_per_episode, num_completed_episodes)
            summary_writer.add_scalar('freq/frequency_plated_per_episode', frequency_plated_per_episode, num_completed_episodes)
            summary_writer.add_scalar('freq/frequency_ingredient_in_pot_per_episode', frequency_ingredient_in_pot_per_episode, num_completed_episodes)
            
            # Log QMIX specific metrics
            if hasattr(agent, 'epsilon'):
                summary_writer.add_scalar('qmix/epsilon', agent.epsilon, num_completed_episodes)
            if hasattr(agent, 'buffer') and len(agent.buffer) > 0:
                summary_writer.add_scalar('qmix/buffer_size', len(agent.buffer), num_completed_episodes)
        
        # Print progress
        if episode % 100 == 0 or episode == num_episodes - 1:
            avg_reward = np.mean(episodes_reward[-100:]) if episodes_reward else 0
            print(f"Episode {episode}/{num_episodes}, Avg Reward (last 100): {avg_reward:.3f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Buffer Size: {len(agent.buffer) if hasattr(agent, 'buffer') else 0}")
    
    freq_dict = {
        'frequency_delivery_per_episode': frequency_delivery_per_episode_list,
        'frequency_plated_per_episode': frequency_plated_per_episode_list,
        'frequency_ingredient_in_pot_per_episode': frequency_ingredient_in_pot_per_episode_list
    }
    
    return episodes_reward, freq_dict




def mpe_environment_loop(agent, env, device, num_episodes=1000, log_dir=None):
    """
    agent: mappo agent
    """
    collect_steps = agent.collect_steps
    num_updates = 5
    obs, info = env.reset()

    obs = torch.stack([   torch.FloatTensor(obs[a]) for a in (env.possible_agents)], dim=0).to(device)  # (num_agents, 18)
    dones = torch.zeros((env.num_agents,)).to(device)
    for episode in range(num_episodes):
        for step in range(collect_steps):
            actions, logprobs, _, values = agent.act(obs)  # with no grad action dim (num_agents,)
            """
            actions dim (num_agents,)
            logprobs dim (num_agents,)
            values dim (num_agents, 1)
            """
            env_action = {a: action.cpu().numpy() for a, action in zip(env.possible_agents, actions)}
            print(f'env_action {env_action}')


            next_obs, rewards, terminated, truncated, info = env.step(env_action)
            """
            next_obs = {agent_id: obs for agent_id in N}          {} if torch.all(dones)
            rewards = {agent_id: R for agent_id in N}             {} if torch.all(dones)
            terminated = {agent_id: terminated for agent_id in N}
            truncated = {agent_id: truncated for agent_id in N}   {} if torch.all(dones)

            TODO: how can I handle truncated and terminated. when rewards is empty.
            Simple_spread reward is sum of all agents distance to the target. So making terminated reward 0 is a problem.
            """
            full_rewards = torch.zeros((env.num_agents)).to(device)
            for aval_agent_str in env.agents:
                agent_id = int(aval_agent_str.split('_')[1])
                full_rewards[agent_id] = rewards[aval_agent_str]
                

            print(f'rewards before {rewards} env.agents {env.possible_agents} done {dones} truncated {truncated}')
            rewards = full_rewards

            #print(f'size of stuff adding to buffer {obs.shape}, {actions.shape}, {rewards.shape}, {dones.shape}, {logprobs.shape}, {values.squeeze(1).shape}')
            agent.add_to_buffer(obs, actions, rewards, dones, logprobs, values.squeeze(1))


            if torch.all(dones):
                # handle reset 
                next_obs, info = env.reset()
                obs = torch.stack([   torch.FloatTensor(next_obs[a]) for a in (env.possible_agents)], dim=0).to(device)
            obs = torch.stack([   torch.FloatTensor(next_obs[a]) for a in (env.possible_agents)], dim=0).to(device)
            dones = torch.tensor([terminated[a] or truncated[a] for a in (env.possible_agents)]).to(device)


        # Update the agent with the collected data
        agent.update(obs)
    return []



"""
Problem: in MPE, 

"""# loop

def agent_environment_sarsa_loop(agent, env, num_episodes, log_dir=None):
    summary_writer = SummaryWriter(log_dir=log_dir)
    n = 5
    episode_returns = []
    for episode in (range(num_episodes)):
        observation, info = env.reset()
        # start your code
        observation = observation['n_agent_overcooked_features'][0]  # get the observation of agent 0
        S = observation  # initialize S
        A = agent.act(observation)  # initlize a (for sarsa)
        episode_return = 0
        while (True):
            observation_next, R, terminated, truncated, info = env.step([A])   # s', r = T(s, a)
            # conversion
            observation_next = observation_next['n_agent_overcooked_features'][0]  # get the observation of agent 0
            R = R[0]  # get the reward of agent 0
            terminated = terminated[0]  # get the terminated of agent 0
            truncated = truncated[0]  # get the truncated of agent 0
            # -----------   
            agent.process_transition(observation_next, R, terminated, truncated)  # update Q
            S = observation_next
            episode_return += R
            if terminated or truncated:
                break

            if n > 1:
                #print(f'len(ab)={len(agent.ab)}')
                A = agent.ab[-1]  # n step takes action A_{t + 1}
            else:
                A = agent.act(S)  # set last_action = A
        summary_writer.add_scalar('episode_rewards', episode_return, episode)
        episode_returns.append(episode_return)
        # end your code
    return episode_returns