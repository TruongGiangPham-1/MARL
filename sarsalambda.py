import hashlib
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tile_coding import IHT, tiles
from overcooked_features import BinaryFeature
from main import make_env
from tqdm import tqdm
import cv2
import time
import argparse
import json
import os
from utils import get_hash_id, get_run_folder
# import summary_writer for logging
from torch.utils.tensorboard import SummaryWriter

import numpy as np

class SarsaLambdaTileCoder:
    def __init__(self, obs_dim, alpha, lmbda, gamma, epsilon,  num_actions=3):
        self.num_features = obs_dim * num_actions
        self.w = np.zeros(self.num_features)      # Weight vector
        self.z = np.zeros(self.num_features)      # Eligibility traces
        self.alpha = alpha      
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.obs_dim = obs_dim

    def get_q(self, features):
        # Q is the dot product of weights and the binary feature vector
        # Optimization: since features is binary, this is just the sum of active weight
        #print(f"np.sum(self.w[features.astype(bool)]) = {np.sum(self.w[features.astype(bool)])}")
        return np.sum(self.w[features.astype(bool)])

    def choose_action(self, obs, extract_func):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions) # MountainCar has 3 actions (0, 1, 2)
        
        # Calculate Q for all possible actions
        qs = []
        for a in range(self.num_actions):
            f = extract_func(obs, a, self.obs_dim, self.num_actions)
            qs.append(self.get_q(f))
        
        # Greedy selection with tie-breaking
        qs = np.array(qs)
        max_q = np.max(qs)
        ties = np.flatnonzero(np.abs(qs - max_q) < 1e-8)
        
        # Safety check: if no ties found, just use argmax
        if len(ties) == 0:
            ties = np.array([np.argmax(qs)])
        
        return np.random.choice(ties)

    def learn(self, f, r, f_next, done):
        q_curr = self.get_q(f)
        q_next = 0 if done else self.get_q(f_next)
        
        # 1. TD Error
        delta = r + self.gamma * q_next - q_curr
        # 2. Update Eligibility Trace
        # We use "Replacing Traces" here as it's more stable with Tile Coding
        self.z *= self.gamma * self.lmbda
        active_indices = f.astype(bool)
        self.z[active_indices] = 1.0 
        
        # 3. Update Weights
        self.w += self.alpha * delta * self.z
        
        if done:
            self.z.fill(0)
def extract_state_action_features(obs, action, iht, num_state_action_features):
    position = obs[0]
    velocity = obs[1]
    active_tiles = tiles(iht, 8, [8 * position / (0.5+1.2), 8 * velocity / (0.07 + 0.07)], [action]) # see footnote 1 on page 246 of http://incompleteideas.net/book/RLbook2020.pdf
    feature_vector = np.zeros(num_state_action_features)
    feature_vector[active_tiles] = 1.0
    return feature_vector

def inference_loop(agent, env, extract_func, num_episodes=1000, episode_id=0, data_path="sarsa_lambda_data"):
    episode_rewards = []
    frame_list = []
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        action = agent.choose_action(obs, extract_func)
        f = extract_func(obs, action, agent.obs_dim, agent.num_actions)
        total_reward = 0
        while True:
            next_obs, reward, terminated, truncated, _ = env.step({0: action})
            img = env.render()
            frame_list.append(img)
            reward = reward[0]  # Assuming reward is a dict with agent IDs as keys
            done = terminated[0] or truncated[0]
            next_action = agent.choose_action(next_obs, extract_func)
            f_next = extract_func(next_obs, next_action, agent.obs_dim, agent.num_actions)

            obs, action, f = next_obs, next_action, f_next
            total_reward += reward
            if done:
                break
            #time.sleep(0.1)  # add a small delay to slow down the rendering
        episode_rewards.append(total_reward)
    import imageio
    imageio.mimsave(f'{data_path}/sarsa_lambda_inference_episode_{episode_id}_num_frames_{len(frame_list)}.gif', frame_list, fps=10)
    return episode_rewards

def main():

    parser = argparse.ArgumentParser(description="SARSA Lambda with Tile Coding in Overcooked")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lambda_", type=float, default=0.9, help="Eligibility trace decay rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Exploration rate")
    parser.add_argument("--data_path", type=str, default="sarsa_lambda_data", help="Path to save model weights and logs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--layout", type=str, default="overcooked_cramped_room_v0", help="Environment layout")
    parser.add_argument("--feature", type=str, default="Binary_feature", help="Feature type for environment")
    args = parser.parse_args()

    # args to dict
    args_dict = vars(args)
    run_folder = get_run_folder(args_dict)

    # create data path if it doesn't exist
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    env = make_env(num_agents=1, layout=args.layout, feature=args.feature, render_mode="rgb_array")
    binary_feature = BinaryFeature(env)
    episode_rewards = []
    # Initialization
    obs_dim = binary_feature.shape[0]  # Observation feature dimension

    def overcooked_extract_func(obs, action, obs_dim, num_actions=7):
        # one shot of action
        # | S| * |A| features, where |S| is the number of state features and |A| is the number of actions
        # append to observation features
        combined_features = np.zeros(obs_dim*num_actions, dtype=np.float32)
        combined_features[obs_dim*action:obs_dim*(action+1)] = obs[0]['n_agent_overcooked_features']  # Assuming obs is a dict with agent IDs as keys
        return combined_features

    agent = SarsaLambdaTileCoder(obs_dim, alpha=args.alpha, lmbda=args.lambda_, gamma=args.gamma, epsilon=args.epsilon, num_actions=7)
    # read sarsa lambda weights and traces from file if they exist
    try:
        #agent.w = np.load(f"{args.data_path}/sarsa_lambda_weights_episode_2000.npy")
        #agent.z = np.load(f"{args.data_path}/sarsa_lambda_traces_episode_2000.npy")
        print("Loaded SARSA Lambda weights and traces from file.")
    except FileNotFoundError:
        print("No saved weights or traces found. Starting with fresh weights and traces.")
    
    #inference_loop(agent, env, overcooked_extract_func, num_episodes=2)
    #return

    freq_dict = {
        "delivery_reward": 0,
        "num_onions_in_pot_reward": 0,
        "num_soups_in_dish_reward": 0,
    }
    summary_writer = SummaryWriter(log_dir="sarsa_lambda_logs")
    for episode in tqdm(range(args.num_episodes)):
        obs, _ = env.reset()
        
        # 1. Pick initial action
        action = agent.choose_action(obs, overcooked_extract_func)
        
        # 2. Get initial features
        #f = extract_state_action_features(obs, action, iht, num_features)
        f = overcooked_extract_func(obs, action, obs_dim, num_actions=7)
        freq_dict = {
            "delivery_reward": 0,
            "num_onions_in_pot_reward": 0,
            "num_soups_in_dish_reward": 0,
        }
        
        total_reward = 0
        count= 0
        while True:
            next_obs, reward, terminated, truncated, _ = env.step({0: action})
            reward = reward[0]  # Assuming reward is a dict with agent IDs as keys
            done = terminated[0] or truncated[0] or reward == 1  # end episode if delivery reward is received
            
            # 3. Pick next action and get next features
            next_action = agent.choose_action(next_obs, overcooked_extract_func)
            #f_next = extract_state_action_features(next_obs, next_action, iht, num_features)
            f_next = overcooked_extract_func(next_obs, next_action, obs_dim, num_actions=7)
            #print(f"Episode {episode}, Reward: {reward}, Done: {done}, Action: {action}, Next Action: {next_action}") 
            # 4. Step the agent
            agent.learn(f, reward, f_next, done)
            
            # 5. Transition
            obs, action, f = next_obs, next_action, f_next
            total_reward += reward
            if reward == 1:
                freq_dict["delivery_reward"] += 1
                print(f"Delivery reward at Episode {episode}, Action: {action}, Next Action: {next_action}")
            elif reward == 0.3:
                print(f"plate reward at Episode {episode}, Action: {action}, Next Action: {next_action}")
                freq_dict["num_soups_in_dish_reward"] += 1
            elif reward == 0.1:
                freq_dict["num_onions_in_pot_reward"] += 1
                count += 1
            if done:
                break
        summary_writer.add_scalar("Total Reward", total_reward, episode)
        summary_writer.add_scalar("Delivery Reward Count", freq_dict["delivery_reward"], episode)
        summary_writer.add_scalar("Onions in Pot Reward Count", freq_dict["num_onions_in_pot_reward"], episode)
        summary_writer.add_scalar("Soups in Dish Reward Count", freq_dict["num_soups_in_dish_reward"], episode)
        summary_writer.add_scalar("weight_mean", np.mean(agent.w), episode)
        if episode % 1000 == 0:
            # save the model weights every 1000 episodes
            np.save(f"{args.data_path}/sarsa_lambda_weights_episode_{episode}.npy", agent.w)
            np.save(f"{args.data_path}/sarsa_lambda_traces_episode_{episode}.npy", agent.z)
            print(f"Episode {episode}, Total Reward: {total_reward}")
        if episode % 1000 == 0:
            # run inference every 100 episodes
            inference_loop(agent, env, overcooked_extract_func, num_episodes=1, episode_id=episode, data_path=run_folder)
        episode_rewards.append(total_reward)

    # Plotting the episode rewards
    # plot the running average of episode rewards
    running_avg = np.convolve(episode_rewards, np.ones((10,))/10, mode='valid')
    plt.plot(running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards over Time')
    # save the plot
    plt.savefig(f"{args.data_path}/sarsa_lambda_rewards.png")
    plt.show()

if __name__ == "__main__":
    main()
