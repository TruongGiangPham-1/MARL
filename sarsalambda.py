import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tile_coding import IHT, tiles
from overcooked_features import BinaryFeature
from main import make_env
from tqdm import tqdm

import numpy as np

class SarsaLambdaTileCoder:
    def __init__(self, num_features, alpha, lmbda, gamma, epsilon, n_tilings=8, num_actions=3):
        self.w = np.zeros(num_features)      # Weight vector
        self.z = np.zeros(num_features)      # Eligibility traces
        self.alpha = alpha / n_tilings       # Rescale alpha by number of tilings
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_features = num_features
        self.num_actions = num_actions

    def get_q(self, features):
        # Q is the dot product of weights and the binary feature vector
        # Optimization: since features is binary, this is just the sum of active weight
        #print(f"np.sum(self.w[features.astype(bool)]) = {np.sum(self.w[features.astype(bool)])}")
        return np.sum(self.w[features.astype(bool)])

    def choose_action(self, obs, iht, extract_func):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions) # MountainCar has 3 actions (0, 1, 2)
        
        # Calculate Q for all possible actions
        qs = []
        for a in range(self.num_actions):
            f = extract_func(obs, a, iht, self.num_features)
            qs.append(self.get_q(f))
        
        # Greedy selection with tie-breaking
        max_q = np.max(qs)
        ties = np.flatnonzero(np.abs(qs - max_q) < 1e-8)
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

def main():
    env = make_env(num_agents=1, layout="overcooked_cramped_room_v0", feature="Binary_feature", render_mode=None)
    binary_feature = BinaryFeature(env)
    episode_rewards = []
    # Initialization
    num_features = 4096
    num_features = binary_feature.shape[0] + 7  # Add 7 for the one-hot encoding of actions
    iht = IHT(num_features) # From your tilecoding library

    def overcooked_extract_func(obs, action, iht, num_features):
        # one shot of action
        action_one_hot = np.zeros(7)  # Assuming 7 possible actions in Overcooked
        action_one_hot[action] = 1.0
        # append to observation features
        combined_features = np.concatenate([obs[0]['n_agent_overcooked_features'], action_one_hot])
        return combined_features

    agent = SarsaLambdaTileCoder(num_features, alpha=0.1, lmbda=0.9, gamma=0.99, epsilon=0.01, num_actions=7)

    for episode in tqdm(range(10000)):
        obs, _ = env.reset()
        
        # 1. Pick initial action
        action = agent.choose_action(obs, iht, overcooked_extract_func)
        
        # 2. Get initial features
        #f = extract_state_action_features(obs, action, iht, num_features)
        f = overcooked_extract_func(obs, action, iht, num_features)
        
        total_reward = 0
        while True:
            next_obs, reward, terminated, truncated, _ = env.step({0: action})
            reward = reward[0]  # Assuming reward is a dict with agent IDs as keys
            done = terminated[0] or truncated[0]
            
            # 3. Pick next action and get next features
            next_action = agent.choose_action(next_obs, iht, overcooked_extract_func)
            #f_next = extract_state_action_features(next_obs, next_action, iht, num_features)
            f_next = overcooked_extract_func(next_obs, next_action, iht, num_features)
            #print(f"Episode {episode}, Reward: {reward}, Done: {done}, Action: {action}, Next Action: {next_action}") 
            # 4. Step the agent
            agent.learn(f, reward, f_next, done)
            
            # 5. Transition
            obs, action, f = next_obs, next_action, f_next
            total_reward += reward
            
            if done:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
        episode_rewards.append(total_reward)

    # Plotting the episode rewards
    # plot the running average of episode rewards
    running_avg = np.convolve(episode_rewards, np.ones((10,))/10, mode='valid')
    plt.plot(running_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards over Time')
    plt.show()

if __name__ == "__main__":
    main()