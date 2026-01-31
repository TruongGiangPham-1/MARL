import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tile_coding import IHT, tiles

import numpy as np

class SarsaLambdaTileCoder:
    def __init__(self, num_features, alpha, lmbda, gamma, epsilon, n_tilings=8):
        self.w = np.zeros(num_features)      # Weight vector
        self.z = np.zeros(num_features)      # Eligibility traces
        self.alpha = alpha / n_tilings       # Rescale alpha by number of tilings
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_features = num_features

    def get_q(self, features):
        # Q is the dot product of weights and the binary feature vector
        # Optimization: since features is binary, this is just the sum of active weights
        return np.sum(self.w[features.astype(bool)])

    def choose_action(self, obs, iht, extract_func):
        if np.random.random() < self.epsilon:
            return np.random.randint(3) # MountainCar has 3 actions (0, 1, 2)
        
        # Calculate Q for all possible actions
        qs = []
        for a in range(3):
            f = extract_func(obs, a, iht, self.num_features)
            qs.append(self.get_q(f))
        return np.argmax(qs)

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
    env = gym.make('MountainCar-v0')
    episode_rewards = []
    # Initialization
    num_features = 4096
    iht = IHT(num_features) # From your tilecoding library
    agent = SarsaLambdaTileCoder(num_features, alpha=0.1, lmbda=0.9, gamma=0.99, epsilon=0.01)

    for episode in range(1000):
        obs, _ = env.reset()
        
        # 1. Pick initial action
        action = agent.choose_action(obs, iht, extract_state_action_features)
        
        # 2. Get initial features
        f = extract_state_action_features(obs, action, iht, num_features)
        
        total_reward = 0
        while True:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Pick next action and get next features
            next_action = agent.choose_action(next_obs, iht, extract_state_action_features)
            f_next = extract_state_action_features(next_obs, next_action, iht, num_features)
            
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