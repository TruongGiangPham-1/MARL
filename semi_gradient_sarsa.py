import numpy as np

class ConstantEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        # Simple epsilon-greedy: if random < epsilon, explore; else exploit
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit: choose best action (break ties randomly)
            return np.random.choice(np.flatnonzero(action_values == action_values.max()))

class SarsaFeatureExtractor:
    """Class that implements feature extraction for SARSA."""

    def __init__(self, obs_dim, num_actions):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.size = obs_dim + num_actions  # one-hot encoding for action

    def __call__(self, state, action):
        """Returns the feature vector for a state-action pair.

        Args:
            state: a torch array of shape (obs_dim,)
            action: an integer in {0, ..., num_actions - 1}
         
        Returns:
            a numpy array of shape (obs_dim * num_actions,)
        """
        # Your code here
        #features = np.zeros(self.obs_dim * self.num_actions)
        #features[action*self.obs_dim:(action+1)*self.obs_dim] = state
        features = np.zeros(self.obs_dim + self.num_actions)
        features[:self.obs_dim] = state
        features[self.obs_dim + action] = 1
        return features
        # end your code

def compute_q_values(state_action_features, weights):
    """Takes in Q-values and produces epsilon-greedy action probabilities

    where ties are broken evenly.

    Args:
        state_action_features: a numpy array of state-action features
        weights: a numpy array of weights
         
    Returns:
        scalar numpy Q-value
    """
    # Your code here
    return np.dot(state_action_features, weights)
    # end your code


def get_action_values(obs, feature_extractor, weights, num_actions):
    """Applies feature_extractor to observation and produces action values

    Args:
        obs: observation
        feature_extractor: extracts features for a state-action pair
        weights: a numpy array of weights
        num_actions: an integer number of actions
         
    Returns:
        a numpy array of Q-values
    """
    action_values = np.zeros(num_actions)
    for action in range(num_actions):
        action_values[action] = compute_q_values(feature_extractor(obs, action), weights)
    return action_values

class SemiGradientSARSA:
    """Class that implements Linear Semi-gradient SARSA."""

    def __init__(self,
                 num_state_action_features,  # 625 * num_actions
                 num_actions,
                 feature_extractor,
                 step_size,
                 explorer,
                 discount,
                 initial_weight_value=0.0, n=1):
        self.num_state_action_features = num_state_action_features
        self.num_actions = num_actions
        self.explorer = explorer
        self.step_size = step_size
        self.feature_extractor = feature_extractor
        self.w = np.full(num_state_action_features, initial_weight_value)
        self.discount = discount
        # Your code here: introduce any variables you may need

        self.last_state = None  # S_t
        self.last_action = None  # A_t


        self.sb = []
        self.ab = []
        self.rb = [0]
        self.t = 0
        self.T = np.inf
        self.n = n
        # End your code here

    def update_q(self, obs, action, reward, next_obs, next_action, terminated):
        # Your code here
        q_s_a_w = compute_q_values(self.feature_extractor(obs, action), self.w)
        target = None
        if terminated or self.n > 1:  # when n > 1, reward=nstep return
            target = reward
        else:
            q_snext_anext_w = compute_q_values(self.feature_extractor(next_obs, next_action), self.w)
            target = reward + self.discount * q_snext_anext_w
        
        grad = self.feature_extractor(obs, action)   # d q(s, a, w) = d [w * phi(s, a)] = phi(s, a)
        self.w += self.step_size*(target - q_s_a_w) * grad

        self.last_action = next_action
        self.last_state = next_obs
        # End your code here
    

    def act(self, obs) -> int:
        """Returns an integer 
        """
        # Your code here
        action = self.explorer.select_action(get_action_values(obs, self.feature_extractor, self.w, self.num_actions))
        # End your code here
        self.last_action = action
        self.last_state = obs

        self.sb.append(obs)
        self.ab.append(action)  # select n store
        return action 
        

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.
            obs: s_{t + 1}   s_t is in self.last_state
        Returns:
            None
        """
        # Your code here
        if self.n == 1:
            state = self.last_state # replace this line
            action = self.last_action # replace this line
            next_state = obs # replace this line
            next_action = self.act(next_state) # replace this line
            self.update_q(state, action, reward, next_state, next_action, terminated) # keep this line
        else:  # n step returns
            if self.t < self.T:
                self.rb.append(reward)  # store 
                if terminated or truncated:
                    self.T = self.t + 1
                else:
                    next_action = self.act(obs) # select and store next_action, obs
                    #self.ab.append(next_action)
                    #self.sb.append(obs)  # store next_state
            
            tau = self.t - self.n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + self.n + 1, self.T + 1)):  # compute n step return
                    #print(f'i {i} len(sb) {len(self.sb)} len(rb) {len(self.rb)}')
                    G += (self.discount**(i - tau - 1)) * self.rb[i ] 
                if tau + self.n < self.T:
                    s = self.sb[tau + self.n]
                    a = self.ab[tau + self.n]
                    G += (self.discount**self.n) * compute_q_values(self.feature_extractor(s, a), self.w)

                s_tau = self.sb[tau ]
                a_tau = self.ab[tau ]
                self.update_q(s_tau, a_tau, G, None, None, terminated)
            self.t += 1
            if terminated or truncated:
                self.reset()

    def reset(self):
        self.t = 0
        self.T = np.inf
        self.sb = []
        self.ab = []
        self.rb = [0]
        return
        # End your code here