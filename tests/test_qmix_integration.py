#!/usr/bin/env python3
"""
Simple integration test to verify QMIX can run with the actual environment
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import argparse

# Import the main function components
from main import make_env, make_vector_env
from QMIX import QMIX
from agent_environment import qmix_environment_loop

def test_qmix_integration():
    """Test QMIX with actual Overcooked environment"""
    print("Testing QMIX integration with Overcooked environment...")
    
    # Set up minimal args
    class TestArgs:
        def __init__(self):
            self.num_agents = 2
            self.num_envs = 1
            self.num_steps = 128
            self.algorithm = 'qmix'
            self.layout = 'overcooked_cramped_room_v0'
            self.feature = 'global_obs'
            self.seed = 1
            self.lr = 1e-3
            self.gamma = 0.99
            self.epsilon_start = 0.9
            self.epsilon_end = 0.1
            self.epsilon_decay = 0.99
            self.target_update_freq = 50
            self.buffer_size = 1000
            self.batch_size_qmix = 16
            self.mixing_embed_dim = 16
            self.hidden_dim = 64
            self.save_path = None
            self.log = False
            self.total_steps = 1000
    
    args = TestArgs()
    
    try:
        # Create environment
        print("Creating environment...")
        env = make_env(args.num_agents, layout=args.layout, feature=args.feature, render_mode=None)
        vec_env = make_vector_env(num_envs=args.num_envs, overcooked_env=env)
        vec_env.reset()
        
        # Get environment specs
        single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape
        single_agent_action_dim = env.action_spaces[0].n
        
        obs_dim = single_agent_obs_dim[0]
        action_dim = single_agent_action_dim
        state_dim = args.num_agents * obs_dim
        
        print(f"Environment specs: obs_dim={obs_dim}, action_dim={action_dim}, state_dim={state_dim}")
        
        # Create QMIX agent
        print("Creating QMIX agent...")
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
            save_path=None,
            log_dir=None,
            log=False,
            args=args
        )
        
        # Test environment interaction
        print("Testing environment interaction...")
        obs, info = vec_env.reset()
        obs_tensor = torch.FloatTensor(obs['n_agent_overcooked_features'])
        
        print(f"Initial observation shape: {obs_tensor.shape}")
        
        # Test action selection
        actions, _, _, _ = agent.act(obs_tensor, training=True)
        print(f"Selected actions: {actions}")
        
        # Test environment step
        next_obs, rewards, terminated, truncated, info = vec_env.step(actions.cpu().numpy())
        next_obs_tensor = torch.FloatTensor(next_obs['n_agent_overcooked_features'])
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.tensor([terminated[i] or truncated[i] for i in range(args.num_agents)])
        
        print(f"Step results: rewards={rewards_tensor}, dones={dones_tensor}")
        
        # Test buffer operations
        print("Testing buffer operations...")
        agent.add_to_buffer(obs_tensor, actions, rewards_tensor, dones_tensor)
        agent.update(next_obs_tensor)
        
        print("✅ QMIX integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ QMIX integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running QMIX integration test...")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = test_qmix_integration()
    
    print("=" * 50)
    if success:
        print("✅ Integration test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Integration test failed!")
        sys.exit(1)
