#!/usr/bin/env python3
"""
Test script for QMIX implementation
"""
import torch
import numpy as np
from QMIX import QMIX, QNetwork, MixingNetwork, QMixBuffer

def test_qnetwork():
    """Test individual Q-network"""
    print("Testing Q-Network...")
    obs_dim = 404
    action_dim = 7
    batch_size = 32
    
    q_net = QNetwork(obs_dim, action_dim)
    obs = torch.randn(batch_size, obs_dim)
    
    q_values = q_net(obs)
    assert q_values.shape == (batch_size, action_dim), f"Expected shape {(batch_size, action_dim)}, got {q_values.shape}"
    print("✓ Q-Network test passed")

def test_mixing_network():
    """Test mixing network"""
    print("Testing Mixing Network...")
    num_agents = 2
    state_dim = 808  # 2 * 404
    batch_size = 32
    
    mixing_net = MixingNetwork(num_agents, state_dim)
    agent_qs = torch.randn(batch_size, num_agents)
    states = torch.randn(batch_size, state_dim)
    
    q_tot = mixing_net(agent_qs, states)
    assert q_tot.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {q_tot.shape}"
    print("✓ Mixing Network test passed")

def test_buffer():
    """Test experience replay buffer"""
    print("Testing QMix Buffer...")
    capacity = 1000
    num_agents = 2
    obs_dim = 404
    state_dim = 808
    
    buffer = QMixBuffer(capacity, num_agents, obs_dim, state_dim)
    
    # Add some experiences
    for i in range(10):
        obs = torch.randn(num_agents, obs_dim)
        actions = torch.randint(0, 7, (num_agents,))
        rewards = torch.randn(num_agents)
        next_obs = torch.randn(num_agents, obs_dim)
        dones = torch.zeros(num_agents)
        state = torch.randn(state_dim)
        next_state = torch.randn(state_dim)
        
        buffer.add(obs, actions, rewards, next_obs, dones, state, next_state)
    
    assert len(buffer) == 10, f"Expected buffer length 10, got {len(buffer)}"
    
    # Sample a batch
    batch_size = 5
    obs, actions, rewards, next_obs, dones, states, next_states = buffer.sample(batch_size)
    
    assert obs.shape == (batch_size, num_agents, obs_dim)
    assert actions.shape == (batch_size, num_agents)
    assert rewards.shape == (batch_size, num_agents)
    assert next_obs.shape == (batch_size, num_agents, obs_dim)
    assert dones.shape == (batch_size, num_agents)
    assert states.shape == (batch_size, state_dim)
    assert next_states.shape == (batch_size, state_dim)
    
    print("✓ QMIX single environment test passed")

def test_qmix_basic():
    """Test basic QMIX functionality"""
    print("Testing QMIX basic functionality...")
    
    # Mock environment class
    class MockEnv:
        def __init__(self):
            self.single_observation_space = type('obj', (object,), {'shape': (404,)})()
            self.single_action_space = type('obj', (object,), {'n': 7})()
    
    # Mock args
    class MockArgs:
        def __init__(self):
            self.num_envs = 1
            self.num_steps = 128
    
    env = MockEnv()
    args = MockArgs()
    
    qmix_agent = QMIX(
        env=env,
        num_agents=2,
        obs_dim=404,
        action_dim=7,
        state_dim=808,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=200,
        buffer_size=1000,
        batch_size=32,
        mixing_embed_dim=32,
        hidden_dim=256,
        save_path=None,
        log_dir=None,
        log=False,
        args=args
    )
    
    # Test action selection
    obs = torch.randn(2, 404)  # 2 agents, 404 obs dim
    actions, _, _, _ = qmix_agent.act(obs, training=True)
    
    assert actions.shape == (2,), f"Expected action shape (2,), got {actions.shape}"
    assert all(0 <= a < 7 for a in actions), "Actions should be in valid range [0, 7)"
    
    print("✓ QMIX basic functionality test passed")

def test_vectorized_env():
    """Test QMIX with single environment (num_envs = 1)"""
    print("Testing QMIX with single environment (num_envs = 1)...")
    
    # Mock environment class
    class MockEnv:
        def __init__(self):
            self.single_observation_space = type('obj', (object,), {'shape': (404,)})()
            self.single_action_space = type('obj', (object,), {'n': 7})()
    
    # Mock args
    class MockArgs:
        def __init__(self):
            self.num_envs = 1
            self.num_steps = 128
    
    env = MockEnv()
    args = MockArgs()
    
    qmix_agent = QMIX(
        env=env,
        num_agents=2,
        obs_dim=404,
        action_dim=7,
        state_dim=808,
        args=args
    )
    
    # Test with single environment format (num_agents, obs_dim)
    obs = torch.randn(2, 404)  # 2 agents, 404 obs_dim
    actions, _, _, _ = qmix_agent.act(obs, training=True)
    
    assert actions.shape == (2,), f"Expected action shape (2,), got {actions.shape}"
    assert all(0 <= a < 7 for a in actions), "Actions should be in valid range [0, 7)"
    
    print("✓ QMIX single environment test passed")

if __name__ == "__main__":
    print("Running QMIX tests...")
    print("=" * 50)
    
    try:
        test_qnetwork()
        test_mixing_network()
        test_buffer()
        test_qmix_basic()
        test_vectorized_env()
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
