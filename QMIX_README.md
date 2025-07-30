# QMIX Implementation

This directory contains an implementation of QMIX (Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning) integrated with the existing MARL codebase.

## Overview

QMIX is a value-based multi-agent reinforcement learning algorithm that:
- Uses individual Q-networks for each agent
- Employs a mixing network to combine individual Q-values into a team Q-value
- Maintains the Individual-Global-Max (IGM) principle
- Uses experience replay for sample efficiency

## Key Components

### 1. QNetwork
Individual Q-networks for each agent that map observations to Q-values for each action.

### 2. MixingNetwork
A hypernetwork that combines individual Q-values while ensuring monotonicity (IGM principle).

### 3. QMixBuffer
Experience replay buffer specifically designed for QMIX to store multi-agent transitions.

### 4. QMIX Agent
Main QMIX class that coordinates training and action selection.

## Usage

### Command Line Arguments

QMIX-specific arguments:
- `--algorithm qmix`: Select QMIX algorithm
- `--epsilon-start`: Initial exploration rate (default: 1.0)
- `--epsilon-end`: Final exploration rate (default: 0.05)
- `--epsilon-decay`: Exploration decay rate (default: 0.995)
- `--target-update-freq`: Target network update frequency (default: 200)
- `--buffer-size`: Experience replay buffer size (default: 5000)
- `--batch-size-qmix`: Batch size for QMIX updates (default: 32)
- `--mixing-embed-dim`: Mixing network embedding dimension (default: 32)
- `--hidden-dim`: Hidden layer dimension (default: 256)

### Makefile Targets

#### Basic Usage
```bash
# Quick test run
make quick-qmix

# Debug run with detailed output
make qmix-debug

# Full training on cramped room
make qmix-cramped

# Full training on forced coordination
make qmix-forced

# Large environment with 4 agents
make qmix-large
```

#### Testing
```bash
# Unit tests
make test-qmix

# Integration test with environment
make test-qmix-integration
```

#### Comparison
```bash
# Compare MAPPO vs QMIX
make compare-cramped
```

### Manual Execution
```bash
python3 main.py --algorithm qmix --num-agents 2 --num-envs 1 \
    --layout overcooked_cramped_room_v0 --total-steps 1000000 \
    --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.995 \
    --buffer-size 10000 --batch-size-qmix 32 --lr 5e-4 \
    --save --log --data-path qmix_results
```

## Implementation Notes

### Environment Assumptions
- **num_envs = 1**: The QMIX implementation assumes single environment training for simplicity
- **Global State**: Uses concatenated agent observations as the global state
- **Team Reward**: Sums individual agent rewards for team reward signal

### Key Differences from MAPPO
1. **Value-based vs Policy-based**: QMIX learns Q-values while MAPPO learns policies
2. **Experience Replay**: QMIX uses experience replay buffer, MAPPO uses on-policy learning
3. **Exploration**: QMIX uses epsilon-greedy exploration, MAPPO uses stochastic policies
4. **Centralized Training**: Both use centralized training but different mechanisms

### Hyperparameter Recommendations

#### Small Environments (2 agents, cramped room)
- Learning rate: 5e-4
- Buffer size: 10,000
- Batch size: 32
- Epsilon decay: 0.995
- Target update frequency: 200

#### Large Environments (4+ agents)
- Learning rate: 5e-4
- Buffer size: 15,000+
- Batch size: 64
- Epsilon decay: 0.9995 (slower decay)
- Target update frequency: 300

## File Structure

```
QMIX.py                    # Main QMIX implementation
test_qmix.py              # Unit tests
test_qmix_integration.py  # Integration tests
main.py                   # Modified to support QMIX
Makefile                  # Added QMIX targets
```

## Expected Performance

QMIX typically shows:
- Good sample efficiency due to experience replay
- Strong coordination in environments requiring explicit cooperation
- Stable learning with proper hyperparameter tuning
- May require more exploration time than policy-based methods initially

## Troubleshooting

### Common Issues
1. **Slow Initial Learning**: Increase epsilon_start or decrease epsilon_decay
2. **Unstable Training**: Reduce learning rate or increase target_update_freq
3. **Memory Issues**: Reduce buffer_size or batch_size_qmix
4. **Poor Coordination**: Increase mixing_embed_dim or hidden_dim

### Debug Mode
Use `make qmix-debug` for a quick diagnostic run with detailed logging.
