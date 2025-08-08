#!/bin/bash

# QMIX Hyperparameter Tuning Script
# Parameter order: layout num_agents num_episodes seed lr gamma epsilon_start epsilon_end epsilon_decay target_update_freq buffer_size batch_size_qmix mixing_embed_dim hidden_dim data_path feature

# Different Learning Rates
echo "=== Testing Different Learning Rates ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 1e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_lr1e4 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_lr5e4 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 1e-3 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_lr1e3 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 2e-3 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_lr2e3 global_obs

# Different Epsilon Decay Rates
echo "=== Testing Different Epsilon Decay Rates ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.90 200 25000 32 32 256 qmix_data_eps90 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_eps95 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.99 200 25000 32 32 256 qmix_data_eps99 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.995 200 25000 32 32 256 qmix_data_eps995 global_obs

# Different Epsilon End Values
echo "=== Testing Different Epsilon End Values ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.001 0.95 200 25000 32 32 256 qmix_data_epsend001 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_epsend01 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.05 0.95 200 25000 32 32 256 qmix_data_epsend05 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.1 0.95 200 25000 32 32 256 qmix_data_epsend1 global_obs

# Different Target Update Frequencies
echo "=== Testing Different Target Update Frequencies ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 100 25000 32 32 256 qmix_data_target100 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_target200 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 500 25000 32 32 256 qmix_data_target500 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 1000 25000 32 32 256 qmix_data_target1000 global_obs

# Different Buffer Sizes
echo "=== Testing Different Buffer Sizes ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 5000 32 32 256 qmix_data_buf5k global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 10000 32 32 256 qmix_data_buf10k global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_buf25k global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 50000 32 32 256 qmix_data_buf50k global_obs

# Different Batch Sizes
echo "=== Testing Different Batch Sizes ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 16 32 256 qmix_data_batch16 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_batch32 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 64 32 256 qmix_data_batch64 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 128 32 256 qmix_data_batch128 global_obs

# Different Network Architectures
echo "=== Testing Different Network Architectures ==="
# Small networks
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 16 128 qmix_data_small global_obs
# Medium networks (baseline)
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_medium global_obs
# Large networks
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 64 64 512 qmix_data_large global_obs
# Extra large networks
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 128 128 1024 qmix_data_xlarge global_obs

# Different Gamma Values
echo "=== Testing Different Gamma Values ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.95 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_gamma95 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_gamma99 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.995 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_gamma995 global_obs

# Different Seeds for Reproducibility
echo "=== Testing Different Seeds ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_seed1 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 2 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_seed2 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 3 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_seed3 global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 42 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_seed42 global_obs

# Different Episode Counts
echo "=== Testing Different Episode Counts ==="
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 2000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_2k_eps global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_5k_eps global_obs
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 10000 1 5e-4 0.99 1.0 0.01 0.95 200 25000 32 32 256 qmix_data_10k_eps global_obs

# Advanced Combinations - Best performing combinations
echo "=== Testing Advanced Combinations ==="
# High exploration, frequent updates
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 5000 1 1e-3 0.99 1.0 0.05 0.99 100 50000 64 64 512 qmix_data_combo1 global_obs
# Conservative exploration, stable learning
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 8000 1 5e-4 0.995 1.0 0.001 0.995 500 25000 32 32 256 qmix_data_combo2 global_obs
# Balanced approach
sbatch CC_script_qmix.sh overcooked_cramped_room_v0 2 6000 1 7e-4 0.99 1.0 0.02 0.97 200 30000 48 48 384 qmix_data_combo3 global_obs

echo "=== All QMIX hyperparameter tuning jobs submitted! ==="
