#!/bin/bash

#echo "num_envs: $1"
#echo "num_steps: $2"
#echo "num_minibatches: $3"
#echo "total_steps: $4"
#echo "seed: $5"
#echo "ppo_epoch: $6"
#echo "clip_param: $7"
#echo "value_loss_coef: $8"
#echo "entropy_coef: $9"
#echo "gamma: ${10}"
#echo "lam: ${11}"
#echo "lr: ${12}"
#echo "data_path: ${13}"

# different LR
sbatch CC_script.sh 16 256 4 20000000 2 5 0.05 0.1 0.1 0.99 0.95 1e-4 data1
sbatch CC_script.sh 16 256 4 20000000 2 5 0.05 0.1 0.1 0.99 0.95 3e-4 data2
sbatch CC_script.sh 16 256 4 20000000 2 5 0.05 0.1 0.1 0.99 0.95 5e-4 data3

# different ppo epoch
sbatch CC_script.sh 16 256 4 20000000 2 7 0.05 0.1 0.1 0.99 0.95 1e-4 data5
sbatch CC_script.sh 16 256 4 20000000 2 10 0.05 0.1 0.1 0.99 0.95 1e-4 data6

# different entropy coef
sbatch CC_script.sh 16 256 4 20000000 2 5 0.05 0.1 0.01 0.99 0.95 1e-4 data7





