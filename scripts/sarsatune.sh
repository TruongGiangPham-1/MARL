#!/bin/bash

#echo $1 # num_episodes
#echo $2 # gamma
#echo $3 # lambda
#echo $4 # epsilon
#echo $5 # alpha
#echo $6 # seed
#echo $7 # data_path
#echo $8 # layout
#echo $9 # feature

EPSILON=(0.001 0.005 0.01 0.05 0.1 0.2)
GAMMA=(0.9 0.95 0.99 0.999)
LAMBDA=(0.1 0.5 0.7 0.9)
ALPHA=(0.01 0.05 0.1 0.2 0.3 0.5)
SEED=(1)
LAYOUT=("overcooked_cramped_room_v0")
FEATURE=("Binary_feature")
DATA_PATH="sarsa_lambda_data"

for epsilon in "${EPSILON[@]}"; do
    for gamma in "${GAMMA[@]}"; do
        for lambda in "${LAMBDA[@]}"; do
            for alpha in "${ALPHA[@]}"; do
                for seed in "${SEED[@]}"; do
                    for layout in "${LAYOUT[@]}"; do
                        for feature in "${FEATURE[@]}"; do
                            sbatch sarsalambda.sh 100000 $gamma $lambda $epsilon $alpha $seed $DATA_PATH $layout $feature
                        done
                    done
                done
            done
        done
    done
done