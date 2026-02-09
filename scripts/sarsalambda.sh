#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=5:00:00
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/home/truonggi/projects/aip-mtaylor3/truonggi/slurm_out/%A.out
#SBATCH --mail-user=truonggi@ualberta.ca
#SBATCH --mail-type=ALL

export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.10
module load gcc opencv/4.9.0
source /home/truonggi/projects/aip-mtaylor3/truonggi/MARL/env/bin/activate 

echo $1 # num_episodes
echo $2 # gamma
echo $3 # lambda
echo $4 # epsilon
echo $5 # alpha
echo $6 # seed
echo ${13} # data_path
echo ${14} # layout
echo ${15} # feature

# parameter tune
python3 ../sarsalambda.py --num_episodes $1 --gamma $2 --lambda_ $3 --epsilon $4 --alpha $5 --seed $6 --data_path ${13} --layout ${14} --feature ${15}


EPSILON=(0.001 0.005 0.01 0.05 0.1 0.2)
GAMMA=(0.9 0.95 0.99 0.999)
LAMBDA=(0.1 0.5 0.7 0.9)
ALPHA=(0.01 0.05 0.1 0.2 0.3 0.5)
SEED=(1)
LAYOUT=("overcooked_cramped_room_v0")
FEATURE=("Binary_feature")

for epsilon in "${EPSILON[@]}"; do
    for gamma in "${GAMMA[@]}"; do
        for lambda in "${LAMBDA[@]}"; do
            for alpha in "${ALPHA[@]}"; do
                for seed in "${SEED[@]}"; do
                    for layout in "${LAYOUT[@]}"; do
                        for feature in "${FEATURE[@]}"; do
                            python3 ../sarsalambda.py --num_episodes $1 --gamma $gamma --lambda_ $lambda --epsilon $epsilon --alpha $alpha --seed $seed --data_path ${13} --layout $layout --feature $feature
                        done
                    done
                done
            done
        done
    done
done