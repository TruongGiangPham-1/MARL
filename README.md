# MARL
Multiagent-PPO (Decentralised/centralised) on [cogrid](https://github.com/chasemcd/cogrid)
## Prerequisite
- `python >= 3.10`
- `virtualenv env && source env/bin/activate`
- pip3 install -r requirements.txt
## Running instruction '
`python3 main.py --save-path models --num-agents 2 --num-envs 16 --layout overcooked_cramped_room_v0  --batch-size 256 --num-minibatches 4 \
	--total-steps 20000000 --seed 2 --log --centralised --ppo-epoch 5 --clip-param 0.2 \
	--value-loss-coef 0.5 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 3e-4 --data-path data`
 
or `make cramped` with makefile (runs the command above)

current supported `layout`s are registered [here](https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/__init__.py#L13)
## Generating plots
After training is finished, a data directory is generated inside `data-path`. This directory contains metric to be plotted.
- `python3 plot.py --folder <data-path> --keyword {returns, delivery, pot}` 
## Test Running the model
`Assuming there exist a model object model/policy.pth`
- `python3 test_load.py --model-path <path-to-policy.pth>`
- Example `python3 test_load.py --model-path model/policy.pth`
