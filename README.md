# MARL
Multiagent-PPO (Decentralised/centralised) on [cogrid](https://github.com/chasemcd/cogrid)
## Prerequisite
- `python >= 3.10`
- `virtualenv env && source env/bin/activate`
- pip3 install -r requirements.txt
## Running instruction '
`python3 main.py --save-path models --num-agents 2 --layout overcooked_cramped_room_v0  --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000 --seed 2 --log --centralised --ppo-epoch 5 --clip-param 0.2 \
	--value-loss-coef 0.5 --entropy-coef 0.3 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 1e-4`
 
or `make cramped` with makefile (runs the command above)

current supported `layout`s are registered [here](https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/__init__.py#L13)
## Test Running the model
`Assuming there exist a model object model/policy.pth`
- `python3 test_load.py --model-path <path-to-policy.pth>`
- Example `python3 test_load.py --model-path model/policy.pth`
