# Overcooked MARL
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

Multiagent-PPO (Decentralised/centralised) on [cogrid](https://github.com/chasemcd/cogrid)'s overcooked. 
## Prerequisite
- `python >= 3.10`
- `virtualenv env && source env/bin/activate`
- pip3 install -r requirements.txt

## File Structures
```bash
├── CentralizedMAPPO.py                      # Multi-agent PPO with Centralized critic.                                                
├── MAPPO.py                                 # Multi-agent PPO with Decentralized critic
├── Makefile                                 # target (cramped | inference )
├── README.md 
├── agent_environment.py                     # RL agent-environment loop
├── buffer.py                                # Experience buffer
├── main.py                                  # entry point for training overcooked
├── model.py                              
├── mpe.py                                
├── overcooked_config.py                     # configures num_agents, layout, reward, etc
├── overcooked_features.py                   # Featurize state
├── plot.py                                  
├── requirements.txt 
├── requirements_cc.txt         
├── scripts                                  # Compute Canada sbatch script
│   ├── CC_script.sh     
│   └── param_tune.sh
├── test_load.py                              
├── utils.py                                  
└── video2gif.py
```

## Basic Usage
To train overcooked-ai, we call `main.py` with specified parameters.


For example, you can train a `cramped-room` layout with 2 agents using:  
```
python3 main.py --save-path models --num-agents 2 --num-envs 16 --layout overcooked_cramped_room_v0  --batch-size 256 --num-minibatches 4 \
	--total-steps 20000000 --seed 2 --log --centralised --ppo-epoch 5 --clip-param 0.2 \
	--value-loss-coef 0.5 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 3e-4 --data-path data \
  --feature global_obs
```
current supported `layout`s are registered [here](https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/__init__.py#L13)

current feature set is `global_obs`, `local_obs`, `only_direction`

```bash
# main.py's arguments
  --num-agents NUM_AGENTS        # number of agents
  --num-envs NUM_ENVS            # number of parallel environment to generate samples
  --layout LAYOUT                # layouts amongst [here](https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/__init__.py#L13)
  --save-path SAVE_PATH          # path to save the NN model
  --data-path DATA_PATH          # path to save the rewards and other metric csv files needed for plotting
  --save                         # weather to Save the model
  --total-steps TOTAL_STEPS      # total number of action agents take in the span of training
  --num-steps NUM_STEPS          # number of steps per environment before updating the NN (PPO thing)
  --num-minibatches NUM_MINIBATCHES
  --log                          # whether to tensorboard log
  --render                       # whether to render the env
  --seed SEED                  
  --lr LR                        # learning rate

## PPO specific hyperparams

  --ppo-epoch PPO_EPOCH
  --clip-param CLIP_PARAM
  --value-loss-coef VALUE_LOSS_COEF
  --entropy-coef ENTROPY_COEF
  --max-grad-norm MAX_GRAD_NORM
  --gamma GAMMA                  # discount factor
  --lam LAM                      # lambda for GAE
  --centralised                  # False is decentralised MAPPO, True is centralised MAPPO
```

## Generating plots
After training is finished, a data directory is generated inside `data-path`. This directory contains metric to be plotted.
```
python3 plot.py --folder <data-path> --keyword {returns, delivery, pot}
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/a240b0dc-2ec6-4586-bdbf-9ad38d3e5f03" alt="Overcooked" width="45%" />
  <img src="https://github.com/user-attachments/assets/b7eac1cf-3dfa-48d9-a71a-19a12cbb64c2" alt="Overcooked Delivery" width="45%" />
</p>
  
## Test Running the model
`Assuming there exist a model object model/policy.pth`
- `python3 test_load.py --model-path <path-to-policy.pth>`
- Example `python3 test_load.py --model-path model/policy.pth`

## Overcooked Features
`AgebtDir`: One hot of size 4. [RIGHT, DOWN, LEFT, UP]
``