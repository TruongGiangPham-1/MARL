# MARL

## Prerequisite
- `python >= 3.10`
- `virtualenv env && source env/bin/activate`
- pip3 install -r requirements.txt
## Running instruction
- `python3 main.py --save --save-path models --num-agents 4 --layout large_overcooked_layout ` or `make all` with makefile

current supported `layout`s are registered [here](https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/__init__.py#L13)
## Test Running the model
`Assuming there exist a model object model/policy.pth`
- `python3 test_load.py --model-path <path-to-policy.pth>`
- Example `python3 test_load.py --model-path model/policy.pth`
