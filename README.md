# MARL

## Prerequisite
- `python >= 3.10`
- `virtualenv env && source env/bin/activate`
- pip3 install -r requirements.txt
## Running instruction
- `python3 main.py --save --save-path models --num-agents 4 --layout large_overcooked_layout ` or `make all` with makefile

current supported `layout` is `large_overcooked_layout`. There should be more but I haven't figure out yet
## Test Running the model
`Assuming there exist a model object model/policy.pth`
- `python3 test_load.py --model-path <path-to-policy.pth>`
- Example `python3 test_load.py --model-path model/policy.pth`
