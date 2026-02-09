import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from supersuit.vector.constructors import MakeCPUAsyncConstructor
import hashlib
import cloudpickle

from io import BytesIO
def evaluate_state(agent, env, device, global_step=1000):
    # load states array from states/*.npy into python list
    states = []
    action_probs = []
    for npy_file in os.listdir("states/"):
        if npy_file.endswith(".npy"):
            state = np.load(os.path.join("states/", npy_file))
            states.append(state)
            break
    with torch.no_grad():
        for state in states:
            state = torch.FloatTensor(state).to(device)
            action_prob = agent.policy.get_prob(state).probs.cpu().numpy()
            action_probs.append(action_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(action_probs[0].squeeze())), action_probs[0].squeeze(), color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title(f"Action Distribution at timestep {global_step}")
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer,format='png')
    plt.close(fig)
    return buffer.getvalue()



## --- my custom supersuit stuff since supersuit.concat_vec_envs_v1 is not working :( 
def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return ([env_fn] * num_envs, env.observation_space, env.action_space)

def concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class="gymnasium"):
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))

    if base_class == "gymnasium":
        return vec_env
    else:
        raise ValueError(
            "supersuit_vec_env only supports 'gymnasium', 'stable_baselines', and 'stable_baselines3' for its base_class"
        )
    
def get_hash_id(args, exclude_keys=("seed", "data_path", "log", "num_episodes"), n=8):
    # remove excluded keys (like seed)
    args_filtered = {k: v for k, v in args.items() if k not in exclude_keys}
    args_json = json.dumps(args_filtered, sort_keys=True)
    h = hashlib.md5(args_json.encode()).hexdigest() 
    return h[:n]

def get_run_folder(args):
    """
    Create run_folder=data_path/layout/hash_id/seed_folder if not exist
    eg file structure for results: 
        data/
        ├── cramped_room/
        │   └── <hashID>/
        │       ├── seed_1/
        │       │   ├── reward.csv
        │       │   ├── config.json
        │       │   └── checkpoints/
        │       ├── seed_2/
        │       │   ├── reward.csv
        │       │   ├── config.json
        │       │   └── checkpoints/
        │       └── seed_3/
    :param args: {key: value} dictionary of command line arguments
    :return: run_folder path
    """
    import os

    base = args.get("data_path", "data")
    layout = args["layout"]
    hash_id = get_hash_id(args)
    seed_folder = f"seed_{args['seed']}"

    run_folder = os.path.join(base, layout, hash_id, seed_folder)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder