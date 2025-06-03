import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from supersuit.vector.constructors import MakeCPUAsyncConstructor
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