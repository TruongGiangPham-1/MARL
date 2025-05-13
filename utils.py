import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio

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
    print(f'action_probs {action_probs[0].shape}') 
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