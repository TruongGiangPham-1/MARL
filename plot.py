
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data", help="path to the folder containing csv")
    parser.add_argument("--keyword", type=str, default="returns", help="keyword to filter csv files")
    args = parser.parse_args()
    folder_name = args.folder
    keyword = args.keyword
    running_avg_range = 100

    if keyword == "returns":
        produce_plots_for_all_configs(folder_name, keyword)
    elif keyword == "pot":
        produce_plots_for_all_configs(folder_name, keyword)
    elif keyword == "delivery":
        produce_plots_for_all_configs(folder_name, keyword)
    return


def extract_config(filename_without_ext):
    configs = ['seed_1', 'seed_2', 'seed_3', 'seed_4']
    for configuration in configs:
        if configuration in filename_without_ext:
            return configuration
    return None


"""
Load all returns.csv files and plot them
"""
def produce_plots_for_all_configs(folder_name="data", keyword="returns"):
    configs = ['seed_1', 'seed_2', 'seed_3', 'seed_4']
    data_dict = {}
    for configuration in configs:
        data_dict[configuration] = []
    files = os.listdir(folder_name)
    for file in files:
        full_path = os.path.join(folder_name, file)
        if os.path.isfile(full_path):
            if os.path.splitext(file)[-1] == '.csv' and keyword  in file:
                config = extract_config(os.path.splitext(file)[0])
                assert config is not None, f"{file} is not in the required format."
                print(f"Reading {full_path}")
                df = pd.read_csv(full_path)
                data_dict[config].append(np.squeeze(df.values))

    for configuration in configs:
        if data_dict[configuration]:
            if keyword == "returns":
                plot_alg_results(data_dict[configuration], f"Overcooked.png", label="Running average")
            elif keyword == "pot":
                plot_ingredients_in_pots(data_dict[configuration], f"Overcooked_ingredients_in_pots.png", label="",title="Overcooked_2 agents in cramped room - Ingredients in Pots",  ylabel="frequency")
            elif keyword == "delivery":
                plot_ingredients_in_pots(data_dict[configuration], f"Overcooked_delivery.png", label="",title="Overcooked_2 agents in cramped room - Delivery",  ylabel="frequency")


def plot_ingredients_in_pots(episode_returns_list, file, label="Algorithm", ylabel="frequency", title="overcooked ingredient in pots", eval_interval=1000):
    """
    episode_returns_list: list of episode returns. If there is 3 seeds, then the list should have 3 lists.
    """
    # Compute running average
    print(len(episode_returns_list))
    running_avg = np.mean(np.array(episode_returns_list), axis=0)  # Average over seeds. dim (1, num_episodes)
    new_running_avg = running_avg.copy()
    #for i in range(len(running_avg)):
    #    new_running_avg[i] = np.mean(running_avg[max(0, i-10):min(len(running_avg), i + 10)])  # each point is the average of itself and its neighbors (+/- 10*eval_interval)
    #running_avg = new_running_avg

    # x axis goes by 1000
    eval_interval = 1
    x_coords = [eval_interval * (i + 1) for i in range(len(running_avg))]
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot individual seeds with light transparency
    for seed_returns in episode_returns_list:
        plt.plot(x_coords, seed_returns,  color='gray', alpha=0.5)
    # Plot the running average
    plt.plot(
        x_coords,
        running_avg,
        color='r',
        label=label
    )
    #plt.plot(x_coords, np.full(len(running_avg), 3500)   , color='b', label='threshold')

    # Adding labels and title
    if 'Ant' in file:
        plt.title(f"")
    else:
        plt.title(f"Overcooked_2 agents in cramped room - Ingredients in Pots")
    plt.xlabel("episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return",title="overcooked rewards",  eval_interval=1000):
    """
    episode_returns_list: list of episode returns. If there is 3 seeds, then the list should have 3 lists.
    """
    # Compute running average
    print(len(episode_returns_list))
    running_avg = np.mean(np.array(episode_returns_list), axis=0)  # Average over seeds. dim (1, num_episodes)
    new_running_avg = running_avg.copy()
    for i in range(len(running_avg)):
        new_running_avg[i] = np.mean(running_avg[max(0, i-10):min(len(running_avg), i + 10)])  # each point is the average of itself and its neighbors (+/- 10*eval_interval)
    running_avg = new_running_avg

    # x axis goes by 1000
    eval_interval = 1
    x_coords = [eval_interval * (i + 1) for i in range(len(running_avg))]
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot individual seeds with light transparency
    for seed_returns in episode_returns_list:
        plt.plot(x_coords, seed_returns,  color='gray', alpha=0.5)
    # Plot the running average
    plt.plot(
        x_coords,
        running_avg,
        color='r',
        label=label
    )
    #plt.plot(x_coords, np.full(len(running_avg), 3500)   , color='b', label='threshold')

    # Adding labels and title
    if 'Ant' in file:
        plt.title(f"")
    else:
        plt.title(f"Overcooked_2 agents in cramped room")
    plt.xlabel("episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == "__main__":
    main()