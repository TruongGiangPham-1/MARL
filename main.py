# BASE CODES FROM https://github.com/chasemcd/n_agent_overcooked/tree/main
import functools

from cogrid.feature_space import feature_space
from cogrid.envs.overcooked import overcooked
from cogrid.core import layouts
from cogrid.envs import registry

# import supersuit
import supersuit as ss
from model import Agent 
import torch
import argparse
from MAPPO import MAPPO
from agent_environment import agent_environment_loop



from overcooked_features import NAgentOvercookedFeatureSpace

# CoGrid is based on a registry system, so we need to register the feature
# that we want to use for the environment. You can do this in any imported
# file, but we're doing it here for clarity.
feature_space.register_feature(
    "n_agent_overcooked_features", NAgentOvercookedFeatureSpace
)



from overcooked_features import NAgentOvercookedFeatureSpace

# CoGrid is based on a registry system, so we need to register the feature
# that we want to use for the environment. You can do this in any imported
# file, but we're doing it here for clarity.
feature_space.register_feature(
    "n_agent_overcooked_features", NAgentOvercookedFeatureSpace
)


# Similarly, we register the layout that we want to use for the environment.
# We have an ascii-based design option, but you can also use a function to
# design, e.g., dynamically generated layouts.
# Each character represents an object that we've defined and registered with CoGrid:
#
#   C -> Counter. An impassable wall but items can be placed on and picked up from it.
#   @ -> Deliver Zone. Cooked dishes can be dopped here to be delivered.
#   = -> Stack of dishes. Agents can pick up from the (unlimited) dish stack to plate cooked onions.
#   O -> Pile of onions. Agents can pick up from the (unlimited) onion pile.
#   U -> Cooking pot. After placing three onions in the pot, they'll cook and can be plated.
#   # -> Walls. Walls are impassable and CoGrid environments must be rectangular and enclosed by walls.
#
# Importantly, we don't specify spawn positions in the environment below. However, if you'd
# like to be able to dictate the exact positions that agents will spawn (e.g., particularly
# in layouts with clear spacial delineations), you can use the "+" character. This will
# randomize agent spawn positions at +'s. In the definition below, spawns are simply selected
# randomly from empty spaces.
large_layout = [
    "#################",
    "#C@CC=CCCCCCCUUC#",
    "#C  C     C    C#",
    "#C  C COO C    C#",
    "#C    CCCCC    C#",
    "#C             C#",
    "#C   CCCCCC    C#",
    "#C   CCOOCC C  C#",
    "#C   C      C  C#",
    "#CUUCCCCCCC=CC@C#",
    "#################",
]

layouts.register_layout("large_overcooked_layout", large_layout)


# Now, we specify the configuration for the environment.
# In the near future this will be a configuration class
# to make the arguments clearer, but for now it's a dictionary:
four_agent_overcooked_config = {
    "name": "FourAgentOvercooked-V0",
    "num_agents": 4,
    # We have two different ways to represent actions in CoGrid.
    # Both have common actions of No-Op, Toggle, and Pickup/Drop.
    # "cardinal_actions" is the default and uses the 4 cardinal directions
    # to move (e.g., move up, down, left, right). This is intuitive for
    # human players, where pressing the right arrow moves you right. There
    # is also "rotation_actions", which only uses forward movement and two
    # rotation actions (e.g., rotate left, rotate right). This is in line
    # with the original Minigrid environments.
    "action_set": "cardinal_actions",
    # We'll use the NAgentOvercookedFeatureSpace that we registered
    # earlier. The features can be specified as a list of features if
    # you have more than one or as a dictionary keyed by agent ID
    # and a list of feature names for each agent if different
    # agents should have different observation spaces.
    "features": "n_agent_overcooked_features",
    # In the same way that we register features and
    # layouts, we can also register reward functions.
    # The delivery reward (common reward of +1 whenever a
    # dish is delivered) has already been registered
    # for overcooked. Some more details are in the documentation
    # on how you could add alternative reward functions. For
    # Overcooked, you can enable reward shaping done by Carroll et al.
    # by specifying "onion_in_pot_reward" and "soup_in_dish_reward"
    # in the rewards list below.
    "rewards": ["delivery_reward"],
    # The scope is used by CoGrid to determine how to
    # map the ascii text in the layout to the environment
    # objects. All objects are registred in the "overcooked"
    # scope.
    "scope": "overcooked",
    # We'll load a single constant layout here, which will be
    # used to instantiate the environment from our registered
    # ASCII layout. You could alternatively pass a "layout_fn",
    # which could generate a layout dynamically.
    "grid": {"layout": "large_overcooked_layout"},
    # Number of steps per episode.
    "max_steps": 1000,
    "num_agents": 4,
}

# Finally, we register the environment with CoGrid. This makes it convenient
# to instantiate the environment from the registry as we do below, but you could
# also just pass the config to the Overcooked constructor directly.
registry.register(
    "FourAgentOvercooked-V0",
    functools.partial(
        overcooked.Overcooked, config=four_agent_overcooked_config
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU for training.",
    )
    args = parser.parse_args()

    env = registry.make("FourAgentOvercooked-V0", render_mode="human")
    env.reset()
    #print(f'env obs {env.observation_spaces[0]["n_agent_overcooked_features"].shape}')  # (404)
    obs, R, terminated, truncated, info = env.step(
        {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.agents
        }
    )  
    print(f'{R}, {terminated}, {truncated}')
    
    """
    obs spaces:
    {0: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    1: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    2: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32)), 
    3: Dict('n_agent_overcooked_features': Box(-inf, inf, (404,), float32))}

    action spaces:
    {0: Discrete(7), 1: Discrete(7), 2: Discrete(7), 3: Discrete(7)}

    o, r, t, t, i = env.step()

    r = {agent_id: R for agent_id in N}
    t = {agent_id: terminated for agent_id in N}
    t = {agent_id: truncated for agent_id in N}
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    net = Agent(env).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


    collect_steps = 5

    single_agent_obs_dim = env.observation_spaces[0]['n_agent_overcooked_features'].shape  # 
    sigle_agent_action_dim = env.action_spaces[0].n  # int
    ppo_agent = MAPPO(env, optimizer, net, single_agent_obs_dim, sigle_agent_action_dim, collect_steps=collect_steps)

    reward = agent_environment_loop(ppo_agent, env, device, num_episodes=1000)

    return
    
if __name__ == "__main__":
    main()