# BASE CODES FROM https://github.com/chasemcd/n_agent_overcooked/tree/main
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.feature_space import feature_space
from cogrid.feature_space import feature
from cogrid.feature_space import features
from cogrid.envs.overcooked import overcooked_features
from cogrid import cogrid_env
from cogrid.core import grid_object
import numpy as np

class globalObs(feature.Feature):
    """
    A wrapper class to generate all encoded Overcooked features as a single array.

    For each agent j, calculate:

        - Agent j Direction
        - Agent j Inventory
        - Agent j Adjacent to Counter
        - Agent j Dist to closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
        - Agent j Pot Features for the two closest pots
            - pot_k_reachable: {0, 1}  # NOTE(chase): This is hardcoded to 1 currently.
            - pot_k_status: onehot of {empty | full | is_cooking | is_ready}
            - pot_k_contents: integer of the number of onions in the pot
            - pot_k_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
            - pot_k_distance: (dy, dx) from the player's location
            - pot_k_location: (row, column) of the pot on the grid
        - Agent j Distance to other agents j != i
        - Agent j Position

    The observation is the concatenation of all these features for all players.
    """

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(),
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # The (dy, dx) distance to the closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Onion, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Plate, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.PlateStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionSoup, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.DeliveryZone, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=grid_object.Counter, n=4
            ),
            # All pot features for the closest two pots
            overcooked_features.NClosestPotFeatures(num_pots=2),
            # The (dy, dx) distance to the closest other agent
            overcooked_features.DistToOtherPlayers(
                num_other_players=num_agents - 1
            ),
            # The (row, column) position of the agent
            features.AgentPosition(),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        )

        #feature_sum = 0
        #feature_dict = {

        #}
        #for feature in self.agent_features:
        #    print(
        #        f"Feature: {feature.name}, shape: {feature.shape}"
        #    )
        #    if feature.name not in feature_dict:
        #        feature_dict[feature.name] = 0
        #    feature_dict[feature.name] += 1
        #    feature_sum += feature.shape[0]
        #print(f"Total feature shape: {feature_sum}")
        #print(f"Feature dict: {feature_dict}")

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]

        for pid in env.agent_ids:
            if pid == player_id:
                continue
            player_encodings.append(self.generate_player_encoding(env, pid))

        encoding = np.hstack(player_encodings).astype(np.float32)

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)

class localObs(feature.Feature):
    """
    For each agent j, calculate:

        - Agent j Direction
        - Agent j Inventory
        - Agent j Adjacent to Counter
        - Agent j Dist to closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
        - Agent j Pot Features for the two closest pots
            - pot_k_reachable: {0, 1}  # NOTE(chase): This is hardcoded to 1 currently.
            - pot_k_status: onehot of {empty | full | is_cooking | is_ready}
            - pot_k_contents: integer of the number of onions in the pot
            - pot_k_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
            - pot_k_distance: (dy, dx) from the player's location
            - pot_k_location: (row, column) of the pot on the grid
        - Agent j Distance to other agents j != i
        - Agent j Position
    """

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(),
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # The (dy, dx) distance to the closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Onion, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Plate, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.PlateStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionSoup, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.DeliveryZone, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=grid_object.Counter, n=4
            ),
            # All pot features for the closest two pots
            overcooked_features.NClosestPotFeatures(num_pots=2),
            # The (dy, dx) distance to the closest other agent
            overcooked_features.DistToOtherPlayers(
                num_other_players=num_agents - 1
            ),
            # The (row, column) position of the agent
            features.AgentPosition(),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        )

        full_shape = np.sum([feature.shape for feature in self.agent_features])  # 101

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]

        encoding = np.hstack(player_encodings).astype(np.float32)

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)
    
class MinimalSpatialOtherAgentAware(feature.Feature):
    """
    MinimalSpatial but knows distance to other agents.
    """

    def __init__(self, env, **kwargs):
        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(),
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            overcooked_features.DistToOtherPlayers(
                num_other_players=num_agents - 1
            ),
            # The (row, column) position of the agent
            features.AgentPosition(),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = np.sum(
            [feature.shape for feature in self.agent_features]
        )
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]
        encoding = np.hstack(player_encodings).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))
        return np.hstack(encoded_features)


class MinimalSpatial(feature.Feature):
    """
    Minimal spatial awareness - only immediate surroundings and self state.
    Good for testing agents with limited environmental awareness.
    """

    def __init__(self, env, **kwargs):
        self.agent_features = [
            features.AgentDir(),
            overcooked_features.OvercookedInventory(),
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            features.AgentPosition(),
            features.CanMoveDirection(),
        ]

        full_shape = np.sum([feat.shape for feat in self.agent_features])
        super().__init__(
            low=-np.inf, high=np.inf, shape=(full_shape,),
            name="n_agent_overcooked_features", **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        encoded_features = []
        for feat in self.agent_features:
            encoded_features.append(feat.generate(env, player_id))
        encoding = np.hstack(encoded_features).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding

class BinaryFeature(feature.Feature):
    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),  # Binary
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(), # Binary
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(), # Binary
            overcooked_features.NextToPot(), # Binary
            # All pot features for the closest two pots
            NClosestBinaryPotFeatures(num_pots=2, grid=env.grid),
            # The (dy, dx) distance to the closest other agent
            #overcooked_features.DistToOtherPlayers(
            #    num_other_players=num_agents - 1
            #),
            # The (row, column) position of the agent
            BinaryAgentPosition(grid=env.grid),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        )

        full_shape = np.sum([feature.shape for feature in self.agent_features])  # 101

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]

        encoding = np.hstack(player_encodings).astype(np.float32)

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)

class TaskFocused(feature.Feature):
    """
    Task-oriented features focusing on cooking workflow.
    Includes ingredients, pots, and delivery but minimal spatial info.
    Missing adjacent tiles information
    """

    def __init__(self, env, **kwargs):
        self.agent_features = [
            features.AgentDir(),
            overcooked_features.OvercookedInventory(),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Onion, n=2),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Plate, n=2),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionStack, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.PlateStack, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionSoup, n=2),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.DeliveryZone, n=1),
            overcooked_features.NClosestPotFeatures(num_pots=1),
            features.AgentPosition(),
            features.CanMoveDirection(),
        ]

        full_shape = np.sum([feat.shape for feat in self.agent_features])
        super().__init__(
            low=-np.inf, high=np.inf, shape=(full_shape,),
            name="task_focused_features", **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        encoded_features = []
        for feat in self.agent_features:
            encoded_features.append(feat.generate(env, player_id))
        encoding = np.hstack(encoded_features).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class ReducedRange(feature.Feature):
    """
    Similar to localObs but with reduced sensing range.
    Tests performance with limited visibility of distant objects.
    """

    def __init__(self, env, **kwargs):
        num_agents = env.config["num_agents"]

        self.agent_features = [
            features.AgentDir(),
            overcooked_features.OvercookedInventory(),
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # Reduced n values for limited range
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Onion, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Plate, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.PlateStack, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionStack, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionSoup, n=1),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.DeliveryZone, n=1),
            overcooked_features.ClosestObj(focal_object_type=grid_object.Counter, n=1),
            overcooked_features.NClosestPotFeatures(num_pots=1),  # Only closest pot
            overcooked_features.DistToOtherPlayers(num_other_players=num_agents - 1),
            features.AgentPosition(),
            features.CanMoveDirection(),
        ]

        full_shape = np.sum([feat.shape for feat in self.agent_features])
        super().__init__(
            low=-np.inf, high=np.inf, shape=(full_shape,),
            name="reduced_range_features", **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        encoded_features = []
        for feat in self.agent_features:
            encoded_features.append(feat.generate(env, player_id))
        encoding = np.hstack(encoded_features).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding

class ExtendedRange(feature.Feature):
    """
    Enhanced sensing range compared to localObs.
    Tests if more environmental information improves performance.
    """

    def __init__(self, env, **kwargs):
        num_agents = env.config["num_agents"]

        self.agent_features = [
            features.AgentDir(),
            overcooked_features.OvercookedInventory(),
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # Extended n values for wider range
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Onion, n=6),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.Plate, n=6),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.PlateStack, n=3),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionStack, n=3),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.OnionSoup, n=6),
            overcooked_features.ClosestObj(focal_object_type=overcooked_grid_objects.DeliveryZone, n=3),
            overcooked_features.ClosestObj(focal_object_type=grid_object.Counter, n=6),
            overcooked_features.NClosestPotFeatures(num_pots=3),  # More pots
            overcooked_features.DistToOtherPlayers(num_other_players=num_agents - 1),
            features.AgentPosition(),
            features.CanMoveDirection(),
        ]

        full_shape = np.sum([feat.shape for feat in self.agent_features])
        super().__init__(
            low=-np.inf, high=np.inf, shape=(full_shape,),
            name="extended_range_features", **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        encoded_features = []
        for feat in self.agent_features:
            encoded_features.append(feat.generate(env, player_id))
        encoding = np.hstack(encoded_features).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class SuccessfullyDeliveredSoup(feature.Feature):
    """
    A feature that returns 1 if the agent has successfully delivered a soup, 0 otherwise.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(1,),
            name="successfully_delivered_soup",
            **kwargs,
        )
        self._is_done = False
    
    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        # env.grid impl see https://github.com/chasemcd/cogrid/blob/main/cogrid/core/grid.py
        # example in reward.py where it checks if the agent is facing delivery zone
        # https://github.com/chasemcd/cogrid/blob/f1beb729cf3ff8a939f385396a235007a5b2dd76/cogrid/envs/overcooked/rewards.py#L63
        agent = env.grid.grid_agents[player_id]
        agent_holding_soup = any(  # whether the agent is holding a soup
            [
                isinstance(obj, overcooked_grid_objects.OnionSoup)
                for obj in agent.inventory
            ]
        )
        # check if agent is facing a delivery zone
        forward_pos = agent.front_pos  # [x, y] of the tile in front of the agent
        forward_tile = env.grid.get(*forward_pos)  # get gridObj at fwd_pos

        agent_facing_delivery_zone = isinstance(
            forward_tile, overcooked_grid_objects.DeliveryZone
        )

        if agent_holding_soup and agent_facing_delivery_zone:
            # UhOH we dont know if the agent will drop the soup.... nvm. we can tell by the reward. was a good exercise though
            return np.array([0], dtype=np.float32)
        else:
            return np.array([1], dtype=np.float32)
        

"""
        for grid_obj in env.grid.grid:
            if grid_obj is None:
                continue
            # Check if the grid obj is what we're looking for
            is_focal_obj = isinstance(
                grid_obj, self.focal_object_type
            ) and not np.array_equal(agent.pos, grid_obj.pos)

            obj_is_placed_on = isinstance(
                grid_obj.obj_placed_on, self.focal_object_type
            )

"""

# -------------------------------
def euclidian_distance(pos_1: tuple[int, int], pos_2: tuple[int, int]) -> int:
    """Calculate the euclidian distance between two points.

    :param pos_1: The first point on the grid.
    :type pos_1: tuple[int, int]
    :param pos_2: The second point on the grid.
    :type pos_2: tuple[int, int]
    :return: The euclidian distance between the two points.
    :rtype: int
    """
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

def _calc_binary_pot_features(pot: overcooked_grid_objects.Pot, agent, grid: cogrid_env.grid) -> np.ndarray:
    # Encode if the pot is reachable (size 1)
    pot_reachable = [1]  # TODO(chase): use search to determine

    # Encode if the pot is empty, cooking, or ready (size 4)
    pot_status = np.zeros((4,), dtype=np.int32)  # empty, cooking, ready, ptr
    if pot.dish_ready:
        pot_status[0] = 1
    elif len(pot.objects_in_pot) == 0:
        pot_status[1] = 1
    elif len(pot.objects_in_pot) == pot.capacity:
        pot_status[2] = 1
    else:
        pot_status[3] = 1

    # Encode the number of each legal content in the pot (size legal_contents)
    #pot_contents = np.zeros((len(pot.legal_contents),))
    #item_types_in_pot = [
    #    pot.legal_contents.index(type(pot_content_obj))
    #    for pot_content_obj in pot.objects_in_pot
    #]
    #for obj_index, obj_count in collections.Counter(item_types_in_pot).items():
    #    pot_contents[obj_index] = obj_count

    # Encode cooking time (size 1)
    #pot_cooking_time = np.array(
    #    (pot.cooking_timer if pot.is_cooking else -1,),
    #    dtype=np.int32,
    #)

    # encode the distance from agent to pot (size 2)
    #pot_distance = np.asarray(agent.pos) - np.asarray(pot.pos)

    # encode the pot location (size 2)
    height = grid.height
    width = grid.width
    pot_location = np.asarray(pot.pos)
    # compute binary encoding of pot location
    binary_pot_location = np.zeros((height * width,), dtype=np.int32)
    flat_index = pot_location[0] * width + pot_location[1]
    binary_pot_location[flat_index] = 1

    pot_features = np.hstack(
        [
            pot_reachable,
            pot_status,
            #pot_contents,
            #pot_cooking_time,
            #pot_distance,
            #pot_location,
            binary_pot_location,
        ]
    )

    return pot_features

class BinaryAgentPosition(feature.Feature):
    def __init__(self, grid=None, **kwargs):
        height = grid.height
        width = grid.width
        super().__init__(
            low=0,
            high=1,
            shape=(height * width,),
            name="binary_agent_position",
            **kwargs,
        )
        self.grid = grid

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        agent = env.grid.grid_agents[player_id]
        agent_pos = np.asarray(agent.pos)
        binary_agent_position = np.zeros((self.shape[0],), dtype=np.int32)
        flat_index = agent_pos[0] * self.grid.width + agent_pos[1]
        binary_agent_position[flat_index] = 1
        return binary_agent_position

class NClosestBinaryPotFeatures(feature.Feature):
    def __init__(self, num_pots=2, grid=None, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(num_pots * (11 + grid.height * grid.width),),
            name="n_closest_pot_features",
            **kwargs,
        )
        self.num_pots = num_pots
        self.grid = grid

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        agent = env.grid.grid_agents[player_id]
        pots_and_dists = []
        for grid_obj in env.grid.grid:
            if not isinstance(grid_obj, overcooked_grid_objects.Pot):
                continue

            euc_dist = euclidian_distance(agent.pos, grid_obj.pos)
            pots_and_dists.append((euc_dist, grid_obj))

        # Retrieve the N closest pots
        closest_pots = [
            pot[1]
            for pot in sorted(pots_and_dists, key=lambda x: x[0])[
                : self.num_pots
            ]
        ]

        pot_features = []
        for pot in closest_pots:
            pot_features.append(_calc_binary_pot_features(pot, agent, env.grid))

        encoding = np.hstack(pot_features)

        # If we're in an environment with less than N pots, pad with zeros
        padded_encoding = np.zeros(self.shape, dtype=np.float32)
        padded_encoding[: len(encoding)] = encoding

        return padded_encoding