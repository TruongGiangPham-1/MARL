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

        for feature in self.agent_features:
            print(
                f"Feature: {feature.name}, shape: {feature.shape}"
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
            NextToDeliveryZone(), # Binary
            NextToPlateStack(), # Binary
            HoldingOnionAndFacingPot(),
            HoldingSoupAndFacingDeliveryZone(),
            HoldingPlateAndFacingReadyPot(),
            HoldingPlateAndPotReady(),
            # The (dy, dx) distance to the closest other agent
            #overcooked_features.DistToOtherPlayers(
            #    num_other_players=num_agents - 1
            #),
            # The (row, column) position of the agent
            #BinaryAgentPosition(grid=env.grid),
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

    # One-hot pot status (size 4): [empty, partially_filled, cooking, ready].
    # These are mutually exclusive: the cooking timer only counts down once the
    # pot is full (== capacity), so a full pot is always either `is_cooking`
    # (timer > 0) or `dish_ready` (timer == 0); anything not full is empty or
    # partially filled. The previous if/elif chain set index 0 for `dish_ready`,
    # left empty pots all-zero, and never reached indices 2/3 (dead branches).
    pot_status = np.zeros((4,), dtype=np.int32)
    num_in_pot = len(pot.objects_in_pot)
    if pot.dish_ready:
        pot_status[3] = 1   # cooked, ready to plate
    elif pot.is_cooking:
        pot_status[2] = 1   # full, still cooking
    elif num_in_pot == 0:
        pot_status[0] = 1   # empty
    else:
        pot_status[1] = 1   # partially filled (0 < n < capacity)

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
            #binary_pot_location,
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
            #shape=(num_pots * (11 + grid.height * grid.width),),
            shape=(num_pots * (11),),
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
    
class NextToDeliveryZone(feature.Feature):
    """
    One hot feature indicating if the agent is adjacent to a delivery zone.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(4,),
            name="next_to_delivery_zone",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        agent = env.grid.grid_agents[player_id]
        adjacent_positions = [
            (agent.pos[0] + dx, agent.pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        one_hot = np.array([0, 0, 0, 0], dtype=np.float32)
        for pos in adjacent_positions:
            tile = env.grid.get(*pos)
            if isinstance(tile, overcooked_grid_objects.DeliveryZone):
                if pos == (agent.pos[0] - 1, agent.pos[1]):
                    one_hot[0] = 1  # Up
                elif pos == (agent.pos[0] + 1, agent.pos[1]):
                    one_hot[1] = 1  # Down
                elif pos == (agent.pos[0], agent.pos[1] - 1):
                    one_hot[2] = 1  # Left
                elif pos == (agent.pos[0], agent.pos[1] + 1):
                    one_hot[3] = 1  # Right
        return one_hot 


class NextToPlateStack(feature.Feature):
    """
    Binary feature indicating if the agent is adjacent to a plate stack.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(4,),
            name="next_to_plate_stack",
            **kwargs,
        )
    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        agent = env.grid.grid_agents[player_id]
        adjacent_positions = [
            (agent.pos[0] + dx, agent.pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        one_hot = np.array([0, 0, 0, 0], dtype=np.float32)
        for pos in adjacent_positions:
            tile = env.grid.get(*pos)
            if isinstance(tile, overcooked_grid_objects.PlateStack):
                if pos == (agent.pos[0] - 1, agent.pos[1]):
                    one_hot[0] = 1  # Up
                elif pos == (agent.pos[0] + 1, agent.pos[1]):
                    one_hot[1] = 1  # Down
                elif pos == (agent.pos[0], agent.pos[1] - 1):
                    one_hot[2] = 1  # Left
                elif pos == (agent.pos[0], agent.pos[1] + 1):
                    one_hot[3] = 1  # Right
        return one_hot  

"""
class Directions(IntEnum):
    Right = 0  
    Down = 1
    Left = 2
    Up = 3

    Directions.Right: np.array((0, 1)),  # Increase col away from 0
    Directions.Down: np.array(
        (1, 0)
    ),  # Down increases the row number (0 is top)
    Directions.Left: np.array(
        (0, -1)
    ),  # Left decreases the col towards 0
    Directions.Up: np.array(
        (-1, 0)
    ),  # Up decreases the row to 0 (move towards the top)


"""
class HoldingOnionAndFacingPot(feature.Feature):
    """
    Binary feature indicating if the agent is holding an onion and adjacent to a pot with an empty slot, and facing that pot.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(1,),
            name="holding_onion_and_facing_empty_pot",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        dir_to_vec = {
            0: (0, 1),  # Right
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (-1, 0), # Up
        }
        agent = env.grid.grid_agents[player_id]
        holding_onion = any(
            [
                isinstance(obj, overcooked_grid_objects.Onion)
                for obj in agent.inventory
            ]
        )
        if not holding_onion:
            return np.array([0], dtype=np.float32)
        adjacent_positions = [
            (agent.pos[0] + dx, agent.pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        agent_dir = env.env_agents[player_id].dir
        # the coordinate in front of the agent based on its orientation
        front_pos = (agent.pos[0] + dir_to_vec[agent_dir][0], agent.pos[1] + dir_to_vec[agent_dir][1])
        for pos in adjacent_positions:
            tile = env.grid.get(*pos)
            if isinstance(tile, overcooked_grid_objects.Pot) and len(tile.objects_in_pot) < tile.capacity and pos == front_pos:
                return np.array([1], dtype=np.float32)
        return np.array([0], dtype=np.float32)

class HoldingPlateAndFacingReadyPot(feature.Feature):
    """
    Binary feature indicating if the agent is holding a plate and facing a ready pot.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(1,),
            name="holding_plate_and_facing_ready_pot",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        dir_to_vec = {
            0: (0, 1),  # Right
            1: (1, 0),  # Down 
            2: (0, -1), # Left
            3: (-1, 0), # Up
        }
        agent = env.grid.grid_agents[player_id]
        holding_plate = any(
            [
                isinstance(obj, overcooked_grid_objects.Plate)
                for obj in agent.inventory
            ]
        )
        if not holding_plate:
            return np.array([0], dtype=np.float32)

        adjacent_positions = [
            (agent.pos[0] + dx, agent.pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        agent_dir = env.env_agents[player_id].dir
        # the coordinate in front of the agent based on its orientation
        front_pos = (agent.pos[0] + dir_to_vec[agent_dir][0], agent.pos[1] + dir_to_vec[agent_dir][1])
        for pos in adjacent_positions:
            tile = env.grid.get(*pos)
            if isinstance(tile, overcooked_grid_objects.Pot) and tile.dish_ready and pos == front_pos:
                return np.array([1], dtype=np.float32)
        return np.array([0], dtype=np.float32)
    
class HoldingSoupAndFacingDeliveryZone(feature.Feature):
    """
    Binary feature indicating if the agent is holding a soup and adjacent to a delivery zone.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(1,),
            name="holding_soup_and_next_to_delivery_zone",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        dir_to_vec = {
            0: (0, 1),  # Right
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (-1, 0), # Up
        }
        agent = env.grid.grid_agents[player_id]
        holding_soup = any(
            [
                isinstance(obj, overcooked_grid_objects.OnionSoup)
                for obj in agent.inventory
            ]
        )
        if not holding_soup:
            return np.array([0], dtype=np.float32)

        adjacent_positions = [
            (agent.pos[0] + dx, agent.pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        agent_dir = env.env_agents[player_id].dir
        # the coordinate in front of the agent based on its orientation
        front_pos = (agent.pos[0] + dir_to_vec[agent_dir][0], agent.pos[1] + dir_to_vec[agent_dir][1])
        for pos in adjacent_positions:
            tile = env.grid.get(*pos)
            if isinstance(tile, overcooked_grid_objects.DeliveryZone) and pos == front_pos:
                return np.array([1], dtype=np.float32)
        return np.array([0], dtype=np.float32)
    
class HoldingPlateAndPotReady(feature.Feature):
    """
    Binary feature indicating if the agent is holding a plate and there is a ready pot adjacent to them.
    Different thatn HoldingPlateAndFacingReadyPot in that the pot doesn't have to be in front of them, just ready anywhere in the map
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(1,),
            name="holding_plate_and_pot_ready",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        agent = env.grid.grid_agents[player_id]
        holding_plate = any(
            [
                isinstance(obj, overcooked_grid_objects.Plate)
                for obj in agent.inventory
            ]
        )
        if not holding_plate:
            return np.array([0], dtype=np.float32)
        for tile in env.grid.grid:
            if isinstance(tile, overcooked_grid_objects.Pot) and tile.dish_ready:
                return np.array([1], dtype=np.float32)
        return np.array([0], dtype=np.float32)


# ===========================================================================
# BinaryFeatureV2
# ---------------------------------------------------------------------------
# Same hand-crafted binary "affordance" features as BinaryFeature, plus two
# additions aimed at generalization and richer value estimation under linear FA:
#   1. RelativeDirToClosestObj: egocentric direction + distance to the nearest
#      onion stack / pot / plate stack / delivery zone. Relative geometry => the
#      navigation signal transfers across layouts (unlike absolute-position
#      one-hots) and removes the state aliasing that caused walking-in-place.
#   2. NClosestBinaryPotFeaturesV2: keeps the fixed pot-status one-hot but adds
#      the onion COUNT (1 vs 2 vs 3) and a COOKING-TIMER bucket ("almost done"),
#      magnitudes the binary-only encoding threw away. Emits exact dims (the V1
#      class declared 11/pot but only filled 5, leaving dead zeros).
# ===========================================================================


def _calc_binary_pot_features_v2(pot: overcooked_grid_objects.Pot) -> np.ndarray:
    """Per-pot binary features, exact size (13), no dead padding.

    reachable(1) + status one-hot(4) + onion-count one-hot(4) + timer bucket(4).
    """
    reachable = [1]  # TODO(chase): use search to determine reachability

    # Status one-hot: [empty, partially_filled, cooking, ready] (see _calc_binary_pot_features).
    status = np.zeros(4, dtype=np.int32)
    num_in_pot = len(pot.objects_in_pot)
    if pot.dish_ready:
        status[3] = 1
    elif pot.is_cooking:
        status[2] = 1
    elif num_in_pot == 0:
        status[0] = 1
    else:
        status[1] = 1

    # Onion-count one-hot: [0, 1, 2, 3]. Distinguishes 1 vs 2 onions, which the
    # "partially_filled" status bit alone cannot.
    count = np.zeros(4, dtype=np.int32)
    count[min(num_in_pot, 3)] = 1

    # Cooking-timer bucket: [not_cooking, t in (20,30], (10,20], (0,10]]. The last
    # bucket is the "almost ready -> go grab a plate" signal. Timer only counts
    # down while the pot is full, so it's only meaningful when is_cooking.
    timer = np.zeros(4, dtype=np.int32)
    if not pot.is_cooking:
        timer[0] = 1
    elif pot.cooking_timer > 20:
        timer[1] = 1
    elif pot.cooking_timer > 10:
        timer[2] = 1
    else:
        timer[3] = 1

    return np.hstack([reachable, status, count, timer]).astype(np.float32)


class NClosestBinaryPotFeaturesV2(feature.Feature):
    """Status + onion-count + cooking-timer-bucket for the closest `num_pots` pots."""

    PER_POT = 13

    def __init__(self, num_pots=2, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(num_pots * self.PER_POT,),
            name="n_closest_pot_features_v2",
            **kwargs,
        )
        self.num_pots = num_pots

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        agent = env.grid.grid_agents[player_id]
        pots_and_dists = []
        for grid_obj in env.grid.grid:
            if not isinstance(grid_obj, overcooked_grid_objects.Pot):
                continue
            pots_and_dists.append((euclidian_distance(agent.pos, grid_obj.pos), grid_obj))

        closest_pots = [
            pot for _, pot in sorted(pots_and_dists, key=lambda x: x[0])[: self.num_pots]
        ]

        out = np.zeros(self.shape, dtype=np.float32)
        if closest_pots:
            enc = np.hstack([_calc_binary_pot_features_v2(pot) for pot in closest_pots])
            out[: len(enc)] = enc  # zero-pad if the layout has fewer than num_pots pots
        return out


class RelativeDirToClosestObj(feature.Feature):
    """Egocentric, layout-agnostic encoding of where the nearest object of
    `focal_object_type` is relative to the agent:

        - row sign one-hot:  [above, same_row, below]                     (3)
        - col sign one-hot:  [left, same_col, right]                      (3)
        - Manhattan-distance one-hot: [<=1, 2-3, 4-6, >=7]                (4)

    All zeros if no such object exists. Encoding relative geometry (rather than
    absolute cells) is what lets a linear policy transfer across layouts.
    """

    def __init__(self, focal_object_type, name="relative_dir_to_obj", **kwargs):
        super().__init__(low=0, high=1, shape=(10,), name=name, **kwargs)
        self.focal_object_type = focal_object_type

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        agent = env.grid.grid_agents[player_id]
        out = np.zeros(self.shape, dtype=np.float32)

        best_obj, best_dist = None, None
        for grid_obj in env.grid.grid:
            if not isinstance(grid_obj, self.focal_object_type):
                continue
            if np.array_equal(agent.pos, grid_obj.pos):
                continue
            d = euclidian_distance(agent.pos, grid_obj.pos)
            if best_dist is None or d < best_dist:
                best_dist, best_obj = d, grid_obj

        if best_obj is None:
            return out

        dr = best_obj.pos[0] - agent.pos[0]  # +ve => target is below (row increases downward)
        dc = best_obj.pos[1] - agent.pos[1]  # +ve => target is to the right

        out[0], out[1], out[2] = (dr < 0), (dr == 0), (dr > 0)   # above / same / below
        out[3], out[4], out[5] = (dc < 0), (dc == 0), (dc > 0)   # left / same / right

        manhattan = abs(dr) + abs(dc)
        if manhattan <= 1:
            out[6] = 1.0
        elif manhattan <= 3:
            out[7] = 1.0
        elif manhattan <= 6:
            out[8] = 1.0
        else:
            out[9] = 1.0
        return out


class BinaryFeatureV2(feature.Feature):
    """BinaryFeature + egocentric direction/distance features + richer pot features."""

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):
        self.agent_features = [
            features.AgentDir(),                          # 4
            overcooked_features.OvercookedInventory(),    # 3
            overcooked_features.NextToCounter(),          # 4
            overcooked_features.NextToPot(),              # 16
            NClosestBinaryPotFeaturesV2(num_pots=2),      # 26 (status + count + timer, x2 pots)
            NextToDeliveryZone(),                         # 4
            NextToPlateStack(),                           # 4
            HoldingOnionAndFacingPot(),                   # 1
            HoldingSoupAndFacingDeliveryZone(),           # 1
            HoldingPlateAndFacingReadyPot(),              # 1
            HoldingPlateAndPotReady(),                    # 1
            # NEW: egocentric "which way + how far" to each key target (generalizable).
            RelativeDirToClosestObj(overcooked_grid_objects.OnionStack, name="dir_to_onion_stack"),    # 10
            RelativeDirToClosestObj(overcooked_grid_objects.Pot, name="dir_to_pot"),                    # 10
            RelativeDirToClosestObj(overcooked_grid_objects.PlateStack, name="dir_to_plate_stack"),     # 10
            RelativeDirToClosestObj(overcooked_grid_objects.DeliveryZone, name="dir_to_delivery_zone"), # 10
            features.CanMoveDirection(),                  # 4
        ]

        full_shape = np.sum([feature.shape for feature in self.agent_features])

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs) -> np.ndarray:
        encoding = np.hstack([self.generate_player_encoding(env, player_id)]).astype(np.float32)
        assert np.array_equal(self.shape, encoding.shape)
        return encoding

    def generate_player_encoding(self, env: cogrid_env.CoGridEnv, player_id: str | int) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))
        return np.hstack(encoded_features)