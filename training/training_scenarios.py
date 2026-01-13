"""
Training scenarios for isolated skill development.

These scenarios create simplified training environments that focus on specific
skills without the complexity of full opponent agents. This allows agents to
master fundamental behaviors before facing complete game situations.

Key scenarios:
- Food collection (single ant, food only)
- Anthill destruction (single ant, enemy anthill only)
- Combat training (ant vs enemy ants, no anthills)
- Survival training (outnumbered scenarios)
- Efficient collection (multiple ants, abundant food, no enemies)
- Defense training (protect anthill from attackers)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from src.utils.game_config import GameConfig
from src.core.game import Game, Position
import random


@dataclass
class TrainingScenario:
    """Configuration for a specific training scenario."""

    name: str
    """Scenario name"""

    description: str
    """What this scenario teaches"""

    board_size: Tuple[int, int] = (20, 20)
    """Board dimensions"""

    max_turns: int = 100
    """Maximum turns per episode"""

    # Player 0 (training agent) setup
    player_ants: int = 1
    """Number of ants for training agent"""

    player_anthill_pos: Optional[Tuple[int, int]] = None
    """Training agent anthill position (None = random)"""

    # Opponent (Player 1) setup
    opponent_ants: int = 0
    """Number of opponent ants"""

    opponent_anthill_pos: Optional[Tuple[int, int]] = None
    """Opponent anthill position (None = random)"""

    # Environment setup
    food_density: float = 0.0
    """Food spawn density"""

    food_positions: Optional[List[Tuple[int, int]]] = None
    """Fixed food positions (overrides density)"""

    rock_density: float = 0.0
    """Rock density"""

    # Reward shaping
    reward_food_collected: float = 10.0
    """Reward for collecting food"""

    reward_food_deposited: float = 50.0
    """Reward for depositing food at anthill"""

    reward_enemy_killed: float = 20.0
    """Reward for killing enemy ant"""

    reward_anthill_damaged: float = 100.0
    """Reward for damaging enemy anthill"""

    reward_survival: float = 0.1
    """Small reward per turn survived"""

    reward_distance_to_food: float = -0.01
    """Penalty for distance to nearest food (encourages approach)"""

    reward_distance_to_anthill: float = -0.01
    """Penalty for distance to enemy anthill (in attack scenarios)"""

    penalty_death: float = -50.0
    """Penalty for ant death"""

    custom_setup: Optional[Callable] = None
    """Custom setup function: (game: Game) -> None"""


# ============================================================================
# PREDEFINED SCENARIOS
# ============================================================================

def get_food_collection_scenario(difficulty: str = "easy") -> TrainingScenario:
    """
    Pure food collection training - single ant, food only, no enemies.

    Teaches: Basic movement, pathfinding to food, returning to anthill

    Args:
        difficulty: 'easy' (lots of food), 'medium' (moderate), 'hard' (sparse)
    """
    food_densities = {
        'easy': 0.20,
        'medium': 0.10,
        'hard': 0.05
    }

    return TrainingScenario(
        name=f"food_collection_{difficulty}",
        description=f"Learn food collection basics ({difficulty} difficulty)",
        player_ants=1,
        opponent_ants=0,
        food_density=food_densities.get(difficulty, 0.10),
        rock_density=0.02,
        max_turns=100,
        reward_food_collected=10.0,
        reward_food_deposited=50.0,
        reward_survival=0.1,
        reward_distance_to_food=-0.01
    )


def get_anthill_attack_scenario(distance: str = "medium") -> TrainingScenario:
    """
    Anthill destruction training - single ant, enemy anthill only.

    Teaches: Pathfinding to enemy anthill, persistent attacking

    Args:
        distance: 'close' (nearby), 'medium', 'far' (opposite corners)
    """
    def setup_anthills(distance_type: str):
        def setup(game: Game):
            board_w, board_h = game.config.board_width, game.config.board_height

            # Place player anthill
            if distance_type == 'close':
                game.board.anthills[0].position = Position(board_w // 2 - 3, board_h // 2)
                game.board.anthills[1].position = Position(board_w // 2 + 3, board_h // 2)
            elif distance_type == 'medium':
                game.board.anthills[0].position = Position(board_w // 4, board_h // 2)
                game.board.anthills[1].position = Position(3 * board_w // 4, board_h // 2)
            else:  # far
                game.board.anthills[0].position = Position(2, 2)
                game.board.anthills[1].position = Position(board_w - 3, board_h - 3)

        return setup

    return TrainingScenario(
        name=f"anthill_attack_{distance}",
        description=f"Learn to find and attack enemy anthill ({distance} distance)",
        player_ants=1,
        opponent_ants=0,
        food_density=0.0,
        rock_density=0.03,
        max_turns=150,
        reward_anthill_damaged=100.0,
        reward_survival=0.1,
        reward_distance_to_anthill=-0.02,
        custom_setup=setup_anthills(distance)
    )


def get_combat_scenario(enemy_count: int = 1) -> TrainingScenario:
    """
    Combat training - single ant vs enemy ants, no anthills to attack.

    Teaches: Combat tactics, when to fight vs flee, survival

    Args:
        enemy_count: Number of enemy ants (1-5)
    """
    return TrainingScenario(
        name=f"combat_1v{enemy_count}",
        description=f"Learn combat against {enemy_count} enemy ant(s)",
        player_ants=1,
        opponent_ants=enemy_count,
        food_density=0.0,
        rock_density=0.05,
        max_turns=100,
        reward_enemy_killed=50.0,
        reward_survival=1.0,  # Higher survival reward in combat
        penalty_death=-100.0,
        # Place anthills far apart so combat happens in middle
        player_anthill_pos=(2, 2),
        opponent_anthill_pos=(18, 18)
    )


def get_survival_scenario(enemy_count: int = 3) -> TrainingScenario:
    """
    Survival training - outnumbered, must survive as long as possible.

    Teaches: Evasion, tactical retreat, staying alive when outnumbered

    Args:
        enemy_count: Number of enemies (3-10)
    """
    return TrainingScenario(
        name=f"survival_1v{enemy_count}",
        description=f"Survive against {enemy_count} enemies",
        player_ants=1,
        opponent_ants=enemy_count,
        food_density=0.0,
        rock_density=0.08,  # More rocks for cover
        max_turns=150,
        reward_survival=2.0,  # High survival reward
        reward_enemy_killed=100.0,  # Big reward if you somehow win
        penalty_death=-200.0,  # Big penalty for dying
        player_anthill_pos=(10, 10),  # Center
        opponent_anthill_pos=(2, 2)
    )


def get_efficient_collection_scenario(ant_count: int = 3) -> TrainingScenario:
    """
    Efficient collection - multiple ants, lots of food, no enemies.

    Teaches: Independent efficient food collection without coordination
    (remember: ants can't coordinate, so they must learn to work in parallel)

    Args:
        ant_count: Number of player ants (2-5)
    """
    return TrainingScenario(
        name=f"efficient_collection_{ant_count}ants",
        description=f"Learn efficient parallel food collection with {ant_count} ants",
        player_ants=ant_count,
        opponent_ants=0,
        food_density=0.15,  # Abundant food
        rock_density=0.02,
        max_turns=150,
        reward_food_collected=10.0,
        reward_food_deposited=50.0,
        reward_survival=0.1
    )


def get_contested_collection_scenario(ant_count: int = 2) -> TrainingScenario:
    """
    Contested collection - multiple ants per side, food competition, no anthill attacks.

    Teaches: Competing for resources, balancing collection vs defense

    Args:
        ant_count: Number of ants per side (2-4)
    """
    return TrainingScenario(
        name=f"contested_collection_{ant_count}v{ant_count}",
        description=f"Compete for food with {ant_count} ants vs {ant_count} enemies",
        player_ants=ant_count,
        opponent_ants=ant_count,
        food_density=0.12,
        rock_density=0.03,
        max_turns=150,
        reward_food_collected=10.0,
        reward_food_deposited=50.0,
        reward_enemy_killed=30.0,
        reward_survival=0.5,
        penalty_death=-30.0,
        player_anthill_pos=(5, 10),
        opponent_anthill_pos=(15, 10)
    )


def get_defense_scenario(attacker_count: int = 2) -> TrainingScenario:
    """
    Defense training - protect your anthill from attackers.

    Teaches: Defensive positioning, anthill protection

    Args:
        attacker_count: Number of attacking enemy ants (2-5)
    """
    def setup_defense(game: Game):
        # Place player anthill in center
        board_w, board_h = game.config.board_width, game.config.board_height
        game.board.anthills[0].position = Position(board_w // 2, board_h // 2)

        # Opponent anthill far away (not the objective)
        game.board.anthills[1].position = Position(0, 0)

        # Place enemy ants in a circle around player anthill
        center_x, center_y = board_w // 2, board_h // 2
        radius = 5
        enemy_ants = [ant for ant in game.board.ants.values() if ant.player_id == 1]

        for i, ant in enumerate(enemy_ants):
            angle = (i / len(enemy_ants)) * 2 * 3.14159
            x = int(center_x + radius * (1 if i % 2 == 0 else -1) * abs(i - len(enemy_ants)/2) / len(enemy_ants) * 2)
            y = int(center_y + radius * (1 if i < len(enemy_ants)/2 else -1))
            x = max(0, min(board_w - 1, x))
            y = max(0, min(board_h - 1, y))
            ant.position = Position(x, y)

    return TrainingScenario(
        name=f"defense_{attacker_count}attackers",
        description=f"Defend anthill from {attacker_count} attackers",
        player_ants=2,  # 2 defenders
        opponent_ants=attacker_count,
        food_density=0.0,
        rock_density=0.03,
        max_turns=100,
        reward_enemy_killed=50.0,
        reward_survival=1.0,
        reward_anthill_damaged=-200.0,  # Big penalty if anthill takes damage
        penalty_death=-50.0,
        custom_setup=setup_defense
    )


def get_maze_scenario() -> TrainingScenario:
    """
    Maze navigation - single ant, food at the end, many rocks creating a maze.

    Teaches: Complex pathfinding, obstacle avoidance
    """
    def setup_maze(game: Game):
        board_w, board_h = game.config.board_width, game.config.board_height

        # Place anthill at start
        game.board.anthills[0].position = Position(1, 1)
        game.board.anthills[1].position = Position(board_w - 2, board_h - 2)

        # Create maze walls with rocks
        # Vertical walls
        for y in range(3, board_h - 3):
            if y != board_h // 2:  # Leave gaps
                game.board.add_rock(Position(board_w // 3, y))
                game.board.add_rock(Position(2 * board_w // 3, y))

        # Horizontal walls
        for x in range(3, board_w - 3):
            if x != board_w // 2:
                game.board.add_rock(Position(x, board_h // 3))
                game.board.add_rock(Position(x, 2 * board_h // 3))

        # Place food at the end
        game.board.add_food(Position(board_w - 2, board_h - 2))
        game.board.add_food(Position(board_w - 3, board_h - 2))
        game.board.add_food(Position(board_w - 2, board_h - 3))

    return TrainingScenario(
        name="maze_navigation",
        description="Navigate through maze to collect food",
        player_ants=1,
        opponent_ants=0,
        food_density=0.0,  # Custom food placement
        rock_density=0.0,  # Custom rock placement
        max_turns=200,
        reward_food_collected=10.0,
        reward_food_deposited=100.0,
        reward_survival=0.1,
        custom_setup=setup_maze
    )


def get_food_race_scenario() -> TrainingScenario:
    """
    Food race - single ant per side, one food item, first to collect wins.

    Teaches: Speed, efficiency, beating opponent to resources
    """
    def setup_race(game: Game):
        board_w, board_h = game.config.board_width, game.config.board_height

        # Place anthills at opposite sides
        game.board.anthills[0].position = Position(2, board_h // 2)
        game.board.anthills[1].position = Position(board_w - 3, board_h // 2)

        # Place single food in center
        game.board.add_food(Position(board_w // 2, board_h // 2))

    return TrainingScenario(
        name="food_race",
        description="Race opponent to collect single food item",
        player_ants=1,
        opponent_ants=1,
        food_density=0.0,
        rock_density=0.05,
        max_turns=100,
        reward_food_collected=50.0,
        reward_food_deposited=200.0,
        reward_survival=0.5,
        custom_setup=setup_race
    )


# ============================================================================
# SCENARIO REGISTRY
# ============================================================================

SCENARIO_REGISTRY = {
    # Food collection (no enemies)
    'food_easy': lambda: get_food_collection_scenario('easy'),
    'food_medium': lambda: get_food_collection_scenario('medium'),
    'food_hard': lambda: get_food_collection_scenario('hard'),

    # Anthill attacks (no enemies)
    'anthill_close': lambda: get_anthill_attack_scenario('close'),
    'anthill_medium': lambda: get_anthill_attack_scenario('medium'),
    'anthill_far': lambda: get_anthill_attack_scenario('far'),

    # Combat training
    'combat_1v1': lambda: get_combat_scenario(1),
    'combat_1v2': lambda: get_combat_scenario(2),
    'combat_1v3': lambda: get_combat_scenario(3),

    # Survival training (outnumbered)
    'survival_1v3': lambda: get_survival_scenario(3),
    'survival_1v5': lambda: get_survival_scenario(5),
    'survival_1v8': lambda: get_survival_scenario(8),

    # Efficient collection (multiple ants, no enemies)
    'efficient_2ants': lambda: get_efficient_collection_scenario(2),
    'efficient_3ants': lambda: get_efficient_collection_scenario(3),
    'efficient_5ants': lambda: get_efficient_collection_scenario(5),

    # Contested collection (competition)
    'contested_2v2': lambda: get_contested_collection_scenario(2),
    'contested_3v3': lambda: get_contested_collection_scenario(3),

    # Defense scenarios
    'defense_2attackers': lambda: get_defense_scenario(2),
    'defense_3attackers': lambda: get_defense_scenario(3),
    'defense_5attackers': lambda: get_defense_scenario(5),

    # Special scenarios
    'maze': get_maze_scenario,
    'food_race': get_food_race_scenario,
}


def get_scenario(name: str) -> TrainingScenario:
    """
    Get a training scenario by name.

    Args:
        name: Scenario name from SCENARIO_REGISTRY

    Returns:
        TrainingScenario instance

    Raises:
        ValueError: If scenario name is unknown
    """
    if name not in SCENARIO_REGISTRY:
        raise ValueError(
            f"Unknown scenario: {name}\n"
            f"Available scenarios: {', '.join(sorted(SCENARIO_REGISTRY.keys()))}"
        )

    return SCENARIO_REGISTRY[name]()


def list_scenarios() -> List[str]:
    """Get list of all available scenario names."""
    return sorted(SCENARIO_REGISTRY.keys())


def get_scenarios_by_category() -> dict:
    """Get scenarios organized by category."""
    return {
        'Food Collection (Solo)': [
            'food_easy', 'food_medium', 'food_hard'
        ],
        'Anthill Attack (Solo)': [
            'anthill_close', 'anthill_medium', 'anthill_far'
        ],
        'Combat Training': [
            'combat_1v1', 'combat_1v2', 'combat_1v3'
        ],
        'Survival (Outnumbered)': [
            'survival_1v3', 'survival_1v5', 'survival_1v8'
        ],
        'Efficient Collection (Multi-ant)': [
            'efficient_2ants', 'efficient_3ants', 'efficient_5ants'
        ],
        'Contested Collection': [
            'contested_2v2', 'contested_3v3'
        ],
        'Defense': [
            'defense_2attackers', 'defense_3attackers', 'defense_5attackers'
        ],
        'Special Challenges': [
            'maze', 'food_race'
        ]
    }
