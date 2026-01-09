"""Greedy heuristic-based agent using intelligent priority system."""

import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action, Position


class GreedyAgent(BaseAgent):
    """
    Greedy agent that uses heuristics and priorities to make intelligent decisions.

    Decision Priority (highest to lowest):
    1. Attack enemy anthill if visible and safe
    2. Collect nearby food
    3. Move toward center of the board (general exploration strategy)
    4. Avoid collisions with ally ants
    5. Retreat when heavily outnumbered

    This agent doesn't require training and serves as a strong baseline.
    """

    def __init__(self, player_id: int, config: Optional[Dict] = None):
        """
        Initialize greedy agent.

        Args:
            player_id: Player ID (0 or 1)
            config: Optional configuration
        """
        super().__init__(player_id, config)

        # Configuration parameters
        self.attack_threshold = self.config.get('attack_threshold', 0.7)  # Probability to attack vs collect
        self.retreat_threshold = self.config.get('retreat_threshold', 2)  # Retreat if outnumbered by this many
        self.exploration_weight = self.config.get('exploration_weight', 0.3)

        # Track ant positions to avoid friendly collisions
        self.planned_moves = {}

    def get_actions(
        self,
        observations: List[AntObservation],
        game_state: GameState
    ) -> Dict[int, int]:
        """
        Get actions for all ants based on greedy heuristics.

        Args:
            observations: List of observations for all living ants
            game_state: Current game state

        Returns:
            Dictionary mapping ant_id -> action
        """
        actions = {}
        self.planned_moves = {}

        # Get board dimensions for exploration
        board_width = game_state.board.width
        board_height = game_state.board.height
        board_center = Position(board_width // 2, board_height // 2)

        # Process each ant
        for obs in observations:
            action = self._decide_action(obs, game_state, board_center)

            # Safety check: never step on own anthill
            action = self._ensure_anthill_safety(obs, action)

            # Track planned move to avoid friendly collisions
            new_pos = Action.get_new_position(obs.position, action)
            self.planned_moves[obs.ant_id] = new_pos

            actions[obs.ant_id] = action

        # Resolve potential friendly collisions
        actions = self._resolve_friendly_collisions(observations, actions)

        return actions

    def _decide_action(
        self,
        obs: AntObservation,
        game_state: GameState,
        board_center: Position
    ) -> int:
        """
        Decide action for a single ant based on priorities.

        Args:
            obs: Ant's observation
            game_state: Current game state
            board_center: Center position of the board

        Returns:
            Action ID
        """
        # Priority 1: Attack enemy anthill (high value target)
        enemy_anthill = self._find_enemy_anthill(obs)
        if enemy_anthill:
            # Check if it's relatively safe to attack
            nearby_enemies = self._count_nearby_enemies(obs)
            nearby_allies = self._count_nearby_allies(obs)

            if nearby_allies >= nearby_enemies or random.random() < self.attack_threshold:
                return self._move_toward(obs, enemy_anthill, game_state)

        # Priority 2: Collect food (growth is important)
        food_pos = self._find_nearest_food(obs)
        if food_pos:
            # Only go for food if not heavily outnumbered
            nearby_enemies = self._count_nearby_enemies(obs)
            nearby_allies = self._count_nearby_allies(obs)

            if nearby_enemies <= nearby_allies + self.retreat_threshold:
                return self._move_toward(obs, food_pos, game_state)

        # Priority 3: Retreat if heavily outnumbered
        nearby_enemies = self._count_nearby_enemies(obs)
        nearby_allies = self._count_nearby_allies(obs)

        if nearby_enemies > nearby_allies + self.retreat_threshold:
            return self._retreat_from_enemies(obs, game_state)

        # Priority 4: Explore toward center (common strategy)
        # This helps discover food and potentially locate enemy
        if random.random() < self.exploration_weight:
            return self._move_toward(obs, board_center, game_state)

        # Priority 5: Random movement (exploration)
        valid_actions = self.get_valid_actions(obs, game_state)
        move_actions = [a for a in valid_actions if a != Action.STAY]

        if move_actions:
            return random.choice(move_actions)

        return Action.STAY

    def _find_enemy_anthill(self, obs: AntObservation) -> Optional[Position]:
        """Find enemy anthill position if visible."""
        enemy_id = 1 - self.player_id
        enemy_anthill_str = f'anthill_{enemy_id}'

        for pos, entity in obs.vision.items():
            if entity == enemy_anthill_str:
                return pos

        return None

    def _find_nearest_food(self, obs: AntObservation) -> Optional[Position]:
        """Find nearest food position."""
        food_positions = [
            pos for pos, entity in obs.vision.items()
            if entity == 'food'
        ]

        if not food_positions:
            return None

        # Return closest food by Manhattan distance
        return min(
            food_positions,
            key=lambda p: abs(p.x - obs.position.x) + abs(p.y - obs.position.y)
        )

    def _count_nearby_enemies(self, obs: AntObservation) -> int:
        """Count visible enemy ants."""
        enemy_id = 1 - self.player_id
        enemy_ant_str = f'ant_{enemy_id}'

        return sum(1 for entity in obs.vision.values() if entity == enemy_ant_str)

    def _count_nearby_allies(self, obs: AntObservation) -> int:
        """Count visible allied ants (excluding self)."""
        ally_ant_str = f'ant_{self.player_id}'

        return sum(1 for entity in obs.vision.values() if entity == ally_ant_str) - 1

    def _move_toward(
        self,
        obs: AntObservation,
        target: Position,
        game_state: GameState
    ) -> int:
        """
        Move toward a target position.

        Args:
            obs: Ant's observation
            target: Target position to move toward
            game_state: Current game state

        Returns:
            Action ID
        """
        dx = target.x - obs.position.x
        dy = target.y - obs.position.y

        # Normalize to unit direction
        if dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1

        if dy > 0:
            dy = 1
        elif dy < 0:
            dy = -1

        # Find matching action
        desired_action = Action.STAY
        for action_id, (adx, ady) in Action.DIRECTIONS.items():
            if adx == dx and ady == dy:
                desired_action = action_id
                break

        # Check if action is valid
        valid_actions = self.get_valid_actions(obs, game_state)
        if desired_action in valid_actions:
            return desired_action

        # If desired action not valid, try alternates
        # Try moving in just x direction
        if dx != 0:
            for action_id, (adx, ady) in Action.DIRECTIONS.items():
                if adx == dx and ady == 0 and action_id in valid_actions:
                    return action_id

        # Try moving in just y direction
        if dy != 0:
            for action_id, (adx, ady) in Action.DIRECTIONS.items():
                if adx == 0 and ady == dy and action_id in valid_actions:
                    return action_id

        # Fall back to any valid move
        move_actions = [a for a in valid_actions if a != Action.STAY]
        if move_actions:
            return random.choice(move_actions)

        return Action.STAY

    def _retreat_from_enemies(self, obs: AntObservation, game_state: GameState) -> int:
        """
        Move away from enemy ants.

        Args:
            obs: Ant's observation
            game_state: Current game state

        Returns:
            Action ID
        """
        enemy_id = 1 - self.player_id
        enemy_ant_str = f'ant_{enemy_id}'

        # Find all enemy positions
        enemy_positions = [
            pos for pos, entity in obs.vision.items()
            if entity == enemy_ant_str
        ]

        if not enemy_positions:
            return Action.STAY

        # Calculate average enemy position
        avg_x = sum(p.x for p in enemy_positions) / len(enemy_positions)
        avg_y = sum(p.y for p in enemy_positions) / len(enemy_positions)

        # Move in opposite direction
        dx = obs.position.x - avg_x
        dy = obs.position.y - avg_y

        # Normalize
        if dx > 0:
            dx = 1
        elif dx < 0:
            dx = -1

        if dy > 0:
            dy = 1
        elif dy < 0:
            dy = -1

        # Find matching action
        for action_id, (adx, ady) in Action.DIRECTIONS.items():
            if adx == dx and ady == dy:
                valid_actions = self.get_valid_actions(obs, game_state)
                if action_id in valid_actions:
                    return action_id

        # If can't retreat in ideal direction, try perpendicular
        valid_actions = self.get_valid_actions(obs, game_state)
        move_actions = [a for a in valid_actions if a != Action.STAY]

        if move_actions:
            return random.choice(move_actions)

        return Action.STAY

    def _ensure_anthill_safety(self, obs: AntObservation, action: int) -> int:
        """
        Ensure we never step on our own anthill.

        Args:
            obs: Ant's observation
            action: Proposed action

        Returns:
            Safe action (original or STAY if unsafe)
        """
        own_anthill_str = f'anthill_{self.player_id}'
        new_pos = Action.get_new_position(obs.position, action)

        # Check if new position has our anthill
        for vision_pos, entity in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity == own_anthill_str:
                    # Would step on own anthill - return STAY instead
                    return Action.STAY

        return action

    def _resolve_friendly_collisions(
        self,
        observations: List[AntObservation],
        actions: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Resolve potential collisions between friendly ants.

        If multiple ants are moving to the same position, make all but one stay.

        Args:
            observations: List of all ant observations
            actions: Current action dictionary

        Returns:
            Updated action dictionary with collisions resolved
        """
        # Count how many ants are moving to each position
        position_counts = defaultdict(list)

        for obs in observations:
            action = actions[obs.ant_id]
            new_pos = Action.get_new_position(obs.position, action)
            position_counts[new_pos].append(obs.ant_id)

        # For positions with multiple ants, make all but one stay
        for pos, ant_ids in position_counts.items():
            if len(ant_ids) > 1:
                # Keep first ant's move, others stay
                for ant_id in ant_ids[1:]:
                    actions[ant_id] = Action.STAY

        return actions

    def reset(self):
        """Reset agent state for new episode."""
        self.planned_moves = {}


class AggressiveGreedyAgent(GreedyAgent):
    """
    More aggressive variant of GreedyAgent that prioritizes attacking.
    """

    def __init__(self, player_id: int, config: Optional[Dict] = None):
        """Initialize aggressive greedy agent."""
        default_config = {
            'attack_threshold': 0.9,  # Almost always attack
            'retreat_threshold': 3,    # Only retreat if very outnumbered
            'exploration_weight': 0.5  # More exploration
        }

        if config:
            default_config.update(config)

        super().__init__(player_id, default_config)


class DefensiveGreedyAgent(GreedyAgent):
    """
    More defensive variant of GreedyAgent that prioritizes survival and food collection.
    """

    def __init__(self, player_id: int, config: Optional[Dict] = None):
        """Initialize defensive greedy agent."""
        default_config = {
            'attack_threshold': 0.3,   # Rarely attack
            'retreat_threshold': 1,    # Retreat if even slightly outnumbered
            'exploration_weight': 0.2  # Less exploration, more cautious
        }

        if config:
            default_config.update(config)

        super().__init__(player_id, default_config)
