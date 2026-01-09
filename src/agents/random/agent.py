"""Random baseline agent - updated for independent ant control."""

import random
from typing import Dict, List, Optional, Any

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action, Position


class RandomAgent(BaseAgent):
    """
    Random agent that takes random valid actions.

    Each ant acts independently with no coordination.
    Useful as a baseline and for testing.
    """

    def __init__(
        self,
        player_id: int,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(player_id, config)

        # Config options
        self.move_probability = self.config.get('move_probability', 0.9)
        self.prefer_food = self.config.get('prefer_food', False)

    def get_action(
        self,
        observation: AntObservation,
        game_state: GameState
    ) -> int:
        """Get random action for a single ant."""
        obs = observation

        if self.prefer_food and self._has_food_nearby(obs):
            # Try to move toward food
            action = self._move_toward_food(obs, game_state)
        else:
            # Pure random
            if random.random() < self.move_probability:
                # Get valid actions and pick one
                valid_actions = self.get_valid_actions(obs, game_state)
                # Remove STAY from options if we want to move
                move_actions = [a for a in valid_actions if a != Action.STAY]

                # CRITICAL: Filter out moves that would step on own anthill
                safe_actions = []
                for action in move_actions:
                    new_pos = Action.get_new_position(obs.position, action)
                    # Check if new position has our anthill (using absolute positions)
                    is_safe = True
                    for vision_pos, entity_type in obs.vision.items():
                        if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                            if entity_type == f'anthill_{self.player_id}':
                                is_safe = False
                                break
                    if is_safe:
                        safe_actions.append(action)

                if safe_actions:
                    action = random.choice(safe_actions)
                else:
                    action = Action.STAY
            else:
                action = Action.STAY

        # Final safety check: ensure we're not moving to our own anthill
        new_pos = Action.get_new_position(obs.position, action)
        for vision_pos, entity_type in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity_type == f'anthill_{self.player_id}':
                    # Would step on own anthill - stay instead!
                    action = Action.STAY
                    break

        return action

    def reset(self):
        """No state to reset for random agent."""
        pass

    def _has_food_nearby(self, obs: AntObservation) -> bool:
        """Check if food is visible."""
        for entity in obs.vision.values():
            if entity == 'food':
                return True
        return False

    def _move_toward_food(self, obs: AntObservation, game_state: GameState) -> int:
        """Try to move toward nearest visible food."""
        # Find food positions
        food_positions = []
        for pos, entity in obs.vision.items():
            if entity == 'food':
                food_positions.append(pos)

        if not food_positions:
            # No food, move randomly
            valid_actions = self.get_valid_actions(obs, game_state)
            return random.choice(valid_actions)

        # Find closest food (by Manhattan distance)
        closest_food = min(
            food_positions,
            key=lambda p: abs(p.x - obs.position.x) + abs(p.y - obs.position.y)
        )

        # Determine direction to move
        dx = closest_food.x - obs.position.x
        dy = closest_food.y - obs.position.y

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

        # Check if this action would step on own anthill
        new_pos = Action.get_new_position(obs.position, desired_action)
        for vision_pos, entity_type in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity_type == f'anthill_{self.player_id}':
                    # Would step on own anthill - return random safe action
                    safe_actions = []
                    for action_id in Action.DIRECTIONS.keys():
                        if action_id == Action.STAY:
                            continue
                        test_pos = Action.get_new_position(obs.position, action_id)
                        is_safe = True
                        for vp, e in obs.vision.items():
                            if vp.x == test_pos.x and vp.y == test_pos.y:
                                if e == f'anthill_{self.player_id}':
                                    is_safe = False
                                    break
                        if is_safe:
                            safe_actions.append(action_id)

                    return random.choice(safe_actions) if safe_actions else Action.STAY

        return desired_action


class SmartRandomAgent(RandomAgent):
    """
    Slightly smarter random agent with basic strategies.

    Each ant makes independent decisions based on local observations.
    """

    def __init__(
        self,
        player_id: int,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(player_id, config)

        # Additional config
        self.attack_probability = self.config.get('attack_probability', 0.3)
        self.retreat_when_outnumbered = self.config.get('retreat_when_outnumbered', True)

    def get_action(
        self,
        observation: AntObservation,
        game_state: GameState
    ) -> int:
        """Get action for single ant with slightly smarter logic."""
        obs = observation
        action = None

        # Priority 1: Attack enemy anthill if visible
        enemy_anthill_pos = self._find_enemy_anthill(obs)
        if enemy_anthill_pos and random.random() < self.attack_probability:
            action = self._move_toward_position(obs, enemy_anthill_pos)

        # Priority 2: Collect nearby food
        elif self._has_food_nearby(obs):
            action = self._move_toward_food(obs, game_state)

        # Priority 3: Retreat if outnumbered
        elif self.retreat_when_outnumbered:
            enemy_count = self._count_nearby_enemies(obs)
            ally_count = self._count_nearby_allies(obs)
            if enemy_count > ally_count + 1:
                action = self._move_away_from_enemies(obs)

        # Default: Random movement
        if action is None:
            valid_actions = self.get_valid_actions(obs, game_state)
            move_actions = [a for a in valid_actions if a != Action.STAY]
            action = random.choice(move_actions) if move_actions else Action.STAY

        # CRITICAL SAFETY CHECK: Never step on own anthill
        new_pos = Action.get_new_position(obs.position, action)
        for vision_pos, entity_type in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity_type == f'anthill_{self.player_id}':
                    # Would step on own anthill - find alternative
                    safe_actions = []
                    for test_action in Action.DIRECTIONS.keys():
                        if test_action == Action.STAY:
                            safe_actions.append(test_action)
                            continue
                        test_pos = Action.get_new_position(obs.position, test_action)
                        is_safe = True
                        for vp, e in obs.vision.items():
                            if vp.x == test_pos.x and vp.y == test_pos.y:
                                if e == f'anthill_{self.player_id}':
                                    is_safe = False
                                    break
                        if is_safe:
                            safe_actions.append(test_action)

                    action = random.choice(safe_actions) if safe_actions else Action.STAY
                    break

        return action

    def _find_enemy_anthill(self, obs: AntObservation) -> Optional[Position]:
        """Find enemy anthill position if visible."""
        enemy_anthill_type = f'anthill_{1 - self.player_id}'
        for pos, entity in obs.vision.items():
            if entity == enemy_anthill_type:
                return pos
        return None

    def _count_nearby_enemies(self, obs: AntObservation) -> int:
        """Count visible enemy ants."""
        enemy_ant_type = f'ant_{1 - self.player_id}'
        return sum(1 for entity in obs.vision.values() if entity == enemy_ant_type)

    def _count_nearby_allies(self, obs: AntObservation) -> int:
        """Count visible allied ants."""
        ally_ant_type = f'ant_{self.player_id}'
        return sum(1 for entity in obs.vision.values() if entity == ally_ant_type)

    def _move_toward_position(self, obs: AntObservation, target: Position) -> int:
        """Move toward a target position."""
        dx = target.x - obs.position.x
        dy = target.y - obs.position.y

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
                return action_id

        return Action.STAY

    def _move_away_from_enemies(self, obs: AntObservation) -> int:
        """Try to move away from enemy ants."""
        enemy_ant_type = f'ant_{1 - self.player_id}'
        enemy_positions = [
            pos for pos, entity in obs.vision.items()
            if entity == enemy_ant_type
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
                return action_id

        # If can't move away, try perpendicular
        return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
