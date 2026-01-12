"""Tactical heuristic agent with advanced decision making - independent ant control."""

import random
from typing import Dict, List, Optional, Tuple

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action, Position


class TacticalAgent(BaseAgent):
    """
    Advanced tactical agent using sophisticated heuristics.

    Each ant makes independent tactical decisions based on:
    - Multi-factor food scoring (distance, risk, competition)
    - Intelligent attack timing (force balance consideration)
    - Dynamic defensive positioning
    - Risk-aware decision making
    - Game phase adaptation

    Designed to outperform SmartRandomAgent through superior tactical decision-making
    without requiring centralized coordination.
    """

    def __init__(self, player_id: int, config: Optional[Dict] = None):
        """Initialize tactical agent."""
        super().__init__(player_id, config)

        # Tactical parameters
        self.food_priority = self.config.get('food_priority', 1.5)
        self.attack_threshold = self.config.get('attack_threshold', 0.6)
        self.defense_priority = self.config.get('defense_priority', 1.2)
        self.risk_aversion = self.config.get('risk_aversion', 0.8)

    def get_action(
        self,
        observation: AntObservation,
        game_state: GameState
    ) -> int:
        """
        Get tactical action for a single ant.

        Args:
            observation: Observation for ONE ant
            game_state: Current game state

        Returns:
            Action ID
        """
        obs = observation

        # Gather tactical information
        enemy_anthill = self._find_enemy_anthill(obs)
        own_anthill = self._find_own_anthill(obs)
        nearby_enemies = self._count_nearby_enemies(obs)
        nearby_allies = self._count_nearby_allies(obs)
        local_force_balance = nearby_allies - nearby_enemies

        # Decision Priority System with Multi-Factor Scoring

        # Priority 1: Critical defense - enemy very close to our anthill
        if own_anthill and self._distance_to(obs.position, own_anthill) <= 3:
            if nearby_enemies > 0:
                # Intercept enemies near our base
                closest_enemy = self._find_closest_enemy(obs)
                if closest_enemy:
                    action = self._move_toward(obs, closest_enemy, game_state)
                    if self._is_action_safe(obs, action):
                        return action

        # Priority 2: Opportunistic attack on enemy anthill
        if enemy_anthill:
            # Calculate attack viability
            distance_to_anthill = self._distance_to(obs.position, enemy_anthill)

            # Attack if:
            # - Very close (adjacent)
            # - OR have local force superiority and random chance
            # - OR no enemies around and random chance
            if distance_to_anthill == 1:
                # Adjacent - go for it!
                action = self._move_toward(obs, enemy_anthill, game_state)
                if self._is_action_safe(obs, action):
                    return action
            elif (local_force_balance >= 0 or nearby_enemies == 0) and random.random() < self.attack_threshold:
                action = self._move_toward(obs, enemy_anthill, game_state)
                if self._is_action_safe(obs, action):
                    return action

        # Priority 3: Intelligent food collection with risk assessment
        food_positions = self._get_all_food_positions(obs)
        if food_positions:
            best_food = self._score_and_select_food(
                obs, food_positions, nearby_enemies, nearby_allies
            )

            if best_food:
                action = self._move_toward(obs, best_food, game_state)
                if self._is_action_safe(obs, action):
                    return action

        # Priority 4: Retreat if heavily outnumbered
        if local_force_balance < -1:  # Outnumbered by 2 or more
            action = self._retreat_from_enemies(obs, game_state)
            if self._is_action_safe(obs, action):
                return action

        # Priority 5: Defensive positioning near own anthill
        if own_anthill and self._distance_to(obs.position, own_anthill) > 5:
            # Too far from base - move closer
            action = self._move_toward(obs, own_anthill, game_state)
            if self._is_action_safe(obs, action):
                return action

        # Priority 6: Exploration (move toward board center)
        board_center = Position(
            game_state.board.width // 2,
            game_state.board.height // 2
        )

        if random.random() < 0.4:  # 40% exploration
            action = self._move_toward(obs, board_center, game_state)
            if self._is_action_safe(obs, action):
                return action

        # Default: Random safe movement
        valid_actions = self.get_valid_actions(obs, game_state)
        move_actions = [a for a in valid_actions if a != Action.STAY]
        safe_moves = [a for a in move_actions if self._is_action_safe(obs, a)]

        if safe_moves:
            return random.choice(safe_moves)

        return Action.STAY

    def _score_and_select_food(
        self,
        obs: AntObservation,
        food_positions: List[Position],
        nearby_enemies: int,
        nearby_allies: int
    ) -> Optional[Position]:
        """
        Score food positions using multiple factors and select best.

        Factors:
        - Distance (closer is better)
        - Risk (fewer enemies nearby is better)
        - Competition (check if enemies are closer)

        Args:
            obs: Ant observation
            food_positions: List of visible food positions
            nearby_enemies: Count of enemies in vision
            nearby_allies: Count of allies in vision

        Returns:
            Best food position or None
        """
        if not food_positions:
            return None

        best_food = None
        best_score = -999999

        for food_pos in food_positions:
            distance = self._distance_to(obs.position, food_pos)

            # Distance score (closer is better)
            distance_score = -distance * 2

            # Risk score (less risky if we have allies or no enemies)
            risk_score = 0
            if nearby_enemies > 0:
                if nearby_allies > nearby_enemies:
                    risk_score = 5  # Safe with ally support
                elif nearby_allies == nearby_enemies:
                    risk_score = 0  # Neutral risk
                else:
                    risk_score = -10 * (nearby_enemies - nearby_allies)  # High risk

            # Competition score (are enemies closer to this food?)
            enemy_distance = self._nearest_enemy_distance_to(obs, food_pos)
            if enemy_distance is not None:
                if enemy_distance > distance:
                    # We're closer - good!
                    competition_score = 5
                elif enemy_distance == distance:
                    # Tied - neutral
                    competition_score = 0
                else:
                    # Enemy closer - bad
                    competition_score = -3
            else:
                # No enemies visible - safe
                competition_score = 3

            total_score = distance_score + risk_score * self.risk_aversion + competition_score

            if total_score > best_score:
                best_score = total_score
                best_food = food_pos

        return best_food

    def _find_own_anthill(self, obs: AntObservation) -> Optional[Position]:
        """Find own anthill if visible."""
        own_anthill_str = f'anthill_{self.player_id}'
        for pos, entity in obs.vision.items():
            if entity == own_anthill_str:
                return pos
        return None

    def _find_enemy_anthill(self, obs: AntObservation) -> Optional[Position]:
        """Find enemy anthill if visible."""
        enemy_anthill_str = f'anthill_{1 - self.player_id}'
        for pos, entity in obs.vision.items():
            if entity == enemy_anthill_str:
                return pos
        return None

    def _find_closest_enemy(self, obs: AntObservation) -> Optional[Position]:
        """Find closest enemy ant."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
        enemy_positions = [
            pos for pos, entity in obs.vision.items()
            if entity == enemy_ant_str
        ]

        if not enemy_positions:
            return None

        return min(
            enemy_positions,
            key=lambda p: self._distance_to(obs.position, p)
        )

    def _get_all_food_positions(self, obs: AntObservation) -> List[Position]:
        """Get all visible food positions."""
        return [
            pos for pos, entity in obs.vision.items()
            if entity == 'food'
        ]

    def _count_nearby_enemies(self, obs: AntObservation) -> int:
        """Count visible enemy ants."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
        return sum(1 for entity in obs.vision.values() if entity == enemy_ant_str)

    def _count_nearby_allies(self, obs: AntObservation) -> int:
        """Count visible allied ants (excluding self)."""
        ally_ant_str = f'ant_{self.player_id}'
        # Subtract 1 because vision includes self
        return sum(1 for entity in obs.vision.values() if entity == ally_ant_str) - 1

    def _nearest_enemy_distance_to(self, obs: AntObservation, target: Position) -> Optional[int]:
        """Find nearest enemy's distance to a target position."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
        enemy_positions = [
            pos for pos, entity in obs.vision.items()
            if entity == enemy_ant_str
        ]

        if not enemy_positions:
            return None

        return min(
            self._distance_to(enemy_pos, target)
            for enemy_pos in enemy_positions
        )

    def _distance_to(self, pos1: Position, pos2: Position) -> int:
        """Calculate Chebyshev distance (king move distance)."""
        return max(abs(pos1.x - pos2.x), abs(pos1.y - pos2.y))

    def _move_toward(
        self,
        obs: AntObservation,
        target: Position,
        game_state: GameState
    ) -> int:
        """Move toward target position."""
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
        """Move away from enemy ants."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
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

    def _is_action_safe(self, obs: AntObservation, action: int) -> bool:
        """Check if action doesn't step on own anthill."""
        own_anthill_str = f'anthill_{self.player_id}'
        new_pos = Action.get_new_position(obs.position, action)

        for vision_pos, entity in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity == own_anthill_str:
                    return False

        return True

    def reset(self):
        """Reset agent state for new episode."""
        pass
