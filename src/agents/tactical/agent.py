"""Elite tactical heuristic agent designed to dominate SmartRandomAgent."""

import random
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action, Position


class TacticalAgent(BaseAgent):
    """
    Advanced tactical agent using sophisticated heuristics and game theory.

    Strategy:
    - Multi-stage gameplay (early/mid/late game tactics)
    - Coordinated swarm attacks when advantageous
    - Intelligent food prioritization with denial tactics
    - Active anthill defense with perimeter control
    - Force projection and territory dominance
    - Risk-aware decision making

    Designed to significantly outperform SmartRandomAgent.
    """

    def __init__(self, player_id: int, config: Optional[Dict] = None):
        """Initialize tactical agent."""
        super().__init__(player_id, config)

        # Tactical parameters
        self.aggression_factor = self.config.get('aggression_factor', 1.2)
        self.defense_radius = self.config.get('defense_radius', 3)
        self.swarm_threshold = self.config.get('swarm_threshold', 2)

        # Internal state
        self.planned_positions = {}  # ant_id -> Position
        self.food_claims = {}  # food_pos -> ant_id
        self.turn_count = 0
        self.game_phase = "early"  # early, mid, late

    def get_actions(
        self,
        observations: List[AntObservation],
        game_state: GameState
    ) -> Dict[int, int]:
        """Get coordinated tactical actions for all ants."""
        self.turn_count += 1
        self._update_game_phase(game_state)

        actions = {}
        self.planned_positions.clear()
        self.food_claims.clear()

        # Get key tactical information
        own_anthill = self._find_own_anthill(observations)
        enemy_anthill = self._find_enemy_anthill(observations)

        # Calculate force distribution
        force_map = self._calculate_force_map(observations)

        # Prioritize ants by role
        defenders, attackers, collectors = self._assign_roles(
            observations, own_anthill, enemy_anthill, game_state
        )

        # Execute roles in priority order
        # 1. Defenders first (critical for survival)
        for obs in defenders:
            action = self._defend_anthill(obs, own_anthill, game_state, force_map)
            actions[obs.ant_id] = action
            self.planned_positions[obs.ant_id] = Action.get_new_position(obs.position, action)

        # 2. Attackers (coordinated assault)
        if len(attackers) >= self.swarm_threshold and enemy_anthill:
            for obs in attackers:
                action = self._swarm_attack(obs, enemy_anthill, attackers, game_state)
                actions[obs.ant_id] = action
                self.planned_positions[obs.ant_id] = Action.get_new_position(obs.position, action)

        # 3. Collectors (intelligent food gathering)
        for obs in collectors:
            action = self._collect_food_smart(obs, game_state, force_map)
            actions[obs.ant_id] = action
            self.planned_positions[obs.ant_id] = Action.get_new_position(obs.position, action)

        # Resolve friendly collisions
        actions = self._resolve_collisions(observations, actions)

        return actions

    def _update_game_phase(self, game_state: GameState):
        """Determine current game phase for adaptive strategy."""
        total_food = game_state.food_collected[0] + game_state.food_collected[1]

        if total_food < 5:
            self.game_phase = "early"
        elif total_food < 15:
            self.game_phase = "mid"
        else:
            self.game_phase = "late"

    def _assign_roles(
        self,
        observations: List[AntObservation],
        own_anthill: Optional[Position],
        enemy_anthill: Optional[Position],
        game_state: GameState
    ) -> Tuple[List[AntObservation], List[AntObservation], List[AntObservation]]:
        """
        Assign tactical roles to ants based on game state.

        Returns:
            (defenders, attackers, collectors)
        """
        defenders = []
        attackers = []
        collectors = []

        # Count nearby threats to our anthill
        threat_level = 0
        if own_anthill:
            for obs in observations:
                enemy_count = self._count_nearby_enemies(obs)
                if obs.position.chebyshev_distance(own_anthill) <= self.defense_radius:
                    threat_level += enemy_count

        # Calculate needed defenders (at least 1, more if threatened)
        needed_defenders = max(1, min(3, threat_level + 1))

        # Sort ants by distance to own anthill
        ants_by_distance = sorted(
            observations,
            key=lambda o: o.position.chebyshev_distance(own_anthill) if own_anthill else 999
        )

        # Assign defenders (closest to own anthill)
        defenders = ants_by_distance[:needed_defenders]
        remaining = ants_by_distance[needed_defenders:]

        # Determine attack vs collect based on game state
        our_food = game_state.food_collected[self.player_id]
        their_food = game_state.food_collected[1 - self.player_id]

        # Attack conditions:
        # 1. Enemy anthill visible
        # 2. We have numerical superiority OR we're ahead in food
        # 3. Late game and we're winning
        should_attack = (
            enemy_anthill is not None and (
                len(observations) > 4 or  # Have enough ants
                our_food > their_food + 2 or  # Significant food lead
                (self.game_phase == "late" and our_food >= their_food)
            )
        )

        if should_attack and enemy_anthill:
            # Send multiple ants to attack (swarm)
            attack_count = max(2, len(remaining) // 2)
            # Sort by distance to enemy anthill
            remaining_by_enemy_dist = sorted(
                remaining,
                key=lambda o: o.position.chebyshev_distance(enemy_anthill)
            )
            attackers = remaining_by_enemy_dist[:attack_count]
            collectors = remaining_by_enemy_dist[attack_count:]
        else:
            # Focus on food collection
            collectors = remaining

        return defenders, attackers, collectors

    def _defend_anthill(
        self,
        obs: AntObservation,
        own_anthill: Optional[Position],
        game_state: GameState,
        force_map: Dict[Position, int]
    ) -> int:
        """Defend own anthill with intelligent positioning."""
        if not own_anthill:
            return Action.STAY

        # Check for immediate threats
        enemy_positions = self._get_enemy_positions(obs)

        if enemy_positions:
            # Find closest enemy to anthill
            closest_enemy = min(
                enemy_positions,
                key=lambda p: p.chebyshev_distance(own_anthill)
            )

            enemy_distance = closest_enemy.chebyshev_distance(own_anthill)

            if enemy_distance <= 2:
                # CRITICAL: Enemy very close to anthill - intercept aggressively
                return self._move_toward(obs, closest_enemy, game_state)
            else:
                # Position between anthill and enemy
                return self._position_between(obs, own_anthill, closest_enemy, game_state)

        # No immediate threat - maintain defensive perimeter
        current_distance = obs.position.chebyshev_distance(own_anthill)

        if current_distance > self.defense_radius:
            # Too far - move closer
            return self._move_toward(obs, own_anthill, game_state)
        elif current_distance < 2:
            # Too close - patrol perimeter
            return self._patrol_perimeter(obs, own_anthill, game_state)
        else:
            # Good position - look for food while defending
            food_pos = self._find_nearest_food(obs)
            if food_pos and food_pos.chebyshev_distance(own_anthill) <= self.defense_radius:
                return self._move_toward(obs, food_pos, game_state)
            return Action.STAY

    def _swarm_attack(
        self,
        obs: AntObservation,
        enemy_anthill: Position,
        all_attackers: List[AntObservation],
        game_state: GameState
    ) -> int:
        """Coordinate multiple ants to attack enemy anthill."""
        # Calculate if we can reach anthill this turn
        distance = obs.position.chebyshev_distance(enemy_anthill)

        if distance == 1:
            # Adjacent to anthill - GO FOR THE KILL
            return self._move_toward(obs, enemy_anthill, game_state)

        # Check for enemy defenders
        enemy_count = self._count_nearby_enemies(obs)
        ally_count = len([a for a in all_attackers
                         if a.position.chebyshev_distance(obs.position) <= 2])

        if enemy_count > 0 and ally_count < enemy_count:
            # Wait for reinforcements - move toward other attackers
            closest_ally = min(
                [a for a in all_attackers if a.ant_id != obs.ant_id],
                key=lambda a: a.position.chebyshev_distance(obs.position),
                default=None
            )
            if closest_ally and obs.position.chebyshev_distance(closest_ally.position) > 1:
                return self._move_toward(obs, closest_ally.position, game_state)

        # Advance toward enemy anthill
        return self._move_toward(obs, enemy_anthill, game_state)

    def _collect_food_smart(
        self,
        obs: AntObservation,
        game_state: GameState,
        force_map: Dict[Position, int]
    ) -> int:
        """Intelligent food collection with denial tactics."""
        food_positions = self._get_all_food_positions(obs)

        if not food_positions:
            # No food visible - explore intelligently
            return self._explore_smart(obs, game_state)

        # Score each food by multiple factors
        best_food = None
        best_score = -999999

        for food_pos in food_positions:
            # Skip if already claimed by ally
            if food_pos in self.food_claims:
                continue

            distance = obs.position.chebyshev_distance(food_pos)

            # Check local force balance
            local_force = force_map.get(food_pos, 0)

            # Check if enemies are also targeting this food
            enemy_distance = self._nearest_enemy_distance_to(obs, food_pos)

            # Scoring factors
            distance_score = -distance * 2  # Closer is better
            force_score = local_force * 3  # Positive if we dominate area
            competition_score = (enemy_distance - distance) * 5 if enemy_distance else 10

            total_score = distance_score + force_score + competition_score

            # Bonus for food that denies enemy expansion
            if enemy_distance and enemy_distance < distance:
                # Enemy is closer - only take if we dominate (denial tactic)
                if local_force > 2:
                    total_score += 15  # Food denial bonus
                else:
                    total_score -= 20  # Avoid contested food when weak

            if total_score > best_score:
                best_score = total_score
                best_food = food_pos

        if best_food:
            # Claim this food
            self.food_claims[best_food] = obs.ant_id
            return self._move_toward(obs, best_food, game_state)

        # No good food options - explore
        return self._explore_smart(obs, game_state)

    def _explore_smart(self, obs: AntObservation, game_state: GameState) -> int:
        """Explore strategically rather than randomly."""
        # Get board dimensions
        board_width = game_state.board.width
        board_height = game_state.board.height

        # Explore toward less-visited areas
        # Simple heuristic: move toward board center with some randomness
        center = Position(board_width // 2, board_height // 2)

        # Add some randomness to avoid predictable patterns
        if random.random() < 0.3:
            # Random exploration
            valid_actions = self.get_valid_actions(obs, game_state)
            move_actions = [a for a in valid_actions if a != Action.STAY]
            return random.choice(move_actions) if move_actions else Action.STAY

        # Move toward center
        return self._move_toward(obs, center, game_state)

    def _calculate_force_map(self, observations: List[AntObservation]) -> Dict[Position, int]:
        """
        Calculate relative force at each position.
        Positive = we dominate, Negative = enemy dominates.
        """
        force_map = defaultdict(int)

        for obs in observations:
            # Add our presence
            for pos in self._get_area_positions(obs.position, radius=2):
                force_map[pos] += 1

            # Subtract enemy presence
            enemy_positions = self._get_enemy_positions(obs)
            for enemy_pos in enemy_positions:
                for pos in self._get_area_positions(enemy_pos, radius=2):
                    force_map[pos] -= 1

        return force_map

    def _get_area_positions(self, center: Position, radius: int) -> List[Position]:
        """Get all positions within radius of center."""
        positions = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) <= radius and abs(dy) <= radius:
                    positions.append(Position(center.x + dx, center.y + dy))
        return positions

    def _position_between(
        self,
        obs: AntObservation,
        protect: Position,
        threat: Position,
        game_state: GameState
    ) -> int:
        """Position between a protected asset and a threat."""
        # Calculate midpoint
        mid_x = (protect.x + threat.x) // 2
        mid_y = (protect.y + threat.y) // 2
        midpoint = Position(mid_x, mid_y)

        return self._move_toward(obs, midpoint, game_state)

    def _patrol_perimeter(
        self,
        obs: AntObservation,
        center: Position,
        game_state: GameState
    ) -> int:
        """Patrol around a perimeter."""
        # Move perpendicular to radial direction
        dx = obs.position.x - center.x
        dy = obs.position.y - center.y

        # Perpendicular direction
        perp_positions = [
            Position(obs.position.x - dy, obs.position.y + dx),
            Position(obs.position.x + dy, obs.position.y - dx)
        ]

        # Choose valid perpendicular move
        for target in perp_positions:
            action = self._move_toward(obs, target, game_state)
            if self._is_action_safe(obs, action):
                return action

        return Action.STAY

    def _move_toward(
        self,
        obs: AntObservation,
        target: Position,
        game_state: GameState
    ) -> int:
        """Move toward target with obstacle avoidance."""
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

        # Primary direction
        primary_action = None
        for action_id, (adx, ady) in Action.DIRECTIONS.items():
            if adx == dx and ady == dy:
                primary_action = action_id
                break

        # Try primary direction
        if primary_action and self._is_action_safe(obs, primary_action):
            valid_actions = self.get_valid_actions(obs, game_state)
            if primary_action in valid_actions:
                return primary_action

        # Try alternate diagonal
        alternates = []

        # Try moving in just x direction
        if dx != 0:
            for action_id, (adx, ady) in Action.DIRECTIONS.items():
                if adx == dx and ady == 0:
                    alternates.append(action_id)

        # Try moving in just y direction
        if dy != 0:
            for action_id, (adx, ady) in Action.DIRECTIONS.items():
                if adx == 0 and ady == dy:
                    alternates.append(action_id)

        # Try alternates
        valid_actions = self.get_valid_actions(obs, game_state)
        for action in alternates:
            if action in valid_actions and self._is_action_safe(obs, action):
                return action

        # Last resort - any safe valid move
        for action in valid_actions:
            if action != Action.STAY and self._is_action_safe(obs, action):
                return action

        return Action.STAY

    def _is_action_safe(self, obs: AntObservation, action: int) -> bool:
        """Check if action is safe (doesn't step on own anthill or planned collision)."""
        new_pos = Action.get_new_position(obs.position, action)

        # Check own anthill
        own_anthill_str = f'anthill_{self.player_id}'
        for vision_pos, entity in obs.vision.items():
            if vision_pos.x == new_pos.x and vision_pos.y == new_pos.y:
                if entity == own_anthill_str:
                    return False

        # Check planned positions (avoid friendly collisions)
        for other_ant_id, other_pos in self.planned_positions.items():
            if other_pos.x == new_pos.x and other_pos.y == new_pos.y:
                return False

        return True

    def _resolve_collisions(
        self,
        observations: List[AntObservation],
        actions: Dict[int, int]
    ) -> Dict[int, int]:
        """Resolve friendly collision conflicts."""
        position_counts = defaultdict(list)

        for obs in observations:
            if obs.ant_id not in actions:
                continue
            action = actions[obs.ant_id]
            new_pos = Action.get_new_position(obs.position, action)
            position_counts[new_pos].append(obs.ant_id)

        # For collisions, keep first ant's move
        for pos, ant_ids in position_counts.items():
            if len(ant_ids) > 1:
                for ant_id in ant_ids[1:]:
                    actions[ant_id] = Action.STAY

        return actions

    # Helper methods for information gathering

    def _find_own_anthill(self, observations: List[AntObservation]) -> Optional[Position]:
        """Find own anthill position."""
        own_anthill_str = f'anthill_{self.player_id}'
        for obs in observations:
            for pos, entity in obs.vision.items():
                if entity == own_anthill_str:
                    return pos
        return None

    def _find_enemy_anthill(self, observations: List[AntObservation]) -> Optional[Position]:
        """Find enemy anthill if visible."""
        enemy_anthill_str = f'anthill_{1 - self.player_id}'
        for obs in observations:
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

        return min(
            food_positions,
            key=lambda p: obs.position.chebyshev_distance(p)
        )

    def _get_all_food_positions(self, obs: AntObservation) -> List[Position]:
        """Get all visible food positions."""
        return [
            pos for pos, entity in obs.vision.items()
            if entity == 'food'
        ]

    def _get_enemy_positions(self, obs: AntObservation) -> List[Position]:
        """Get all visible enemy ant positions."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
        return [
            pos for pos, entity in obs.vision.items()
            if entity == enemy_ant_str
        ]

    def _count_nearby_enemies(self, obs: AntObservation) -> int:
        """Count visible enemy ants."""
        enemy_ant_str = f'ant_{1 - self.player_id}'
        return sum(1 for entity in obs.vision.values() if entity == enemy_ant_str)

    def _nearest_enemy_distance_to(self, obs: AntObservation, target: Position) -> Optional[int]:
        """Find nearest enemy's distance to target position."""
        enemy_positions = self._get_enemy_positions(obs)

        if not enemy_positions:
            return None

        return min(
            enemy_pos.chebyshev_distance(target)
            for enemy_pos in enemy_positions
        )

    def reset(self):
        """Reset agent state."""
        self.planned_positions.clear()
        self.food_claims.clear()
        self.turn_count = 0
        self.game_phase = "early"
