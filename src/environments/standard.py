"""Standard game environment with balanced rewards."""

from typing import List, Optional
import numpy as np

from src.environments.base import BaseEnvironment
from src.utils.game_config import GameConfig
from src.core.game import GameState
from src.agents.base import AntObservation


class StandardEnvironment(BaseEnvironment):
    """
    Standard game environment with balanced reward structure.
    
    Rewards:
    - Food collection: +1.0
    - Ant spawned (from food): +0.5
    - Enemy ant killed: +0.3
    - Own ant lost: -0.3
    - Win game: +10.0
    - Lose game: -10.0
    - Draw: 0.0
    - Small step penalty: -0.01 (encourage efficiency)
    """
    
    def get_default_config(self) -> GameConfig:
        """Standard 100x100 game configuration."""
        return GameConfig(
            board_width=100,
            board_height=100,
            food_density=0.10,
            rock_density=0.05,
            initial_ants_per_player=5,
            ant_vision_radius=1,
            max_turns=1000,
            min_anthill_distance=30,
            spawn_radius=2
        )
    
    def calculate_reward(
        self,
        player_id: int,
        prev_state: GameState,
        curr_state: GameState,
        observations: List[AntObservation]
    ) -> float:
        """Calculate standard reward."""
        reward = 0.0
        
        # Food collection reward
        prev_food = prev_state.food_collected.get(player_id, 0)
        curr_food = curr_state.food_collected.get(player_id, 0)
        food_delta = curr_food - prev_food
        reward += food_delta * 1.0
        
        # Ant spawning bonus (happens when food is collected)
        prev_ants = len(prev_state.board.get_ants_by_player(player_id))
        curr_ants = len(curr_state.board.get_ants_by_player(player_id))
        spawned = max(0, curr_ants - prev_ants + (curr_state.ants_lost.get(player_id, 0) - prev_state.ants_lost.get(player_id, 0)))
        reward += spawned * 0.5
        
        # Combat rewards/penalties
        prev_lost = prev_state.ants_lost.get(player_id, 0)
        curr_lost = curr_state.ants_lost.get(player_id, 0)
        own_lost = curr_lost - prev_lost
        reward -= own_lost * 0.3
        
        # Enemy losses (our gains)
        enemy_id = 1 - player_id
        prev_enemy_lost = prev_state.ants_lost.get(enemy_id, 0)
        curr_enemy_lost = curr_state.ants_lost.get(enemy_id, 0)
        enemy_lost = curr_enemy_lost - prev_enemy_lost
        reward += enemy_lost * 0.3
        
        # Game ending rewards
        if curr_state.is_terminal():
            if curr_state.winner == player_id:
                reward += 10.0
            elif curr_state.winner == enemy_id:
                reward -= 10.0
            # Draw gives 0
        
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
        return reward


class SparseRewardEnvironment(StandardEnvironment):
    """
    Environment with sparse rewards - only at game end.
    
    Good for testing credit assignment in RL algorithms.
    """
    
    def calculate_reward(
        self,
        player_id: int,
        prev_state: GameState,
        curr_state: GameState,
        observations: List[AntObservation]
    ) -> float:
        """Only reward at game end."""
        if curr_state.is_terminal():
            if curr_state.winner == player_id:
                return 100.0
            elif curr_state.winner == (1 - player_id):
                return -100.0
            else:
                # Draw - slight reward based on food collected
                return curr_state.food_collected.get(player_id, 0) - \
                       curr_state.food_collected.get(1 - player_id, 0)
        return 0.0


class ShapedRewardEnvironment(StandardEnvironment):
    """
    Environment with heavily shaped rewards to guide learning.
    
    Includes distance-based rewards and exploration bonuses.
    """
    
    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.visited_positions = {0: set(), 1: set()}
    
    def reset(self, seed: Optional[int] = None):
        """Reset and clear visit history."""
        self.visited_positions = {0: set(), 1: set()}
        return super().reset(seed)
    
    def calculate_reward(
        self,
        player_id: int,
        prev_state: GameState,
        curr_state: GameState, 
        observations: List[AntObservation]
    ) -> float:
        """Calculate shaped reward with exploration bonus."""
        # Start with standard reward
        reward = super().calculate_reward(
            player_id, prev_state, curr_state, observations
        )
        
        # Add exploration bonus
        exploration_bonus = 0.0
        for obs in observations:
            pos_tuple = (obs.position.x, obs.position.y)
            if pos_tuple not in self.visited_positions[player_id]:
                exploration_bonus += 0.02
                self.visited_positions[player_id].add(pos_tuple)
        
        reward += exploration_bonus
        
        # Add proximity rewards
        if not curr_state.is_terminal():
            # Reward for being near food
            for obs in observations:
                for pos, entity in obs.vision.items():
                    if entity == 'food':
                        distance = abs(pos.x - obs.position.x) + abs(pos.y - obs.position.y)
                        if distance == 1:
                            reward += 0.05  # Adjacent to food
                        elif distance == 2:
                            reward += 0.02  # Near food
            
            # Reward for being near enemy anthill
            enemy_anthill = curr_state.board.anthills.get(1 - player_id)
            if enemy_anthill and enemy_anthill.alive:
                for obs in observations:
                    distance = obs.position.manhattan_distance(enemy_anthill.position)
                    if distance < 10:
                        reward += 0.1 * (10 - distance) / 10  # Closer = more reward
        
        return reward