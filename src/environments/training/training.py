"""Training environment for reinforcement learning agents."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.environments.base import BaseEnvironment, StepResult
from src.environments.standard import StandardEnvironment
from src.utils.game_config import GameConfig
from src.agents.base import BaseAgent


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    episode: int
    total_reward: float
    food_collected: int
    ants_lost: int
    winner: Optional[int]
    steps: int
    anthill_destroyed: bool


class TrainingEnvironment(StandardEnvironment):
    """Extended environment for training with additional metrics and features."""
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        reward_shaping: bool = True,
        normalize_rewards: bool = True,
        track_metrics: bool = True
    ):
        """
        Initialize training environment.
        
        Args:
            config: Game configuration
            seed: Random seed
            reward_shaping: Whether to use reward shaping
            normalize_rewards: Whether to normalize rewards
            track_metrics: Whether to track detailed metrics
        """
        super().__init__(config, seed)
        
        self.reward_shaping = reward_shaping
        self.normalize_rewards = normalize_rewards
        self.track_metrics = track_metrics
        
        # Metrics tracking
        self.episode_count = 0
        self.episode_metrics = []
        self.current_episode_rewards = {0: 0.0, 1: 0.0}
        self.current_episode_steps = 0
        
        # Reward normalization parameters (running statistics)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def reset(self, seed: Optional[int] = None) -> Dict[int, List]:
        """Reset environment and metrics."""
        observations = super().reset(seed)
        
        # Reset episode tracking
        self.current_episode_rewards = {0: 0.0, 1: 0.0}
        self.current_episode_steps = 0
        
        return observations
    
    def step(self, actions: Dict[int, Dict[int, int]]) -> StepResult:
        """Step environment with enhanced tracking."""
        # Get previous state for reward calculation
        prev_state = self.game.get_state()
        
        # Execute step
        result = super().step(actions)
        
        # Add reward shaping if enabled
        if self.reward_shaping:
            shaped_rewards = self._shape_rewards(
                result.rewards,
                prev_state,
                self.game.get_state(),
                result.observations
            )
            result.rewards = shaped_rewards
        
        # Normalize rewards if enabled
        if self.normalize_rewards:
            result.rewards = self._normalize_rewards(result.rewards)
        
        # Track metrics
        if self.track_metrics:
            self.current_episode_rewards[0] += result.rewards[0]
            self.current_episode_rewards[1] += result.rewards[1]
            self.current_episode_steps += 1
            
            if result.done:
                self._record_episode_metrics(result)
        
        return result
    
    def _shape_rewards(
        self,
        base_rewards: Dict[int, float],
        prev_state,
        curr_state,
        observations: Dict[int, List]
    ) -> Dict[int, float]:
        """Add reward shaping to encourage good behavior."""
        shaped = base_rewards.copy()
        
        for player_id in [0, 1]:
            # Distance to enemy anthill bonus
            enemy_id = 1 - player_id
            enemy_anthill = self.game.board.anthills[enemy_id]
            
            if enemy_anthill.alive:
                min_distance = float('inf')
                for obs in observations[player_id]:
                    dist = obs.position.manhattan_distance(enemy_anthill.position)
                    min_distance = min(min_distance, dist)
                
                # Small bonus for being closer to enemy anthill
                proximity_bonus = max(0, (20 - min_distance) * 0.01)
                shaped[player_id] += proximity_bonus
            
            # Exploration bonus
            # Track unique positions visited (would need state tracking)
            
            # Ant survival bonus (small negative for losing ants)
            prev_ants = len([a for a in prev_state.board.ants.values() 
                           if a.player_id == player_id and a.alive])
            curr_ants = len([a for a in curr_state.board.ants.values() 
                           if a.player_id == player_id and a.alive])
            if curr_ants < prev_ants:
                shaped[player_id] -= 0.1 * (prev_ants - curr_ants)
            
            # Food proximity bonus
            for obs in observations[player_id]:
                for pos, entity in obs.vision.items():
                    if entity == 'food':
                        # Small bonus for being near food
                        shaped[player_id] += 0.02
                        break
        
        return shaped
    
    def _normalize_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        """Normalize rewards using running statistics."""
        normalized = {}
        
        for player_id, reward in rewards.items():
            # Update running statistics
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            self.reward_std = np.sqrt(
                ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean)) 
                / self.reward_count
            )
            
            # Normalize
            if self.reward_std > 0:
                normalized[player_id] = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            else:
                normalized[player_id] = reward
        
        return normalized
    
    def _record_episode_metrics(self, final_result: StepResult):
        """Record metrics for completed episode."""
        self.episode_count += 1
        
        # Check if anthill was destroyed
        anthill_destroyed = False
        for player_id in [0, 1]:
            if not self.game.board.anthills[player_id].alive:
                anthill_destroyed = True
                break
        
        metrics = TrainingMetrics(
            episode=self.episode_count,
            total_reward=self.current_episode_rewards[0],  # Track player 0 by default
            food_collected=final_result.info['food_collected'][0],
            ants_lost=final_result.info['ants_lost'][0],
            winner=final_result.winner,
            steps=self.current_episode_steps,
            anthill_destroyed=anthill_destroyed
        )
        
        self.episode_metrics.append(metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if not self.episode_metrics:
            return {}
        
        recent_metrics = self.episode_metrics[-100:]  # Last 100 episodes
        
        wins = [m.winner == 0 for m in recent_metrics]
        
        return {
            'total_episodes': self.episode_count,
            'win_rate': np.mean(wins) if wins else 0,
            'avg_reward': np.mean([m.total_reward for m in recent_metrics]),
            'avg_food': np.mean([m.food_collected for m in recent_metrics]),
            'avg_ants_lost': np.mean([m.ants_lost for m in recent_metrics]),
            'avg_steps': np.mean([m.steps for m in recent_metrics]),
            'anthill_destruction_rate': np.mean([m.anthill_destroyed for m in recent_metrics])
        }


class SelfPlayEnvironment(TrainingEnvironment):
    """Environment for self-play training."""
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        update_opponent_freq: int = 100,
        **kwargs
    ):
        """
        Initialize self-play environment.
        
        Args:
            config: Game configuration
            seed: Random seed
            update_opponent_freq: Episodes between opponent updates
            **kwargs: Additional arguments for TrainingEnvironment
        """
        super().__init__(config, seed, **kwargs)
        
        self.update_opponent_freq = update_opponent_freq
        self.opponent_version = 0
        
    def should_update_opponent(self) -> bool:
        """Check if opponent should be updated."""
        return self.episode_count % self.update_opponent_freq == 0
    
    def update_opponent(self, agent: BaseAgent):
        """Update opponent with current agent version."""
        self.opponent_version += 1
        # In practice, you'd clone the agent here
        return self.opponent_version


class CurriculumEnvironment(TrainingEnvironment):
    """Environment with curriculum learning."""
    
    def __init__(
        self,
        initial_config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        difficulty_schedule: Optional[List[Tuple[int, GameConfig]]] = None,
        **kwargs
    ):
        """
        Initialize curriculum environment.
        
        Args:
            initial_config: Starting game configuration
            seed: Random seed
            difficulty_schedule: List of (episode, config) pairs
            **kwargs: Additional arguments for TrainingEnvironment
        """
        super().__init__(initial_config, seed, **kwargs)
        
        self.initial_config = initial_config or self.config
        self.difficulty_schedule = difficulty_schedule or self._default_schedule()
        self.current_difficulty_level = 0
    
    def _default_schedule(self) -> List[Tuple[int, GameConfig]]:
        """Create default difficulty schedule."""
        return [
            (0, GameConfig(board_width=10, board_height=10, max_turns=50)),
            (100, GameConfig(board_width=20, board_height=20, max_turns=100)),
            (500, GameConfig(board_width=50, board_height=50, max_turns=200)),
            (1000, GameConfig(board_width=100, board_height=100, max_turns=500))
        ]
    
    def reset(self, seed: Optional[int] = None) -> Dict[int, List]:
        """Reset with appropriate difficulty level."""
        # Check if we should increase difficulty
        for level, (episode_threshold, config) in enumerate(self.difficulty_schedule):
            if self.episode_count >= episode_threshold:
                self.current_difficulty_level = level
                self.config = config
        
        return super().reset(seed)
    
    def get_current_difficulty(self) -> str:
        """Get description of current difficulty."""
        config = self.config
        return f"Level {self.current_difficulty_level}: {config.board_width}x{config.board_height}, {config.max_turns} turns"