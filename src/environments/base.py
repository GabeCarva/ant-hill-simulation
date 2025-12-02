"""Base environment interface for different game variants."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from src.utils.game_config import GameConfig
from src.core.game import Game, GameState
from src.agents.base import BaseAgent, AntObservation, AgentWrapper


@dataclass
class StepResult:
    """Result of an environment step."""
    observations: Dict[int, List[AntObservation]]  # player_id -> observations
    rewards: Dict[int, float]  # player_id -> reward
    done: bool
    info: Dict[str, Any]
    winner: Optional[int]  # None if not done, player_id or -1 for draw


class BaseEnvironment(ABC):
    """
    Abstract base class for game environments.
    
    Environments define:
    - Board setup and initialization
    - Reward structure
    - Termination conditions
    - Observation processing
    """
    
    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        """
        Initialize environment.
        
        Args:
            config: Game configuration
            seed: Random seed for reproducibility
        """
        self.config = config or self.get_default_config()
        self.seed = seed
        self.game = None
        self.agent_wrappers = {}
        self.episode_steps = 0
        self.cumulative_rewards = {0: 0.0, 1: 0.0}
        
    @abstractmethod
    def get_default_config(self) -> GameConfig:
        """Get default configuration for this environment."""
        pass
    
    @abstractmethod
    def calculate_reward(
        self,
        player_id: int,
        prev_state: GameState,
        curr_state: GameState,
        observations: List[AntObservation]
    ) -> float:
        """
        Calculate reward for a player after a step.
        
        Args:
            player_id: Player to calculate reward for
            prev_state: State before actions
            curr_state: State after actions
            observations: Current observations for player's ants
            
        Returns:
            Reward value
        """
        pass
    
    def reset(self, seed: Optional[int] = None) -> Dict[int, List[AntObservation]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Initial observations for both players
        """
        if seed is not None:
            self.seed = seed
            
        self.game = Game(self.config, seed=self.seed)
        self.episode_steps = 0
        self.cumulative_rewards = {0: 0.0, 1: 0.0}
        
        # Create agent wrappers if agents are attached
        for player_id, wrapper in self.agent_wrappers.items():
            wrapper.game = self.game
        
        # Get initial observations
        observations = {}
        for player_id in [0, 1]:
            if player_id in self.agent_wrappers:
                observations[player_id] = self.agent_wrappers[player_id].get_observations()
            else:
                # Manual observation collection if no agent attached
                observations[player_id] = self._get_player_observations(player_id)
        
        return observations
    
    def step(self, actions: Dict[int, Dict[int, int]]) -> StepResult:
        """
        Execute one environment step.
        
        Args:
            actions: Dict mapping player_id -> (ant_id -> action)
            
        Returns:
            StepResult with observations, rewards, done flag, and info
        """
        # Store previous state for reward calculation
        prev_state = self.game.get_state()
        
        # Apply actions for both players
        for player_id, player_actions in actions.items():
            for ant_id, action in player_actions.items():
                self.game.set_ant_action(ant_id, action)
        
        # Execute game step
        curr_state = self.game.step()
        self.episode_steps += 1
        
        # Collect observations and calculate rewards
        observations = {}
        rewards = {}
        
        for player_id in [0, 1]:
            if player_id in self.agent_wrappers:
                observations[player_id] = self.agent_wrappers[player_id].get_observations()
            else:
                observations[player_id] = self._get_player_observations(player_id)
            
            rewards[player_id] = self.calculate_reward(
                player_id, prev_state, curr_state, observations[player_id]
            )
            self.cumulative_rewards[player_id] += rewards[player_id]
        
        # Check termination
        done = curr_state.is_terminal()
        
        # Compile info
        info = {
            'turn': curr_state.turn,
            'food_collected': dict(curr_state.food_collected),
            'ants_lost': dict(curr_state.ants_lost),
            'ants_alive': {
                0: len(self.game.board.get_ants_by_player(0)),
                1: len(self.game.board.get_ants_by_player(1))
            },
            'cumulative_rewards': dict(self.cumulative_rewards)
        }
        
        return StepResult(
            observations=observations,
            rewards=rewards,
            done=done,
            info=info,
            winner=curr_state.winner
        )
    
    def attach_agent(self, player_id: int, agent: BaseAgent):
        """Attach an agent to control a player."""
        self.agent_wrappers[player_id] = AgentWrapper(agent, self.game)
    
    def _get_player_observations(self, player_id: int) -> List[AntObservation]:
        """Get observations for all ants of a player."""
        observations = []
        
        for ant in self.game.board.get_ants_by_player(player_id):
            if not ant.alive:
                continue
            
            raw_obs = self.game.get_ant_observation(ant)
            vision_array = BaseAgent.encode_vision(
                raw_obs['vision'],
                player_id,
                self.game.config.ant_vision_radius
            )
            
            observations.append(AntObservation(
                ant_id=ant.ant_id,
                player_id=player_id,
                position=ant.position,
                vision=raw_obs['vision'],
                vision_array=vision_array
            ))
        
        return observations
    
    def render(self, mode: str = 'ascii'):
        """
        Render current game state.
        
        Args:
            mode: Rendering mode ('ascii', 'gui', 'rgb_array')
        """
        if mode == 'ascii':
            from src.visualization.ascii_viz import ASCIIVisualizer
            viz = ASCIIVisualizer()
            return viz.render(self.game)
        elif mode == 'gui':
            # TODO: Implement GUI visualization
            raise NotImplementedError("GUI rendering not yet implemented")
        elif mode == 'rgb_array':
            # TODO: Implement RGB array for video recording
            raise NotImplementedError("RGB array rendering not yet implemented")
        else:
            raise ValueError(f"Unknown render mode: {mode}")