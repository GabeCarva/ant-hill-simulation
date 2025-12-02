"""Base agent interface and utilities."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

from src.utils.game_config import Position, Action
from src.core.game import Game, GameState
from src.core.entities import Ant


@dataclass
class AntObservation:
    """
    Structured observation for a single ant.
    
    Attributes:
        ant_id: Unique identifier for this ant
        player_id: Which player controls this ant (0 or 1)
        position: Current position on the board
        vision: Raw vision data as dictionary
        vision_array: Encoded vision for neural networks
                     Shape: (grid_size, grid_size, 3)
                     Channel 0: Entity type (0=empty, 1=wall, 2=rock, 3=food, 4=ant, 5=anthill)
                     Channel 1: Team affiliation (-1=enemy, 0=neutral, 1=allied)
                     Channel 2: Mobility (0=blocked, 1=passable)
    """
    ant_id: int
    player_id: int
    position: Position
    vision: Dict[Position, Optional[str]]  # Raw vision data
    vision_array: np.ndarray  # Encoded vision for neural networks
    
    @property
    def vision_radius(self) -> int:
        """Get vision radius from vision array shape."""
        # Assuming square vision grid
        return (self.vision_array.shape[0] - 1) // 2


class BaseAgent(ABC):
    """
    Abstract base class for all agent types.
    
    Each agent controls all ants for one player and must decide
    actions for each ant based on observations.
    """
    
    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent.
        
        Args:
            player_id: Which player this agent controls (0 or 1)
            config: Agent-specific configuration
        """
        self.player_id = player_id
        self.config = config or {}
        self.is_training = False
        
    @abstractmethod
    def get_actions(
        self, 
        observations: List[AntObservation], 
        game_state: GameState
    ) -> Dict[int, int]:
        """
        Get actions for all ants.
        
        Args:
            observations: List of observations, one per living ant
            game_state: Current game state
            
        Returns:
            Dictionary mapping ant_id -> action
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass
    
    def train_mode(self, training: bool = True):
        """Set training mode (affects exploration vs exploitation)."""
        self.is_training = training
    
    def save(self, path: str):
        """Save agent state/model to disk."""
        pass
    
    def load(self, path: str):
        """Load agent state/model from disk."""
        pass
    
    
    @staticmethod
    def encode_vision(
        vision: Dict[Position, Optional[str]], 
        player_id: int,
        vision_radius: int = 1
    ) -> np.ndarray:
        """
        Encode vision dictionary to neural network input using factored representation.
        
        Channel layout:
        0: Entity type encoding
           - 0: empty
           - 1: wall
           - 2: rock  
           - 3: food
           - 4: ant
           - 5: anthill
        1: Team affiliation
           - -1: enemy
           -  0: neutral/none
           - +1: allied
        2: Mobility (can move through)
           - 0: blocked
           - 1: passable
           
        This factorization lets the network learn entity-team interactions.
        
        Args:
            vision: Raw vision dictionary from game
            player_id: Current player ID
            vision_radius: Vision radius
            
        Returns:
            numpy array of shape (2*radius+1, 2*radius+1, 3)
        """
        grid_size = 2 * vision_radius + 1
        encoded = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
        
        center_x, center_y = 0, 0
        
        for pos, entity in vision.items():
            grid_x = pos.x - center_x + vision_radius
            grid_y = pos.y - center_y + vision_radius
            
            if not (0 <= grid_x < grid_size and 0 <= grid_y < grid_size):
                continue
            
            if entity == 'wall':
                encoded[grid_y, grid_x, 0] = 1  # Entity type: wall
                encoded[grid_y, grid_x, 1] = 0  # Team: neutral
                encoded[grid_y, grid_x, 2] = 0  # Mobility: blocked
            elif entity == 'rock':
                encoded[grid_y, grid_x, 0] = 2  # Entity type: rock
                encoded[grid_y, grid_x, 1] = 0  # Team: neutral
                encoded[grid_y, grid_x, 2] = 0  # Mobility: blocked
            elif entity is None:
                encoded[grid_y, grid_x, 0] = 0  # Entity type: empty
                encoded[grid_y, grid_x, 1] = 0  # Team: neutral
                encoded[grid_y, grid_x, 2] = 1  # Mobility: passable
            elif entity == 'food':
                encoded[grid_y, grid_x, 0] = 3  # Entity type: food
                encoded[grid_y, grid_x, 1] = 0  # Team: neutral
                encoded[grid_y, grid_x, 2] = 1  # Mobility: passable
            elif entity.startswith('ant_'):
                ant_player = int(entity.split('_')[1])
                encoded[grid_y, grid_x, 0] = 4  # Entity type: ant
                encoded[grid_y, grid_x, 1] = 1 if ant_player == player_id else -1
                encoded[grid_y, grid_x, 2] = 1  # Mobility: passable (but collision)
            elif entity.startswith('anthill_'):
                hill_player = int(entity.split('_')[1])
                encoded[grid_y, grid_x, 0] = 5  # Entity type: anthill
                encoded[grid_y, grid_x, 1] = 1 if hill_player == player_id else -1
                encoded[grid_y, grid_x, 2] = 1  # Mobility: passable
        
        return encoded
    
    @staticmethod
    def get_valid_actions(
        observation: AntObservation,
        game_state: GameState
    ) -> List[int]:
        """
        Get list of valid actions for an ant (for action masking).
        
        Args:
            observation: Ant's observation
            game_state: Current game state
            
        Returns:
            List of valid action indices
        """
        valid = []
        board = game_state.board
        
        for action_id, (dx, dy) in Action.DIRECTIONS.items():
            new_pos = Position(
                observation.position.x + dx,
                observation.position.y + dy
            )
            
            # Check if move is valid
            if not board.is_valid_position(new_pos):
                continue
                
            # Check for rocks (can't move into them)
            entity = board.get_entity_at(new_pos)
            if entity and entity.__class__.__name__ == 'Rock':
                continue
                
            valid.append(action_id)
        
        # STAY is always valid
        if Action.STAY not in valid:
            valid.append(Action.STAY)
            
        return valid


class AgentWrapper:
    """
    Wrapper to handle agent-game interaction.
    
    Manages observation collection and action distribution for an agent.
    """
    
    def __init__(self, agent: BaseAgent, game: Game):
        self.agent = agent
        self.game = game
        
    def get_observations(self) -> List[AntObservation]:
        """Collect observations for all of agent's ants."""
        observations = []
        
        for ant in self.game.board.get_ants_by_player(self.agent.player_id):
            if not ant.alive:
                continue
                
            raw_obs = self.game.get_ant_observation(ant)
            
            # Encode vision for neural network using factored encoding
            vision_array = BaseAgent.encode_vision(
                raw_obs['vision'],
                self.agent.player_id,
                self.game.config.ant_vision_radius
            )
            
            observations.append(AntObservation(
                ant_id=ant.ant_id,
                player_id=ant.player_id,
                position=ant.position,
                vision=raw_obs['vision'],
                vision_array=vision_array
            ))
        
        return observations
    
    def apply_actions(self, actions: Dict[int, int]):
        """Apply agent's actions to the game."""
        for ant_id, action in actions.items():
            self.game.set_ant_action(ant_id, action)
    
    def step(self):
        """
        Execute one step: get observations, decide actions, apply them.
        
        Note: This doesn't call game.step() - that should be done after
        both agents have submitted their actions.
        """
        observations = self.get_observations()
        game_state = self.game.get_state()
        
        if observations:  # Only get actions if we have living ants
            actions = self.agent.get_actions(observations, game_state)
            self.apply_actions(actions)