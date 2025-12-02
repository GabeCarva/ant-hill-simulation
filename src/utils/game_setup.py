"""Game setup and match configuration."""

from dataclasses import dataclass
from typing import Optional, Any, Callable, Tuple
from src.utils.game_config import GameConfig


@dataclass
class GameSetup:
    """Complete game setup including agents and visualization."""
    
    # Core game configuration
    config: GameConfig = None
    
    # Agent specifications
    player1_agent: str = "random"  # Agent type identifier
    player1_model_path: Optional[str] = None  # Path to saved model weights
    player2_agent: str = "random"  
    player2_model_path: Optional[str] = None
    
    # Visualization
    visualization: str = "none"  # "none", "ascii", "gui"
    render_every_n_turns: int = 1  # Only render every N turns for performance
    
    # Match settings
    seed: Optional[int] = None  # Random seed for reproducibility
    record_replay: bool = False  # Whether to save replay data
    
    def __post_init__(self):
        if self.config is None:
            self.config = GameConfig()
    
    @classmethod
    def quick_match(
        cls,
        board_size: Tuple[int, int] = (100, 100),
        food_density: float = 0.10,
        rock_density: float = 0.05,
        max_turns: Optional[int] = None,
        player1: str = "random",
        player2: str = "random",
        visualization: str = "ascii"
    ) -> 'GameSetup':
        """Convenience method for quick game setup."""
        config = GameConfig(
            board_width=board_size[0],
            board_height=board_size[1],
            food_density=food_density,
            rock_density=rock_density,
            max_turns=max_turns
        )
        
        return cls(
            config=config,
            player1_agent=player1,
            player2_agent=player2,
            visualization=visualization
        )