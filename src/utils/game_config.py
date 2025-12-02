"""Game configuration and constants."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GameConfig:
    """Core game configuration."""
    # Board parameters
    board_width: int = 20
    board_height: int = 20
    food_density: float = 0.10  # Percentage of squares with food
    rock_density: float = 0.05  # Percentage of squares with rocks
    
    # Game mechanics
    initial_ants_per_player: int = 2
    ant_vision_radius: int = 1  # Chebyshev distance (see all 8 neighbors)
    max_turns: Optional[int] = None  # None = no turn limit
    
    # Board generation constraints
    min_anthill_distance: int = 20  # Minimum Manhattan distance between anthills
    spawn_radius: int = 2  # Radius around anthill for new ant spawning
    
    @property
    def board_size(self) -> Tuple[int, int]:
        return (self.board_width, self.board_height)
    
    @property
    def total_squares(self) -> int:
        return self.board_width * self.board_height
    
    @property
    def initial_food_count(self) -> int:
        return int(self.total_squares * self.food_density)
    
    @property
    def initial_rock_count(self) -> int:
        return int(self.total_squares * self.rock_density)


@dataclass
class Position:
    """2D position on the board."""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def manhattan_distance(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def chebyshev_distance(self, other: 'Position') -> int:
        """Distance for diagonal movement (king move in chess)."""
        return max(abs(self.x - other.x), abs(self.y - other.y))
    
    def get_neighbors(self, include_diagonals: bool = True) -> list['Position']:
        """Get all valid neighbor positions (doesn't check board bounds)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if not include_diagonals and dx != 0 and dy != 0:
                    continue
                neighbors.append(Position(self.x + dx, self.y + dy))
        return neighbors


# Action space for ants
class Action:
    """Possible actions for an ant."""
    STAY = 0
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7
    NORTHWEST = 8
    
    # Movement vectors for each action
    DIRECTIONS = {
        STAY: (0, 0),
        NORTH: (0, -1),
        NORTHEAST: (1, -1),
        EAST: (1, 0),
        SOUTHEAST: (1, 1),
        SOUTH: (0, 1),
        SOUTHWEST: (-1, 1),
        WEST: (-1, 0),
        NORTHWEST: (-1, -1),
    }
    
    @classmethod
    def get_new_position(cls, pos: Position, action: int) -> Position:
        """Get new position after taking an action."""
        dx, dy = cls.DIRECTIONS[action]
        return Position(pos.x + dx, pos.y + dy)