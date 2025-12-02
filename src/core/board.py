"""Board representation and management."""

import random
from typing import Dict, List, Set, Optional, Tuple
from src.utils.game_config import Position, GameConfig
from src.core.entities import Entity, Rock, Food, Ant, Anthill


class Board:
    """Game board managing entity positions and interactions."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.width = config.board_width
        self.height = config.board_height
        
        # Entity tracking
        self.rocks: Set[Rock] = set()
        self.food: Set[Food] = set()
        self.anthills: Dict[int, Anthill] = {}  # player_id -> Anthill
        self.ants: Dict[int, Ant] = {}  # ant_id -> Ant
        
        # Position lookup for efficient collision detection
        self._position_map: Dict[Position, Entity] = {}
        
        # Track next ant ID for unique identification
        self._next_ant_id = 0
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within board bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_empty(self, pos: Position) -> bool:
        """Check if position is empty (no entities)."""
        return pos not in self._position_map
    
    def get_entity_at(self, pos: Position) -> Optional[Entity]:
        """Get entity at given position."""
        return self._position_map.get(pos)
    
    def add_rock(self, pos: Position) -> bool:
        """Add rock at position if empty."""
        if not self.is_valid_position(pos) or not self.is_empty(pos):
            return False
        rock = Rock(pos)
        self.rocks.add(rock)
        self._position_map[pos] = rock
        return True
    
    def add_food(self, pos: Position) -> bool:
        """Add food at position if empty."""
        if not self.is_valid_position(pos) or not self.is_empty(pos):
            return False
        food = Food(pos)
        self.food.add(food)
        self._position_map[pos] = food
        return True
    
    def remove_food(self, food: Food):
        """Remove food from board."""
        self.food.discard(food)
        self._position_map.pop(food.position, None)
    
    def add_anthill(self, pos: Position, player_id: int) -> Optional[Anthill]:
        """Add anthill for player at position."""
        if not self.is_valid_position(pos) or not self.is_empty(pos):
            return None
        anthill = Anthill(pos, player_id)
        self.anthills[player_id] = anthill
        self._position_map[pos] = anthill
        return anthill
    
    def add_ant(self, pos: Position, player_id: int) -> Optional[Ant]:
        """Add ant at position."""
        if not self.is_valid_position(pos) or not self.is_empty(pos):
            return None
        ant = Ant(pos, player_id, self._next_ant_id)
        self._next_ant_id += 1
        self.ants[ant.ant_id] = ant
        self._position_map[pos] = ant
        return ant
    
    def move_ant(self, ant: Ant, new_pos: Position) -> bool:
        """Move ant to new position. Returns False if move invalid."""
        if not self.is_valid_position(new_pos):
            return False
        
        # Remove ant from old position (if it exists there)
        if ant.position in self._position_map and self._position_map[ant.position] == ant:
            del self._position_map[ant.position]
        
        # Update ant position
        ant.position = new_pos
        
        # Add to new position (collision handling done elsewhere)
        self._position_map[new_pos] = ant
        return True
    
    def remove_ant(self, ant: Ant):
        """Remove ant from board."""
        ant.alive = False
        self.ants.pop(ant.ant_id, None)
        self._position_map.pop(ant.position, None)
    
    def get_ants_by_player(self, player_id: int) -> List[Ant]:
        """Get all alive ants for a player."""
        return [ant for ant in self.ants.values() 
                if ant.player_id == player_id and ant.alive]
    
    def get_vision(self, pos: Position, radius: int = 1) -> Dict[Position, Optional[str]]:
        """
        Get visible entities around a position.
        Returns dict of position -> entity type string or None if empty.
        """
        vision = {}
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = Position(pos.x + dx, pos.y + dy)
                if not self.is_valid_position(check_pos):
                    vision[check_pos] = 'wall'
                elif check_pos in self._position_map:
                    entity = self._position_map[check_pos]
                    if isinstance(entity, Rock):
                        vision[check_pos] = 'rock'
                    elif isinstance(entity, Food):
                        vision[check_pos] = 'food'
                    elif isinstance(entity, Ant):
                        vision[check_pos] = f'ant_{entity.player_id}'
                    elif isinstance(entity, Anthill):
                        vision[check_pos] = f'anthill_{entity.player_id}'
                else:
                    vision[check_pos] = None
        return vision
    
    def find_empty_neighbor(self, center: Position, max_distance: int = 1) -> Optional[Position]:
        """Find random empty position near center within max_distance."""
        positions = []
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                pos = Position(center.x + dx, center.y + dy)
                if self.is_valid_position(pos) and self.is_empty(pos):
                    positions.append(pos)
        
        return random.choice(positions) if positions else None
    
    def get_random_empty_position(self) -> Optional[Position]:
        """Get random empty position on board."""
        empty_positions = []
        for x in range(self.width):
            for y in range(self.height):
                pos = Position(x, y)
                if self.is_empty(pos):
                    empty_positions.append(pos)
        
        return random.choice(empty_positions) if empty_positions else None