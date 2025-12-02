"""Game entities: Rock, Food, Ant, Anthill."""

from dataclasses import dataclass
from typing import Optional
from src.utils.game_config import Position


@dataclass
class Entity:
    """Base class for all game entities."""
    position: Position


@dataclass
class Rock(Entity):
    """Immovable obstacle."""
    
    def __hash__(self):
        return hash(('Rock', self.position.x, self.position.y))
    
    def __eq__(self, other):
        return isinstance(other, Rock) and self.position == other.position


@dataclass
class Food(Entity):
    """Collectible resource that creates new ants."""
    
    def __hash__(self):
        return hash(('Food', self.position.x, self.position.y))
    
    def __eq__(self, other):
        return isinstance(other, Food) and self.position == other.position


@dataclass
class Anthill(Entity):
    """Base for each player. If destroyed, player loses."""
    player_id: int
    alive: bool = True
    
    def __hash__(self):
        return hash(('Anthill', self.position.x, self.position.y, self.player_id))
    
    def __eq__(self, other):
        return isinstance(other, Anthill) and self.position == other.position and self.player_id == other.player_id


@dataclass
class Ant(Entity):
    """Mobile unit that can move and collect food."""
    player_id: int
    ant_id: int  # Unique ID within the game
    alive: bool = True
    
    def __hash__(self):
        return hash(self.ant_id)
    
    def __eq__(self, other):
        return isinstance(other, Ant) and self.ant_id == other.ant_id