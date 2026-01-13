"""Core game logic and state management."""

import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from src.utils.game_config import GameConfig, Position, Action
from src.core.board import Board
from src.core.entities import Ant, Food, Rock


@dataclass
class GameState:
    """Current game state snapshot."""
    turn: int
    winner: Optional[int]  # None if game ongoing, player_id if won, -1 if draw
    board: Board
    food_collected: Dict[int, int]  # player_id -> food count
    ants_lost: Dict[int, int]  # player_id -> ants lost count
    distance_traveled: Dict[int, int] = None  # player_id -> total distance
    win_condition: Optional[str] = None  # 'anthill_kill', 'anthill_suicide', 'timeout'

    def is_terminal(self) -> bool:
        return self.winner is not None


class Game:
    """Main game controller handling rules and state transitions."""
    
    def __init__(self, config: GameConfig, seed: Optional[int] = None):
        self.config = config
        if seed is not None:
            random.seed(seed)
        
        self.board = Board(config)
        self.turn = 0
        self.winner = None
        
        # Statistics
        self.food_collected = defaultdict(int)
        self.ants_lost = defaultdict(int)
        self.distance_traveled = defaultdict(int)  # Total distance traveled per player
        self.win_condition = None  # 'anthill_kill', 'anthill_suicide', 'timeout', or None
        
        # Action queue for simultaneous movement
        self.pending_actions: Dict[int, int] = {}  # ant_id -> action
        
        self._initialize_board()
    
    def _initialize_board(self):
        """Set up initial board state."""
        # Place anthills for two players
        self._place_anthills()
        
        # Place initial ants
        for player_id in [0, 1]:
            anthill = self.board.anthills[player_id]
            for _ in range(self.config.initial_ants_per_player):
                pos = self.board.find_empty_neighbor(
                    anthill.position, 
                    self.config.spawn_radius
                )
                if pos:
                    self.board.add_ant(pos, player_id)
        
        # Place rocks
        for _ in range(self.config.initial_rock_count):
            pos = self.board.get_random_empty_position()
            if pos:
                self.board.add_rock(pos)
        
        # Place food
        for _ in range(self.config.initial_food_count):
            pos = self.board.get_random_empty_position()
            if pos:
                self.board.add_food(pos)
    
    def _place_anthills(self):
        """Place anthills with minimum distance constraint."""
        width = self.config.board_width
        height = self.config.board_height
        
        # Try to place anthills at opposite corners first
        positions = [
            Position(2, 2),
            Position(width - 3, height - 3)
        ]
        
        # If corners don't work, try random positions
        max_attempts = 100
        for attempt in range(max_attempts):
            if all(self.board.is_empty(pos) for pos in positions):
                # Check distance constraint
                if positions[0].manhattan_distance(positions[1]) >= self.config.min_anthill_distance:
                    break
            
            # Generate new random positions
            positions = [
                Position(random.randint(1, width - 2), random.randint(1, height - 2)),
                Position(random.randint(1, width - 2), random.randint(1, height - 2))
            ]
        
        # Place anthills
        for player_id, pos in enumerate(positions):
            self.board.add_anthill(pos, player_id)
    
    def get_state(self) -> GameState:
        """Get current game state snapshot."""
        return GameState(
            turn=self.turn,
            winner=self.winner,
            board=self.board,
            food_collected={0: self.food_collected[0], 1: self.food_collected[1]},
            ants_lost={0: self.ants_lost[0], 1: self.ants_lost[1]},
            distance_traveled={0: self.distance_traveled[0], 1: self.distance_traveled[1]},
            win_condition=self.win_condition
        )
    
    def get_ant_observation(self, ant: Ant) -> Dict:
        """Get observation for a single ant."""
        return {
            'ant_id': ant.ant_id,
            'player_id': ant.player_id,
            'position': ant.position,  # Return Position object, not tuple
            'vision': self.board.get_vision(ant.position, self.config.ant_vision_radius)
        }
    
    def set_ant_action(self, ant_id: int, action: int):
        """Queue action for an ant."""
        if ant_id in self.board.ants and self.board.ants[ant_id].alive:
            self.pending_actions[ant_id] = action
    
    def step(self) -> GameState:
        """Execute one game turn with all pending actions."""
        if self.winner is not None:
            return self.get_state()
        
        # Phase 1: Collect all new positions
        new_positions: Dict[int, Position] = {}  # ant_id -> new position
        position_occupants: Dict[Position, List[int]] = defaultdict(list)  # position -> [ant_ids]
        
        for ant_id, ant in self.board.ants.items():
            if not ant.alive:
                continue
            
            action = self.pending_actions.get(ant_id, Action.STAY)
            new_pos = Action.get_new_position(ant.position, action)
            
            # Check if move is valid (not into wall or rock)
            if not self.board.is_valid_position(new_pos):
                new_pos = ant.position  # Stay in place
            elif isinstance(self.board.get_entity_at(new_pos), Rock):
                new_pos = ant.position  # Stay in place
            
            new_positions[ant_id] = new_pos
            position_occupants[new_pos].append(ant_id)
        
        # Phase 2: Resolve collisions (multiple ants moving to same square)
        ants_to_remove = set()
        food_spawn_positions = []
        
        for pos, ant_ids in position_occupants.items():
            if len(ant_ids) > 1:
                # Multiple ants collide - all die, spawn food
                for ant_id in ant_ids:
                    ant = self.board.ants[ant_id]
                    self.ants_lost[ant.player_id] += 1
                    ants_to_remove.add(ant_id)
                food_spawn_positions.append(pos)
        
        # Phase 3: Move surviving ants and check interactions
        for ant_id, new_pos in new_positions.items():
            if ant_id in ants_to_remove:
                continue
            
            ant = self.board.ants[ant_id]
            entity_at_target = self.board.get_entity_at(new_pos)
            
            # Check for anthill collision
            if entity_at_target and entity_at_target.__class__.__name__ == 'Anthill':
                anthill = entity_at_target
                if anthill.player_id != ant.player_id:
                    # Enemy ant reached anthill - game over
                    anthill.alive = False
                    self.winner = ant.player_id  # Ant's player wins
                    self.win_condition = 'anthill_kill'
                    break
                else:
                    # Ant stepped on own anthill - that player loses!
                    anthill.alive = False
                    self.winner = 1 - ant.player_id  # Other player wins
                    self.win_condition = 'anthill_suicide'
                    break
            
            # Check for food collection
            if entity_at_target and isinstance(entity_at_target, Food):
                # Collect food
                self.board.remove_food(entity_at_target)
                self.food_collected[ant.player_id] += 1
                
                # Spawn new ant near anthill
                anthill = self.board.anthills[ant.player_id]
                spawn_pos = self.board.find_empty_neighbor(
                    anthill.position,
                    self.config.spawn_radius
                )
                if spawn_pos:
                    self.board.add_ant(spawn_pos, ant.player_id)
            
            # Move ant and track distance
            if ant.ant_id not in ants_to_remove:
                # Calculate distance traveled (Chebyshev distance for 8-directional movement)
                distance = max(abs(new_pos.x - ant.position.x), abs(new_pos.y - ant.position.y))
                self.distance_traveled[ant.player_id] += distance
                self.board.move_ant(ant, new_pos)
        
        # Phase 4: Remove dead ants and spawn collision food
        for ant_id in ants_to_remove:
            if ant_id in self.board.ants:
                self.board.remove_ant(self.board.ants[ant_id])
        
        for pos in food_spawn_positions:
            if self.board.is_empty(pos):
                self.board.add_food(pos)
        
        # Clear action queue
        self.pending_actions.clear()
        
        # Increment turn
        self.turn += 1
        
        # Check for draw condition
        if self.config.max_turns is not None and self.turn >= self.config.max_turns:
            # Game ends - winner is player with most food collected
            self.win_condition = 'timeout'
            if self.food_collected[0] > self.food_collected[1]:
                self.winner = 0
            elif self.food_collected[1] > self.food_collected[0]:
                self.winner = 1
            else:
                self.winner = -1  # Draw if equal food
        
        return self.get_state()
    
    def reset(self, seed: Optional[int] = None) -> GameState:
        """Reset game to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.board = Board(self.config)
        self.turn = 0
        self.winner = None
        self.food_collected.clear()
        self.ants_lost.clear()
        self.pending_actions.clear()
        
        self._initialize_board()
        return self.get_state()