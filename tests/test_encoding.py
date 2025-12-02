#!/usr/bin/env python3
"""Utility to visualize the factored vision encoding."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.utils.game_config import GameConfig, Position
from src.core.game import Game
from src.agents.base import BaseAgent


def create_sample_vision() -> dict:
    """Create a sample vision dictionary for testing."""
    vision = {}
    
    # Create a 3x3 vision grid (radius=1)
    # Layout:
    # [wall] [food] [empty]
    # [ally] [self] [enemy]
    # [rock] [ally_hill] [enemy_hill]
    
    positions = [
        (Position(-1, -1), 'wall'),
        (Position(0, -1), 'food'),
        (Position(1, -1), None),  # empty
        (Position(-1, 0), 'ant_0'),  # ally
        (Position(0, 0), 'ant_0'),  # self (center)
        (Position(1, 0), 'ant_1'),  # enemy
        (Position(-1, 1), 'rock'),
        (Position(0, 1), 'anthill_0'),  # ally anthill
        (Position(1, 1), 'anthill_1'),  # enemy anthill
    ]
    
    for pos, entity in positions:
        vision[pos] = entity
    
    return vision


def print_channel(array: np.ndarray, channel: int, name: str):
    """Pretty print a single channel."""
    print(f"\n{name}:")
    print("-" * 20)
    channel_data = array[:, :, channel]
    
    for row in channel_data:
        row_str = ""
        for val in row:
            if val == 0:
                row_str += "  .  "
            elif val == -1:
                row_str += " -1  "
            elif val == 1:
                row_str += " +1  "
            else:
                row_str += f"{val:4.1f} "
        print(row_str)


def test_vision_encoding():
    """Test and visualize the factored encoding."""
    print("=" * 50)
    print("FACTORED VISION ENCODING VISUALIZATION")
    print("=" * 50)
    
    vision = create_sample_vision()
    player_id = 0  # We are player 0
    
    # Print raw vision
    print("\nRaw Vision Grid:")
    print("-" * 20)
    print("[wall]      [food]      [empty]")
    print("[ally ant]  [self]      [enemy ant]")
    print("[rock]      [ally hill] [enemy hill]")
    
    # Encode vision
    encoded = BaseAgent.encode_vision(vision, player_id, vision_radius=1)
    
    print(f"\nEncoded shape: {encoded.shape}")
    print("Channels: [Entity Type, Team Affiliation, Mobility]")
    
    # Print each channel
    print_channel(encoded, 0, "Channel 0: Entity Type (0=empty, 1=wall, 2=rock, 3=food, 4=ant, 5=anthill)")
    print_channel(encoded, 1, "Channel 1: Team (-1=enemy, 0=neutral, +1=allied)")
    print_channel(encoded, 2, "Channel 2: Mobility (0=blocked, 1=passable)")


def test_in_game():
    """Test encoding with actual game state."""
    print("\n" + "=" * 50)
    print("IN-GAME ENCODING TEST")
    print("=" * 50)
    
    # Create a small game
    config = GameConfig(
        board_width=10,
        board_height=10,
        food_density=0.1,
        rock_density=0.1,
        initial_ants_per_player=2
    )
    
    game = Game(config, seed=42)
    
    # Get an ant
    ants = game.board.get_ants_by_player(0)
    if ants:
        ant = ants[0]
        obs = game.get_ant_observation(ant)
        
        print(f"\nAnt {ant.ant_id} at position ({ant.position.x}, {ant.position.y})")
        print("\nVisible entities:")
        for pos, entity in obs['vision'].items():
            if entity is not None:
                print(f"  ({pos.x:+2}, {pos.y:+2}): {entity}")
        
        # Encode vision
        encoded = BaseAgent.encode_vision(
            obs['vision'], 
            ant.player_id,
            config.ant_vision_radius
        )
        
        print(f"\nEncoded vision shape: {encoded.shape}")
        print_channel(encoded, 0, "Entity Types")
        print_channel(encoded, 1, "Team Affiliations")
        print_channel(encoded, 2, "Mobility Map")


def explain_encoding():
    """Explain the benefits of factored encoding."""
    print("\n" + "=" * 50)
    print("FACTORED ENCODING EXPLAINED")
    print("=" * 50)
    
    print("""
The factored encoding separates entity properties into independent channels:

CHANNEL 0 - Entity Type (categorical):
  0 = Empty space
  1 = Wall (boundary)
  2 = Rock (obstacle)
  3 = Food (resource)
  4 = Ant (unit)
  5 = Anthill (base)

CHANNEL 1 - Team Affiliation (continuous):
  -1.0 = Enemy
   0.0 = Neutral/None
  +1.0 = Allied

CHANNEL 2 - Mobility (binary):
  0 = Blocked (cannot move through)
  1 = Passable (can move through)

BENEFITS:
1. Clean separation of "what" (entity) from "whose" (team)
2. Network can learn entity-team interactions
3. Explicit mobility helps with pathfinding
4. Reduces feature correlation
5. Compact representation (3 channels)

EXAMPLE INTERPRETATIONS:
- Entity=4, Team=+1 → Allied ant (coordinate)
- Entity=4, Team=-1 → Enemy ant (avoid or attack)
- Entity=5, Team=-1 → Enemy anthill (primary target!)
- Entity=3, Team=0  → Food (collect)
- Mobility=0 → Cannot move here (wall or rock)
""")


if __name__ == "__main__":
    test_vision_encoding()
    test_in_game()
    explain_encoding()