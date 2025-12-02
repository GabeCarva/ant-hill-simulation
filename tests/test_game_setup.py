"""Test script to verify game setup and basic mechanics."""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.game_setup import GameSetup
from src.utils.game_config import GameConfig, Action
from src.core.game import Game
import random


def test_basic_game():
    """Test basic game initialization and run a few turns."""
    
    # Method 1: Using GameSetup.quick_match
    print("Test 1: Quick match setup")
    setup = GameSetup.quick_match(
        board_size=(20, 20),  # Smaller for testing
        food_density=0.15,
        rock_density=0.10,
        max_turns=100,
        player1="random",
        player2="random",
        visualization="ascii"
    )
    
    game = Game(setup.config, seed=42)
    state = game.get_state()
    
    print(f"Board size: {setup.config.board_width}x{setup.config.board_height}")
    print(f"Total squares: {setup.config.total_squares}")
    print(f"Initial food: {setup.config.initial_food_count} ({setup.config.food_density*100:.1f}%)")
    print(f"Initial rocks: {setup.config.initial_rock_count} ({setup.config.rock_density*100:.1f}%)")
    print(f"Max turns: {setup.config.max_turns}")
    print(f"Ants per player: {setup.config.initial_ants_per_player}")
    print()
    
    # Method 2: Custom configuration
    print("Test 2: Custom configuration")
    custom_config = GameConfig(
        board_width=50,
        board_height=50,
        food_density=0.20,
        rock_density=0.03,
        initial_ants_per_player=3,
        max_turns=500
    )
    
    custom_setup = GameSetup(
        config=custom_config,
        player1_agent="dqn",
        player1_model_path="models/dqn_v1.pth",
        player2_agent="ppo",
        player2_model_path="models/ppo_v2.pth",
        visualization="gui"
    )
    
    print(f"Board: {custom_config.board_width}x{custom_config.board_height}")
    print(f"Food count: {custom_config.initial_food_count}")
    print(f"Rock count: {custom_config.initial_rock_count}")
    print(f"Player 1: {custom_setup.player1_agent} ({custom_setup.player1_model_path})")
    print(f"Player 2: {custom_setup.player2_agent} ({custom_setup.player2_model_path})")
    print(f"Visualization: {custom_setup.visualization}")
    print()
    
    # Method 3: Test actual game play
    print("Test 3: Running a mini game")
    mini_setup = GameSetup.quick_match(
        board_size=(10, 10),
        food_density=0.10,
        rock_density=0.05,
        max_turns=20
    )
    
    mini_game = Game(mini_setup.config, seed=123)
    
    # Run some turns with random actions
    for turn in range(20):
        # Get all ants and give them random actions
        for player_id in [0, 1]:
            ants = mini_game.board.get_ants_by_player(player_id)
            for ant in ants:
                action = random.choice(list(Action.DIRECTIONS.keys()))
                mini_game.set_ant_action(ant.ant_id, action)
        
        state = mini_game.step()
        
        if turn % 5 == 0:
            print(f"Turn {turn}: Ants P0={len(mini_game.board.get_ants_by_player(0))}, "
                  f"P1={len(mini_game.board.get_ants_by_player(1))}, "
                  f"Food collected P0={mini_game.food_collected[0]}, "
                  f"P1={mini_game.food_collected[1]}")
        
        if state.is_terminal():
            if state.winner == -1:
                print(f"Game ended in a draw at turn {turn}")
            else:
                print(f"Player {state.winner} wins at turn {turn}!")
            break
    
    if not state.is_terminal():
        # Max turns reached
        if mini_game.food_collected[0] > mini_game.food_collected[1]:
            print(f"Player 0 wins by food collection: {mini_game.food_collected[0]} vs {mini_game.food_collected[1]}")
        elif mini_game.food_collected[1] > mini_game.food_collected[0]:
            print(f"Player 1 wins by food collection: {mini_game.food_collected[1]} vs {mini_game.food_collected[0]}")
        else:
            print(f"Draw by equal food collection: {mini_game.food_collected[0]} each")


if __name__ == "__main__":
    test_basic_game()