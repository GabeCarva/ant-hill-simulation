#!/usr/bin/env python3
"""Demo script showing a game with ASCII visualization."""

import sys
import os
import time
import random

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.game_setup import GameSetup
from src.utils.game_config import Action
from src.core.game import Game
from src.visualization.ascii_viz import ASCIIVisualizer
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.base import AntObservation


def get_victory_details(game, state):
    """Get detailed victory information."""
    if state.winner == -1:
        # Draw
        p0_food = state.food_collected[0]
        p1_food = state.food_collected[1]
        if p0_food == p1_food:
            return f"DRAW - Both players collected {p0_food} food after {state.turn} turns"
        else:
            return f"DRAW - P0: {p0_food} food, P1: {p1_food} food after {state.turn} turns"
    
    # Check if an anthill was destroyed
    for player_id in [0, 1]:
        if not game.board.anthills[player_id].alive:
            if player_id == state.winner:
                # Winner's anthill destroyed? Must be self-destruction by opponent
                other_player = 1 - player_id
                return f"PLAYER {state.winner} WINS! Player {other_player} self-destructed (stepped on own anthill)"
            else:
                # Normal offensive victory
                return f"PLAYER {state.winner} WINS! Successfully destroyed Player {player_id}'s anthill"
    
    # No anthill destroyed - timeout victory
    p0_food = state.food_collected[0]
    p1_food = state.food_collected[1]
    winner_food = p0_food if state.winner == 0 else p1_food
    loser_food = p1_food if state.winner == 0 else p0_food
    return f"PLAYER {state.winner} WINS by timeout! More food collected ({winner_food} vs {loser_food})"


def run_demo():
    """Run a demo game with visualization."""
    
    # Create a small game for clear visualization
    setup = GameSetup.quick_match(
        board_size=(40, 20),  # Small enough to see everything
        food_density=0.10,
        rock_density=0.05,
        max_turns=200,
        player1="random",
        player2="smart_random",
        visualization="ascii"
    )
    
    # Initialize game
    game = Game(setup.config, seed=random.randint(0, 10000))
    viz = ASCIIVisualizer(use_colors=True)
    
    # Create agents
    agent0 = RandomAgent(player_id=0, config={
        'move_probability': 0.85,
        'prefer_food': False
    })
    agent1 = SmartRandomAgent(player_id=1, config={
        'prefer_food': True,
        'attack_probability': 0.4,
        'retreat_when_outnumbered': True
    })
    
    print("AI Ant Colony - Demo Game")
    print("=" * 40)
    print(f"Board: {setup.config.board_width}x{setup.config.board_height}")
    print(f"Food density: {setup.config.food_density*100:.1f}%")
    print(f"Rock density: {setup.config.rock_density*100:.1f}%")
    print(f"Max turns: {setup.config.max_turns}")
    print()
    print("Agents:")
    print("  Player 0 (Blue): RandomAgent (basic random movement)")
    print("  Player 1 (Red): SmartRandomAgent (seeks food, attacks anthills)")
    print()
    print("Legend:")
    print("  H/h = Anthills (Player 0/1)")
    print("  A/a = Ants (Player 0/1)")
    print("  * = Food")
    print("  # = Rock")
    print()
    print("Victory Conditions:")
    print("  1. Destroy enemy anthill (offensive victory)")
    print("  2. Opponent steps on own anthill (defensive victory)")
    print("  3. Collect more food by timeout")
    print()
    input("Press Enter to start...")
    
    # Game loop
    turn = 0
    while turn < setup.config.max_turns:
        # Clear screen and show current state
        print("\033[2J\033[H")  # Clear screen
        print(viz.render(game, clear_screen=False))
        
        # Check if game is over
        if game.get_state().is_terminal():
            break
        
        # Get observations for each player's ants
        observations = {0: [], 1: []}
        for ant_id, ant in game.board.ants.items():
            if ant.alive:
                obs_data = game.get_ant_observation(ant)
                # Create AntObservation object
                obs = AntObservation(
                    ant_id=obs_data['ant_id'],
                    player_id=obs_data['player_id'],
                    position=obs_data['position'],
                    vision=obs_data['vision'],
                    vision_array=None  # Agents will compute if needed
                )
                observations[ant.player_id].append(obs)
        
        # Get actions from agents
        state = game.get_state()
        actions_p0 = agent0.get_actions(observations[0], state) if observations[0] else {}
        actions_p1 = agent1.get_actions(observations[1], state) if observations[1] else {}
        
        # Set actions in game
        for ant_id, action in actions_p0.items():
            game.set_ant_action(ant_id, action)
        for ant_id, action in actions_p1.items():
            game.set_ant_action(ant_id, action)
        
        # Step the game
        game.step()
        turn += 1
        
        # Save snapshot every 50 turns
        if turn % 50 == 0:
            filename = f"snapshot_turn_{turn}.txt"
            viz.render_to_file(game, filename)
            
        # Small delay for animation
        time.sleep(0.05)
    
    # Game over - show final state
    print("\033[2J\033[H")  # Clear screen
    print(viz.render(game, clear_screen=False))
    
    print("\n" + "=" * 40)
    print("GAME OVER!")
    print("=" * 40)
    
    # Get detailed victory information
    final_state = game.get_state()
    victory_details = get_victory_details(game, final_state)
    print(f"\n{victory_details}")
    
    print(f"\nFinal Statistics:")
    print(f"  Total turns: {final_state.turn}")
    print(f"  Food collected: Player 0={final_state.food_collected[0]}, Player 1={final_state.food_collected[1]}")
    print(f"  Ants lost: Player 0={final_state.ants_lost[0]}, Player 1={final_state.ants_lost[1]}")
    
    # Count remaining ants
    ants_alive = {0: 0, 1: 0}
    for ant in game.board.ants.values():
        if ant.alive:
            ants_alive[ant.player_id] += 1
    print(f"  Final ant count: Player 0={ants_alive[0]}, Player 1={ants_alive[1]}")
    
    # Show anthill status
    print(f"\nAnthill Status:")
    for player_id in [0, 1]:
        anthill = game.board.anthills[player_id]
        status = "ALIVE" if anthill.alive else "DESTROYED"
        pos = f"({anthill.position.x}, {anthill.position.y})"
        print(f"  Player {player_id}: {status} at {pos}")
    
    # Save final state
    filename = f"game_final_turn_{turn}.txt"
    viz.render_to_file(game, filename)
    print(f"\nFinal game state saved to {filename}")


if __name__ == "__main__":
    run_demo()