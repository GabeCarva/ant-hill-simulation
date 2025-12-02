#!/usr/bin/env python3
"""Integration test demonstrating agents playing in environments."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environments.standard import StandardEnvironment, ShapedRewardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.utils.game_config import GameConfig
import time


def test_agent_vs_agent():
    """Test two agents playing against each other."""
    
    print("=" * 60)
    print("AGENT VS AGENT TEST")
    print("=" * 60)
    
    # Create a small environment for quick testing
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.15,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )
    
    env = StandardEnvironment(config=config, seed=42)
    
    # Create agents
    agent1 = RandomAgent(player_id=0, config={'move_probability': 0.9})
    agent2 = SmartRandomAgent(
        player_id=1, 
        config={
            'move_probability': 0.95,
            'attack_probability': 0.5,
            'prefer_food': True
        }
    )
    
    # Reset environment
    observations = env.reset()
    
    print(f"Starting game:")
    print(f"  Board: {config.board_width}x{config.board_height}")
    print(f"  Food: {config.initial_food_count} pieces")
    print(f"  Rocks: {config.initial_rock_count} pieces")
    print(f"  Initial ants per player: {config.initial_ants_per_player}")
    print()
    
    # Play game
    total_rewards = {0: 0.0, 1: 0.0}
    step_count = 0
    
    while True:
        # Get actions from both agents
        actions = {}
        actions[0] = agent1.get_actions(observations[0], env.game.get_state())
        actions[1] = agent2.get_actions(observations[1], env.game.get_state())
        
        # Step environment
        result = env.step(actions)
        step_count += 1
        
        # Accumulate rewards
        total_rewards[0] += result.rewards[0]
        total_rewards[1] += result.rewards[1]
        
        # Print progress every 20 steps
        if step_count % 20 == 0:
            print(f"Step {step_count}:")
            print(f"  Ants: P0={result.info['ants_alive'][0]}, P1={result.info['ants_alive'][1]}")
            print(f"  Food: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")
            print(f"  Rewards: P0={result.rewards[0]:.2f}, P1={result.rewards[1]:.2f}")
        
        # Check if done
        if result.done:
            print()
            print("=" * 40)
            print("GAME OVER!")
            print("=" * 40)
            
            if result.winner == -1:
                print("Result: DRAW")
            else:
                print(f"Winner: Player {result.winner}")
            
            print(f"\nFinal Statistics:")
            print(f"  Total steps: {step_count}")
            print(f"  Food collected: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")
            print(f"  Ants lost: P0={result.info['ants_lost'][0]}, P1={result.info['ants_lost'][1]}")
            print(f"  Total rewards: P0={total_rewards[0]:.2f}, P1={total_rewards[1]:.2f}")
            break
        
        # Update observations for next step
        observations = result.observations


def test_environment_comparison():
    """Compare different reward structures."""
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT COMPARISON TEST")
    print("=" * 60)
    
    config = GameConfig(
        board_width=15,
        board_height=15,
        food_density=0.20,
        rock_density=0.03,
        initial_ants_per_player=2,
        max_turns=50
    )
    
    environments = [
        ("Standard", StandardEnvironment(config=config, seed=123)),
        ("Shaped", ShapedRewardEnvironment(config=config, seed=123))
    ]
    
    for env_name, env in environments:
        print(f"\n{env_name} Environment:")
        print("-" * 30)
        
        # Create simple agents
        agent1 = RandomAgent(player_id=0)
        agent2 = RandomAgent(player_id=1)
        
        # Reset
        observations = env.reset()
        
        # Play short game
        total_rewards = {0: 0.0, 1: 0.0}
        for step in range(50):
            actions = {
                0: agent1.get_actions(observations[0], env.game.get_state()),
                1: agent2.get_actions(observations[1], env.game.get_state())
            }
            
            result = env.step(actions)
            total_rewards[0] += result.rewards[0]
            total_rewards[1] += result.rewards[1]
            
            if result.done:
                break
            
            observations = result.observations
        
        print(f"  Steps played: {step + 1}")
        print(f"  Total rewards: P0={total_rewards[0]:.2f}, P1={total_rewards[1]:.2f}")
        print(f"  Food collected: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")


def test_visualization():
    """Test game with ASCII visualization."""
    
    print("\n" + "=" * 60)
    print("VISUALIZATION TEST")  
    print("=" * 60)
    print("Running a quick visual game...")
    print()
    
    config = GameConfig(
        board_width=25,
        board_height=15,
        food_density=0.12,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=30
    )
    
    env = StandardEnvironment(config=config, seed=456)
    
    agent1 = SmartRandomAgent(player_id=0, config={'prefer_food': True})
    agent2 = SmartRandomAgent(player_id=1, config={'prefer_food': True})
    
    observations = env.reset()
    
    # Run a few steps with visualization
    for step in range(30):
        # Get actions
        actions = {
            0: agent1.get_actions(observations[0], env.game.get_state()),
            1: agent2.get_actions(observations[1], env.game.get_state())
        }
        
        # Step
        result = env.step(actions)
        
        # Show board every 10 steps
        if step % 10 == 0:
            print(f"\n--- Step {step} ---")
            print(env.render(mode='ascii'))
            time.sleep(0.5)  # Pause to see output
        
        if result.done:
            print(f"\n--- Final State (Step {step}) ---")
            print(env.render(mode='ascii'))
            break
        
        observations = result.observations


if __name__ == "__main__":
    test_agent_vs_agent()
    test_environment_comparison()
    test_visualization()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)