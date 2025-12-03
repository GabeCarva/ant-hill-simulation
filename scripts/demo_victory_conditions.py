"""Demo showing different victory conditions."""

import sys
import os

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.utils.game_config import GameConfig, Position, Action
from src.core.game import Game
from src.visualization.ascii_viz import ASCIIVisualizer
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.environments.standard import StandardEnvironment
from demo_simple import get_victory_reason


def demo_offensive_victory():
    """Demo where one player destroys the enemy anthill."""
    print("\n" + "=" * 60)
    print("DEMO 1: Offensive Victory (Destroying Enemy Anthill)")
    print("=" * 60)
    
    config = GameConfig(
        board_width=15,
        board_height=10,
        food_density=0.1,
        rock_density=0.03,
        initial_ants_per_player=3,
        max_turns=100
    )
    
    env = StandardEnvironment(config=config, seed=0)
    agent1 = RandomAgent(player_id=0, config={'move_probability': 0.7})
    agent2 = SmartRandomAgent(player_id=1, config={
        'prefer_food': False,
        'attack_probability': 0.9  # Very aggressive
    })
    
    viz = ASCIIVisualizer(use_colors=False)
    observations = env.reset()
    
    print("\nInitial State:")
    print(viz.render(env.game, clear_screen=False))
    
    # Run until game ends
    for turn in range(100):
        actions = {
            0: agent1.get_actions(observations[0], env.game.get_state()),
            1: agent2.get_actions(observations[1], env.game.get_state())
        }
        
        result = env.step(actions)
        
        if result.done:
            print(f"\nFinal State (Turn {turn + 1}):")
            print(viz.render(env.game, clear_screen=False))
            print(f"\nVictory: {get_victory_reason(env, result)}")
            break
        
        observations = result.observations


def demo_timeout_victory():
    """Demo where game ends by timeout with food difference."""
    print("\n" + "=" * 60)
    print("DEMO 2: Timeout Victory (More Food Collected)")
    print("=" * 60)
    
    config = GameConfig(
        board_width=20,
        board_height=15,
        food_density=0.15,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=20  # Short game for timeout
    )
    
    env = StandardEnvironment(config=config, seed=42)
    agent1 = RandomAgent(player_id=0, config={
        'move_probability': 0.9,
        'prefer_food': True  # Focus on food
    })
    agent2 = RandomAgent(player_id=1, config={
        'move_probability': 0.6,
        'prefer_food': False  # Less focused
    })
    
    viz = ASCIIVisualizer(use_colors=False)
    observations = env.reset()
    
    print("\nInitial State:")
    print(viz.render(env.game, clear_screen=False))
    
    # Run exactly max_turns
    for turn in range(config.max_turns):
        actions = {
            0: agent1.get_actions(observations[0], env.game.get_state()),
            1: agent2.get_actions(observations[1], env.game.get_state())
        }
        
        result = env.step(actions)
        observations = result.observations
        
        if result.done:
            print(f"\nFinal State (Turn {turn + 1}):")
            print(viz.render(env.game, clear_screen=False))
            print(f"\nVictory: {get_victory_reason(env, result)}")
            break


def demo_draw():
    """Demo where game ends in a draw."""
    print("\n" + "=" * 60)
    print("DEMO 3: Draw (Equal Food at Timeout)")
    print("=" * 60)
    
    # Create a very small game with no food to force a draw
    config = GameConfig(
        board_width=10,
        board_height=10,
        food_density=0.0,  # No food!
        rock_density=0.1,
        initial_ants_per_player=2,
        max_turns=10  # Very short
    )
    
    env = StandardEnvironment(config=config, seed=123)
    agent1 = RandomAgent(player_id=0)
    agent2 = RandomAgent(player_id=1)
    
    viz = ASCIIVisualizer(use_colors=False)
    observations = env.reset()
    
    print("\nInitial State (no food on board):")
    print(viz.render(env.game, clear_screen=False))
    
    # Run exactly max_turns
    for turn in range(config.max_turns):
        actions = {
            0: agent1.get_actions(observations[0], env.game.get_state()),
            1: agent2.get_actions(observations[1], env.game.get_state())
        }
        
        result = env.step(actions)
        observations = result.observations
        
        if result.done:
            print(f"\nFinal State (Turn {turn + 1}):")
            print(viz.render(env.game, clear_screen=False))
            print(f"\nVictory: {get_victory_reason(env, result)}")
            break


def demo_self_destruction():
    """Demo where a player could self-destruct (but agents prevent it)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Self-Destruction Prevention")
    print("=" * 60)
    print("(Agents are programmed to NEVER step on their own anthill)")
    
    # Create scenario where ant is next to its own anthill
    config = GameConfig(
        board_width=5,
        board_height=5,
        food_density=0.0,
        rock_density=0.0,
        initial_ants_per_player=1,
        max_turns=20
    )
    
    game = Game(config, seed=42)
    viz = ASCIIVisualizer(use_colors=False)
    
    # Get player 0's ant and anthill
    ant = list(game.board.get_ants_by_player(0))[0]
    anthill = game.board.anthills[0]
    
    # Move ant right next to its own anthill
    game.board.move_ant(ant, Position(anthill.position.x - 1, anthill.position.y))
    
    print(f"\nAnt positioned next to own anthill:")
    print(f"  Ant at: ({ant.position.x}, {ant.position.y})")
    print(f"  Own Anthill at: ({anthill.position.x}, {anthill.position.y})")
    print(viz.render(game, clear_screen=False))
    
    # Create agent and test
    agent = RandomAgent(player_id=0, config={'move_probability': 1.0})
    
    # Try 10 moves to show agent never moves east to anthill
    print("\nTesting 10 random moves (East would be self-destruction):")
    action_names = {0: 'STAY', 1: 'N', 2: 'NE', 3: 'E', 4: 'SE', 5: 'S', 6: 'SW', 7: 'W', 8: 'NW'}
    
    for i in range(10):
        obs_data = game.get_ant_observation(ant)
        from src.agents.base import AntObservation
        obs = AntObservation(
            ant_id=ant.ant_id,
            player_id=0,
            position=ant.position,
            vision=obs_data['vision'],
            vision_array=None
        )
        
        actions = agent.get_actions([obs], game.get_state())
        action = actions[ant.ant_id]
        print(f"  Move {i+1}: {action_names[action]}")
    
    print("\nNotice: The agent NEVER chooses East (would destroy own anthill)")


def main():
    """Run all demos."""
    print("=" * 60)
    print("AI ANT COLONY - VICTORY CONDITIONS DEMO")
    print("=" * 60)
    
    demo_offensive_victory()
    demo_timeout_victory()
    demo_draw()
    demo_self_destruction()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Victory conditions demonstrated:")
    print("1. Offensive Victory: Destroy enemy anthill")
    print("2. Timeout Victory: Collect more food when time runs out")
    print("3. Draw: Equal food at timeout")
    print("4. Self-Destruction: Would lose if step on own anthill (prevented)")


if __name__ == "__main__":
    main()