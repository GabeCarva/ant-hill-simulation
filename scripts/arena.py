"""Generic arena script for testing trained agents against opponents.

This script provides an easy interface to:
- Load any trained model
- Fight against various opponents
- Run multiple games with statistics
- Optionally visualize games
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, List

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.adaptive_q_learning.agent import AdaptiveQLearningAgent
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.agents.greedy.agent import GreedyAgent, AggressiveGreedyAgent, DefensiveGreedyAgent
from src.agents.tactical.agent import TacticalAgent
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.visualization.ascii_viz import ASCIIVisualizer


def load_trained_agent(model_path: str, player_id: int):
    """
    Load a trained agent from a .pkl file.

    Automatically detects the agent type and loads it.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Try AdaptiveQLearningAgent first (most common)
    try:
        agent = AdaptiveQLearningAgent(player_id=player_id)
        agent.load(model_path)
        agent.train_mode(False)  # Disable exploration
        return agent, 'AdaptiveQLearningAgent'
    except Exception as e:
        pass

    # Try SimpleQLearningAgent
    try:
        agent = SimpleQLearningAgent(player_id=player_id)
        agent.load(model_path)
        agent.train_mode(False)
        return agent, 'SimpleQLearningAgent'
    except Exception as e:
        pass

    raise ValueError(f"Could not load model from {model_path}. Unsupported agent type.")


def create_opponent(opponent_name: str, player_id: int):
    """Create an opponent agent by name."""
    opponents = {
        'random': RandomAgent(player_id=player_id),
        'smart_random': SmartRandomAgent(player_id=player_id),
        'greedy': GreedyAgent(player_id=player_id),
        'aggressive': AggressiveGreedyAgent(player_id=player_id),
        'defensive': DefensiveGreedyAgent(player_id=player_id),
        'tactical': TacticalAgent(player_id=player_id),
    }

    if opponent_name not in opponents:
        raise ValueError(f"Unknown opponent: {opponent_name}. "
                        f"Available: {', '.join(opponents.keys())}")

    return opponents[opponent_name]


def run_single_game(
    agent1,
    agent2,
    config: GameConfig,
    visualize: bool = False,
    delay: float = 0.3
) -> Dict:
    """
    Run a single game between two agents.

    Returns:
        Dict with game results (winner, turns, food_collected, etc.)
    """
    env = StandardEnvironment(config=config)
    observations = env.reset()

    if visualize:
        viz = ASCIIVisualizer()
        print("\n" + "=" * 70)
        print("INITIAL BOARD")
        print("=" * 70)
        print(viz.render(env.game))
        print()
        time.sleep(delay)

    turn = 0
    done = False

    while not done:
        turn += 1

        # Get actions for both agents
        actions = {}
        game_state = env.game.get_state()

        for player_id in [0, 1]:
            agent = agent1 if player_id == 0 else agent2
            player_actions = {}

            for obs in observations[player_id]:
                action = agent.get_action(obs, game_state)
                player_actions[obs.ant_id] = action

            actions[player_id] = player_actions

        # Execute step
        result = env.step(actions)
        observations = result.observations
        done = result.done

        # Visualize if requested
        if visualize:
            print("\n" + "=" * 70)
            print(f"TURN {turn}")
            print("=" * 70)
            print(viz.render(env.game))

            ants_0 = len([a for a in env.game.board.ants.values() if a.player_id == 0])
            ants_1 = len([a for a in env.game.board.ants.values() if a.player_id == 1])

            print(f"\nFood: P0={env.game.food_collected[0]:3d}  P1={env.game.food_collected[1]:3d}")
            print(f"Ants: P0={ants_0:3d}  P1={ants_1:3d}")

            time.sleep(delay)

    # Compile results
    return {
        'winner': result.winner,
        'turns': turn,
        'food_0': env.game.food_collected[0],
        'food_1': env.game.food_collected[1],
        'ants_lost_0': env.game.ants_lost[0],
        'ants_lost_1': env.game.ants_lost[1],
    }


def run_arena(
    model_path: str,
    opponent: str,
    num_games: int,
    board_size: tuple = (20, 20),
    max_turns: int = 500,
    visualize_first: bool = False,
    visualize_all: bool = False,
    delay: float = 0.3
):
    """
    Run arena matches between a trained model and an opponent.

    Args:
        model_path: Path to trained model (.pkl file)
        opponent: Opponent name (random, greedy, tactical, etc.)
        num_games: Number of games to run
        board_size: Board dimensions (width, height)
        max_turns: Maximum turns per game
        visualize_first: Whether to visualize the first game
        visualize_all: Whether to visualize all games
        delay: Delay between turns when visualizing (seconds)
    """
    print("=" * 70)
    print("ARENA - TRAINED AGENT VS OPPONENT")
    print("=" * 70)

    # Load trained agent
    print(f"\n📦 Loading trained agent from: {model_path}")
    agent1, agent_type = load_trained_agent(model_path, player_id=0)
    print(f"   ✓ Loaded {agent_type}")
    if hasattr(agent1, 'q_table'):
        print(f"   ✓ Q-table size: {len(agent1.q_table)} states")

    # Create opponent
    print(f"\n🎯 Creating opponent: {opponent}")
    agent2 = create_opponent(opponent, player_id=1)
    print(f"   ✓ Opponent ready")

    # Game configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Board size: {board_size[0]}x{board_size[1]}")
    print(f"   Max turns: {max_turns}")
    print(f"   Games: {num_games}")
    print(f"   Visualize: {'First game only' if visualize_first else ('All games' if visualize_all else 'No')}")

    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=max_turns
    )

    # Run games
    print("\n" + "=" * 70)
    print("RUNNING MATCHES")
    print("=" * 70)

    results = []
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(1, num_games + 1):
        # Decide whether to visualize this game
        visualize = visualize_all or (visualize_first and game_num == 1)

        if visualize:
            print(f"\n{'=' * 70}")
            print(f"GAME {game_num}/{num_games} (VISUALIZED)")
            print(f"{'=' * 70}")
        else:
            print(f"\nGame {game_num}/{num_games}... ", end='', flush=True)

        # Reset agents
        agent1.reset()
        agent2.reset()

        # Run game
        result = run_single_game(agent1, agent2, config, visualize=visualize, delay=delay)
        results.append(result)

        # Update statistics
        if result['winner'] == 0:
            wins += 1
            outcome = "WIN"
        elif result['winner'] == 1:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"

        if not visualize:
            print(f"{outcome} (Food: {result['food_0']}-{result['food_1']}, Turns: {result['turns']})")
        else:
            print(f"\n{'=' * 70}")
            print(f"GAME {game_num} RESULT: {outcome}")
            print(f"{'=' * 70}")
            print(f"Winner: {'Your Agent (P0)' if result['winner'] == 0 else ('Opponent (P1)' if result['winner'] == 1 else 'Draw')}")
            print(f"Food: P0={result['food_0']}, P1={result['food_1']}")
            print(f"Turns: {result['turns']}")
            print(f"Ants Lost: P0={result['ants_lost_0']}, P1={result['ants_lost_1']}")
            if game_num < num_games:
                input("\nPress Enter to continue to next game...")

    # Final statistics
    print("\n" + "=" * 70)
    print("ARENA RESULTS")
    print("=" * 70)

    total = num_games
    win_rate = (wins / total) * 100
    loss_rate = (losses / total) * 100
    draw_rate = (draws / total) * 100

    print(f"\n📊 Overall Performance:")
    print(f"   Wins:   {wins:3d} / {total} ({win_rate:5.1f}%)")
    print(f"   Losses: {losses:3d} / {total} ({loss_rate:5.1f}%)")
    print(f"   Draws:  {draws:3d} / {total} ({draw_rate:5.1f}%)")

    # Detailed statistics
    avg_food_0 = sum(r['food_0'] for r in results) / total
    avg_food_1 = sum(r['food_1'] for r in results) / total
    avg_turns = sum(r['turns'] for r in results) / total
    avg_ants_lost_0 = sum(r['ants_lost_0'] for r in results) / total
    avg_ants_lost_1 = sum(r['ants_lost_1'] for r in results) / total

    print(f"\n📈 Average Statistics:")
    print(f"   Your Agent Food:     {avg_food_0:.1f}")
    print(f"   Opponent Food:       {avg_food_1:.1f}")
    print(f"   Game Length:         {avg_turns:.1f} turns")
    print(f"   Your Ants Lost:      {avg_ants_lost_0:.1f}")
    print(f"   Opponent Ants Lost:  {avg_ants_lost_1:.1f}")

    # Performance summary
    print(f"\n🏆 Performance Summary:")
    if win_rate >= 75:
        print("   EXCELLENT! Your agent dominates this opponent.")
    elif win_rate >= 60:
        print("   VERY GOOD! Your agent performs well.")
    elif win_rate >= 50:
        print("   GOOD! Your agent is competitive.")
    elif win_rate >= 40:
        print("   FAIR. Your agent needs improvement.")
    else:
        print("   NEEDS WORK. Consider more training.")

    print("\n" + "=" * 70)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Arena: Test trained agents against opponents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 games against greedy opponent
  python scripts/arena.py --model models/curriculum_hybrid_final.pkl --opponent greedy --games 10

  # Visualize the first game
  python scripts/arena.py --model models/curriculum_hybrid_final.pkl --opponent tactical --games 5 --visualize-first

  # Visualize all games with slower playback
  python scripts/arena.py --model models/curriculum_hybrid_final.pkl --opponent random --games 3 --visualize-all --delay 0.5

  # Large board, many games
  python scripts/arena.py --model models/curriculum_hybrid_final.pkl --opponent aggressive --games 100 --board-size 30 30

Available opponents:
  random, smart_random, greedy, aggressive, defensive, tactical
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model (.pkl file)'
    )

    parser.add_argument(
        '--opponent', '-o',
        type=str,
        default='greedy',
        choices=['random', 'smart_random', 'greedy', 'aggressive', 'defensive', 'tactical'],
        help='Opponent type (default: greedy)'
    )

    parser.add_argument(
        '--games', '-g',
        type=int,
        default=10,
        help='Number of games to run (default: 10)'
    )

    parser.add_argument(
        '--board-size', '-b',
        type=int,
        nargs=2,
        default=[20, 20],
        metavar=('WIDTH', 'HEIGHT'),
        help='Board size (default: 20 20)'
    )

    parser.add_argument(
        '--max-turns', '-t',
        type=int,
        default=500,
        help='Maximum turns per game (default: 500)'
    )

    parser.add_argument(
        '--visualize-first',
        action='store_true',
        help='Visualize only the first game'
    )

    parser.add_argument(
        '--visualize-all',
        action='store_true',
        help='Visualize all games (can be slow)'
    )

    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=0.3,
        help='Delay between turns when visualizing (seconds, default: 0.3)'
    )

    args = parser.parse_args()

    # Run arena
    try:
        run_arena(
            model_path=args.model,
            opponent=args.opponent,
            num_games=args.games,
            board_size=tuple(args.board_size),
            max_turns=args.max_turns,
            visualize_first=args.visualize_first,
            visualize_all=args.visualize_all,
            delay=args.delay
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
