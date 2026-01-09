"""Evaluation script for TacticalAgent vs SmartRandomAgent and others."""

import os
import sys
import argparse
from typing import Dict, List

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.greedy.agent import GreedyAgent, AggressiveGreedyAgent, DefensiveGreedyAgent
from src.agents.tactical.agent import TacticalAgent
from src.visualization.ascii_viz import ASCIIVisualizer


def evaluate_matchup(
    agent1_type: str,
    agent2_type: str,
    num_games: int,
    board_size: tuple = (20, 20),
    verbose: bool = False
) -> Dict:
    """
    Evaluate one agent type against another.

    Args:
        agent1_type: Type of first agent (the one being evaluated)
        agent2_type: Type of second agent (opponent)
        num_games: Number of games to play
        board_size: Board dimensions
        verbose: Print detailed game info

    Returns:
        Dictionary with results
    """
    # Create config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=150
    )

    env = StandardEnvironment(config=config)

    # Create agents
    agent_map = {
        'tactical': TacticalAgent,
        'smart_random': SmartRandomAgent,
        'random': RandomAgent,
        'greedy': GreedyAgent,
        'greedy_aggressive': AggressiveGreedyAgent,
        'greedy_defensive': DefensiveGreedyAgent
    }

    # Stats
    wins = 0
    draws = 0
    losses = 0
    total_food = 0
    total_enemy_food = 0
    total_turns = 0
    total_ants_lost = 0
    total_enemy_ants_lost = 0
    win_methods = {'anthill': 0, 'food': 0, 'draw': 0}

    for game in range(num_games):
        # Create fresh agents
        if agent1_type in agent_map:
            agent1 = agent_map[agent1_type](player_id=0)
        else:
            raise ValueError(f"Unknown agent type: {agent1_type}")

        if agent2_type in agent_map:
            if agent2_type == 'smart_random':
                agent2 = SmartRandomAgent(
                    player_id=1,
                    config={'prefer_food': True, 'attack_probability': 0.3}
                )
            else:
                agent2 = agent_map[agent2_type](player_id=1)
        else:
            raise ValueError(f"Unknown agent type: {agent2_type}")

        # Reset
        observations = env.reset()
        agent1.reset()
        agent2.reset()

        done = False
        turns = 0

        while not done:
            state = env.game.get_state()

            # Get actions
            actions1 = agent1.get_actions(observations[0], state)
            actions2 = agent2.get_actions(observations[1], state)

            # Step
            actions = {0: actions1, 1: actions2}
            result = env.step(actions)

            observations = result.observations
            done = result.done
            turns += 1

        # Record results
        total_turns += turns
        total_food += result.info['food_collected'][0]
        total_enemy_food += result.info['food_collected'][1]
        total_ants_lost += result.info.get('ants_lost', {}).get(0, 0)
        total_enemy_ants_lost += result.info.get('ants_lost', {}).get(1, 0)

        if result.winner == 0:
            wins += 1
            # Determine win method
            if not state.board.anthills[1].alive:
                win_methods['anthill'] += 1
            else:
                win_methods['food'] += 1
        elif result.winner == -1:
            draws += 1
            win_methods['draw'] += 1
        else:
            losses += 1

        if verbose and (game + 1) % 10 == 0:
            print(f"  Games {game + 1}/{num_games}: "
                  f"W:{wins} D:{draws} L:{losses} "
                  f"({wins/(game+1)*100:.1f}% win rate)")

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'loss_rate': losses / num_games,
        'avg_food': total_food / num_games,
        'avg_enemy_food': total_enemy_food / num_games,
        'avg_turns': total_turns / num_games,
        'avg_ants_lost': total_ants_lost / num_games,
        'avg_enemy_ants_lost': total_enemy_ants_lost / num_games,
        'win_by_anthill': win_methods['anthill'],
        'win_by_food': win_methods['food'],
        'total_draws': win_methods['draw']
    }


def main():
    """Main evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate TacticalAgent")

    parser.add_argument(
        "--agent",
        type=str,
        default="tactical",
        choices=['tactical', 'smart_random', 'greedy'],
        help="Agent to evaluate"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per opponent"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Board size"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"AGENT EVALUATION: {args.agent.upper()}")
    print("=" * 80)
    print(f"Games per opponent: {args.games}")
    print(f"Board size: {args.board_size[0]}x{args.board_size[1]}")
    print()

    # Define opponents to test against
    opponents = [
        ('random', 'Random'),
        ('smart_random', 'SmartRandom'),
        ('greedy', 'Greedy'),
        ('greedy_aggressive', 'Greedy (Aggressive)'),
        ('greedy_defensive', 'Greedy (Defensive)')
    ]

    all_results = {}

    for opp_type, opp_name in opponents:
        print(f"Testing vs {opp_name}...")

        results = evaluate_matchup(
            agent1_type=args.agent,
            agent2_type=opp_type,
            num_games=args.games,
            board_size=tuple(args.board_size),
            verbose=args.verbose
        )

        all_results[opp_name] = results

        print(f"\nResults vs {opp_name}:")
        print(f"  Win rate:      {results['win_rate']:.1%}")
        print(f"  Draw rate:     {results['draw_rate']:.1%}")
        print(f"  Loss rate:     {results['loss_rate']:.1%}")
        print(f"  Avg food:      {results['avg_food']:.1f}")
        print(f"  Enemy food:    {results['avg_enemy_food']:.1f}")
        print(f"  Avg turns:     {results['avg_turns']:.1f}")
        print(f"  Ants lost:     {results['avg_ants_lost']:.1f}")
        print(f"  Enemy lost:    {results['avg_enemy_ants_lost']:.1f}")
        print(f"  Win methods:   Anthill:{results['win_by_anthill']} "
              f"Food:{results['win_by_food']} Draw:{results['total_draws']}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall_wins = sum(r['wins'] for r in all_results.values())
    overall_games = args.games * len(opponents)
    overall_win_rate = overall_wins / overall_games

    print(f"Overall win rate: {overall_win_rate:.1%} ({overall_wins}/{overall_games})")
    print()

    # Performance by opponent
    print("Performance by opponent:")
    for opp_name, results in all_results.items():
        print(f"  {opp_name:25s}: {results['win_rate']:6.1%} "
              f"({results['wins']:3d}W / {results['draws']:2d}D / {results['losses']:3d}L)")

    print()

    # Highlight key matchup (vs SmartRandom)
    if 'SmartRandom' in all_results:
        sr_results = all_results['SmartRandom']
        print("=" * 80)
        print(f"KEY MATCHUP: {args.agent.upper()} vs SMARTRANDOM")
        print("=" * 80)
        print(f"Win rate:  {sr_results['win_rate']:.1%}")
        print(f"Avg food:  {sr_results['avg_food']:.1f} vs {sr_results['avg_enemy_food']:.1f}")
        print(f"Food diff: +{sr_results['avg_food'] - sr_results['avg_enemy_food']:.1f} per game")
        print()

        if sr_results['win_rate'] >= 0.70:
            print("✓ DOMINANT PERFORMANCE!")
        elif sr_results['win_rate'] >= 0.60:
            print("✓ Strong performance")
        elif sr_results['win_rate'] >= 0.50:
            print("± Competitive performance")
        else:
            print("✗ Needs improvement")


if __name__ == "__main__":
    main()
