"""Training script for AdaptiveQLearningAgent with mixed opponents and evaluation."""

import os
import sys
import time
import json
import argparse
from typing import Optional, Dict, List
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.greedy.agent import GreedyAgent, AggressiveGreedyAgent, DefensiveGreedyAgent
from src.agents.adaptive_q_learning.agent import AdaptiveQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer


def create_opponent(opponent_type: str, player_id: int = 1):
    """Create opponent agent based on type."""
    opponents_map = {
        'random': RandomAgent(player_id=player_id),
        'smart_random': SmartRandomAgent(
            player_id=player_id,
            config={'prefer_food': True, 'attack_probability': 0.3}
        ),
        'greedy': GreedyAgent(player_id=player_id),
        'greedy_aggressive': AggressiveGreedyAgent(player_id=player_id),
        'greedy_defensive': DefensiveGreedyAgent(player_id=player_id)
    }

    if opponent_type not in opponents_map:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    return opponents_map[opponent_type]


def evaluate_against_opponents(
    agent: AdaptiveQLearningAgent,
    env: StandardEnvironment,
    opponent_types: List[str],
    num_games: int = 20
) -> Dict[str, Dict]:
    """
    Evaluate agent against multiple opponent types.

    Args:
        agent: Agent to evaluate
        env: Environment
        opponent_types: List of opponent type names
        num_games: Games to play against each opponent

    Returns:
        Dictionary mapping opponent_type -> stats
    """
    agent.train_mode(False)  # Disable exploration
    results = {}

    for opp_type in opponent_types:
        opponent = create_opponent(opp_type)

        wins = 0
        draws = 0
        total_food = 0
        total_steps = 0

        for _ in range(num_games):
            observations = env.reset()
            agent.reset()
            opponent.reset()

            done = False
            steps = 0

            while not done:
                state = env.game.get_state()
                agent_actions = agent.get_actions(observations[0], state)
                opponent_actions = opponent.get_actions(observations[1], state)

                actions = {0: agent_actions, 1: opponent_actions}
                result = env.step(actions)

                observations = result.observations
                done = result.done
                steps += 1

            if result.winner == 0:
                wins += 1
            elif result.winner == -1:
                draws += 1

            total_food += result.info['food_collected'][0]
            total_steps += steps

        results[opp_type] = {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': (num_games - wins - draws) / num_games,
            'avg_food': total_food / num_games,
            'avg_steps': total_steps / num_games
        }

    agent.train_mode(True)  # Re-enable training
    return results


def train_adaptive_agent(
    episodes: int = 10000,
    board_size: tuple = (20, 20),
    opponent_mix: List[str] = None,
    learning_rate: float = 0.3,
    learning_rate_end: float = 0.001,
    learning_rate_decay: str = "polynomial",
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9994,
    save_freq: int = 500,
    eval_freq: int = 200,
    eval_games: int = 20,
    render: bool = False,
    model_path: Optional[str] = None
):
    """
    Train an AdaptiveQLearningAgent with mixed opponents.

    Args:
        episodes: Number of training episodes
        board_size: Board dimensions
        opponent_mix: List of opponent types to train against (randomly selected)
        learning_rate: Initial learning rate
        learning_rate_end: Final learning rate
        learning_rate_decay: Type of LR decay ('exponential', 'polynomial', 'step', 'none')
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay multiplier
        save_freq: Save model every N episodes
        eval_freq: Evaluate every N episodes
        eval_games: Number of games per opponent in evaluation
        render: Whether to render games
        model_path: Path to pre-trained model
    """

    # Default opponent mix if not specified
    if opponent_mix is None:
        opponent_mix = ['greedy', 'smart_random']

    print("=" * 80)
    print("ADAPTIVE Q-LEARNING TRAINING")
    print("=" * 80)
    print(f"Board size: {board_size[0]}x{board_size[1]}")
    print(f"Episodes: {episodes}")
    print(f"Opponent mix: {', '.join(opponent_mix)}")
    print(f"Learning rate: {learning_rate} -> {learning_rate_end} ({learning_rate_decay} decay)")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    print()

    # Create game config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=150
    )

    # Create environment
    env = StandardEnvironment(config=config)

    # Create agent
    agent = AdaptiveQLearningAgent(
        player_id=0,
        learning_rate=learning_rate,
        learning_rate_end=learning_rate_end,
        learning_rate_decay_type=learning_rate_decay,
        gamma=0.95,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        max_episodes=episodes,
        model_path=model_path
    )

    # Training stats
    stats = {
        'episodes': [],
        'rewards': [],
        'wins': [],
        'draws': [],
        'losses': [],
        'food_collected': [],
        'steps': [],
        'learning_rates': [],
        'epsilons': [],
        'q_table_sizes': [],
        'opponent_types': [],
        'eval_results': []  # Periodic evaluation results
    }

    # Visualizer
    viz = ASCIIVisualizer(use_colors=False) if render else None

    # All opponent types for evaluation
    eval_opponent_types = ['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive']

    # Training loop
    start_time = time.time()

    for episode in range(episodes):
        # Start episode (updates learning rate and epsilon)
        agent.start_episode()

        # Select random opponent from mix
        opponent_type = np.random.choice(opponent_mix)
        opponent = create_opponent(opponent_type)

        # Reset
        observations = env.reset()
        agent.reset()
        opponent.reset()

        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            # Get current state
            state = env.game.get_state()

            # Get actions
            agent_actions = agent.get_actions(observations[0], state)
            opponent_actions = opponent.get_actions(observations[1], state)

            # Step
            actions = {0: agent_actions, 1: opponent_actions}
            result = env.step(actions)

            # Update Q-values
            ant_rewards = {}
            for ant_id in agent_actions.keys():
                ant_rewards[ant_id] = result.rewards[0]

            agent.update_q_values(ant_rewards, result.observations[0], result.done)

            episode_reward += result.rewards[0]
            episode_steps += 1

            # Render if requested
            if render and episode % 100 == 0 and episode_steps % 10 == 0:
                print("\033[2J\033[H")  # Clear screen
                print(viz.render(env.game, clear_screen=False))
                print(f"Episode: {episode}, Step: {episode_steps}")
                print(f"Opponent: {opponent_type}")
                time.sleep(0.05)

            observations = result.observations
            done = result.done

        # Record stats
        stats['episodes'].append(episode)
        stats['rewards'].append(episode_reward)
        stats['wins'].append(1 if result.winner == 0 else 0)
        stats['draws'].append(1 if result.winner == -1 else 0)
        stats['losses'].append(1 if result.winner == 1 else 0)
        stats['food_collected'].append(result.info['food_collected'][0])
        stats['steps'].append(episode_steps)
        stats['learning_rates'].append(agent.learning_rate)
        stats['epsilons'].append(agent.epsilon)
        stats['q_table_sizes'].append(len(agent.q_table))
        stats['opponent_types'].append(opponent_type)

        # Periodic evaluation
        if (episode + 1) % eval_freq == 0 and episode > 0:
            print(f"\n{'='*80}")
            print(f"Episode {episode + 1}/{episodes} - Evaluation")
            print(f"{'='*80}")

            # Training stats
            window = min(eval_freq, episode + 1)
            recent_wins = np.mean(stats['wins'][-window:])
            recent_draws = np.mean(stats['draws'][-window:])
            recent_reward = np.mean(stats['rewards'][-window:])
            recent_food = np.mean(stats['food_collected'][-window:])

            print(f"\nTraining performance (last {window} episodes):")
            print(f"  Win rate:   {recent_wins:.2%}")
            print(f"  Draw rate:  {recent_draws:.2%}")
            print(f"  Avg reward: {recent_reward:.2f}")
            print(f"  Avg food:   {recent_food:.1f}")
            print(f"\nAgent state:")
            print(f"  Learning rate: {agent.learning_rate:.4f}")
            print(f"  Epsilon:       {agent.epsilon:.4f}")
            print(f"  Q-table size:  {len(agent.q_table)} states")

            # Evaluate against all opponent types
            print(f"\nEvaluation vs all opponents ({eval_games} games each):")
            eval_results = evaluate_against_opponents(
                agent, env, eval_opponent_types, num_games=eval_games
            )

            for opp_type, results in eval_results.items():
                print(f"  {opp_type:20s}: "
                      f"Win {results['win_rate']:5.1%} | "
                      f"Draw {results['draw_rate']:5.1%} | "
                      f"Loss {results['loss_rate']:5.1%} | "
                      f"Food {results['avg_food']:4.1f}")

            # Store evaluation results
            stats['eval_results'].append({
                'episode': episode + 1,
                'results': eval_results
            })

            print()

        # Save periodically
        if (episode + 1) % save_freq == 0:
            save_path = f"models/adaptive_q_ep{episode + 1}.pkl"
            os.makedirs("models", exist_ok=True)
            agent.save(save_path)

            # Save stats
            stats_path = "logs/adaptive_q_stats.json"
            os.makedirs("logs", exist_ok=True)
            with open(stats_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_stats = {}
                for key, value in stats.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
                        json_stats[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
                    else:
                        json_stats[key] = value
                json.dump(json_stats, f, indent=2)

    # Final save
    final_path = "models/adaptive_q_final.pkl"
    agent.save(final_path)

    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")

    final_eval = evaluate_against_opponents(
        agent, env, eval_opponent_types, num_games=100
    )

    for opp_type, results in final_eval.items():
        print(f"  {opp_type:20s}: "
              f"Win {results['win_rate']:5.1%} | "
              f"Draw {results['draw_rate']:5.1%} | "
              f"Loss {results['loss_rate']:5.1%} | "
              f"Food {results['avg_food']:4.1f}")

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/episodes:.2f} sec/episode)")
    print(f"Final Q-table size: {len(agent.q_table)} states")
    print(f"Final learning rate: {agent.learning_rate:.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"\nModel saved to: {final_path}")
    print(f"Stats saved to: logs/adaptive_q_stats.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train AdaptiveQLearningAgent with mixed opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (mixed opponents)
  python training/train_adaptive.py --episodes 10000

  # Train only against GreedyAgent
  python training/train_adaptive.py --episodes 10000 --opponents greedy

  # Train with multiple opponent types
  python training/train_adaptive.py --episodes 20000 --opponents greedy smart_random greedy_aggressive

  # Custom learning rate decay
  python training/train_adaptive.py --episodes 10000 --lr 0.5 --lr-end 0.0001 --lr-decay exponential

  # Higher minimum exploration
  python training/train_adaptive.py --episodes 10000 --epsilon-end 0.1

  # Continue from checkpoint
  python training/train_adaptive.py --episodes 5000 --load models/adaptive_q_ep10000.pkl
        """
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Board size (width height)"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs='+',
        default=['greedy', 'smart_random'],
        choices=['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive'],
        help="Opponent types to train against (randomly selected each episode)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.3,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=0.001,
        help="Final learning rate"
    )
    parser.add_argument(
        "--lr-decay",
        type=str,
        default="polynomial",
        choices=['exponential', 'polynomial', 'step', 'none'],
        help="Learning rate decay type"
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon"
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon"
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9994,
        help="Epsilon decay multiplier"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=500,
        help="Save model every N episodes"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=200,
        help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="Number of games per opponent during evaluation"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load pre-trained model to continue training"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render games during training"
    )

    args = parser.parse_args()

    train_adaptive_agent(
        episodes=args.episodes,
        board_size=tuple(args.board_size),
        opponent_mix=args.opponents,
        learning_rate=args.lr,
        learning_rate_end=args.lr_end,
        learning_rate_decay=args.lr_decay,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        render=args.render,
        model_path=args.load
    )


if __name__ == "__main__":
    main()
