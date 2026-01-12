"""
Curriculum-based training for Q-learning agents.

This script implements structured training curricula that progressively build
agent capabilities through multiple phases with different opponents and settings.

Usage Examples:
    # Full standard training (recommended)
    python training/train_curriculum.py --mode standard --episodes 20000

    # Quick testing
    python training/train_curriculum.py --mode rapid --episodes 2000

    # Intensive training for competition
    python training/train_curriculum.py --mode intensive --episodes 50000

    # Custom curriculum
    python training/train_curriculum.py --custom "basics:random:1000,mid:greedy:2000"

    # Continue from checkpoint
    python training/train_curriculum.py --mode standard --episodes 20000 --load models/checkpoint.pkl
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
from src.utils.game_config import GameConfig
from src.utils.agent_helpers import get_agent_actions
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.greedy.agent import GreedyAgent, AggressiveGreedyAgent, DefensiveGreedyAgent
from src.agents.adaptive_q_learning.agent import AdaptiveQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer

from training.curriculum_config import (
    TrainingCurriculum,
    CurriculumPhase,
    get_curriculum,
    parse_custom_curriculum
)


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


def evaluate_agent(
    agent: AdaptiveQLearningAgent,
    env: StandardEnvironment,
    opponent_types: List[str],
    num_games: int = 20,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate agent against multiple opponent types.

    Args:
        agent: Agent to evaluate
        env: Environment
        opponent_types: List of opponent type names
        num_games: Games to play against each opponent
        verbose: Whether to print results

    Returns:
        Dictionary mapping opponent_type -> stats
    """
    agent.train_mode(False)
    results = {}

    for opp_type in opponent_types:
        opponent = create_opponent(opp_type)

        wins = 0
        draws = 0
        total_food = 0
        total_enemy_food = 0
        total_steps = 0

        for _ in range(num_games):
            observations = env.reset()
            agent.reset()
            opponent.reset()

            done = False
            steps = 0

            while not done:
                state = env.game.get_state()
                agent_actions = get_agent_actions(agent, observations[0], state)
                opponent_actions = get_agent_actions(opponent, observations[1], state)

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
            total_enemy_food += result.info['food_collected'][1]
            total_steps += steps

        results[opp_type] = {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': (num_games - wins - draws) / num_games,
            'avg_food': total_food / num_games,
            'avg_enemy_food': total_enemy_food / num_games,
            'avg_steps': total_steps / num_games
        }

        if verbose:
            res = results[opp_type]
            print(f"  {opp_type:20s}: "
                  f"Win {res['win_rate']:5.1%} | "
                  f"Draw {res['draw_rate']:5.1%} | "
                  f"Loss {res['loss_rate']:5.1%} | "
                  f"Food {res['avg_food']:4.1f}")

    agent.train_mode(True)
    return results


def train_phase(
    phase: CurriculumPhase,
    agent: AdaptiveQLearningAgent,
    env: StandardEnvironment,
    phase_number: int,
    total_phases: int,
    stats: Dict,
    render: bool = False,
    eval_opponents: List[str] = None,
    eval_games: int = 20
) -> None:
    """
    Train a single curriculum phase.

    Args:
        phase: Phase configuration
        agent: Agent to train
        env: Environment
        phase_number: Current phase index (1-indexed)
        total_phases: Total number of phases
        stats: Global stats dictionary to update
        render: Whether to render games
        eval_opponents: Opponents to evaluate against
        eval_games: Number of games per opponent in evaluation
    """
    print(f"\n{'='*80}")
    print(f"PHASE {phase_number}/{total_phases}: {phase.name.upper()}")
    print(f"{'='*80}")
    print(f"Episodes: {phase.episodes}")
    print(f"Opponents: {', '.join(phase.opponents)}")
    if phase.description:
        print(f"Goal: {phase.description}")
    print()

    # Override phase-specific settings
    if phase.epsilon is not None:
        agent.epsilon = phase.epsilon
    if phase.epsilon_decay is not None:
        original_decay = agent.epsilon_decay
        agent.epsilon_decay = phase.epsilon_decay
    if phase.learning_rate is not None:
        agent.learning_rate = phase.learning_rate

    viz = ASCIIVisualizer(use_colors=False) if render else None
    eval_opponents = eval_opponents or ['random', 'smart_random', 'greedy']

    phase_start_episode = len(stats['episodes'])
    episodes_in_phase = 0

    for episode_in_phase in range(phase.episodes):
        # Start episode
        agent.start_episode()

        # Select opponent
        opponent_type = np.random.choice(phase.opponents)
        opponent = create_opponent(opponent_type)

        # Reset
        observations = env.reset()
        agent.reset()
        opponent.reset()

        done = False
        episode_reward = 0
        episode_steps = 0

        # Run episode
        while not done:
            state = env.game.get_state()

            agent_actions = get_agent_actions(agent, observations[0], state)
            opponent_actions = get_agent_actions(opponent, observations[1], state)

            actions = {0: agent_actions, 1: opponent_actions}
            result = env.step(actions)

            # Update Q-values
            ant_rewards = {ant_id: result.rewards[0] for ant_id in agent_actions.keys()}
            agent.update_q_values(ant_rewards, result.observations[0], result.done)

            episode_reward += result.rewards[0]
            episode_steps += 1

            # Render if requested
            if render and episode_in_phase % 100 == 0 and episode_steps % 10 == 0:
                print("\033[2J\033[H")
                print(viz.render(env.game, clear_screen=False))
                print(f"Phase: {phase.name} | Episode: {episode_in_phase}/{phase.episodes}")
                time.sleep(0.05)

            observations = result.observations
            done = result.done

        # Record stats
        stats['episodes'].append(phase_start_episode + episode_in_phase)
        stats['phases'].append(phase.name)
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
        episodes_in_phase += 1

        # Periodic evaluation within phase
        if (episode_in_phase + 1) % phase.eval_freq == 0:
            window = min(phase.eval_freq, episode_in_phase + 1)
            recent_wins = np.mean(stats['wins'][-window:])
            recent_draws = np.mean(stats['draws'][-window:])
            recent_food = np.mean(stats['food_collected'][-window:])

            print(f"  Episode {episode_in_phase + 1}/{phase.episodes} "
                  f"| Win {recent_wins:.2%} "
                  f"| Draw {recent_draws:.2%} "
                  f"| Food {recent_food:.1f} "
                  f"| ε={agent.epsilon:.4f} "
                  f"| LR={agent.learning_rate:.4f} "
                  f"| Q={len(agent.q_table)}")

    # End-of-phase evaluation
    print(f"\n--- Phase {phase_number} Complete: Evaluation ---")
    eval_results = evaluate_agent(agent, env, eval_opponents, num_games=eval_games)

    stats['phase_evals'].append({
        'phase': phase.name,
        'episode': phase_start_episode + episodes_in_phase,
        'results': eval_results
    })

    # Restore settings if they were overridden
    if phase.epsilon_decay is not None:
        agent.epsilon_decay = original_decay

    # Save checkpoint
    if phase.save_checkpoint:
        checkpoint_path = f"models/curriculum_{phase.name}_ep{phase_start_episode + episodes_in_phase}.pkl"
        os.makedirs("models", exist_ok=True)
        agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def train_curriculum(
    curriculum: TrainingCurriculum,
    board_size: tuple = (20, 20),
    max_turns: int = 150,
    render: bool = False,
    eval_games: int = 20,
    model_path: Optional[str] = None,
    output_prefix: str = "curriculum"
):
    """
    Train agent using a curriculum.

    Args:
        curriculum: Training curriculum to follow
        board_size: Board dimensions
        max_turns: Maximum turns per game
        render: Whether to render games
        eval_games: Number of games per opponent in evaluation
        model_path: Path to pre-trained model to continue from
        output_prefix: Prefix for output files
    """
    print("=" * 80)
    print(f"CURRICULUM TRAINING: {curriculum.name.upper()}")
    print("=" * 80)
    print(f"Description: {curriculum.description}")
    print(f"Total episodes: {curriculum.total_episodes()}")
    print(f"Phases: {len(curriculum.phases)}")
    print(f"Board size: {board_size[0]}x{board_size[1]}")
    print()

    # Print phase breakdown
    print("Phase breakdown:")
    for i, phase in enumerate(curriculum.phases, 1):
        pct = phase.episodes / curriculum.total_episodes() * 100
        print(f"  {i}. {phase.name:20s}: {phase.episodes:6d} episodes ({pct:5.1f}%)")
    print()

    # Create environment
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=max_turns
    )
    env = StandardEnvironment(config=config)

    # Create agent
    agent = AdaptiveQLearningAgent(
        player_id=0,
        learning_rate=curriculum.initial_learning_rate,
        learning_rate_end=curriculum.final_learning_rate,
        learning_rate_decay_type=curriculum.learning_rate_decay_type,
        gamma=0.95,
        epsilon_start=curriculum.initial_epsilon,
        epsilon_end=curriculum.final_epsilon,
        epsilon_decay=curriculum.epsilon_decay,
        max_episodes=curriculum.total_episodes(),
        model_path=model_path
    )

    # Global stats
    stats = {
        'curriculum': curriculum.name,
        'total_episodes': curriculum.total_episodes(),
        'episodes': [],
        'phases': [],
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
        'phase_evals': []
    }

    # All opponent types for evaluation
    eval_opponents = ['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive']

    # Training loop
    start_time = time.time()

    for phase_num, phase in enumerate(curriculum.phases, 1):
        train_phase(
            phase=phase,
            agent=agent,
            env=env,
            phase_number=phase_num,
            total_phases=len(curriculum.phases),
            stats=stats,
            render=render,
            eval_opponents=eval_opponents,
            eval_games=eval_games
        )

    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    final_eval = evaluate_agent(agent, env, eval_opponents, num_games=100)

    stats['final_eval'] = final_eval

    # Save final model
    final_model_path = f"models/{output_prefix}_{curriculum.name}_final.pkl"
    agent.save(final_model_path)

    # Save stats
    stats_path = f"logs/{output_prefix}_{curriculum.name}_stats.json"
    os.makedirs("logs", exist_ok=True)

    # Convert numpy types for JSON
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (np.integer, np.floating)):
                json_stats[key] = [float(v) for v in value]
            else:
                json_stats[key] = value
        else:
            json_stats[key] = value

    with open(stats_path, 'w') as f:
        json.dump(json_stats, f, indent=2)

    # Print summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Curriculum: {curriculum.name}")
    print(f"Episodes: {curriculum.total_episodes()}")
    print(f"Time: {hours}h {minutes}m {seconds}s ({elapsed/curriculum.total_episodes():.2f} sec/episode)")
    print(f"Final Q-table size: {len(agent.q_table)} states")
    print(f"Final learning rate: {agent.learning_rate:.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"\nModel saved: {final_model_path}")
    print(f"Stats saved: {stats_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Curriculum-based training for ant agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes (--mode):
  rapid      : Ultra-fast 2K episodes for quick testing (~2-5 min)
  basic      : Quick 5K episodes for basic training (~5-10 min)
  standard   : Balanced 20K episodes - RECOMMENDED (~20-40 min)
  intensive  : Deep 50K episodes for competition (~1-2 hours)
  aggressive : 15K episodes focused on aggressive play
  defensive  : 15K episodes focused on defensive play
  adaptive   : 15K episodes focused on adaptability

Examples:
  # Standard full training (recommended)
  python training/train_curriculum.py --mode standard --episodes 20000

  # Quick test
  python training/train_curriculum.py --mode rapid

  # Intensive competition training
  python training/train_curriculum.py --mode intensive --episodes 50000

  # Custom curriculum
  python training/train_curriculum.py --custom "basics:random:1000,mid:greedy:2000,hard:greedy_aggressive+greedy_defensive:3000"

  # Continue from checkpoint
  python training/train_curriculum.py --mode standard --load models/checkpoint.pkl
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=['rapid', 'basic', 'standard', 'intensive', 'aggressive', 'defensive', 'adaptive'],
        help="Training mode (default: standard)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Total training episodes (overrides mode default)"
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Custom curriculum specification (format: phase:opponents:episodes,...)"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Board size (width height)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=150,
        help="Maximum turns per game"
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
        "--output-prefix",
        type=str,
        default="curriculum",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render games during training"
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List all available training modes and exit"
    )

    args = parser.parse_args()

    # List modes
    if args.list_modes:
        print("\nAvailable Training Modes:\n")
        for mode_name in ['rapid', 'basic', 'standard', 'intensive', 'aggressive', 'defensive', 'adaptive']:
            curriculum = get_curriculum(mode_name)
            print(f"{mode_name:12s} ({curriculum.total_episodes():5d} episodes): {curriculum.description}")
            for phase in curriculum.phases:
                pct = phase.episodes / curriculum.total_episodes() * 100
                print(f"  - {phase.name:18s}: {phase.episodes:5d} eps ({pct:4.1f}%) | {', '.join(phase.opponents)}")
            print()
        return

    # Get curriculum
    if args.custom:
        curriculum = parse_custom_curriculum(args.custom)
    else:
        curriculum = get_curriculum(args.mode, args.episodes)

    # Train
    train_curriculum(
        curriculum=curriculum,
        board_size=tuple(args.board_size),
        max_turns=args.max_turns,
        render=args.render,
        eval_games=args.eval_games,
        model_path=args.load,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()
