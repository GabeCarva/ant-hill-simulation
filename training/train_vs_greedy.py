"""Training script for agents playing against the GreedyAgent."""

import os
import sys
import time
import json
import argparse
from typing import Optional
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
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer


def train_against_greedy(
    agent_type: str = "q_learning",
    episodes: int = 2000,
    board_size: tuple = (20, 20),
    greedy_variant: str = "standard",
    save_freq: int = 100,
    eval_freq: int = 50,
    render: bool = False,
    model_path: Optional[str] = None
):
    """
    Train an agent against GreedyAgent.

    Args:
        agent_type: Type of agent to train ('q_learning')
        episodes: Number of training episodes
        board_size: Board dimensions (width, height)
        greedy_variant: GreedyAgent variant ('standard', 'aggressive', 'defensive')
        save_freq: Save model every N episodes
        eval_freq: Evaluate and print stats every N episodes
        render: Whether to render games
        model_path: Path to load pre-trained model (optional)
    """

    print("=" * 70)
    print("TRAINING AGAINST GREEDY AGENT")
    print("=" * 70)
    print(f"Agent type: {agent_type}")
    print(f"Board size: {board_size[0]}x{board_size[1]}")
    print(f"Opponent: GreedyAgent ({greedy_variant} variant)")
    print(f"Episodes: {episodes}")
    print()

    # Create game config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=150  # Slightly longer for more complex opponent
    )

    # Create environment
    env = StandardEnvironment(config=config)

    # Create training agent
    if agent_type == "q_learning":
        agent = SimpleQLearningAgent(
            player_id=0,
            learning_rate=0.1,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.9975,  # Slower decay for harder opponent
            model_path=model_path
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create opponent
    if greedy_variant == "aggressive":
        opponent = AggressiveGreedyAgent(player_id=1)
    elif greedy_variant == "defensive":
        opponent = DefensiveGreedyAgent(player_id=1)
    else:
        opponent = GreedyAgent(player_id=1)

    # Training stats
    stats = {
        'episodes': [],
        'rewards': [],
        'wins': [],
        'draws': [],
        'losses': [],
        'food_collected': [],
        'steps': [],
        'epsilon': [],
        'ants_lost': []
    }

    # Visualizer
    viz = ASCIIVisualizer(use_colors=False) if render else None

    # Training loop
    agent.train_mode(True)
    start_time = time.time()

    for episode in range(episodes):
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

            # Update Q-values for Q-learning agent
            if hasattr(agent, 'update_q_values'):
                # Calculate ant-specific rewards
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
                print(f"Reward: {episode_reward:.1f}")
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
        stats['ants_lost'].append(result.info.get('ants_lost', {}).get(0, 0))

        if hasattr(agent, 'epsilon'):
            stats['epsilon'].append(agent.epsilon)

        # Evaluate and print progress
        if (episode + 1) % eval_freq == 0:
            recent_wins = np.mean(stats['wins'][-eval_freq:])
            recent_draws = np.mean(stats['draws'][-eval_freq:])
            recent_losses = np.mean(stats['losses'][-eval_freq:])
            recent_reward = np.mean(stats['rewards'][-eval_freq:])
            recent_food = np.mean(stats['food_collected'][-eval_freq:])
            recent_ants_lost = np.mean(stats['ants_lost'][-eval_freq:])

            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Win rate:   {recent_wins:.2%}")
            print(f"  Draw rate:  {recent_draws:.2%}")
            print(f"  Loss rate:  {recent_losses:.2%}")
            print(f"  Avg reward: {recent_reward:.2f}")
            print(f"  Avg food:   {recent_food:.1f}")
            print(f"  Avg ants lost: {recent_ants_lost:.1f}")

            if hasattr(agent, 'epsilon'):
                print(f"  Epsilon:    {agent.epsilon:.3f}")
            if hasattr(agent, 'q_table'):
                print(f"  Q-table size: {len(agent.q_table)} states")
            print()

        # Save periodically
        if (episode + 1) % save_freq == 0:
            save_path = f"models/{agent_type}_vs_greedy_ep{episode + 1}.pkl"
            os.makedirs("models", exist_ok=True)
            agent.save(save_path)

            # Save stats
            stats_path = f"logs/{agent_type}_vs_greedy_stats.json"
            os.makedirs("logs", exist_ok=True)
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

    # Final save
    final_path = f"models/{agent_type}_vs_greedy_final.pkl"
    agent.save(final_path)

    # Print final stats
    elapsed = time.time() - start_time
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f} seconds ({elapsed/episodes:.2f} sec/episode)")
    print(f"Final win rate:  {np.mean(stats['wins'][-100:]):.2%}")
    print(f"Final draw rate: {np.mean(stats['draws'][-100:]):.2%}")
    print(f"Final avg reward: {np.mean(stats['rewards'][-100:]):.2f}")
    print(f"Final avg food:   {np.mean(stats['food_collected'][-100:]):.1f}")

    if hasattr(agent, 'q_table'):
        print(f"Q-table size: {len(agent.q_table)} states")

    print(f"\nModel saved to: {final_path}")
    print(f"Stats saved to: logs/{agent_type}_vs_greedy_stats.json")
    print()

    # Suggest evaluation command
    print("To evaluate the trained agent, run:")
    print(f"  python training/train_vs_greedy.py --evaluate {final_path}")


def evaluate_agent(model_path: str, agent_type: str = "q_learning", num_games: int = 100):
    """
    Evaluate a trained agent against multiple opponents.

    Args:
        model_path: Path to trained model
        agent_type: Type of agent
        num_games: Number of games to play against each opponent
    """

    print("=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Games per opponent: {num_games}")
    print()

    # Create environment
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=150
    )

    env = StandardEnvironment(config=config)

    # Load agent
    if agent_type == "q_learning":
        agent = SimpleQLearningAgent(player_id=0)
        agent.load(model_path)
        agent.train_mode(False)  # Disable exploration
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create opponents
    opponents = {
        'Random': RandomAgent(player_id=1),
        'Smart Random': SmartRandomAgent(
            player_id=1,
            config={'prefer_food': True, 'attack_probability': 0.3}
        ),
        'Greedy (Standard)': GreedyAgent(player_id=1),
        'Greedy (Aggressive)': AggressiveGreedyAgent(player_id=1),
        'Greedy (Defensive)': DefensiveGreedyAgent(player_id=1)
    }

    # Evaluate against each opponent
    for opp_name, opponent in opponents.items():
        wins = 0
        draws = 0
        losses = 0
        total_food = 0
        total_steps = 0
        total_ants_lost = 0

        for game in range(num_games):
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
            else:
                losses += 1

            total_food += result.info['food_collected'][0]
            total_steps += steps
            total_ants_lost += result.info.get('ants_lost', {}).get(0, 0)

        print(f"\nVs {opp_name}:")
        print(f"  Win rate:  {wins/num_games:.2%}")
        print(f"  Draw rate: {draws/num_games:.2%}")
        print(f"  Loss rate: {losses/num_games:.2%}")
        print(f"  Avg food:  {total_food/num_games:.1f}")
        print(f"  Avg steps: {total_steps/num_games:.1f}")
        print(f"  Avg ants lost: {total_ants_lost/num_games:.1f}")

    print()
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train agents against GreedyAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Q-learning agent vs standard GreedyAgent
  python training/train_vs_greedy.py --episodes 2000

  # Train vs aggressive variant
  python training/train_vs_greedy.py --episodes 2000 --greedy-variant aggressive

  # Continue training from checkpoint
  python training/train_vs_greedy.py --episodes 1000 --load models/q_learning_vs_greedy_ep1000.pkl

  # Evaluate trained agent
  python training/train_vs_greedy.py --evaluate models/q_learning_vs_greedy_final.pkl

  # Train with visualization (slower)
  python training/train_vs_greedy.py --episodes 100 --render
        """
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="q_learning",
        choices=["q_learning"],
        help="Type of agent to train"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
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
        "--greedy-variant",
        type=str,
        default="standard",
        choices=["standard", "aggressive", "defensive"],
        help="GreedyAgent variant to train against"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        help="Evaluate model from path"
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
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Number of games for evaluation"
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate_agent(
            model_path=args.evaluate,
            agent_type=args.agent,
            num_games=args.eval_games
        )
    else:
        train_against_greedy(
            agent_type=args.agent,
            episodes=args.episodes,
            board_size=tuple(args.board_size),
            greedy_variant=args.greedy_variant,
            render=args.render,
            model_path=args.load
        )


if __name__ == "__main__":
    main()
