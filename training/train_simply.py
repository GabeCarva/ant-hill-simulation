"""Simple training script for Q-learning agents (no PyTorch required) - Updated for independent ant control."""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer


def get_agent_actions(agent, observations, game_state):
    """
    Get actions from agent for all ants (handles new independent control API).

    Args:
        agent: Agent instance
        observations: List of AntObservation objects
        game_state: Current game state

    Returns:
        Dictionary mapping ant_id -> action
    """
    actions = {}
    for obs in observations:
        action = agent.get_action(obs, game_state)
        actions[obs.ant_id] = action
    return actions


def train_q_learning(
    episodes: int = 1000,
    board_size: tuple = (20, 20),
    opponent_type: str = "random",
    save_freq: int = 100,
    eval_freq: int = 50,
    render: bool = False
):
    """Train a Q-learning agent."""

    print("=" * 60)
    print("Q-LEARNING TRAINING")
    print("=" * 60)
    print(f"Board size: {board_size[0]}x{board_size[1]}")
    print(f"Opponent: {opponent_type}")
    print(f"Episodes: {episodes}")
    print()

    # Create game config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )

    # Create environment
    env = StandardEnvironment(config=config)

    # Create agents
    agent = SimpleQLearningAgent(
        player_id=0,
        learning_rate=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    if opponent_type == "smart":
        opponent = SmartRandomAgent(
            player_id=1,
            config={'prefer_food': True, 'attack_probability': 0.3}
        )
    else:
        opponent = RandomAgent(player_id=1)

    # Training stats
    stats = {
        'episodes': [],
        'rewards': [],
        'wins': [],
        'food_collected': [],
        'steps': [],
        'epsilon': []
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

            # Get actions (using new independent control API)
            agent_actions = get_agent_actions(agent, observations[0], state)
            opponent_actions = get_agent_actions(opponent, observations[1], state)

            # Step
            actions = {0: agent_actions, 1: opponent_actions}
            result = env.step(actions)

            # Update Q-values
            if hasattr(agent, 'update_q_values'):
                # Calculate ant-specific rewards
                ant_rewards = {}
                for ant_id in agent_actions.keys():
                    ant_rewards[ant_id] = result.rewards[0]  # Simple: same reward for all ants

                agent.update_q_values(ant_rewards, result.observations[0], result.done)

            episode_reward += result.rewards[0]
            episode_steps += 1

            # Render if requested
            if render and episode % 100 == 0 and episode_steps % 10 == 0:
                print("\033[2J\033[H")  # Clear screen
                print(viz.render(env.game, clear_screen=False))
                print(f"Episode: {episode}, Step: {episode_steps}")
                time.sleep(0.05)

            observations = result.observations
            done = result.done

        # Record stats
        stats['episodes'].append(episode)
        stats['rewards'].append(episode_reward)
        stats['wins'].append(1 if result.winner == 0 else 0)
        stats['food_collected'].append(result.info['food_collected'][0])
        stats['steps'].append(episode_steps)
        stats['epsilon'].append(agent.epsilon)

        # Evaluate and print progress
        if (episode + 1) % eval_freq == 0:
            recent_wins = np.mean(stats['wins'][-eval_freq:])
            recent_reward = np.mean(stats['rewards'][-eval_freq:])
            recent_food = np.mean(stats['food_collected'][-eval_freq:])

            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Win rate: {recent_wins:.2%}")
            print(f"  Avg reward: {recent_reward:.2f}")
            print(f"  Avg food: {recent_food:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {len(agent.q_table)} states")
            print()

        # Save periodically
        if (episode + 1) % save_freq == 0:
            save_path = f"models/q_learning_ep{episode + 1}.pkl"
            os.makedirs("models", exist_ok=True)
            agent.save(save_path)

            # Save stats
            stats_path = "logs/q_learning_stats.json"
            os.makedirs("logs", exist_ok=True)
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

    # Final save
    agent.save("models/q_learning_final.pkl")

    # Print final stats
    elapsed = time.time() - start_time
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Final win rate: {np.mean(stats['wins'][-100:]):.2%}")
    print(f"Final avg reward: {np.mean(stats['rewards'][-100:]):.2f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"Model saved to: models/q_learning_final.pkl")


def evaluate_agent(model_path: str, num_games: int = 100):
    """Evaluate a trained Q-learning agent."""

    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)

    # Create environment
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )

    env = StandardEnvironment(config=config)

    # Load agent
    agent = SimpleQLearningAgent(player_id=0)
    agent.load(model_path)
    agent.train_mode(False)  # Disable exploration

    # Create opponents
    opponents = {
        'random': RandomAgent(player_id=1),
        'smart': SmartRandomAgent(
            player_id=1,
            config={'prefer_food': True, 'attack_probability': 0.3}
        )
    }

    for opp_name, opponent in opponents.items():
        wins = 0
        total_food = 0
        total_steps = 0

        for game in range(num_games):
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
            total_food += result.info['food_collected'][0]
            total_steps += steps

        print(f"\nVs {opp_name} opponent:")
        print(f"  Win rate: {wins/num_games:.2%}")
        print(f"  Avg food: {total_food/num_games:.1f}")
        print(f"  Avg steps: {total_steps/num_games:.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Q-learning ant agents")

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
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
        "--opponent",
        type=str,
        default="random",
        choices=["random", "smart"],
        help="Opponent type"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        help="Evaluate model from path"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render games during training"
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate_agent(args.evaluate)
    else:
        train_q_learning(
            episodes=args.episodes,
            board_size=tuple(args.board_size),
            opponent_type=args.opponent,
            render=args.render
        )


if __name__ == "__main__":
    main()
