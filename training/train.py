"""Main training script for ant colony agents."""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
from src.utils.game_config import GameConfig
from src.environments.training import TrainingEnvironment, SelfPlayEnvironment, CurriculumEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.visualization.ascii_viz import ASCIIVisualizer

# Try to import deep learning agents
try:
    from src.agents.dqn.agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Warning: DQN agent not available (PyTorch not installed)")

try:
    from src.agents.ppo.agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: PPO agent not available (PyTorch not installed)")


class Trainer:
    """Main trainer class for ant colony agents."""
    
    def __init__(
        self,
        agent_type: str,
        opponent_type: str = "random",
        env_type: str = "standard",
        config: Optional[GameConfig] = None,
        save_dir: str = "models",
        log_dir: str = "logs",
        device: str = "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            agent_type: Type of agent to train (dqn, ppo)
            opponent_type: Type of opponent (random, smart_random, self)
            env_type: Type of environment (standard, self_play, curriculum)
            config: Game configuration
            save_dir: Directory to save models
            log_dir: Directory to save logs
            device: Device to use (cpu, cuda)
        """
        self.agent_type = agent_type
        self.opponent_type = opponent_type
        self.env_type = env_type
        self.config = config or GameConfig()
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize agents
        self.agent = self._create_agent(player_id=0)
        self.opponent = self._create_opponent(player_id=1)
        
        # Training stats
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'win_rates': [],
            'food_collected': [],
            'steps': [],
            'losses': []
        }
        
        # Visualization
        self.visualizer = ASCIIVisualizer(use_colors=False)
    
    def _create_environment(self):
        """Create training environment."""
        if self.env_type == "self_play":
            return SelfPlayEnvironment(
                config=self.config,
                reward_shaping=True,
                normalize_rewards=True,
                track_metrics=True
            )
        elif self.env_type == "curriculum":
            return CurriculumEnvironment(
                initial_config=self.config,
                reward_shaping=True,
                normalize_rewards=True,
                track_metrics=True
            )
        else:
            return TrainingEnvironment(
                config=self.config,
                reward_shaping=True,
                normalize_rewards=True,
                track_metrics=True
            )
    
    def _create_agent(self, player_id: int):
        """Create agent to train."""
        if self.agent_type == "dqn":
            if not DQN_AVAILABLE:
                raise ValueError("DQN agent requires PyTorch")
            return DQNAgent(
                player_id=player_id,
                input_shape=(3, 3, 3),
                learning_rate=0.001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                device=self.device
            )
        elif self.agent_type == "ppo":
            if not PPO_AVAILABLE:
                raise ValueError("PPO agent requires PyTorch")
            return PPOAgent(
                player_id=player_id,
                input_shape=(3, 3, 3),
                learning_rate=0.0003,
                gamma=0.99,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def _create_opponent(self, player_id: int):
        """Create opponent agent."""
        if self.opponent_type == "random":
            return RandomAgent(player_id=player_id)
        elif self.opponent_type == "smart_random":
            return SmartRandomAgent(
                player_id=player_id,
                config={'prefer_food': True, 'attack_probability': 0.3}
            )
        elif self.opponent_type == "self":
            # For self-play, create same type as main agent
            return self._create_agent(player_id)
        else:
            raise ValueError(f"Unknown opponent type: {self.opponent_type}")
    
    def train(
        self,
        num_episodes: int = 1000,
        save_freq: int = 100,
        eval_freq: int = 50,
        render_freq: int = 0,
        verbose: int = 1
    ):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            save_freq: Save model every N episodes
            eval_freq: Evaluate every N episodes
            render_freq: Render game every N episodes (0 = never)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        # Set agent to training mode
        self.agent.train_mode(True)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observations = self.env.reset()
            
            # Reset agents
            self.agent.reset()
            self.opponent.reset()
            
            done = False
            episode_rewards = []
            episode_steps = 0
            
            while not done:
                # Get actions
                agent_actions = self.agent.get_actions(
                    observations[0],
                    self.env.game.get_state()
                )
                opponent_actions = self.opponent.get_actions(
                    observations[1],
                    self.env.game.get_state()
                )
                
                actions = {0: agent_actions, 1: opponent_actions}
                
                # Step environment
                result = self.env.step(actions)
                
                # Update agent with rewards
                if hasattr(self.agent, 'update_rewards'):
                    self.agent.update_rewards(
                        {ant_id: result.rewards[0] for ant_id in agent_actions.keys()},
                        result.done
                    )
                
                episode_rewards.append(result.rewards[0])
                episode_steps += 1
                
                # Render if requested
                if render_freq > 0 and episode % render_freq == 0:
                    print(self.visualizer.render(self.env.game, clear_screen=True))
                    time.sleep(0.1)
                
                observations = result.observations
                done = result.done
            
            # Train agent
            if hasattr(self.agent, 'train_step'):
                loss = self.agent.train_step()
                if loss is not None:
                    self.training_stats['losses'].append(loss)
            
            # Update PPO agent
            if self.agent_type == "ppo" and hasattr(self.agent, 'train_step'):
                self.agent.train_step()
            
            # Record episode stats
            total_reward = sum(episode_rewards)
            self.training_stats['episodes'].append(episode)
            self.training_stats['rewards'].append(total_reward)
            self.training_stats['steps'].append(episode_steps)
            
            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                eval_stats = self.evaluate(num_games=10)
                self.training_stats['win_rates'].append(eval_stats['win_rate'])
                self.training_stats['food_collected'].append(eval_stats['avg_food'])
                
                if verbose >= 1:
                    print(f"\nEpisode {episode + 1}/{num_episodes}")
                    print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
                    print(f"  Avg Reward: {np.mean(self.training_stats['rewards'][-eval_freq:]):.2f}")
                    print(f"  Avg Food: {eval_stats['avg_food']:.1f}")
                    if self.training_stats['losses']:
                        print(f"  Avg Loss: {np.mean(self.training_stats['losses'][-eval_freq:]):.4f}")
                    if hasattr(self.agent, 'epsilon'):
                        print(f"  Epsilon: {self.agent.epsilon:.3f}")
            
            # Save periodically
            if (episode + 1) % save_freq == 0:
                self.save(episode + 1)
            
            # Update opponent for self-play
            if self.env_type == "self_play" and hasattr(self.env, 'should_update_opponent'):
                if self.env.should_update_opponent():
                    # Clone current agent as opponent
                    if verbose >= 1:
                        print(f"Updating self-play opponent at episode {episode + 1}")
                    # In practice, you'd properly clone the model here
        
        # Final save
        self.save(num_episodes)
        
        # Print final stats
        if verbose >= 1:
            elapsed_time = time.time() - start_time
            print(f"\nTraining completed in {elapsed_time:.1f} seconds")
            print(f"Final win rate: {self.training_stats['win_rates'][-1]:.2%}")
            print(f"Final avg reward: {np.mean(self.training_stats['rewards'][-100:]):.2f}")
    
    def evaluate(self, num_games: int = 10) -> Dict:
        """Evaluate agent performance."""
        # Set agent to eval mode
        self.agent.train_mode(False)
        
        wins = 0
        total_food = 0
        total_steps = 0
        
        for _ in range(num_games):
            observations = self.env.reset()
            self.agent.reset()
            self.opponent.reset()
            
            done = False
            steps = 0
            
            while not done:
                agent_actions = self.agent.get_actions(
                    observations[0],
                    self.env.game.get_state()
                )
                opponent_actions = self.opponent.get_actions(
                    observations[1],
                    self.env.game.get_state()
                )
                
                actions = {0: agent_actions, 1: opponent_actions}
                result = self.env.step(actions)
                
                observations = result.observations
                done = result.done
                steps += 1
            
            # Record results
            if result.winner == 0:
                wins += 1
            total_food += result.info['food_collected'][0]
            total_steps += steps
        
        # Set back to training mode
        self.agent.train_mode(True)
        
        return {
            'win_rate': wins / num_games,
            'avg_food': total_food / num_games,
            'avg_steps': total_steps / num_games
        }
    
    def save(self, episode: int):
        """Save model and training stats."""
        # Save model
        model_path = os.path.join(
            self.save_dir,
            f"{self.agent_type}_ep{episode}.pth"
        )
        self.agent.save(model_path)
        
        # Save training stats
        stats_path = os.path.join(
            self.log_dir,
            f"training_stats_{self.agent_type}.json"
        )
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """Load a saved model."""
        self.agent.load(model_path)
        print(f"Model loaded from {model_path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ant colony agents")
    
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["dqn", "ppo"],
        help="Agent type to train"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "smart_random", "self"],
        help="Opponent type"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="standard",
        choices=["standard", "self_play", "curriculum"],
        help="Environment type"
    )
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Save model every N episodes"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render games during training"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load model from path"
    )
    
    args = parser.parse_args()
    
    # Create game config
    config = GameConfig(
        board_width=args.board_size[0],
        board_height=args.board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=200
    )
    
    # Create trainer
    trainer = Trainer(
        agent_type=args.agent,
        opponent_type=args.opponent,
        env_type=args.env,
        config=config,
        device=args.device
    )
    
    # Load model if specified
    if args.load:
        trainer.load(args.load)
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        render_freq=100 if args.render else 0,
        verbose=1
    )


if __name__ == "__main__":
    main()