"""Adaptive Q-Learning agent with learning rate decay and improved hyperparameters."""

import pickle
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action, Position


class AdaptiveQLearningAgent(BaseAgent):
    """
    Q-Learning agent with adaptive learning rate and epsilon decay.

    Key improvements over SimpleQLearningAgent:
    - Learning rate decay for stable convergence
    - Configurable epsilon decay based on episode count
    - Better default hyperparameters
    - Performance tracking for debugging
    """

    def __init__(
        self,
        player_id: int,
        learning_rate: float = 0.3,
        learning_rate_end: float = 0.001,
        learning_rate_decay_type: str = "exponential",
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9994,
        max_episodes: Optional[int] = None,
        config: Optional[Dict] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize adaptive Q-learning agent.

        Args:
            player_id: Player ID (0 or 1)
            learning_rate: Initial learning rate (alpha)
            learning_rate_end: Final learning rate (for decay)
            learning_rate_decay_type: Type of decay ('exponential', 'polynomial', 'step')
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay multiplier per episode
            max_episodes: Total episodes for training (used for polynomial decay)
            config: Optional configuration dictionary
            model_path: Path to load pre-trained model
        """
        super().__init__(player_id, config)

        # Q-learning parameters
        self.learning_rate_initial = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_end = learning_rate_end
        self.learning_rate_decay_type = learning_rate_decay_type
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training configuration
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.training = True

        # Q-table: (state, action) -> Q-value
        self.q_table: Dict[Tuple, float] = defaultdict(float)

        # Experience tracking for updates
        self.previous_states: Dict[int, Tuple] = {}  # ant_id -> state
        self.previous_actions: Dict[int, int] = {}  # ant_id -> action

        # Performance tracking
        self.stats = {
            'learning_rates': [],
            'epsilons': [],
            'q_table_sizes': [],
            'episodes': []
        }

        # Load pre-trained model if provided
        if model_path:
            self.load(model_path)

    def _get_state_key(self, obs: AntObservation) -> Tuple:
        """
        Convert observation to hashable state key.

        Uses local grid representation around ant.
        """
        # Get local grid (3x3 around ant)
        local_grid = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                pos = Position(obs.position.x + dx, obs.position.y + dy)
                entity = obs.vision.get(pos, 'empty')
                local_grid.append(entity)

        return tuple(local_grid)

    def _decay_learning_rate(self):
        """Decay learning rate based on configured strategy."""
        if self.max_episodes is None:
            # Use exponential decay without episode limit
            decay_rate = 0.9995
            self.learning_rate = max(
                self.learning_rate_end,
                self.learning_rate * decay_rate
            )
        elif self.learning_rate_decay_type == "exponential":
            # Exponential decay: lr = lr_init * decay^episode
            decay_rate = (self.learning_rate_end / self.learning_rate_initial) ** (1 / self.max_episodes)
            self.learning_rate = max(
                self.learning_rate_end,
                self.learning_rate_initial * (decay_rate ** self.current_episode)
            )
        elif self.learning_rate_decay_type == "polynomial":
            # Polynomial decay: lr = lr_init * (1 - episode/max)^power
            power = 0.9
            progress = min(1.0, self.current_episode / self.max_episodes)
            self.learning_rate = max(
                self.learning_rate_end,
                self.learning_rate_initial * ((1 - progress) ** power)
            )
        elif self.learning_rate_decay_type == "step":
            # Step decay: lr = lr_init / (1 + decay * episode)
            decay_factor = 0.0001
            self.learning_rate = max(
                self.learning_rate_end,
                self.learning_rate_initial / (1 + decay_factor * self.current_episode)
            )
        else:
            # No decay
            self.learning_rate = self.learning_rate_initial

    def _decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_mode(self, training: bool):
        """Set training mode."""
        self.training = training
        if not training:
            self.epsilon = 0.0  # No exploration during evaluation

    def start_episode(self):
        """Call at the start of each episode to update hyperparameters."""
        self.current_episode += 1
        self._decay_learning_rate()
        self._decay_epsilon()

        # Track stats every 50 episodes
        if self.current_episode % 50 == 0:
            self.stats['episodes'].append(self.current_episode)
            self.stats['learning_rates'].append(self.learning_rate)
            self.stats['epsilons'].append(self.epsilon)
            self.stats['q_table_sizes'].append(len(self.q_table))

    def get_actions(
        self,
        observations: List[AntObservation],
        game_state: GameState
    ) -> Dict[int, int]:
        """
        Get actions for all ants using epsilon-greedy Q-learning.

        Args:
            observations: List of observations for living ants
            game_state: Current game state

        Returns:
            Dictionary mapping ant_id -> action
        """
        actions = {}

        for obs in observations:
            state = self._get_state_key(obs)

            # Epsilon-greedy action selection
            if self.training and random.random() < self.epsilon:
                # Explore: random valid action
                valid_actions = self.get_valid_actions(obs, game_state)
                action = random.choice(valid_actions)
            else:
                # Exploit: choose best action
                action = self._get_best_action(state, obs, game_state)

            actions[obs.ant_id] = action

            # Store for Q-value update
            self.previous_states[obs.ant_id] = state
            self.previous_actions[obs.ant_id] = action

        return actions

    def _get_best_action(
        self,
        state: Tuple,
        obs: AntObservation,
        game_state: GameState
    ) -> int:
        """Get action with highest Q-value."""
        valid_actions = self.get_valid_actions(obs, game_state)

        # Get Q-values for all valid actions
        q_values = {
            action: self.q_table.get((state, action), 0.0)
            for action in valid_actions
        }

        # Return action with highest Q-value
        return max(q_values, key=q_values.get)

    def update_q_values(
        self,
        rewards: Dict[int, float],
        next_observations: List[AntObservation],
        done: bool
    ):
        """
        Update Q-values based on observed rewards.

        Args:
            rewards: Dictionary mapping ant_id -> reward
            next_observations: Observations for next state
            done: Whether episode is finished
        """
        if not self.training:
            return

        # Create mapping of ant_id -> next_state
        next_states = {}
        for obs in next_observations:
            next_states[obs.ant_id] = self._get_state_key(obs)

        # Update Q-values for each ant
        for ant_id, reward in rewards.items():
            if ant_id not in self.previous_states:
                continue

            state = self.previous_states[ant_id]
            action = self.previous_actions[ant_id]

            # Get current Q-value
            current_q = self.q_table.get((state, action), 0.0)

            # Calculate target Q-value
            if done or ant_id not in next_states:
                # Terminal state or ant died
                target_q = reward
            else:
                # Non-terminal: use Bellman equation
                next_state = next_states[ant_id]

                # Get max Q-value for next state
                max_next_q = max(
                    self.q_table.get((next_state, a), 0.0)
                    for a in range(5)  # 5 possible actions
                )

                target_q = reward + self.gamma * max_next_q

            # Q-learning update with current learning rate
            self.q_table[(state, action)] = (
                current_q + self.learning_rate * (target_q - current_q)
            )

    def reset(self):
        """Reset episode-specific state."""
        self.previous_states.clear()
        self.previous_actions.clear()

    def save(self, path: str):
        """Save agent state to file."""
        state = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'learning_rate_initial': self.learning_rate_initial,
            'learning_rate_end': self.learning_rate_end,
            'learning_rate_decay_type': self.learning_rate_decay_type,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'current_episode': self.current_episode,
            'max_episodes': self.max_episodes,
            'stats': self.stats
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load agent state from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.q_table = defaultdict(float, state['q_table'])
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.learning_rate_initial = state.get('learning_rate_initial', self.learning_rate_initial)
        self.learning_rate_end = state.get('learning_rate_end', self.learning_rate_end)
        self.learning_rate_decay_type = state.get('learning_rate_decay_type', 'exponential')
        self.epsilon = state.get('epsilon', self.epsilon)
        self.epsilon_start = state.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = state.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay = state.get('epsilon_decay', self.epsilon_decay)
        self.gamma = state.get('gamma', self.gamma)
        self.current_episode = state.get('current_episode', 0)
        self.max_episodes = state.get('max_episodes', self.max_episodes)
        self.stats = state.get('stats', self.stats)

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'current_episode': self.current_episode,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'stats_history': self.stats
        }
