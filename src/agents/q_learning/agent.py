"""Simple Q-learning agent that doesn't require PyTorch."""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action


class SimpleQLearningAgent(BaseAgent):
    """Simple Q-learning agent using state discretization."""
    
    def __init__(
        self,
        player_id: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        model_path: Optional[str] = None
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            player_id: Player ID (0 or 1)
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            model_path: Path to load pre-trained model
        """
        super().__init__(player_id)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state_hash -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience tracking
        self.last_states = {}
        self.last_actions = {}
        
        # Training stats
        self.episodes = 0
        self.steps = 0
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _discretize_state(self, obs: AntObservation) -> str:
        """
        Discretize observation into a hashable state representation.
        
        This creates a simplified state representation based on:
        - What's in each direction (empty, wall, food, ant, anthill)
        - Team affiliations
        """
        # Create a simple string representation of what's around the ant
        state_parts = []
        
        # Check each direction
        directions = [
            ("N", (0, -1)),
            ("NE", (1, -1)),
            ("E", (1, 0)),
            ("SE", (1, 1)),
            ("S", (0, 1)),
            ("SW", (-1, 1)),
            ("W", (-1, 0)),
            ("NW", (-1, -1))
        ]
        
        for dir_name, (dx, dy) in directions:
            check_pos = obs.position
            entity = None
            
            # Find entity at this relative position
            for pos, ent in obs.vision.items():
                if pos.x == check_pos.x + dx and pos.y == check_pos.y + dy:
                    entity = ent
                    break
            
            # Categorize entity
            if entity is None:
                state_parts.append(f"{dir_name}:empty")
            elif entity == 'wall':
                state_parts.append(f"{dir_name}:wall")
            elif entity == 'rock':
                state_parts.append(f"{dir_name}:rock")
            elif entity == 'food':
                state_parts.append(f"{dir_name}:food")
            elif entity.startswith('ant_'):
                if int(entity.split('_')[1]) == self.player_id:
                    state_parts.append(f"{dir_name}:ally_ant")
                else:
                    state_parts.append(f"{dir_name}:enemy_ant")
            elif entity.startswith('anthill_'):
                if int(entity.split('_')[1]) == self.player_id:
                    state_parts.append(f"{dir_name}:ally_hill")
                else:
                    state_parts.append(f"{dir_name}:enemy_hill")
        
        return "|".join(state_parts)
    
    def get_actions(self, observations: List[AntObservation], game_state: GameState) -> Dict[int, int]:
        """Get actions for all ants using Q-learning."""
        actions = {}
        
        for obs in observations:
            # Get discretized state
            state = self._discretize_state(obs)
            
            # Get valid actions
            valid_actions = self.get_valid_actions(obs, game_state)
            
            # Choose action using epsilon-greedy
            if self.is_training and random.random() < self.epsilon:
                # Exploration: random valid action
                action = random.choice(valid_actions) if valid_actions else Action.STAY
            else:
                # Exploitation: choose best Q-value
                action = self._get_best_action(state, valid_actions)
            
            actions[obs.ant_id] = action
            
            # Store for learning
            if self.is_training:
                self.last_states[obs.ant_id] = state
                self.last_actions[obs.ant_id] = action
        
        return actions
    
    def _get_best_action(self, state: str, valid_actions: List[int]) -> int:
        """Get action with highest Q-value for given state."""
        if not valid_actions:
            return Action.STAY
        
        # Get Q-values for valid actions
        q_values = {
            action: self.q_table[state][action]
            for action in valid_actions
        }
        
        # Return action with max Q-value
        if q_values:
            max_q = max(q_values.values())
            # Break ties randomly
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
        else:
            return random.choice(valid_actions)
    
    def update_q_values(self, rewards: Dict[int, float], next_observations: List[AntObservation], done: bool):
        """Update Q-values based on rewards received."""
        if not self.is_training:
            return
        
        for obs in next_observations:
            ant_id = obs.ant_id
            
            # Check if we have previous state/action for this ant
            if ant_id not in self.last_states:
                continue
            
            prev_state = self.last_states[ant_id]
            prev_action = self.last_actions[ant_id]
            reward = rewards.get(ant_id, 0.0)
            
            # Get next state
            next_state = self._discretize_state(obs)
            
            # Get max Q-value for next state
            if done:
                max_next_q = 0
            else:
                valid_next_actions = list(range(len(Action.DIRECTIONS)))
                if valid_next_actions:
                    max_next_q = max(
                        self.q_table[next_state][a]
                        for a in valid_next_actions
                    )
                else:
                    max_next_q = 0
            
            # Q-learning update
            old_q = self.q_table[prev_state][prev_action]
            new_q = old_q + self.learning_rate * (reward + self.gamma * max_next_q - old_q)
            self.q_table[prev_state][prev_action] = new_q
        
        self.steps += 1
    
    def reset(self):
        """Reset agent for new episode."""
        self.last_states = {}
        self.last_actions = {}
        self.episodes += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save Q-table to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'steps': self.steps
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str):
        """Load Q-table from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), save_data['q_table'])
        self.epsilon = save_data['epsilon']
        self.episodes = save_data['episodes']
        self.steps = save_data['steps']


class TabularAgent(BaseAgent):
    """Even simpler tabular agent for very small state spaces."""
    
    def __init__(
        self,
        player_id: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        model_path: Optional[str] = None
    ):
        """Initialize tabular agent."""
        super().__init__(player_id)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Simple policy: map situations to preferred actions
        self.policy = self._initialize_policy()
        
        # Load if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _initialize_policy(self) -> Dict[str, int]:
        """Initialize a simple hand-crafted policy."""
        return {
            'food_north': Action.NORTH,
            'food_south': Action.SOUTH,
            'food_east': Action.EAST,
            'food_west': Action.WEST,
            'enemy_ant_near': Action.STAY,  # Avoid combat
            'enemy_hill_near': Action.NORTH,  # Attack
            'default': Action.NORTH
        }
    
    def _get_situation(self, obs: AntObservation) -> str:
        """Determine current situation."""
        # Check for food
        for pos, entity in obs.vision.items():
            if entity == 'food':
                dx = pos.x - obs.position.x
                dy = pos.y - obs.position.y
                
                if dy < 0:
                    return 'food_north'
                elif dy > 0:
                    return 'food_south'
                elif dx > 0:
                    return 'food_east'
                elif dx < 0:
                    return 'food_west'
            
            elif entity and entity.startswith('ant_'):
                if int(entity.split('_')[1]) != self.player_id:
                    return 'enemy_ant_near'
            
            elif entity and entity.startswith('anthill_'):
                if int(entity.split('_')[1]) != self.player_id:
                    return 'enemy_hill_near'
        
        return 'default'
    
    def get_actions(self, observations: List[AntObservation], game_state: GameState) -> Dict[int, int]:
        """Get actions based on simple policy."""
        actions = {}
        
        for obs in observations:
            situation = self._get_situation(obs)
            preferred_action = self.policy.get(situation, Action.NORTH)
            
            # Check if action is valid
            valid_actions = self.get_valid_actions(obs, game_state)
            
            if preferred_action in valid_actions:
                actions[obs.ant_id] = preferred_action
            elif valid_actions:
                actions[obs.ant_id] = random.choice(valid_actions)
            else:
                actions[obs.ant_id] = Action.STAY
        
        return actions
    
    def reset(self):
        """Reset agent."""
        pass
    
    def save(self, path: str):
        """Save policy."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)
    
    def load(self, path: str):
        """Load policy."""
        with open(path, 'rb') as f:
            self.policy = pickle.load(f)