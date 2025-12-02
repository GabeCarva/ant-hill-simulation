"""Deep Q-Network (DQN) agent implementation."""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action

# PyTorch imports (will be conditional based on availability)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN agent will not work.")


class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int, hidden_size: int = 256):
        """
        Initialize DQN.
        
        Args:
            input_shape: Shape of input (height, width, channels)
            num_actions: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        
        # Input is (3, 3, 3) for vision_radius=1
        h, w, c = input_shape
        self.input_size = h * w * c
        
        # Convolutional layers for spatial processing
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = h * w * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        
        # Value and advantage streams (for Dueling DQN)
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch, height, width, channels)
        # Reshape to (batch, channels, height, width) for PyTorch
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Dueling DQN: separate value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """DQN-based ant colony agent."""
    
    def __init__(
        self,
        player_id: int,
        input_shape: Tuple[int, int, int] = (3, 3, 3),
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        device: str = "cpu",
        model_path: Optional[str] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            player_id: Player ID (0 or 1)
            input_shape: Shape of observation input
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            target_update_freq: Steps between target network updates
            device: Device to use (cpu or cuda)
            model_path: Path to load pre-trained model
        """
        super().__init__(player_id)
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DQN agent")
        
        self.input_shape = input_shape
        self.num_actions = len(Action.DIRECTIONS)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DQN(input_shape, self.num_actions).to(self.device)
        self.target_network = DQN(input_shape, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.training_history = {
            'losses': [],
            'rewards': [],
            'epsilon': [],
            'win_rate': []
        }
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def get_actions(self, observations: List[AntObservation], game_state: GameState) -> Dict[int, int]:
        """Get actions for all ants."""
        actions = {}
        
        for obs in observations:
            # Encode observation
            if obs.vision_array is None:
                obs.vision_array = self.encode_vision(
                    obs.vision,
                    obs.player_id,
                    vision_radius=1
                )
            
            # Choose action using epsilon-greedy
            if self.is_training and random.random() < self.epsilon:
                # Exploration: random valid action
                valid_actions = self.get_valid_actions(obs, game_state)
                action = random.choice(valid_actions) if valid_actions else Action.STAY
            else:
                # Exploitation: use Q-network
                action = self._get_best_action(obs.vision_array, obs, game_state)
            
            actions[obs.ant_id] = action
            
            # Store experience if training
            if self.is_training and hasattr(self, 'last_observations'):
                last_obs = self.last_observations.get(obs.ant_id)
                if last_obs is not None:
                    # Calculate reward (will be updated by environment)
                    reward = 0  # Placeholder, actual reward comes from environment
                    done = False  # Will be updated
                    
                    self.replay_buffer.push(
                        last_obs['vision_array'],
                        last_obs['action'],
                        reward,
                        obs.vision_array,
                        done
                    )
        
        # Store current observations for next step
        if self.is_training:
            self.last_observations = {
                obs.ant_id: {
                    'vision_array': obs.vision_array,
                    'action': actions[obs.ant_id]
                }
                for obs in observations
            }
        
        return actions
    
    def _get_best_action(self, vision_array: np.ndarray, obs: AntObservation, game_state: GameState) -> int:
        """Get best action according to Q-network."""
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            state_tensor = torch.FloatTensor(vision_array).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values = self.q_network(state_tensor)
            
            # Get valid actions
            valid_actions = self.get_valid_actions(obs, game_state)
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            for action in range(self.num_actions):
                if action not in valid_actions:
                    masked_q_values[0, action] = -float('inf')
            
            # Get best action
            action = masked_q_values.argmax(dim=1).item()
            
            # Fallback to STAY if no valid actions
            if action not in valid_actions:
                action = Action.STAY
            
            return action
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Record loss
        self.training_history['losses'].append(loss.item())
        self.training_history['epsilon'].append(self.epsilon)
        
        return loss.item()
    
    def update_rewards(self, rewards: Dict[int, float], done: bool):
        """Update stored experiences with actual rewards."""
        if not self.is_training:
            return
        
        # Update the last experiences in buffer with actual rewards
        # This is a simplified approach - in practice, you might want to
        # track experiences more carefully
        if hasattr(self, 'last_observations'):
            for ant_id, reward in rewards.items():
                if ant_id in self.last_observations:
                    # Update the most recent experience for this ant
                    # Note: This is a simplification
                    pass
    
    def reset(self):
        """Reset agent for new episode."""
        self.last_observations = {}
        self.episodes += 1
    
    def save(self, path: str):
        """Save model and training history."""
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        
        # Save training history
        history_path = path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        # Load training history if exists
        history_path = path.replace('.pth', '_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)