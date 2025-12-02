"""Proximal Policy Optimization (PPO) agent implementation."""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os

from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState
from src.utils.game_config import Action

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. PPO agent will not work.")


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int, hidden_size: int = 256):
        """
        Initialize Actor-Critic network.
        
        Args:
            input_shape: Shape of input (height, width, channels)
            num_actions: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        h, w, c = input_shape
        self.input_size = h * w * c
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = h * w * 64
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(conv_output_size, hidden_size)
        
        # Actor head (policy)
        self.fc_actor = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_head = nn.Linear(hidden_size // 2, num_actions)
        
        # Critic head (value)
        self.fc_critic = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch, height, width, channels)
        # Reshape to (batch, channels, height, width) for PyTorch
        x = x.permute(0, 3, 1, 2)
        
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected
        x = F.relu(self.fc_shared(x))
        
        # Actor head
        actor = F.relu(self.fc_actor(x))
        logits = self.policy_head(actor)
        
        # Critic head
        critic = F.relu(self.fc_critic(x))
        value = self.value_head(critic)
        
        return logits, value


class PPOMemory:
    """Memory buffer for PPO."""
    
    def __init__(self):
        """Initialize memory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def store(self, state, action, reward, value, log_prob, done):
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """Clear the memory."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_batch(self):
        """Get all stored data as batch."""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )


class PPOAgent(BaseAgent):
    """PPO-based ant colony agent."""
    
    def __init__(
        self,
        player_id: int,
        input_shape: Tuple[int, int, int] = (3, 3, 3),
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 0.5,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        model_path: Optional[str] = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            player_id: Player ID (0 or 1)
            input_shape: Shape of observation input
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            device: Device to use (cpu or cuda)
            model_path: Path to load pre-trained model
        """
        super().__init__(player_id)
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PPO agent")
        
        self.input_shape = input_shape
        self.num_actions = len(Action.DIRECTIONS)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device(device)
        
        # Network
        self.ac_network = ActorCritic(input_shape, self.num_actions).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.training_history = {
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'rewards': [],
            'win_rate': []
        }
        
        # Current episode data
        self.current_episode_rewards = []
        
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
            
            # Get action from policy
            action, value, log_prob = self._get_action(obs.vision_array, obs, game_state)
            actions[obs.ant_id] = action
            
            # Store in memory if training
            if self.is_training:
                # We'll update rewards later
                self.memory.store(
                    obs.vision_array,
                    action,
                    0,  # Placeholder reward
                    value,
                    log_prob,
                    False  # Placeholder done
                )
        
        return actions
    
    def _get_action(self, vision_array: np.ndarray, obs: AntObservation, game_state: GameState) -> Tuple[int, float, float]:
        """Get action from policy network."""
        # Convert to tensor
        state_tensor = torch.FloatTensor(vision_array).unsqueeze(0).to(self.device)
        
        # Get logits and value
        with torch.no_grad():
            logits, value = self.ac_network(state_tensor)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(obs, game_state)
        
        # Mask invalid actions
        masked_logits = logits.clone()
        for action in range(self.num_actions):
            if action not in valid_actions:
                masked_logits[0, action] = -float('inf')
        
        # Sample action from policy
        probs = F.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)
        
        if self.is_training:
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)
        
        log_prob = dist.log_prob(action).item()
        action = action.item()
        value = value.item()
        
        # Fallback to STAY if invalid
        if action not in valid_actions:
            action = Action.STAY
        
        return action, value, log_prob
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory.states) == 0:
            return None
        
        # Get batch
        states, actions, rewards, values, old_log_probs, dones = self.memory.get_batch()
        
        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current logits and values
                logits, values = self.ac_network(batch_states)
                
                # Compute current log probs
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory
        self.memory.clear()
        
        # Update steps
        self.steps += 1
        
        # Record stats
        num_batches = self.epochs * (len(states) // self.batch_size + 1)
        self.training_history['actor_losses'].append(total_actor_loss / num_batches)
        self.training_history['critic_losses'].append(total_critic_loss / num_batches)
        self.training_history['entropies'].append(total_entropy / num_batches)
        
        return total_actor_loss / num_batches
    
    def update_rewards(self, rewards: Dict[int, float], done: bool):
        """Update stored experiences with actual rewards."""
        if not self.is_training:
            return
        
        # Update the rewards in memory
        # This is simplified - in practice you'd track ant-specific rewards
        if len(self.memory.rewards) > 0:
            # Distribute rewards among recent actions
            for i in range(len(self.memory.rewards)):
                if i >= len(self.memory.rewards) - len(rewards):
                    # Update recent rewards
                    self.memory.rewards[i] = sum(rewards.values()) / len(rewards)
            
            # Update done flags
            if done:
                self.memory.dones[-1] = True
        
        # Track episode rewards
        self.current_episode_rewards.extend(rewards.values())
    
    def reset(self):
        """Reset agent for new episode."""
        self.episodes += 1
        
        # Record episode reward
        if self.current_episode_rewards:
            episode_reward = sum(self.current_episode_rewards)
            self.training_history['rewards'].append(episode_reward)
        
        self.current_episode_rewards = []
    
    def save(self, path: str):
        """Save model and training history."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'ac_network_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.ac_network.load_state_dict(checkpoint['ac_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        # Load training history if exists
        history_path = path.replace('.pth', '_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)