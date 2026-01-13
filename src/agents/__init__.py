"""Agent implementations for the ant hill simulation.

This package contains various agent types that can control ants:
- BaseAgent: Abstract base class for all agents
- AdaptiveQLearningAgent: Q-learning with adaptive decay
- SimpleQLearningAgent: Basic tabular Q-learning
- GreedyAgent: Various greedy strategies
- TacticalAgent: Multi-objective tactical decision-making
- RandomAgent: Random baseline
- DQNAgent: Deep Q-Network (requires PyTorch)
- PPOAgent: Proximal Policy Optimization (requires PyTorch)
"""

from src.agents.base import BaseAgent, AntObservation, AgentWrapper
from src.agents.adaptive_q_learning.agent import AdaptiveQLearningAgent
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.agents.greedy.agent import GreedyAgent, AggressiveGreedyAgent, DefensiveGreedyAgent
from src.agents.tactical.agent import TacticalAgent
from src.agents.random.agent import RandomAgent, SmartRandomAgent

__all__ = [
    'BaseAgent',
    'AntObservation',
    'AgentWrapper',
    'AdaptiveQLearningAgent',
    'SimpleQLearningAgent',
    'GreedyAgent',
    'AggressiveGreedyAgent',
    'DefensiveGreedyAgent',
    'TacticalAgent',
    'RandomAgent',
    'SmartRandomAgent',
]

# Optional imports (require PyTorch)
try:
    from src.agents.dqn.agent import DQNAgent
    __all__.append('DQNAgent')
except ImportError:
    pass

try:
    from src.agents.ppo.agent import PPOAgent
    __all__.append('PPOAgent')
except ImportError:
    pass
