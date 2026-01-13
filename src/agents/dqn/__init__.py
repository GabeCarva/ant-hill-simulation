"""Deep Q-Network agent implementation (requires PyTorch)."""

try:
    from src.agents.dqn.agent import DQNAgent
    __all__ = ['DQNAgent']
except ImportError:
    # PyTorch not installed
    __all__ = []
