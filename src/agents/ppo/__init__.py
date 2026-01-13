"""Proximal Policy Optimization agent implementation (requires PyTorch)."""

try:
    from src.agents.ppo.agent import PPOAgent
    __all__ = ['PPOAgent']
except ImportError:
    # PyTorch not installed
    __all__ = []
