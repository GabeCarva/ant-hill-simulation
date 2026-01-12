"""Helper functions for working with agents in the new independent control API."""

from typing import Dict, List
from src.agents.base import BaseAgent, AntObservation
from src.core.game import GameState


def get_agent_actions(agent: BaseAgent, observations: List[AntObservation], game_state: GameState) -> Dict[int, int]:
    """
    Get actions from agent for all ants (handles new independent control API).

    This helper function bridges the gap between the old API (get_actions returning all actions at once)
    and the new API (get_action called once per ant).

    Args:
        agent: Agent instance
        observations: List of AntObservation objects for this agent's ants
        game_state: Current game state

    Returns:
        Dictionary mapping ant_id -> action
    """
    actions = {}
    for obs in observations:
        action = agent.get_action(obs, game_state)
        actions[obs.ant_id] = action
    return actions
