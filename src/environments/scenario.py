"""
Scenario-based training environment.

This environment runs isolated training scenarios that focus on specific skills
without the complexity of full opponent agents.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.core.game import Game, Position
from src.utils.game_config import GameConfig
from src.agents.base import AntObservation


@dataclass
class ScenarioStepResult:
    """Result from a scenario training step."""

    observations: List[List[AntObservation]]
    """Observations for each player"""

    rewards: Dict[int, float]
    """Rewards for each player"""

    done: bool
    """Whether episode is complete"""

    winner: int
    """Winner: 0, 1, or -1 for draw"""

    info: dict
    """Additional information"""


class ScenarioEnvironment:
    """
    Environment for running training scenarios.

    Unlike StandardEnvironment which requires two agents, this environment
    can run with just one agent (player 0) in simplified scenarios, or with
    a simple opponent for combat scenarios.
    """

    def __init__(self, scenario_config):
        """
        Initialize scenario environment.

        Args:
            scenario_config: TrainingScenario configuration
        """
        self.scenario = scenario_config

        # Create game config from scenario
        self.config = GameConfig(
            board_width=scenario_config.board_size[0],
            board_height=scenario_config.board_size[1],
            food_density=scenario_config.food_density,
            rock_density=scenario_config.rock_density,
            initial_ants_per_player=max(scenario_config.player_ants, scenario_config.opponent_ants),
            max_turns=scenario_config.max_turns
        )

        self.game: Optional[Game] = None
        self.turn_count = 0
        self.last_food_counts = {0: 0, 1: 0}

    def reset(self) -> List[List[AntObservation]]:
        """
        Reset environment for new episode.

        Returns:
            Initial observations for both players
        """
        # Create new game
        self.game = Game(self.config)
        self.turn_count = 0

        # Apply custom setup if provided
        if self.scenario.custom_setup:
            self.scenario.custom_setup(self.game)
        else:
            # Standard setup based on scenario config
            self._setup_scenario()

        # Place fixed food if specified
        if self.scenario.food_positions:
            for x, y in self.scenario.food_positions:
                if 0 <= x < self.config.board_width and 0 <= y < self.config.board_height:
                    self.game.board.add_food(Position(x, y))

        self.last_food_counts = {
            0: self.game.food_collected[0],
            1: self.game.food_collected[1]
        }

        # Get initial observations
        observations = [[], []]
        for ant_id, ant in self.game.board.ants.items():
            obs = self.game.get_ant_observation(ant_id)
            observations[ant.player_id].append(obs)

        return observations

    def _setup_scenario(self):
        """Setup scenario based on config."""
        # Adjust ant counts
        current_ants = {0: [], 1: []}
        for ant_id, ant in list(self.game.board.ants.items()):
            current_ants[ant.player_id].append(ant_id)

        # Remove excess ants for player 0
        while len(current_ants[0]) > self.scenario.player_ants:
            ant_id = current_ants[0].pop()
            ant = self.game.board.ants[ant_id]
            self.game.board._position_map.pop(ant.position, None)
            del self.game.board.ants[ant_id]

        # Remove excess ants for player 1
        while len(current_ants[1]) > self.scenario.opponent_ants:
            ant_id = current_ants[1].pop()
            ant = self.game.board.ants[ant_id]
            self.game.board._position_map.pop(ant.position, None)
            del self.game.board.ants[ant_id]

        # Position anthills if specified
        if self.scenario.player_anthill_pos:
            x, y = self.scenario.player_anthill_pos
            self.game.board.anthills[0].position = Position(x, y)

        if self.scenario.opponent_anthill_pos:
            x, y = self.scenario.opponent_anthill_pos
            self.game.board.anthills[1].position = Position(x, y)

    def step(self, actions: Dict[int, Dict[int, int]]) -> ScenarioStepResult:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary mapping player_id -> {ant_id: action}

        Returns:
            ScenarioStepResult with observations, rewards, done status, and info
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Track state before actions
        ants_before = {pid: len([a for a in self.game.board.ants.values() if a.player_id == pid])
                      for pid in [0, 1]}
        food_before = {pid: self.game.food_collected[pid] for pid in [0, 1]}
        anthill_alive_before = {pid: self.game.board.anthills[pid].alive for pid in [0, 1]}

        # Execute actions
        self.game.execute_turn(actions)
        self.turn_count += 1

        # Track state after actions
        ants_after = {pid: len([a for a in self.game.board.ants.values() if a.player_id == pid])
                     for pid in [0, 1]}
        food_after = {pid: self.game.food_collected[pid] for pid in [0, 1]}
        anthill_alive_after = {pid: self.game.board.anthills[pid].alive for pid in [0, 1]}

        # Calculate scenario-specific rewards
        rewards = {0: 0.0, 1: 0.0}

        for pid in [0, 1]:
            # Food collection rewards
            food_collected = food_after[pid] - food_before[pid]
            if food_collected > 0:
                rewards[pid] += food_collected * self.scenario.reward_food_deposited

            # Enemy killed rewards
            enemies_killed = ants_before[1 - pid] - ants_after[1 - pid]
            if enemies_killed > 0:
                rewards[pid] += enemies_killed * self.scenario.reward_enemy_killed

            # Anthill destruction rewards (alive -> dead)
            if anthill_alive_before[1 - pid] and not anthill_alive_after[1 - pid]:
                # Enemy anthill was destroyed
                rewards[pid] += self.scenario.reward_anthill_damaged

            # Own anthill destruction penalty
            if anthill_alive_before[pid] and not anthill_alive_after[pid]:
                # Own anthill was destroyed
                rewards[pid] -= abs(self.scenario.reward_anthill_damaged)

            # Death penalty
            ants_died = ants_before[pid] - ants_after[pid]
            if ants_died > 0:
                rewards[pid] += ants_died * self.scenario.penalty_death

            # Survival reward (per ant alive)
            if ants_after[pid] > 0:
                rewards[pid] += ants_after[pid] * self.scenario.reward_survival

        # Distance-based reward shaping (for player 0 only, the training agent)
        if self.scenario.reward_distance_to_food != 0 and len(self.game.board.ants) > 0:
            player_ants = [a for a in self.game.board.ants.values() if a.player_id == 0]
            if player_ants and len(self.game.board.food) > 0:
                # Find average distance to nearest food
                total_dist = 0
                for ant in player_ants:
                    min_dist = min(
                        abs(ant.position.x - food.position.x) + abs(ant.position.y - food.position.y)
                        for food in self.game.board.food
                    )
                    total_dist += min_dist
                avg_dist = total_dist / len(player_ants)
                rewards[0] += avg_dist * self.scenario.reward_distance_to_food

        if self.scenario.reward_distance_to_anthill != 0 and self.scenario.opponent_ants > 0:
            player_ants = [a for a in self.game.board.ants.values() if a.player_id == 0]
            enemy_anthill = self.game.board.anthills[1]
            if player_ants:
                total_dist = 0
                for ant in player_ants:
                    dist = abs(ant.position.x - enemy_anthill.position.x) + \
                           abs(ant.position.y - enemy_anthill.position.y)
                    total_dist += dist
                avg_dist = total_dist / len(player_ants)
                rewards[0] += avg_dist * self.scenario.reward_distance_to_anthill

        # Check if done
        done = False
        winner = -1

        # Standard win conditions
        if not self.game.board.anthills[0].alive:
            done = True
            winner = 1
        elif not self.game.board.anthills[1].alive:
            done = True
            winner = 0
        elif self.turn_count >= self.scenario.max_turns:
            done = True
            # Winner is player with most food
            if food_after[0] > food_after[1]:
                winner = 0
            elif food_after[1] > food_after[0]:
                winner = 1
            else:
                winner = -1

        # Scenario-specific win conditions
        # If no opponent ants and all player ants died, scenario failed
        if self.scenario.opponent_ants == 0 and ants_after[0] == 0:
            done = True
            winner = 1  # Loss

        # If survival scenario, player wins if they survive to the end
        if 'survival' in self.scenario.name.lower() and done and winner == -1 and ants_after[0] > 0:
            winner = 0

        # Get observations
        observations = [[], []]
        for ant_id, ant in self.game.board.ants.items():
            obs = self.game.get_ant_observation(ant_id)
            observations[ant.player_id].append(obs)

        # Build info dict
        info = {
            'turn': self.turn_count,
            'food_collected': {0: food_after[0], 1: food_after[1]},
            'ants_alive': {0: ants_after[0], 1: ants_after[1]},
            'anthill_alive': {0: anthill_alive_after[0], 1: anthill_alive_after[1]},
            'scenario': self.scenario.name
        }

        self.last_food_counts = food_after

        return ScenarioStepResult(
            observations=observations,
            rewards=rewards,
            done=done,
            winner=winner,
            info=info
        )

    def get_state(self):
        """Get current game state."""
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self.game.get_state()
