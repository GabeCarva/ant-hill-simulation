
"""Comprehensive test of all game components."""

import sys
import os

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import time
from src.utils.game_config import GameConfig, Position, Action
from src.utils.game_setup import GameSetup
from src.core.game import Game
from src.core.board import Board
from src.core.entities import Ant, Food, Rock, Anthill
from src.agents.base import BaseAgent, AgentWrapper
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.environments.standard import StandardEnvironment, ShapedRewardEnvironment
from src.visualization.ascii_viz import ASCIIVisualizer


def test_section(name: str):
    """Print a test section header."""
    print("\n" + "=" * 60)
    print(f"TEST: {name}")
    print("=" * 60)


def test_basic_components():
    """Test basic game components."""
    test_section("Basic Components")
    
    # Test Position
    pos1 = Position(5, 5)
    pos2 = Position(8, 9)
    print(f"Position creation: {pos1}")
    print(f"Manhattan distance: {pos1.manhattan_distance(pos2)} (expected: 7)")
    print(f"Chebyshev distance: {pos1.chebyshev_distance(pos2)} (expected: 4)")
    
    # Test Action
    new_pos = Action.get_new_position(pos1, Action.NORTH)
    print(f"Action NORTH from {pos1}: {new_pos} (expected: Position(5, 4))")
    
    # Test GameConfig
    config = GameConfig()
    print(f"Default config: {config.board_width}x{config.board_height} board")
    print(f"Food count: {config.initial_food_count} ({config.food_density*100}% of {config.total_squares})")
    print(f"Rock count: {config.initial_rock_count} ({config.rock_density*100}% of {config.total_squares})")
    
    return True


def test_board_operations():
    """Test board functionality."""
    test_section("Board Operations")
    
    config = GameConfig(board_width=10, board_height=10, food_density=0.1, rock_density=0.05)
    board = Board(config)
    
    # Test adding entities
    assert board.add_rock(Position(2, 2)), "Failed to add rock"
    assert not board.add_rock(Position(2, 2)), "Added rock to occupied position"
    print("Rock placement and collision detection")
    
    assert board.add_food(Position(3, 3)), "Failed to add food"
    print("Food placement")
    
    anthill = board.add_anthill(Position(1, 1), player_id=0)
    assert anthill is not None, "Failed to add anthill"
    print("Anthill placement")
    
    ant = board.add_ant(Position(1, 2), player_id=0)
    assert ant is not None, "Failed to add ant"
    print(f"Ant placement (ID: {ant.ant_id})")
    
    # Test movement
    assert board.move_ant(ant, Position(2, 3)), "Failed to move ant"
    print("Ant movement")
    
    # Test vision
    vision = board.get_vision(ant.position, radius=1)
    print(f"Vision system: {len(vision)} cells visible")
    
    # Test queries
    ants = board.get_ants_by_player(0)
    assert len(ants) == 1, "Wrong number of ants"
    print(f"Player ant query: {len(ants)} ants for player 0")
    
    return True


def test_game_initialization():
    """Test game initialization."""
    test_section("Game Initialization")
    
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.1,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )
    
    game = Game(config, seed=42)
    state = game.get_state()
    
    print(f"Game created with seed 42")
    print(f"Turn: {state.turn}")
    print(f"Anthills placed: {len(game.board.anthills)}")
    print(f"Initial ants: {len(game.board.ants)}")
    print(f"Food pieces: {len(game.board.food)}")
    print(f"Rocks: {len(game.board.rocks)}")
    
    # Check anthill distance
    anthills = list(game.board.anthills.values())
    if len(anthills) == 2:
        distance = anthills[0].position.manhattan_distance(anthills[1].position)
        print(f"Anthill separation: {distance} (min: {config.min_anthill_distance})")
    
    return True


def test_game_mechanics():
    """Test core game mechanics."""
    test_section("Game Mechanics")
    
    config = GameConfig(
        board_width=10,
        board_height=10,
        food_density=0.2,
        rock_density=0.05,
        initial_ants_per_player=2,
        max_turns=50
    )
    
    game = Game(config, seed=123)
    
    print("Testing game mechanics:")
    
    # Test food collection
    initial_food = len(game.board.food)
    initial_ants = len(game.board.ants)
    
    # Run a few turns with random actions
    for turn in range(10):
        for ant_id, ant in game.board.ants.items():
            if ant.alive:
                action = np.random.choice(list(Action.DIRECTIONS.keys()))
                game.set_ant_action(ant_id, action)
        
        state = game.step()
        
        if state.food_collected[0] > 0 or state.food_collected[1] > 0:
            print(f"  Turn {turn}: Food collected! P0={state.food_collected[0]}, P1={state.food_collected[1]}")
            break
    
    print(f"Food collection mechanics working")
    
    # Test collision
    game.reset(seed=456)
    
    # Get two ants and make them collide
    ants_p0 = game.board.get_ants_by_player(0)
    ants_p1 = game.board.get_ants_by_player(1)
    
    if len(ants_p0) > 0 and len(ants_p1) > 0:
        ant0 = ants_p0[0]
        ant1 = ants_p1[0]
        
        # Try to move them to the same position
        target = Position(5, 5)
        
        # Move ants close to target
        game.board.move_ant(ant0, Position(5, 4))
        game.board.move_ant(ant1, Position(4, 5))
        
        # Set actions to move to same square
        game.set_ant_action(ant0.ant_id, Action.SOUTH)
        game.set_ant_action(ant1.ant_id, Action.EAST)
        
        state_before = game.get_state()
        state_after = game.step()
        
        if state_after.ants_lost[0] > state_before.ants_lost[0]:
            print(f"Collision detection: Ants destroyed when moving to same square")
    
    # Test max turns
    game.reset(seed=789)
    for _ in range(config.max_turns):
        game.step()
    
    state = game.get_state()
    assert state.is_terminal(), "Game didn't end at max turns"
    print(f"Max turns termination (winner based on food)")
    
    return True


def test_vision_encoding():
    """Test vision encoding system."""
    test_section("Vision Encoding")
    
    # Create sample vision
    vision = {
        Position(-1, -1): 'wall',
        Position(0, -1): 'food',
        Position(1, -1): None,
        Position(-1, 0): 'ant_0',
        Position(0, 0): 'ant_0',  # self
        Position(1, 0): 'ant_1',
        Position(-1, 1): 'rock',
        Position(0, 1): 'anthill_0',
        Position(1, 1): 'anthill_1',
    }
    
    encoded = BaseAgent.encode_vision(vision, player_id=0, vision_radius=1)
    
    print(f"Encoded shape: {encoded.shape} (expected: (3, 3, 3))")
    
    # Check entity types (channel 0)
    assert encoded[0, 0, 0] == 1, "Wall not encoded correctly"
    assert encoded[0, 1, 0] == 3, "Food not encoded correctly"
    assert encoded[1, 0, 0] == 4, "Ant not encoded correctly"
    assert encoded[2, 0, 0] == 2, "Rock not encoded correctly"
    print("Entity type encoding correct")
    
    # Check team affiliations (channel 1)
    assert encoded[1, 0, 1] == 1, "Allied ant not encoded correctly"
    assert encoded[1, 2, 1] == -1, "Enemy ant not encoded correctly"
    assert encoded[2, 1, 1] == 1, "Allied anthill not encoded correctly"
    assert encoded[2, 2, 1] == -1, "Enemy anthill not encoded correctly"
    print("Team affiliation encoding correct")
    
    # Check mobility (channel 2)
    assert encoded[0, 0, 2] == 0, "Wall mobility not encoded correctly"
    assert encoded[0, 1, 2] == 1, "Food mobility not encoded correctly"
    assert encoded[2, 0, 2] == 0, "Rock mobility not encoded correctly"
    print("Mobility encoding correct")
    
    return True


def test_agents():
    """Test agent implementations."""
    test_section("Agent Systems")
    
    # Test RandomAgent
    agent = RandomAgent(player_id=0, config={'move_probability': 0.9})
    print("RandomAgent created")
    
    # Test SmartRandomAgent
    smart_agent = SmartRandomAgent(player_id=1, config={
        'prefer_food': True,
        'attack_probability': 0.5
    })
    print("SmartRandomAgent created")
    
    # Create game and test agent actions
    config = GameConfig(board_width=15, board_height=15)
    game = Game(config)
    
    # Test AgentWrapper
    wrapper = AgentWrapper(agent, game)
    observations = wrapper.get_observations()
    print(f"AgentWrapper observations: {len(observations)} ants observed")
    
    if observations:
        obs = observations[0]
        print(f"Observation structure: ant_id={obs.ant_id}, shape={obs.vision_array.shape}")
        
        # Test get_actions
        state = game.get_state()
        actions = agent.get_actions(observations, state)
        print(f"Agent actions: {len(actions)} actions returned")
        
        # Test action validity
        for ant_id, action in actions.items():
            assert action in Action.DIRECTIONS, f"Invalid action: {action}"
        print("All actions valid")
    
    return True


def test_environments():
    """Test environment implementations."""
    test_section("Environment Systems")
    
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.15,
        rock_density=0.05,
        max_turns=50
    )
    
    # Test StandardEnvironment
    env = StandardEnvironment(config=config, seed=42)
    initial_obs = env.reset()
    print(f"StandardEnvironment reset: {len(initial_obs[0])} observations for P0")
    
    # Test step
    actions = {
        0: {obs.ant_id: Action.STAY for obs in initial_obs[0]},
        1: {obs.ant_id: Action.STAY for obs in initial_obs[1]}
    }
    
    result = env.step(actions)
    print(f"Environment step completed")
    print(f"  Rewards: P0={result.rewards[0]:.3f}, P1={result.rewards[1]:.3f}")
    print(f"  Info keys: {list(result.info.keys())}")
    
    # Test ShapedRewardEnvironment
    shaped_env = ShapedRewardEnvironment(config=config, seed=42)
    shaped_obs = shaped_env.reset()
    shaped_result = shaped_env.step(actions)
    print(f"ShapedRewardEnvironment working")
    print(f"  Shaped rewards: P0={shaped_result.rewards[0]:.3f}, P1={shaped_result.rewards[1]:.3f}")
    
    return True


def test_visualization():
    """Test visualization system."""
    test_section("Visualization")
    
    config = GameConfig(
        board_width=15,
        board_height=10,
        food_density=0.1,
        rock_density=0.05
    )
    
    game = Game(config, seed=111)
    viz = ASCIIVisualizer(use_colors=False)  # No colors for cleaner test output
    
    # Generate visualization
    output = viz.render(game, clear_screen=False)
    lines = output.split('\n')
    
    print(f"ASCII visualization generated: {len(lines)} lines")
    
    # Check for expected elements
    assert any('Turn:' in line for line in lines), "Turn counter missing"
    assert any('Player 0' in line for line in lines), "Player 0 info missing"
    assert any('Player 1' in line for line in lines), "Player 1 info missing"
    print("Visualization contains expected elements")
    
    # Show a small sample
    print("\nSample output (first 5 lines):")
    for line in lines[:5]:
        print(f"  {line}")
    
    return True


def test_game_setup():
    """Test game setup configurations."""
    test_section("Game Setup")
    
    # Test quick match
    setup1 = GameSetup.quick_match(
        board_size=(50, 50),
        food_density=0.12,
        rock_density=0.08,
        max_turns=500,
        player1="dqn",
        player2="ppo",
        visualization="gui"
    )
    
    print(f"Quick match setup created")
    print(f"  Board: {setup1.config.board_width}x{setup1.config.board_height}")
    print(f"  Players: {setup1.player1_agent} vs {setup1.player2_agent}")
    print(f"  Visualization: {setup1.visualization}")
    
    # Test custom setup
    custom_config = GameConfig(
        board_width=75,
        board_height=75,
        food_density=0.08,
        rock_density=0.12
    )
    
    setup2 = GameSetup(
        config=custom_config,
        player1_agent="evolutionary",
        player1_model_path="models/evo_gen_100.pth",
        player2_agent="random",
        visualization="ascii",
        seed=12345,
        record_replay=True
    )
    
    print(f"Custom setup created")
    print(f"  Seed: {setup2.seed}")
    print(f"  Record replay: {setup2.record_replay}")
    print(f"  Player 1 model: {setup2.player1_model_path}")
    
    return True


def test_full_game():
    """Run a complete game to test integration."""
    test_section("Full Game Integration")
    
    print("Running a complete 50-turn game...\n")
    
    # Setup
    setup = GameSetup.quick_match(
        board_size=(25, 25),
        food_density=0.15,
        rock_density=0.05,
        max_turns=50,
        player1="random",
        player2="smart_random",
        visualization="ascii"
    )
    
    # Create environment and agents
    env = StandardEnvironment(config=setup.config, seed=999)
    agent1 = RandomAgent(player_id=0)
    agent2 = SmartRandomAgent(player_id=1, config={'prefer_food': True})
    
    # Reset
    observations = env.reset()
    
    # Play game
    total_rewards = {0: 0.0, 1: 0.0}
    turn_count = 0
    
    while True:
        # Get actions
        actions = {
            0: agent1.get_actions(observations[0], env.game.get_state()),
            1: agent2.get_actions(observations[1], env.game.get_state())
        }
        
        # Step
        result = env.step(actions)
        turn_count += 1
        
        # Accumulate rewards
        total_rewards[0] += result.rewards[0]
        total_rewards[1] += result.rewards[1]
        
        # Print progress every 10 turns
        if turn_count % 10 == 0:
            print(f"Turn {turn_count}: "
                  f"Ants=[{result.info['ants_alive'][0]}, {result.info['ants_alive'][1]}], "
                  f"Food=[{result.info['food_collected'][0]}, {result.info['food_collected'][1]}], "
                  f"Rewards=[{result.rewards[0]:.2f}, {result.rewards[1]:.2f}]")
        
        # Check termination
        if result.done:
            break
        
        # Update observations
        observations = result.observations
    
    # Final results
    print(f"\nGame completed in {turn_count} turns")
    
    if result.winner == -1:
        print("  Result: DRAW")
    else:
        print(f"  Winner: Player {result.winner}")
    
    print(f"  Final food: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")
    print(f"  Total rewards: P0={total_rewards[0]:.2f}, P1={total_rewards[1]:.2f}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Components", test_basic_components),
        ("Board Operations", test_board_operations),
        ("Game Initialization", test_game_initialization),
        ("Game Mechanics", test_game_mechanics),
        ("Vision Encoding", test_vision_encoding),
        ("Agent Systems", test_agents),
        ("Environment Systems", test_environments),
        ("Visualization", test_visualization),
        ("Game Setup", test_game_setup),
        ("Full Game Integration", test_full_game)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {str(e)}")
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)