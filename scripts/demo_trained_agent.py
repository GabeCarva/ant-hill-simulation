"""Demo script to visualize trained agents playing the game."""

import os
import sys
import time
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer

# Try to import deep learning agents
try:
    from src.agents.dqn.agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

try:
    from src.agents.ppo.agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False


def load_agent(agent_type: str, model_path: str, player_id: int):
    """Load a trained agent from file."""
    if agent_type == "q_learning":
        agent = SimpleQLearningAgent(player_id=player_id)
        agent.load(model_path)
        agent.train_mode(False)  # Disable exploration
        return agent
    
    elif agent_type == "dqn":
        if not DQN_AVAILABLE:
            raise ValueError("DQN requires PyTorch")
        agent = DQNAgent(player_id=player_id)
        agent.load(model_path)
        agent.train_mode(False)
        return agent
    
    elif agent_type == "ppo":
        if not PPO_AVAILABLE:
            raise ValueError("PPO requires PyTorch")
        agent = PPOAgent(player_id=player_id)
        agent.load(model_path)
        agent.train_mode(False)
        return agent
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_demo_game(
    agent_type: str,
    model_path: str,
    opponent_type: str = "random",
    board_size: tuple = (20, 20),
    max_turns: int = 100,
    delay: float = 0.1,
    save_replay: bool = False
):
    """Run a demo game with a trained agent."""
    
    print("=" * 60)
    print("TRAINED AGENT DEMO")
    print("=" * 60)
    print(f"Agent: {agent_type} (from {model_path})")
    print(f"Opponent: {opponent_type}")
    print(f"Board: {board_size[0]}x{board_size[1]}")
    print(f"Max turns: {max_turns}")
    print()
    input("Press Enter to start...")
    
    # Create game config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=max_turns
    )
    
    # Create environment
    env = StandardEnvironment(config=config)
    
    # Load trained agent
    agent = load_agent(agent_type, model_path, player_id=0)
    
    # Create opponent
    if opponent_type == "smart":
        opponent = SmartRandomAgent(
            player_id=1,
            config={'prefer_food': True, 'attack_probability': 0.3}
        )
    elif opponent_type == "trained":
        # Load another trained agent as opponent
        opponent = load_agent(agent_type, model_path, player_id=1)
    else:
        opponent = RandomAgent(player_id=1)
    
    # Visualizer
    viz = ASCIIVisualizer(use_colors=True)
    
    # Replay storage
    replay = [] if save_replay else None
    
    # Reset
    observations = env.reset()
    agent.reset()
    opponent.reset()
    
    done = False
    turn = 0
    
    while not done and turn < max_turns:
        # Clear screen and show state
        print("\033[2J\033[H")  # Clear screen
        print(viz.render(env.game, clear_screen=False))
        
        # Get actions
        state = env.game.get_state()
        agent_actions = agent.get_actions(observations[0], state)
        opponent_actions = opponent.get_actions(observations[1], state)
        
        # Show some agent decisions
        if observations[0]:
            print("\nAgent decisions:")
            for i, obs in enumerate(observations[0][:3]):  # Show first 3 ants
                ant_id = obs.ant_id
                if ant_id in agent_actions:
                    action_name = ['STAY', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][agent_actions[ant_id]]
                    print(f"  Ant {ant_id}: {action_name}")
        
        # Step
        actions = {0: agent_actions, 1: opponent_actions}
        result = env.step(actions)
        
        # Save replay frame
        if save_replay:
            replay.append({
                'turn': turn,
                'board': viz.render(env.game, clear_screen=False),
                'actions': actions,
                'food_collected': result.info['food_collected'].copy()
            })
        
        observations = result.observations
        done = result.done
        turn += 1
        
        # Delay for visualization
        time.sleep(delay)
    
    # Game over
    print("\033[2J\033[H")  # Clear screen
    print(viz.render(env.game, clear_screen=False))
    
    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    
    # Determine victory
    if result.winner == -1:
        print("Result: DRAW")
    elif result.winner == 0:
        print("Result: TRAINED AGENT WINS!")
    else:
        print("Result: OPPONENT WINS!")
    
    print(f"\nFinal Statistics:")
    print(f"  Turns played: {turn}")
    print(f"  Food collected: Agent={result.info['food_collected'][0]}, Opponent={result.info['food_collected'][1]}")
    print(f"  Ants lost: Agent={result.info['ants_lost'][0]}, Opponent={result.info['ants_lost'][1]}")
    
    # Check anthill status
    for player_id in [0, 1]:
        if not env.game.board.anthills[player_id].alive:
            print(f"  Player {player_id}'s anthill was destroyed!")
    
    # Save replay
    if save_replay:
        import json
        replay_file = f"replay_{agent_type}_{int(time.time())}.json"
        with open(replay_file, 'w') as f:
            json.dump(replay, f, indent=2, default=str)
        print(f"\nReplay saved to {replay_file}")


def run_tournament(models: list, num_games: int = 10):
    """Run a tournament between multiple trained agents."""
    
    print("=" * 60)
    print("AGENT TOURNAMENT")
    print("=" * 60)
    
    # Parse model entries
    agents_info = []
    for model_entry in models:
        parts = model_entry.split(':')
        if len(parts) != 2:
            print(f"Invalid model entry: {model_entry} (expected format: type:path)")
            continue
        agent_type, model_path = parts
        agents_info.append((agent_type, model_path))
    
    print(f"Participants: {len(agents_info)} agents")
    for i, (agent_type, model_path) in enumerate(agents_info):
        print(f"  {i+1}. {agent_type} ({os.path.basename(model_path)})")
    print()
    
    # Create environment
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )
    env = StandardEnvironment(config=config)
    
    # Tournament matrix
    num_agents = len(agents_info)
    wins = [[0] * num_agents for _ in range(num_agents)]
    
    # Run matches
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue  # Skip self-play
            
            agent_type_i, model_path_i = agents_info[i]
            agent_type_j, model_path_j = agents_info[j]
            
            print(f"Match: Agent {i+1} vs Agent {j+1}")
            
            # Load agents
            agent_i = load_agent(agent_type_i, model_path_i, player_id=0)
            agent_j = load_agent(agent_type_j, model_path_j, player_id=1)
            
            # Run games
            for game in range(num_games):
                observations = env.reset()
                agent_i.reset()
                agent_j.reset()
                
                done = False
                while not done:
                    state = env.game.get_state()
                    actions_i = agent_i.get_actions(observations[0], state)
                    actions_j = agent_j.get_actions(observations[1], state)
                    
                    actions = {0: actions_i, 1: actions_j}
                    result = env.step(actions)
                    
                    observations = result.observations
                    done = result.done
                
                if result.winner == 0:
                    wins[i][j] += 1
            
            print(f"  Result: {wins[i][j]}/{num_games} wins for Agent {i+1}")
    
    # Print results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    # Calculate total wins
    total_wins = [sum(row) for row in wins]
    rankings = sorted(range(num_agents), key=lambda i: total_wins[i], reverse=True)
    
    print("\nRankings:")
    for rank, i in enumerate(rankings):
        agent_type, model_path = agents_info[i]
        print(f"{rank+1}. Agent {i+1} ({agent_type}): {total_wins[i]} total wins")
    
    print("\nHead-to-head results:")
    print("     ", end="")
    for j in range(num_agents):
        print(f"  A{j+1} ", end="")
    print()
    
    for i in range(num_agents):
        print(f"A{i+1}:  ", end="")
        for j in range(num_agents):
            if i == j:
                print("  -  ", end="")
            else:
                print(f" {wins[i][j]:2d}  ", end="")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demo trained ant agents")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demo game')
    demo_parser.add_argument(
        '--agent-type',
        type=str,
        default='q_learning',
        choices=['q_learning', 'dqn', 'ppo'],
        help='Type of agent'
    )
    demo_parser.add_argument(
        '--model',
        type=str,
        default='models/q_learning_final.pkl',
        help='Path to trained model'
    )
    demo_parser.add_argument(
        '--opponent',
        type=str,
        default='random',
        choices=['random', 'smart', 'trained'],
        help='Opponent type'
    )
    demo_parser.add_argument(
        '--board-size',
        type=int,
        nargs=2,
        default=[20, 20],
        help='Board size'
    )
    demo_parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between turns (seconds)'
    )
    demo_parser.add_argument(
        '--save-replay',
        action='store_true',
        help='Save game replay'
    )
    
    # Tournament command
    tournament_parser = subparsers.add_parser('tournament', help='Run a tournament')
    tournament_parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Models to compete (format: type:path)'
    )
    tournament_parser.add_argument(
        '--games',
        type=int,
        default=10,
        help='Games per matchup'
    )
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo_game(
            agent_type=args.agent_type,
            model_path=args.model,
            opponent_type=args.opponent,
            board_size=tuple(args.board_size),
            delay=args.delay,
            save_replay=args.save_replay
        )
    elif args.command == 'tournament':
        run_tournament(
            models=args.models,
            num_games=args.games
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
"""Demo script to visualize trained agents playing."""

import os
import sys
import time
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.game_config import GameConfig
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent, SmartRandomAgent
from src.agents.q_learning.agent import SimpleQLearningAgent
from src.visualization.ascii_viz import ASCIIVisualizer
from demo_simple import get_victory_reason

# Try to import deep learning agents
try:
    from src.agents.dqn.agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

try:
    from src.agents.ppo.agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False


def demo_trained_agent(
    model_path: str,
    agent_type: str = "q_learning",
    opponent_type: str = "random",
    board_size: tuple = (20, 20),
    num_games: int = 1,
    delay: float = 0.1
):
    """Demo a trained agent."""
    
    print("=" * 60)
    print("TRAINED AGENT DEMO")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Agent type: {agent_type}")
    print(f"Opponent: {opponent_type}")
    print(f"Board size: {board_size[0]}x{board_size[1]}")
    print()
    
    # Create game config
    config = GameConfig(
        board_width=board_size[0],
        board_height=board_size[1],
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )
    
    # Create environment
    env = StandardEnvironment(config=config)
    
    # Load trained agent
    if agent_type == "q_learning":
        agent = SimpleQLearningAgent(player_id=0)
        agent.load(model_path)
        agent.train_mode(False)  # Disable exploration
    elif agent_type == "dqn" and DQN_AVAILABLE:
        agent = DQNAgent(player_id=0)
        agent.load(model_path)
        agent.train_mode(False)
    elif agent_type == "ppo" and PPO_AVAILABLE:
        agent = PPOAgent(player_id=0)
        agent.load(model_path)
        agent.train_mode(False)
    else:
        print(f"Agent type {agent_type} not available!")
        return
    
    # Create opponent
    if opponent_type == "smart":
        opponent = SmartRandomAgent(
            player_id=1,
            config={'prefer_food': True, 'attack_probability': 0.3}
        )
    else:
        opponent = RandomAgent(player_id=1)
    
    # Visualizer
    viz = ASCIIVisualizer(use_colors=True)
    
    # Play games
    for game_num in range(num_games):
        if num_games > 1:
            print(f"\n{'='*40}")
            print(f"GAME {game_num + 1}/{num_games}")
            print(f"{'='*40}")
        
        # Reset
        observations = env.reset()
        agent.reset()
        opponent.reset()
        
        done = False
        steps = 0
        
        print("\nInitial State:")
        print(viz.render(env.game, clear_screen=False))
        input("Press Enter to start...")
        
        while not done:
            # Clear screen
            print("\033[2J\033[H")
            
            # Get actions
            state = env.game.get_state()
            agent_actions = agent.get_actions(observations[0], state)
            opponent_actions = opponent.get_actions(observations[1], state)
            
            # Step
            actions = {0: agent_actions, 1: opponent_actions}
            result = env.step(actions)
            
            # Render
            print(viz.render(env.game, clear_screen=False))
            print(f"Step: {steps + 1}")
            print(f"Food: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")
            print(f"Ants: P0={result.info['ants_alive'][0]}, P1={result.info['ants_alive'][1]}")
            
            observations = result.observations
            done = result.done
            steps += 1
            
            # Delay
            time.sleep(delay)
        
        # Show final result
        print("\n" + "=" * 40)
        print("GAME OVER!")
        print("=" * 40)
        
        victory_reason = get_victory_reason(env, result)
        print(f"\n{victory_reason}")
        
        print(f"\nFinal Statistics:")
        print(f"  Total steps: {steps}")
        print(f"  Food collected: P0={result.info['food_collected'][0]}, P1={result.info['food_collected'][1]}")
        print(f"  Ants lost: P0={result.info['ants_lost'][0]}, P1={result.info['ants_lost'][1]}")
        
        if num_games > 1 and game_num < num_games - 1:
            input("\nPress Enter for next game...")


def compare_agents(
    model_paths: list,
    agent_types: list,
    num_games: int = 10
):
    """Compare multiple trained agents."""
    
    print("=" * 60)
    print("AGENT COMPARISON")
    print("=" * 60)
    
    # Create environment
    config = GameConfig(
        board_width=20,
        board_height=20,
        food_density=0.10,
        rock_density=0.05,
        initial_ants_per_player=3,
        max_turns=100
    )
    
    env = StandardEnvironment(config=config)
    
    # Test each agent
    results = {}
    
    for model_path, agent_type in zip(model_paths, agent_types):
        print(f"\nTesting {agent_type} from {model_path}")
        
        # Load agent
        if agent_type == "q_learning":
            agent = SimpleQLearningAgent(player_id=0)
            agent.load(model_path)
            agent.train_mode(False)
        else:
            print(f"Skipping {agent_type} (not available)")
            continue
        
        # Test against different opponents
        agent_results = {}
        
        for opp_name, opponent in [
            ("random", RandomAgent(player_id=1)),
            ("smart", SmartRandomAgent(player_id=1, config={'prefer_food': True}))
        ]:
            wins = 0
            total_food = 0
            
            for _ in range(num_games):
                observations = env.reset()
                agent.reset()
                opponent.reset()
                
                done = False
                while not done:
                    state = env.game.get_state()
                    agent_actions = agent.get_actions(observations[0], state)
                    opponent_actions = opponent.get_actions(observations[1], state)
                    
                    actions = {0: agent_actions, 1: opponent_actions}
                    result = env.step(actions)
                    
                    observations = result.observations
                    done = result.done
                
                if result.winner == 0:
                    wins += 1
                total_food += result.info['food_collected'][0]
            
            agent_results[opp_name] = {
                'win_rate': wins / num_games,
                'avg_food': total_food / num_games
            }
        
        results[agent_type] = agent_results
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for agent_type, agent_results in results.items():
        print(f"\n{agent_type}:")
        for opp_name, stats in agent_results.items():
            print(f"  vs {opp_name}:")
            print(f"    Win rate: {stats['win_rate']:.1%}")
            print(f"    Avg food: {stats['avg_food']:.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Demo trained ant agents")
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/q_learning_final.pkl",
        help="Path to trained model"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="q_learning",
        choices=["q_learning", "dqn", "ppo"],
        help="Agent type"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "smart"],
        help="Opponent type"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Board size (width height)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps (seconds)"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple models (format: path:type)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Parse model paths and types
        model_paths = []
        agent_types = []
        for item in args.compare:
            if ':' in item:
                path, agent_type = item.split(':')
                model_paths.append(path)
                agent_types.append(agent_type)
            else:
                model_paths.append(item)
                agent_types.append("q_learning")
        
        compare_agents(model_paths, agent_types)
    else:
        demo_trained_agent(
            model_path=args.model,
            agent_type=args.type,
            opponent_type=args.opponent,
            board_size=tuple(args.board_size),
            num_games=args.games,
            delay=args.delay
        )


if __name__ == "__main__":
    main()