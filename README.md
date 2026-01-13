# Ant Hill Simulation

A reinforcement learning environment where ant colonies compete using deep learning models for decision-making.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ant-hill-simulation.git
cd ant-hill-simulation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run comprehensive tests to verify everything works
python tests/test_comprehensive.py

# Run a simple demo game
python scripts/demo_game.py

# Train a Q-learning agent with hybrid curriculum (recommended)
python training/train_curriculum.py --mode hybrid --episodes 10000

# Or train simply without curriculum
python training/train_simply.py --episodes 1000

# Evaluate trained agent
python scripts/evaluate_agents.py --agents models/curriculum_hybrid_final.pkl --opponents random smart_random greedy --games 100

# Visualize a trained agent
python scripts/demo_trained_agent.py demo --agent-path models/curriculum_hybrid_final.pkl
```

## Project Structure

```
ant-hill-simulation/
├── src/                     # Core game implementation
│   ├── core/               # Game mechanics (game.py, board.py, entities.py)
│   ├── agents/             # Agent implementations
│   │   ├── base.py         # Abstract base agent & observation encoding
│   │   ├── adaptive_q_learning/  # Adaptive Q-learning with decay
│   │   ├── q_learning/     # Simple Q-learning agent
│   │   ├── greedy/         # Greedy baseline agents
│   │   ├── tactical/       # Tactical decision-making agent
│   │   ├── random/         # Random baseline agents
│   │   ├── dqn/            # Deep Q-Network (needs PyTorch)
│   │   └── ppo/            # PPO agent (needs PyTorch)
│   ├── environments/       # RL environments
│   │   ├── base.py         # Base environment class
│   │   ├── standard.py     # Standard game environment
│   │   ├── scenario.py     # Scenario-based training environment
│   │   └── training.py     # Training environments with reward shaping
│   ├── utils/              # Utilities and configurations
│   ├── arena/              # Matchmaking and tournaments
│   └── visualization/      # Display systems (ASCII, GUI)
├── training/               # Training scripts and configurations
│   ├── train_curriculum.py      # Main curriculum-based trainer
│   ├── curriculum_config.py     # Curriculum phase definitions
│   ├── training_scenarios.py    # 25+ training scenarios
│   ├── train_simply.py          # Simple training without curriculum
│   └── train.py                 # Alternative training script
├── scripts/                # Demo and utility scripts
│   ├── demo_game.py             # Interactive game visualization
│   ├── demo_trained_agent.py    # Visualize trained agents
│   ├── evaluate_agents.py       # Evaluate agent performance
│   └── evaluate_tactical.py     # Tactical agent evaluation
├── tests/                  # Test suite
│   ├── test_comprehensive.py    # Full test suite (523 lines)
│   ├── test_integration.py      # Integration tests
│   ├── test_encoding.py         # Observation encoding tests
│   └── test_game_setup.py       # Game setup tests
├── models/                 # Saved trained models (.pkl files)
├── logs/                   # Training logs and statistics
├── CURRICULUM_GUIDE.md     # Comprehensive curriculum training guide
├── SCENARIO_TRAINING_GUIDE.md  # 25+ scenario descriptions
├── TRAINING_QUICK_REF.md   # Quick reference for training commands
└── README.md
```

## Game Rules

- **Board**: Configurable size grid (default 100x100)
- **Entities**:
  - **Ants**: Mobile units that can move one square per turn (including diagonally)
  - **Anthills**: Player bases - if destroyed, that player loses
  - **Food**: Collectible resources that spawn new ants
  - **Rocks**: Immovable obstacles
  
- **Mechanics**:
  - Ants have limited vision (1 square radius by default)
  - Collecting food spawns a new ant at the anthill
  - Multiple ants moving to the same square results in mutual destruction
  - Game ends when an anthill is destroyed or max turns reached
  - Winner determined by anthill destruction or food collected

## Observation Space

Ants receive observations as a 3-channel tensor:
- **Channel 0**: Entity type (0=empty, 1=wall, 2=rock, 3=food, 4=ant, 5=anthill)
- **Channel 1**: Team affiliation (-1=enemy, 0=neutral, 1=allied)
- **Channel 2**: Mobility (0=blocked, 1=passable)

See `src/agents/Observation encoding.md` for detailed documentation of the factored encoding system.

## Curriculum & Scenario-Based Training

The project includes a comprehensive curriculum training system with **25+ specialized training scenarios** and multiple training modes:

### Training Modes:
- **Hybrid Mode** (Recommended): Alternates between isolated scenario training and full opponent games
- **Opponent Mode**: Traditional training against other agents
- **Scenario Mode**: Focused skill development in simplified environments

### Key Features:
- Progressive difficulty through 6 training phases
- Isolated scenarios for food collection, combat, anthill attacks, survival, maze navigation, and more
- Automatic learning rate and epsilon decay schedules
- Phase-based checkpointing and evaluation
- Comprehensive performance tracking

### Quick Training:
```bash
# Recommended: Hybrid curriculum training (10,000 episodes ≈ 45 minutes)
python training/train_curriculum.py --mode hybrid --episodes 10000

# Fast baseline training (1,000 episodes ≈ 3 minutes)
python training/train_simply.py --episodes 1000

# Other modes: rapid, basic, standard, intensive, specialized, scenario
python training/train_curriculum.py --mode standard --episodes 5000
```

### Documentation:
- **[CURRICULUM_GUIDE.md](CURRICULUM_GUIDE.md)** - Complete curriculum system guide with 8 predefined modes
- **[SCENARIO_TRAINING_GUIDE.md](SCENARIO_TRAINING_GUIDE.md)** - Detailed descriptions of all 25+ scenarios
- **[TRAINING_QUICK_REF.md](TRAINING_QUICK_REF.md)** - Quick reference card with commands

### Available Agents:
- **AdaptiveQLearningAgent**: Tabular Q-learning with adaptive decay (no dependencies)
- **GreedyAgent**: Greedy baseline (food-focused, aggressive, defensive variants)
- **TacticalAgent**: Multi-objective tactical decision-making
- **RandomAgent**: Random baseline for benchmarking
- **DQN/PPO**: Neural network agents (requires PyTorch)

See **[AGENT_COMPARISON.md](AGENT_COMPARISON.md)** for detailed agent comparisons and when to use each.

## Creating Custom Agents

```python
from src.agents.base import BaseAgent
from src.core.game import GameState
from typing import Dict, List

class MyAgent(BaseAgent):
    def get_actions(self, observations: List[AntObservation], 
                   game_state: GameState) -> Dict[int, int]:
        """Return actions for all ants."""
        actions = {}
        for obs in observations:
            # Your logic here
            actions[obs.ant_id] = Action.NORTH
        return actions
    
    def reset(self):
        """Reset agent state for new game."""
        pass
```

## Running Experiments

```python
from src.environments.standard import StandardEnvironment
from src.agents.random.agent import RandomAgent
from src.utils.game_config import GameConfig

# Configure game
config = GameConfig(
    board_width=100,
    board_height=100,
    food_density=0.10,
    rock_density=0.05,
    max_turns=1000
)

# Create environment and agents
env = StandardEnvironment(config=config)
agent1 = RandomAgent(player_id=0)
agent2 = RandomAgent(player_id=1)

# Run game
observations = env.reset()
done = False

while not done:
    actions = {
        0: agent1.get_actions(observations[0], env.game.get_state()),
        1: agent2.get_actions(observations[1], env.game.get_state())
    }
    
    result = env.step(actions)
    observations = result.observations
    done = result.done
    
    # Process rewards, update models, etc.
```

## Test Results

**All systems tested and working:**
- Basic Components
- Board Operations  
- Game Initialization
- Game Mechanics
- Vision Encoding
- Agent Systems
- Environment Systems
- Visualization
- Game Setup
- Full Integration

## Future Work

- [x] Implement DQN agent (requires PyTorch)
- [x] Implement PPO agent (requires PyTorch)
- [x] Create comprehensive training curricula with 25+ scenarios
- [ ] Implement evolutionary strategies
- [ ] Complete GUI visualization (currently ASCII only)
- [ ] Build automated tournament system
- [ ] Add replay recording/playback
- [ ] Expand neural network agent training (DQN/PPO optimization)
- [ ] Add multi-agent cooperative training

## License

MIT