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
python scripts/demo_simple.py

# Train a Q-learning agent (no dependencies needed!)
python training/train_simple.py --episodes 1000

# Visualize a trained agent
python scripts/demo_trained_agent.py demo --agent-type q_learning --model models/q_learning_final.pkl
```

## Project Structure

```
ant-hill-simulation/
├── src/                     # Core game implementation
│   ├── core/               # Game mechanics
│   │   ├── game.py        # Main game loop and state management
│   │   ├── board.py       # Board representation and operations
│   │   └── entities.py    # Game entities (Ant, Food, Rock, Anthill)
│   ├── agents/            # Agent implementations
│   │   ├── base.py        # Abstract base agent class
│   │   ├── random/        # Random baseline agents
│   │   ├── q_learning/    # Q-learning agent
│   │   ├── dqn/           # Deep Q-Network (needs PyTorch)
│   │   └── ppo/           # PPO agent (needs PyTorch)
│   ├── environments/      # RL environments
│   │   ├── base.py        # Base environment class
│   │   ├── standard.py    # Standard game environment
│   │   └── training.py    # Training environments with reward shaping
│   ├── utils/             # Utilities
│   │   ├── game_config.py # Game configuration
│   │   └── game_setup.py  # Quick setup helpers
│   └── visualization/     # Display systems
│       └── ascii_viz.py   # ASCII visualization
├── training/              # Training scripts
│   ├── train_simple.py    # Q-learning training (no dependencies)
│   └── train.py          # Full training script (needs PyTorch)
├── scripts/              # Demo and utility scripts
│   ├── demo_simple.py    # Basic game demo
│   ├── demo_game.py      # Interactive visualization
│   └── demo_trained_agent.py # Visualize trained agents
├── tests/                # Test suite
│   ├── test_comprehensive.py # Full test suite
│   └── test_anthill_safety.py # Safety tests
├── models/               # Saved trained models
├── logs/                 # Training logs
└── docs/                 # Additional documentation
├── tests/                   # Test files
├── docs/                    # Documentation
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

- [ ] Implement DQN agent
- [ ] Implement PPO agent
- [ ] Implement evolutionary strategies
- [ ] Add GUI visualization
- [ ] Create training curricula
- [ ] Build tournament system
- [ ] Add replay recording/playback

## License

MIT