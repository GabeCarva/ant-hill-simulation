# Training Guide for AI Ant Colony

This guide explains how to train reinforcement learning agents for the ant colony game.

## Quick Start

### Simple Q-Learning (No PyTorch Required)

Train a basic Q-learning agent that doesn't require any deep learning libraries:

```bash
# Basic training
python train_simple.py --episodes 1000

# Train against smart opponent
python train_simple.py --episodes 2000 --opponent smart

# Larger board
python train_simple.py --episodes 1000 --board-size 30 30

# With visualization
python train_simple.py --episodes 100 --render

# Evaluate trained model
python train_simple.py --evaluate models/q_learning_final.pkl
```

### Deep Learning Agents (Requires PyTorch)

For more advanced agents, install PyTorch first:

```bash
# Install training dependencies
pip install -r requirements_training.txt

# Train DQN agent
python train.py --agent dqn --episodes 5000

# Train PPO agent
python train.py --agent ppo --episodes 5000 --device cuda

# Self-play training
python train.py --agent dqn --opponent self --env self_play

# Curriculum learning
python train.py --agent ppo --env curriculum --episodes 10000
```

## Training Agents

### Available Agents

1. **SimpleQLearningAgent** (No dependencies)
   - Tabular Q-learning with state discretization
   - Good for small boards and learning basics
   - Fast training, limited to simple strategies

2. **DQNAgent** (Requires PyTorch)
   - Deep Q-Network with experience replay
   - Handles continuous state space
   - Good for medium to large boards

3. **PPOAgent** (Requires PyTorch)
   - Proximal Policy Optimization
   - State-of-the-art policy gradient method
   - Best for complex strategies

### Training Environments

1. **Standard Environment**
   - Fixed opponent (random or smart_random)
   - Consistent difficulty
   - Good for initial training

2. **Self-Play Environment**
   - Agent plays against itself
   - Automatically increases difficulty
   - Prevents overfitting to specific opponents

3. **Curriculum Environment**
   - Gradually increases board size and complexity
   - Starts with 10x10, progresses to 100x100
   - Best for learning robust policies

## Training Parameters

### Q-Learning Parameters

- `learning_rate`: How quickly the agent learns (default: 0.1)
- `gamma`: Discount factor for future rewards (default: 0.95)
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_end`: Final exploration rate (default: 0.01)
- `epsilon_decay`: How quickly exploration decreases (default: 0.995)

### DQN Parameters

- `learning_rate`: Neural network learning rate (default: 0.001)
- `batch_size`: Replay buffer batch size (default: 32)
- `buffer_size`: Experience replay buffer size (default: 100000)
- `target_update_freq`: Steps between target network updates (default: 1000)

### PPO Parameters

- `learning_rate`: Policy network learning rate (default: 0.0003)
- `gae_lambda`: GAE lambda for advantage estimation (default: 0.95)
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `epochs`: PPO epochs per update (default: 4)

## Reward Structure

The training environments use the following reward structure:

### Base Rewards
- Food collection: +1.0
- New ant spawned: +0.5
- Enemy ant killed: +0.3
- Own ant lost: -0.3
- Win game: +10.0
- Lose game: -10.0
- Step penalty: -0.01 (encourages efficiency)

### Reward Shaping (Optional)
- Proximity to enemy anthill: +0.01 to +0.2
- Proximity to food: +0.02
- Exploration bonus: +0.02 per new square visited

## Monitoring Training

Training progress is saved to:
- `models/`: Saved model checkpoints
- `logs/`: Training statistics and metrics

Monitor training metrics:
```python
import json
import matplotlib.pyplot as plt

# Load training stats
with open('logs/q_learning_stats.json', 'r') as f:
    stats = json.load(f)

# Plot win rate
plt.plot(stats['episodes'], stats['wins'])
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Training Progress')
plt.show()
```

## Evaluating Agents

Test your trained agents:

```python
# Evaluate Q-learning agent
python train_simple.py --evaluate models/q_learning_final.pkl

# Run a demo game with trained agent
python demo_trained_agent.py --model models/q_learning_final.pkl
```

## Tips for Better Training

1. **Start Small**: Begin with small boards (10x10) before scaling up
2. **Tune Hyperparameters**: Adjust learning rate and epsilon decay
3. **Use Reward Shaping**: Enable reward shaping for faster learning
4. **Monitor Progress**: Check win rate and average reward regularly
5. **Save Checkpoints**: Save models frequently during training
6. **Vary Opponents**: Train against different opponent types
7. **Use Curriculum Learning**: Gradually increase difficulty

## Common Issues

### Training is Slow
- Reduce board size
- Increase learning rate (carefully)
- Use reward shaping
- Decrease max_turns

### Agent Not Learning
- Check exploration rate (epsilon)
- Verify rewards are being calculated correctly
- Try different state representation
- Increase training episodes

### Memory Issues
- Reduce replay buffer size for DQN
- Use smaller batch sizes
- Train on smaller boards
- Use SimpleQLearningAgent instead

## Example Training Runs

### Basic Q-Learning
```bash
# 1. Train on small board
python train_simple.py --episodes 500 --board-size 10 10

# 2. Train on medium board
python train_simple.py --episodes 1000 --board-size 20 20 --opponent smart

# 3. Evaluate
python train_simple.py --evaluate models/q_learning_final.pkl
```

### Advanced DQN Training (Requires PyTorch)
```bash
# 1. Initial training
python train.py --agent dqn --episodes 1000 --board-size 15 15

# 2. Self-play refinement
python train.py --agent dqn --opponent self --episodes 2000

# 3. Curriculum learning
python train.py --agent dqn --env curriculum --episodes 5000
```

## Next Steps

Once you have trained agents:
1. Test them against each other
2. Visualize their strategies
3. Fine-tune hyperparameters
4. Implement custom reward functions
5. Create tournaments between different agents

Happy training!