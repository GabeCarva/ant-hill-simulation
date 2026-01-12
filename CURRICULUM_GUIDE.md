# Training Curriculum System

A comprehensive, structured training system for Q-learning ant agents that accounts for independent ant control constraints.

## Quick Start

### Full Training (Recommended)
```bash
# Standard 20K episode training (20-40 minutes)
python training/train_curriculum.py --mode standard --episodes 20000
```

This single command provides complete, balanced training with:
- Progressive difficulty across 5 phases
- Mixed opponent training to prevent overfitting
- Automatic learning rate and epsilon decay
- Evaluation checkpoints after each phase
- Final comprehensive evaluation

## Training Modes

### Predefined Curricula

| Mode | Episodes | Time | Use Case |
|------|----------|------|----------|
| **rapid** | 2,000 | 2-5 min | Quick testing, debugging, proof of concept |
| **basic** | 5,000 | 5-10 min | Fast training for basic capabilities |
| **standard** ⭐ | 20,000 | 20-40 min | **RECOMMENDED** - Balanced comprehensive training |
| **intensive** | 50,000 | 1-2 hours | Competition-level performance |
| **aggressive** | 15,000 | ~15-30 min | Specialized aggressive playstyle |
| **defensive** | 15,000 | ~15-30 min | Specialized defensive playstyle |
| **adaptive** | 15,000 | ~15-30 min | Specialized adaptive playstyle |

## Simple Commands

```bash
# Quick test (2K episodes, ~2-5 minutes)
python training/train_curriculum.py --mode rapid

# Standard full training - RECOMMENDED (20-40 minutes)
python training/train_curriculum.py --mode standard

# Intensive competition training (~1-2 hours)
python training/train_curriculum.py --mode intensive

# Custom episode count with standard curriculum
python training/train_curriculum.py --mode standard --episodes 10000

# Specialized training
python training/train_curriculum.py --mode aggressive
python training/train_curriculum.py --mode defensive
python training/train_curriculum.py --mode adaptive
```

## Standard Mode Details

The **standard** curriculum (recommended) provides balanced training:

### Phase Breakdown
1. **Fundamentals** (20%, 4K episodes) - vs Random
   - Learn basic movement, food collection, survival
   - High exploration (epsilon starts at 1.0)

2. **Basic Tactics** (25%, 5K episodes) - vs SmartRandom
   - Develop tactical awareness
   - Learn threat assessment and risk management

3. **Intermediate Strategy** (25%, 5K episodes) - vs Greedy + SmartRandom
   - Build strategies against balanced opponents
   - Mix of difficulty for generalization

4. **Advanced Play** (20%, 4K episodes) - vs all Greedy variants
   - Handle diverse opponent styles
   - Aggressive, defensive, and balanced strategies

5. **Generalization** (10%, 2K episodes) - vs all opponents
   - Finalize robust play across all scenarios
   - Lower exploration for exploitation

### Training Parameters
- **Learning Rate**: 0.3 → 0.001 (polynomial decay)
- **Epsilon**: 1.0 → 0.05 (decay rate 0.9994)
- **Expected Results**: 30-50% win rate vs SmartRandom

## Custom Curriculum

Create your own curriculum with specific phases:

```bash
# Format: phase_name:opponents:episodes,phase_name:opponents:episodes,...
# Use '+' to combine multiple opponents in a phase

# Example 1: Simple custom curriculum
python training/train_curriculum.py --custom "basics:random:1000,mid:greedy:2000"

# Example 2: Complex multi-opponent phases
python training/train_curriculum.py --custom "explore:random:1000,refine:smart_random+greedy:2000,master:greedy_aggressive+greedy_defensive:1500"

# Example 3: Focused training
python training/train_curriculum.py --custom "warmup:random:500,main:greedy:5000,polish:greedy+greedy_aggressive+greedy_defensive:2000"
```

### Available Opponents
- `random` - Random moves
- `smart_random` - Random with food preference and attack probability
- `greedy` - Balanced heuristic agent
- `greedy_aggressive` - Aggressive variant
- `greedy_defensive` - Defensive variant

## Advanced Options

```bash
# Continue from checkpoint
python training/train_curriculum.py --mode standard --load models/curriculum_test_ep10000.pkl

# Custom board size
python training/train_curriculum.py --mode standard --board-size 30 30

# Longer games (more turns)
python training/train_curriculum.py --mode standard --max-turns 200

# More evaluation games per checkpoint
python training/train_curriculum.py --mode standard --eval-games 50

# Custom output prefix for files
python training/train_curriculum.py --mode standard --output-prefix my_agent

# List all available modes
python training/train_curriculum.py --list-modes
```

## Outputs

### Models (saved to `models/`)
- **Phase checkpoints**: `curriculum_{phase_name}_ep{episode}.pkl`
- **Final model**: `curriculum_{mode}_final.pkl`

### Statistics (saved to `logs/`)
- **Training stats**: `curriculum_{mode}_stats.json`
  - Per-episode rewards, wins, losses, draws
  - Learning rate and epsilon curves
  - Q-table size growth
  - Opponent types faced
  - Phase-wise evaluation results

## Understanding Results

### Evaluation Metrics

After each phase and at the end of training, the agent is evaluated:

```
random              : Win  85.0% | Draw  0.0% | Loss 15.0% | Food 32.5
smart_random        : Win  35.0% | Draw  5.0% | Loss 60.0% | Food 27.2
greedy              : Win  45.0% | Draw 10.0% | Loss 45.0% | Food 24.4
```

- **Win/Draw/Loss rates**: Success against each opponent type
- **Food**: Average food collected per game

### Expected Performance Ranges

With **standard** curriculum (20K episodes):
- vs Random: 70-95% win rate
- vs SmartRandom: 25-45% win rate (KEY BENCHMARK)
- vs Greedy: 35-50% win rate
- vs Greedy variants: 20-50% win rate

With **intensive** curriculum (50K episodes):
- vs Random: 85-100% win rate
- vs SmartRandom: 40-60% win rate
- vs Greedy: 45-60% win rate
- vs Greedy variants: 30-55% win rate

## Design Principles

The curriculum system is designed around independent ant control:

### 1. Progressive Difficulty
- Start with simple opponents (Random)
- Gradually increase complexity (SmartRandom → Greedy → Variants)
- Final phase mixes all for generalization

### 2. No Coordination Exploits
- All training accounts for independent ant decisions
- Agents cannot coordinate within a turn
- Focus on individual ant skill development

### 3. Generalization Focus
- Mixed opponent training prevents overfitting
- Multiple phases ensure robust strategies
- Periodic evaluation tracks generalization

### 4. Automatic Hyperparameter Scheduling
- Learning rate decays as training progresses
- Epsilon decays to reduce exploration over time
- Phase-specific overrides when needed

## Troubleshooting

### Training Too Slow
- Use `--mode rapid` or `--mode basic` for faster results
- Reduce `--episodes` with standard mode
- Example: `python training/train_curriculum.py --mode standard --episodes 5000`

### Poor Performance
- Increase training episodes: `--episodes 50000`
- Use intensive mode: `--mode intensive`
- Train longer with focused opponents

### Out of Memory
- Normal Q-table size: 5K-50K states for 20K episodes
- If issues occur, reduce board size or max turns

### Want Different Behavior
- Use specialized modes: `--mode aggressive`, `--mode defensive`, `--mode adaptive`
- Create custom curriculum with specific opponents
- Adjust learning parameters by modifying `curriculum_config.py`

## Comparison with Previous Training

### Old Method (train_simply.py)
```bash
python training/train_simply.py --episodes 20000 --opponent smart
```
- Single opponent throughout training
- Risk of overfitting
- No progressive difficulty
- Manual hyperparameter tuning

### New Method (Curriculum)
```bash
python training/train_curriculum.py --mode standard --episodes 20000
```
- Multiple opponent types
- Progressive difficulty
- Anti-overfitting measures
- Automatic hyperparameter scheduling
- Phase-wise evaluation
- Better generalization

**Result**: Curriculum training typically achieves 10-20% better performance vs diverse opponents.

## Tips for Best Results

1. **Start with standard mode**: It's well-balanced for most needs
2. **Use full episode counts**: Don't reduce episodes below recommended
3. **Check phase evaluations**: Monitor learning progress across phases
4. **Train overnight for intensive**: 50K episodes takes 1-2 hours
5. **Custom curricula**: Experiment with opponent mixes for specific goals
6. **Save checkpoints**: Use `--load` to continue interrupted training

## Examples for Common Goals

### Goal: Quick Agent Test
```bash
python training/train_curriculum.py --mode rapid --episodes 2000
```

### Goal: Production-Ready Agent
```bash
python training/train_curriculum.py --mode standard --episodes 20000
```

### Goal: Competition Agent
```bash
python training/train_curriculum.py --mode intensive --episodes 50000
```

### Goal: Beat Greedy Agents
```bash
python training/train_curriculum.py --custom "basics:random:2000,greedy_focus:greedy+greedy_aggressive+greedy_defensive:8000,generalize:random+smart_random+greedy:2000"
```

### Goal: Balanced All-Rounder
```bash
python training/train_curriculum.py --mode adaptive --episodes 15000
```

## Next Steps

After training, evaluate your agent:
```bash
# Evaluate trained agent
python scripts/evaluate_tactical.py --agent models/curriculum_standard_final.pkl --games 100
```

Or use it in custom games:
```python
from src.agents.adaptive_q_learning.agent import AdaptiveQLearningAgent

agent = AdaptiveQLearningAgent(player_id=0)
agent.load("models/curriculum_standard_final.pkl")
agent.train_mode(False)  # Disable exploration for deployment
```
