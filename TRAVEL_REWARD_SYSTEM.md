# Travel Reward System Documentation

## Overview

The travel reward system is designed to incentivize agent exploration and efficient movement patterns. By rewarding movement, agents can learn to explore the map more effectively, find resources faster, and develop better positioning strategies.

## Motivation

Without movement rewards, agents may develop overly passive strategies:
- **Camping**: Staying near the anthill waiting for food to spawn nearby
- **Minimal Exploration**: Only moving when food is visible
- **Inefficient Paths**: Not learning optimal routes to resources
- **Poor Map Coverage**: Leaving large areas unexplored

Movement rewards encourage:
- **Active Exploration**: Discovering more of the map
- **Resource Discovery**: Finding food and enemy positions faster
- **Map Control**: Establishing presence across the board
- **Strategic Positioning**: Learning advantageous locations

## Current Tracking

The game now tracks **total distance traveled** for each player:
- **Metric**: `game.distance_traveled[player_id]`
- **Calculation**: Chebyshev distance (max of abs(dx), abs(dy))
  - Suitable for 8-directional movement
  - 1 for orthogonal moves (N, S, E, W)
  - 1 for diagonal moves (NE, NW, SE, SW)
  - 0 for staying in place
- **Accumulation**: Summed across all ants for the entire game

### Example:
```python
# Ant at (5, 5) moves to (6, 7)
dx = abs(6 - 5) = 1
dy = abs(7 - 5) = 2
distance = max(1, 2) = 2  # Chebyshev distance

# Total for player:
game.distance_traveled[player_id] += 2
```

## Proposed Reward Structures

### 1. **Simple Distance Reward** (Recommended Starting Point)

Reward a small amount for each unit of distance traveled.

```python
# In environment's calculate_reward()
distance_moved = curr_state.distance_traveled[player_id] - prev_state.distance_traveled[player_id]
travel_reward = distance_moved * TRAVEL_REWARD_WEIGHT

# Example weights:
TRAVEL_REWARD_WEIGHT = 0.01  # Small reward per unit distance
```

**Pros:**
- Simple to implement
- Encourages active movement
- Scales naturally with game length

**Cons:**
- May encourage aimless wandering
- Doesn't distinguish between useful and useless movement

**Recommended For:** Initial experiments, baseline exploration behavior

---

### 2. **Exploration Reward** (Territory Discovery)

Reward visiting new positions (first-time visits).

```python
class ExplorationRewardEnvironment(BaseEnvironment):
    def __init__(self, config=None, seed=None):
        super().__init__(config, seed)
        self.visited_positions = {0: set(), 1: set()}  # player_id -> set of (x, y) tuples

    def calculate_reward(self, player_id, prev_state, curr_state, observations):
        # Base rewards (food, combat, etc.)
        reward = self._calculate_base_reward(...)

        # Exploration bonus
        exploration_bonus = 0.0
        for obs in observations:
            pos_tuple = (obs.position.x, obs.position.y)
            if pos_tuple not in self.visited_positions[player_id]:
                exploration_bonus += EXPLORATION_REWARD
                self.visited_positions[player_id].add(pos_tuple)

        return reward + exploration_bonus

# Example weight:
EXPLORATION_REWARD = 0.05  # Reward for each new position visited
```

**Pros:**
- Encourages map coverage
- Rewards actual exploration, not just movement
- Natural diminishing returns (finite board)

**Cons:**
- Requires tracking visited positions
- More memory usage
- May encourage spread over concentration

**Recommended For:** Agents that need to learn map control and positioning

---

### 3. **Diminishing Distance Reward** (Efficient Movement)

Reward movement with diminishing returns to prevent aimless wandering.

```python
def calculate_travel_reward(distance_moved, max_useful_distance=10):
    """
    Reward distance with diminishing returns.

    Uses logarithmic scaling so early movement is rewarded more than
    excessive movement.
    """
    if distance_moved == 0:
        return 0.0

    # Logarithmic scaling
    import math
    normalized_distance = min(distance_moved / max_useful_distance, 1.0)
    reward = TRAVEL_BASE_REWARD * math.log(1 + normalized_distance * 9)  # log(1) to log(10)

    return reward

# Example weights:
TRAVEL_BASE_REWARD = 0.02
```

**Pros:**
- Encourages movement without overvaluing it
- Natural balance between exploration and other objectives
- Prevents infinite travel rewards

**Cons:**
- More complex to tune
- Requires understanding of typical game distances

**Recommended For:** Advanced agents, post-initial training

---

### 4. **Directed Movement Reward** (Goal-Oriented)

Reward movement toward strategic objectives (food, enemy anthill).

```python
def calculate_reward(self, player_id, prev_state, curr_state, observations):
    # Base rewards
    reward = self._calculate_base_reward(...)

    # Directed movement bonus
    for obs in observations:
        # Reward moving toward visible food
        closest_food = self._find_closest_food(obs)
        if closest_food:
            prev_distance = self._distance(prev_ant_positions[obs.ant_id], closest_food)
            curr_distance = self._distance(obs.position, closest_food)
            if curr_distance < prev_distance:
                reward += APPROACH_REWARD

        # Reward moving toward enemy anthill
        enemy_anthill = self._find_enemy_anthill(curr_state, player_id)
        if enemy_anthill:
            prev_distance = self._distance(prev_ant_positions[obs.ant_id], enemy_anthill)
            curr_distance = self._distance(obs.position, enemy_anthill)
            if curr_distance < prev_distance:
                reward += AGGRESSION_REWARD

    return reward

# Example weights:
APPROACH_REWARD = 0.03   # Moving toward food
AGGRESSION_REWARD = 0.02  # Moving toward enemy
```

**Pros:**
- Encourages goal-oriented behavior
- Learns strategic positioning
- Combines exploration with objectives

**Cons:**
- Complex to implement
- Requires tracking previous positions
- May overfit to visible objectives

**Recommended For:** Specialized training scenarios, tactical behavior

---

### 5. **Efficiency Reward** (Movement per Food)

Reward low distance-to-food ratio (efficient resource collection).

```python
def calculate_efficiency_reward(food_collected, distance_traveled):
    """
    Reward efficiency: more food with less movement is better.

    This encourages finding food quickly and taking efficient paths.
    """
    if distance_traveled == 0:
        return 0.0

    efficiency = food_collected / (distance_traveled + 1)  # +1 to avoid division by zero
    reward = efficiency * EFFICIENCY_WEIGHT

    return reward

# Calculate at game end or periodically
game_end_efficiency_bonus = calculate_efficiency_reward(
    game.food_collected[player_id],
    game.distance_traveled[player_id]
)

# Example weight:
EFFICIENCY_WEIGHT = 5.0  # Scales with food/distance ratio
```

**Pros:**
- Encourages smart, purposeful movement
- Rewards finding nearby food
- Discourages excessive wandering

**Cons:**
- Only meaningful with multiple food collections
- May discourage long-distance exploration
- Requires careful weight tuning

**Recommended For:** Advanced training, resource collection scenarios

---

## Implementation Guidelines

### Phase 1: Baseline (No Travel Rewards)
Train agents without travel rewards to establish baseline performance.

```bash
# Standard curriculum training
python training/train_curriculum.py --mode hybrid --episodes 10000
```

### Phase 2: Add Simple Travel Reward
Modify environment to add basic distance rewards.

```python
# In StandardEnvironment or custom environment
class TravelRewardEnvironment(StandardEnvironment):
    TRAVEL_REWARD_WEIGHT = 0.01

    def calculate_reward(self, player_id, prev_state, curr_state, observations):
        # Base reward
        reward = super().calculate_reward(player_id, prev_state, curr_state, observations)

        # Add travel reward
        distance_moved = (curr_state.distance_traveled[player_id] -
                         prev_state.distance_traveled[player_id])
        travel_reward = distance_moved * self.TRAVEL_REWARD_WEIGHT

        return reward + travel_reward
```

### Phase 3: Experiment and Tune
- Start with very small weights (0.001 - 0.01)
- Monitor behavior changes:
  - Average distance traveled per game
  - Food collection rate
  - Win rate vs baselines
- Adjust weights based on desired behavior

### Phase 4: Compare Performance
Use arena script to compare agents with/without travel rewards:

```bash
# Test baseline agent
python scripts/arena.py -m models/baseline.pkl -o greedy -g 100

# Test travel-reward agent
python scripts/arena.py -m models/travel_reward.pkl -o greedy -g 100

# Compare distance statistics
# Expected: Travel-reward agent moves more and explores better
```

---

## Recommended Weight Ranges

Based on standard reward values:

| Reward Type | Typical Range | Notes |
|-------------|---------------|-------|
| Food Collection | 1.0 - 10.0 | Primary objective |
| Enemy Ant Kill | 0.3 - 3.0 | Combat reward |
| Win Game | 10.0 - 100.0 | Game-end bonus |
| **Travel (Simple)** | **0.001 - 0.05** | **Keep very small** |
| **Exploration** | **0.01 - 0.1** | **Per new position** |
| **Directed Movement** | **0.01 - 0.05** | **Per step toward goal** |

**Key Principle:** Travel rewards should be **significantly smaller** than primary objectives (food, kills, wins). Otherwise, agents may prioritize movement over winning.

---

## Anti-Patterns to Avoid

### 1. **Over-Rewarding Movement**
```python
TRAVEL_REWARD = 0.5  # ❌ TOO HIGH
# Result: Agents run around aimlessly, ignoring food and combat
```

### 2. **Rewarding Total Distance Only**
```python
# ❌ Only reward at game end
if game_over:
    reward += total_distance * 0.1
# Problem: No feedback during game, hard to learn
```

### 3. **Ignoring Context**
```python
# ❌ Reward all movement equally
reward += distance * 0.01
# Problem: Moving away from food is rewarded same as toward it
```

### 4. **No Cap on Rewards**
```python
# ❌ Unbounded travel rewards
reward += infinite_distance_reward
# Problem: Agent may prioritize movement over all else
```

---

## Expected Behavioral Changes

### With Travel Rewards:
✅ **More Exploration**: Agents visit more of the map
✅ **Faster Discovery**: Find food and enemies sooner
✅ **Active Play**: Less waiting, more proactive movement
✅ **Better Positioning**: Learn strategic locations

### Potential Negative Effects:
⚠️ **Aimless Wandering**: If weight too high
⚠️ **Ignoring Objectives**: If movement overvalued
⚠️ **Inefficient Paths**: If not combined with other rewards
⚠️ **Higher Collision Rate**: More movement = more ant collisions

---

## Testing and Evaluation

### Metrics to Track:
1. **Average Distance Traveled**: Should increase with travel rewards
2. **Food Collection Rate**: Should not decrease (or increase if exploration helps)
3. **Win Rate**: Primary metric - must not degrade
4. **Map Coverage**: Percentage of board visited
5. **Time to First Food**: Should decrease with better exploration

### Comparison Script:
```bash
# Generate comparison report
python scripts/arena.py -m models/baseline.pkl -o greedy -g 100 > baseline_stats.txt
python scripts/arena.py -m models/travel_reward.pkl -o greedy -g 100 > travel_stats.txt

# Compare distance_traveled values
# Travel reward agent should have higher average distance
```

---

## Implementation Checklist

- [ ] **Baseline Performance**: Train agent without travel rewards
- [ ] **Environment Setup**: Create TravelRewardEnvironment class
- [ ] **Weight Selection**: Start with TRAVEL_REWARD_WEIGHT = 0.01
- [ ] **Training**: Train with travel rewards enabled
- [ ] **Evaluation**: Compare using arena script
- [ ] **Distance Analysis**: Verify increased movement
- [ ] **Win Rate Check**: Ensure performance not degraded
- [ ] **Behavior Review**: Visualize games to check for aimless wandering
- [ ] **Weight Tuning**: Adjust based on observations
- [ ] **Documentation**: Record findings and optimal weights

---

## Future Enhancements

1. **Adaptive Travel Rewards**: Decrease weight as agent improves
2. **Context-Aware Rewards**: Different weights for different game phases
3. **Multi-Objective Optimization**: Balance exploration vs exploitation dynamically
4. **Curriculum for Movement**: Start high, decrease over training
5. **Position-Based Rewards**: Reward visiting strategic areas more

---

## References

- **Current Implementation**: `src/core/game.py` lines 41-42, 210-211
- **Statistics Tracking**: `scripts/arena.py` win condition and distance reporting
- **Example Environment**: `src/environments/training.py` (ShapedRewardEnvironment includes exploration bonus)

---

## Summary

**Status**: 🟡 **Ready for Implementation**

The infrastructure is in place:
- ✅ Distance tracking implemented in `Game` class
- ✅ Statistics reported in arena script
- ✅ Documentation completed
- 🔜 Environment implementation needed
- 🔜 Training and evaluation needed

**Next Steps**:
1. Create `TravelRewardEnvironment` in `src/environments/`
2. Train agent with simple distance reward (weight = 0.01)
3. Evaluate using arena script
4. Compare distance and win rate metrics
5. Tune weight based on results

**Contact:** See training team for implementation assistance.
