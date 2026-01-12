# Scenario-Based Training Guide

## What Are Training Scenarios?

Training scenarios are **simplified, isolated environments** that focus on specific skills without the complexity of full opponent agents. Think of them as "training drills" where agents master fundamental behaviors before facing complete game situations.

### Key Insight

You were absolutely right! Training only against full opponent models means agents must learn everything at once:
- Food collection
- Combat tactics
- Anthill attacks
- Multi-ant efficiency
- Defensive positioning

**Scenario-based training** breaks this down into manageable pieces.

## Quick Start

### Scenario-Only Training
```bash
# Train in isolated scenarios, then transition to full games
python training/train_curriculum.py --mode scenario --episodes 10000
```

**What this does:**
1. Food collection practice (no enemies)
2. Combat basics (1v1, 1v2)
3. Anthill attack drills
4. Multi-ant efficiency
5. Then full opponent games

### Hybrid Training (Recommended)
```bash
# Mix scenarios and opponent training throughout
python training/train_curriculum.py --mode hybrid --episodes 20000
```

**What this does:**
- Alternates between isolated skill training and full games
- Learn skill → apply in game → learn next skill → apply
- Best of both worlds

## Available Scenarios

### Food Collection (Solo)
Single ant, abundant food, no enemies - learn basic movement and collection.

| Scenario | Difficulty | Description |
|----------|------------|-------------|
| `food_easy` | Easy | 20% food density - learn basics |
| `food_medium` | Medium | 10% food density - standard collection |
| `food_hard` | Hard | 5% food density - sparse food pathfinding |

**Teaches:** Movement, pathfinding to food, returning to anthill

### Anthill Attack (Solo)
Single ant, enemy anthill only, no defenders - learn to find and attack.

| Scenario | Distance | Description |
|----------|----------|-------------|
| `anthill_close` | Close | Enemy anthill nearby |
| `anthill_medium` | Medium | Mid-range pathfinding |
| `anthill_far` | Far | Long-distance navigation |

**Teaches:** Pathfinding to enemy anthill, persistent attacking

### Combat Training
Single ant vs enemy ants, no anthills to attack - pure combat.

| Scenario | Enemies | Description |
|----------|---------|-------------|
| `combat_1v1` | 1 | Fair fight - learn basic combat |
| `combat_1v2` | 2 | Outnumbered 2:1 - learn tactics |
| `combat_1v3` | 3 | Heavily outnumbered - when to fight |

**Teaches:** Combat tactics, when to fight vs flee

### Survival (Outnumbered)
Stay alive as long as possible when heavily outnumbered.

| Scenario | Enemies | Description |
|----------|---------|-------------|
| `survival_1v3` | 3 | 3:1 - tactical retreat |
| `survival_1v5` | 5 | 5:1 - evasion tactics |
| `survival_1v8` | 8 | 8:1 - extreme survival |

**Teaches:** Evasion, tactical retreat, survival when outnumbered

### Efficient Collection (Multi-ant)
Multiple ants, lots of food, no enemies - learn independent efficiency.

| Scenario | Ants | Description |
|----------|------|-------------|
| `efficient_2ants` | 2 | Two ants working independently |
| `efficient_3ants` | 3 | Three ants parallel collection |
| `efficient_5ants` | 5 | Five ants maximum efficiency |

**Teaches:** Independent efficiency (remember: ants can't coordinate!)

### Contested Collection
Multiple ants per side, compete for food, no anthill attacks.

| Scenario | Teams | Description |
|----------|-------|-------------|
| `contested_2v2` | 2v2 | Resource competition |
| `contested_3v3` | 3v3 | Larger team competition |

**Teaches:** Competing for resources, balancing collection vs defense

### Defense Scenarios
Protect your anthill from attackers.

| Scenario | Attackers | Description |
|----------|-----------|-------------|
| `defense_2attackers` | 2 | Defend against 2 enemies |
| `defense_3attackers` | 3 | Defend against 3 enemies |
| `defense_5attackers` | 5 | Defend against 5 enemies |

**Teaches:** Defensive positioning, anthill protection

### Special Challenges
Advanced scenarios for specific skills.

| Scenario | Description |
|----------|-------------|
| `maze` | Navigate complex maze with rocks to collect food |
| `food_race` | Race opponent to single food item |

**Teaches:** Complex pathfinding, speed, efficiency

## Training Modes Comparison

### Opponent-Only Training (Traditional)
```bash
python training/train_curriculum.py --mode standard --episodes 20000
```
- ✅ Learns complete game behavior
- ✅ Tested approach
- ❌ Must learn everything at once
- ❌ Slow early progress
- ❌ May develop bad habits before learning basics

### Scenario-Only Training
```bash
python training/train_curriculum.py --mode scenario --episodes 10000
```
- ✅ Master fundamentals first
- ✅ Faster initial learning
- ✅ Clear skill progression
- ❌ May need extra time for full-game integration
- ❌ Scenario-specific strategies might not transfer

### Hybrid Training (Recommended!)
```bash
python training/train_curriculum.py --mode hybrid --episodes 20000
```
- ✅ Best of both worlds
- ✅ Learn skill → apply immediately
- ✅ Continuous validation
- ✅ Good skill transfer
- ✅ Balanced development

## Hybrid Curriculum Breakdown

The hybrid mode alternates between scenarios and opponents:

### Phase 1: Fundamentals Scenarios (15%, 3K episodes)
- `food_easy`, `food_medium`, `combat_1v1`
- Learn basics in isolation

### Phase 2: Fundamentals Opponents (15%, 3K episodes)
- vs Random, SmartRandom
- Apply learned basics

### Phase 3: Intermediate Scenarios (15%, 3K episodes)
- `anthill_medium`, `combat_1v2`, `contested_2v2`
- Learn intermediate tactics

### Phase 4: Intermediate Opponents (20%, 4K episodes)
- vs SmartRandom, Greedy
- Apply tactics in full games

### Phase 5: Advanced Scenarios (10%, 2K episodes)
- `survival_1v3`, `defense_3attackers`, `maze`
- Master difficult challenges

### Phase 6: Advanced Opponents (25%, 5K episodes)
- vs all Greedy variants
- Final integration and mastery

## How Scenarios Work

### Reward Shaping

Scenarios use custom rewards to guide learning:

```python
# Food collection scenarios
+10  per food item picked up
+50  per food deposited at anthill
+0.1 per turn survived

# Combat scenarios
+50  per enemy killed
+1.0 per turn survived
-100 if you die

# Anthill attack scenarios
+100 per point of anthill damage
-0.02 per unit distance from enemy anthill (encourages approach)

# Survival scenarios
+2.0 per turn survived (high reward for staying alive)
+100 if you kill enemy (bonus for unlikely win)
-200 if you die (big penalty)
```

### Environment Simplification

Scenarios remove complexity:
- **Food collection**: No enemies to worry about
- **Combat**: No food collection distraction
- **Anthill attack**: No defenders to fight
- **Multi-ant**: No enemy interference

This lets agents focus on ONE skill at a time.

## When to Use Each Mode

### Use `--mode scenario` when:
- Agent struggles with basic food collection
- Want strong fundamentals before competition
- Have time for 10K scenario + 10K opponent training
- Building agent from scratch

### Use `--mode hybrid` when:
- Want balanced skill development
- Have 20K episodes to spare
- Want continuous skill → application cycle
- Best general-purpose training

### Use `--mode standard` when:
- Want traditional opponent-only training
- Agent already has good fundamentals
- Limited time (opponents-only is faster)
- Comparing against baseline approaches

## Expected Performance

### After Scenario Training (10K episodes)
- **Strong fundamentals**: Excellent food collection, basic combat
- **Weak integration**: May not apply skills effectively in full games initially
- **Needs follow-up**: Should train 5-10K more episodes vs opponents

### After Hybrid Training (20K episodes)
- **Balanced skills**: Good fundamentals + game integration
- **Better generalization**: Skills transfer well
- **Expected**: 35-55% win rate vs SmartRandom (potentially better than standard)

### After Standard Training (20K episodes)
- **Complete game skills**: Learned everything in context
- **Expected**: 30-50% win rate vs SmartRandom
- **Solid baseline**: Proven approach

## Advanced: Custom Scenario Curricula

You can mix scenarios and opponents however you like:

```bash
# Heavy scenario focus
python training/train_curriculum.py --custom "food:food_easy+food_medium:2000,combat:combat_1v1+combat_1v2:2000,integrate:random+smart_random:3000"

# Scenario sandwich (scenarios → opponents → scenarios)
python training/train_curriculum.py --custom "basics:food_easy:1000,games1:random:2000,advanced:survival_1v3+maze:1000,games2:greedy:3000"

# Specific skill focus
python training/train_curriculum.py --custom "food_drill:food_hard:3000,anthill_drill:anthill_far:2000,full_game:greedy:5000"
```

**Note:** Custom curricula require manual specification of scenario vs opponent phases in the code. The examples above assume phases are created with appropriate `training_type`.

## Scenario Design Philosophy

### Independent Ant Control

All scenarios are designed for **independent ant control**:
- Each ant makes decisions without knowledge of teammates
- No coordination within a single turn
- Ants cannot share plans or information

### Progressive Difficulty

Scenarios increase in difficulty:
1. **Solo + No Enemies**: Master basics (food collection)
2. **Solo + Enemies**: Learn combat (1v1, 1v2)
3. **Multi-ant + No Enemies**: Learn efficiency (parallel work)
4. **Multi-ant + Enemies**: Learn competition (contested resources)
5. **Complex Scenarios**: Master advanced skills (maze, survival)

### Skill Transfer

Scenarios are designed so skills transfer to full games:
- Food collection → Full game food collection
- Combat tactics → Full game combat
- Anthill pathfinding → Full game anthill attacks
- Independent efficiency → Full game multi-ant play

## Tips for Best Results

1. **Start with hybrid**: It's the most balanced approach
2. **Monitor phase evaluations**: Check if skills are transferring
3. **Extend scenario phases**: If agent struggles, add more episodes
4. **Mix difficulties**: Include easy scenarios even in late training
5. **Don't skip opponent training**: Scenarios alone aren't enough
6. **Use scenarios for debugging**: If agent fails at specific skill, train that scenario

## Troubleshooting

### Agent learns scenarios but fails in full games
- **Problem**: Skills not transferring
- **Solution**: Use hybrid mode for better integration
- **Alternative**: Add more opponent training after scenarios

### Agent ignores learned skills
- **Problem**: Opponent training overwriting scenario learning
- **Solution**: Reduce learning rate for opponent phases
- **Alternative**: Interleave scenarios throughout training (hybrid mode)

### Training too slow
- **Problem**: Creating new scenario environments each episode
- **Solution**: This is normal; scenarios are lightweight
- **Note**: Scenario episodes are typically faster than opponent episodes

### Rewards seem wrong
- **Problem**: Scenario rewards don't match game rewards
- **Solution**: This is intentional - scenarios use reward shaping
- **Note**: Shaped rewards guide learning; agent re-learns proper rewards in full games

## Summary

**Scenario-based training** lets agents master fundamental skills in isolation before tackling the complexity of full games. The **hybrid mode** provides the best balance by alternating between isolated skill training and full game experience.

Try it out:
```bash
# Recommended: Hybrid training
python training/train_curriculum.py --mode hybrid --episodes 20000
```

This will train your agent with the best combination of isolated skill development and full game experience!
