# Observation Encoding Documentation

## Overview

Each ant receives observations about its local environment within its vision radius. These observations are encoded into a numerical format suitable for neural network processing using a factored representation.

## Factored Encoding

The vision encoding separates entity properties into three independent channels:

### Input Shape
- Shape: `(2*vision_radius+1, 2*vision_radius+1, 3)`
- Default vision_radius = 1, so shape is `(3, 3, 3)`

### Channel Layout

#### Channel 0: Entity Type (Categorical)
- `0` = Empty space
- `1` = Wall (boundary)
- `2` = Rock (obstacle)
- `3` = Food (resource)
- `4` = Ant (unit)
- `5` = Anthill (base)

#### Channel 1: Team Affiliation (Continuous)
- `-1.0` = Enemy
- `0.0` = Neutral/None
- `+1.0` = Allied

#### Channel 2: Mobility (Binary)
- `0` = Blocked (cannot move through)
- `1` = Passable (can move through)

## Example

For an ant with vision_radius=1, seeing:
```
[wall]      [food]      [empty]
[ally ant]  [self]      [enemy ant]
[rock]      [ally hill] [enemy hill]
```

The encoded observation would be:

**Channel 0 (Entity Type):**
```
1  3  0
4  4  4
2  5  5
```

**Channel 1 (Team):**
```
 0  0  0
+1 +1 -1
 0 +1 -1
```

**Channel 2 (Mobility):**
```
0  1  1
1  1  1
0  1  1
```

## Benefits of Factored Encoding

1. **Clean Separation**: Separates "what" (entity type) from "whose" (team affiliation)
2. **Compositional Learning**: Neural networks can learn interactions between entity types and teams
3. **Explicit Navigation**: Mobility channel provides direct pathfinding information
4. **Reduced Correlation**: Less correlation between features compared to one-hot encoding
5. **Compact Representation**: Only 3 channels for complete information

## Usage in Neural Networks

### Input Layer Design
```python
# For CNN-based models
input_shape = (3, 3, 3)  # (height, width, channels)

# PyTorch
self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

# TensorFlow/Keras
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 3, 3)))
```

### Interpreting the Encoding
The factored encoding allows networks to learn patterns like:
- "Entity type 4 (ant) with team -1 (enemy)" → Avoid or attack
- "Entity type 5 (anthill) with team -1 (enemy)" → High-priority target
- "Entity type 3 (food) with team 0 (neutral)" → Collect
- "Mobility 0" → Plan path around obstacle

## Implementation Details

The encoding is implemented in `src/agents/base.py` in the `BaseAgent.encode_vision()` static method. This method is called automatically when observations are collected for agents in both the `AgentWrapper` and `BaseEnvironment` classes.