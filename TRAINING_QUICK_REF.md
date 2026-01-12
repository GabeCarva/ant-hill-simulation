# Training Quick Reference

## Most Common Commands

### Recommended: Hybrid Training (NEW!)
```bash
python training/train_curriculum.py --mode hybrid --episodes 20000
```
Time: ~20-40 minutes | Mixes scenarios + opponents | Expected: 35-55% win rate

### Standard Training (Opponent-Only)
```bash
python training/train_curriculum.py --mode standard --episodes 20000
```
Time: ~20-40 minutes | Traditional approach | Expected: 30-50% win rate

### Scenario-Based Training (NEW!)
```bash
python training/train_curriculum.py --mode scenario --episodes 10000
```
Time: ~10-20 minutes | Learn fundamentals first | Best for beginners

### Quick Test
```bash
python training/train_curriculum.py --mode rapid
```
Time: ~2-5 minutes | Good for: Testing, debugging

### Competition Training
```bash
python training/train_curriculum.py --mode intensive --episodes 50000
```
Time: ~1-2 hours | Expected: 40-60% win rate vs SmartRandom

## All Modes at a Glance

| Command | Episodes | Time | Use |
|---------|----------|------|-----|
| `--mode rapid` | 2K | ~2-5 min | Quick test |
| `--mode basic` | 5K | ~5-10 min | Basic training |
| `--mode standard` | 20K | ~20-40 min | Opponent-only (traditional) |
| `--mode intensive` | 50K | ~1-2 hours | Competition |
| `--mode scenario` 🆕 | 10K | ~10-20 min | Isolated skills first |
| `--mode hybrid` ⭐ | 20K | ~20-40 min | **RECOMMENDED** (scenarios + opponents) |
| `--mode aggressive` | 15K | ~15-30 min | Aggressive style |
| `--mode defensive` | 15K | ~15-30 min | Defensive style |
| `--mode adaptive` | 15K | ~15-30 min | Adaptive style |

## Custom Curriculum

```bash
# Format: phase_name:opponents:episodes,...
python training/train_curriculum.py --custom "basics:random:1000,mid:greedy:2000,hard:greedy_aggressive+greedy_defensive:3000"
```

**Available Opponents:**
- `random` - Random agent
- `smart_random` - Smart random with food preference
- `greedy` - Balanced heuristic
- `greedy_aggressive` - Aggressive heuristic
- `greedy_defensive` - Defensive heuristic

## Useful Options

```bash
# List all modes
python training/train_curriculum.py --list-modes

# Continue from checkpoint
python training/train_curriculum.py --mode standard --load models/checkpoint.pkl

# Custom episode count
python training/train_curriculum.py --mode standard --episodes 10000

# Custom board size
python training/train_curriculum.py --mode standard --board-size 30 30

# More evaluation games
python training/train_curriculum.py --mode standard --eval-games 50
```

## Output Files

**Models:** `models/curriculum_{mode}_final.pkl`
**Stats:** `logs/curriculum_{mode}_stats.json`
**Checkpoints:** `models/curriculum_{phase}_ep{N}.pkl`

## Expected Performance

### After Standard Training (20K episodes)
- vs Random: 70-95% win rate
- vs SmartRandom: 25-45% win rate ← KEY BENCHMARK
- vs Greedy: 35-50% win rate

### After Intensive Training (50K episodes)
- vs Random: 85-100% win rate
- vs SmartRandom: 40-60% win rate ← KEY BENCHMARK
- vs Greedy: 45-60% win rate

## New: Scenario Training 🆕

**What?** Train in isolated environments focusing on specific skills.

**Why?** Like the user said: "models may need to train in special scenarios - like a single ant on a board with just food, or a single ant on a board with only enemy ant hills"

**Available Scenarios:**
- Food collection (no enemies)
- Anthill attacks (no defenders)
- Combat training (1v1, 1v2, 1v3)
- Survival (outnumbered)
- Multi-ant efficiency
- Defense scenarios
- Special challenges (maze, races)

**How to use:**
```bash
# Scenario-only training
python training/train_curriculum.py --mode scenario

# Hybrid (scenarios + opponents) - RECOMMENDED
python training/train_curriculum.py --mode hybrid
```

See: `SCENARIO_TRAINING_GUIDE.md` for details

## Tips

1. **Start with hybrid**: Best approach for new agents (scenarios + opponents)
2. **Use full episodes**: Don't reduce below recommended
3. **Monitor phase evals**: Check progress between phases
4. **Try scenarios**: If agent struggles with basics, use scenario training
5. **Save checkpoints**: Resume interrupted training with `--load`
6. **Custom for goals**: Use custom curriculum for specific objectives

For full documentation, see:
- `CURRICULUM_GUIDE.md` - Complete curriculum system guide
- `SCENARIO_TRAINING_GUIDE.md` - Scenario-based training guide
