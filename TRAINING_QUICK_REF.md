# Training Quick Reference

## Most Common Commands

### Recommended: Full Standard Training
```bash
python training/train_curriculum.py --mode standard --episodes 20000
```
Time: ~20-40 minutes | Expected: 30-50% win rate vs SmartRandom

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
| `--mode standard` ⭐ | 20K | ~20-40 min | **RECOMMENDED** |
| `--mode intensive` | 50K | ~1-2 hours | Competition |
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

## Tips

1. **Start with standard**: Best balance of time/performance
2. **Use full episodes**: Don't reduce below recommended
3. **Monitor phase evals**: Check progress between phases
4. **Save checkpoints**: Resume interrupted training with `--load`
5. **Custom for goals**: Use custom curriculum for specific objectives

For full documentation, see: `CURRICULUM_GUIDE.md`
