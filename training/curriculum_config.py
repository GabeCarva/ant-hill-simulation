"""
Training curriculum configurations for ant agents.

This module defines structured training curricula that progressively build
agent capabilities while accounting for independent ant control constraints.

Key Design Principles:
1. Progressive Difficulty: Start with simple opponents, advance to complex ones
2. Generalization: Mix opponents to prevent overfitting
3. Exploration-Exploitation Balance: Adjust epsilon across phases
4. Learning Rate Scheduling: Decay learning rate as training progresses
5. Independent Control: All curricula designed for agents that control
   individual ants without coordination
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CurriculumPhase:
    """A single phase in a training curriculum."""

    name: str
    """Name of this phase (e.g., 'basic_behaviors', 'intermediate_tactics')"""

    episodes: int
    """Number of episodes in this phase"""

    opponents: Optional[List[str]] = None
    """List of opponent types to train against (randomly selected each episode)"""

    scenarios: Optional[List[str]] = None
    """List of scenario names to train in (randomly selected each episode)"""

    training_type: str = "opponent"
    """Type of training: 'opponent' (vs agents) or 'scenario' (isolated scenarios)"""

    learning_rate: Optional[float] = None
    """Learning rate for this phase (None = use agent's current LR)"""

    epsilon: Optional[float] = None
    """Epsilon override for this phase (None = use agent's natural decay)"""

    epsilon_decay: Optional[float] = None
    """Epsilon decay rate for this phase (None = use agent's default)"""

    eval_freq: int = 200
    """Evaluate agent every N episodes during this phase"""

    save_checkpoint: bool = True
    """Whether to save a checkpoint at the end of this phase"""

    description: str = ""
    """Human-readable description of what this phase teaches"""


@dataclass
class TrainingCurriculum:
    """A complete training curriculum consisting of multiple phases."""

    name: str
    """Curriculum name"""

    phases: List[CurriculumPhase]
    """Ordered list of training phases"""

    description: str = ""
    """Description of the overall curriculum"""

    initial_learning_rate: float = 0.3
    """Starting learning rate"""

    final_learning_rate: float = 0.001
    """Ending learning rate"""

    learning_rate_decay_type: str = "polynomial"
    """Type of LR decay: 'exponential', 'polynomial', 'step', 'none'"""

    initial_epsilon: float = 1.0
    """Starting exploration rate"""

    final_epsilon: float = 0.05
    """Minimum exploration rate"""

    epsilon_decay: float = 0.9994
    """Default epsilon decay rate"""

    def total_episodes(self) -> int:
        """Calculate total episodes across all phases."""
        return sum(phase.episodes for phase in self.phases)

    def get_phase_distribution(self) -> Dict[str, float]:
        """Get the percentage of episodes in each phase."""
        total = self.total_episodes()
        return {
            phase.name: phase.episodes / total
            for phase in self.phases
        }


# ============================================================================
# PREDEFINED CURRICULA
# ============================================================================

def get_basic_curriculum(total_episodes: int = 5000) -> TrainingCurriculum:
    """
    Basic curriculum for quick training and testing.

    Focus: Learn fundamental behaviors against simple opponents
    Time: ~5-10 minutes for 5000 episodes
    Use: Quick iteration, debugging, baseline testing
    """
    return TrainingCurriculum(
        name="basic",
        description="Fast training against simple opponents for testing",
        phases=[
            CurriculumPhase(
                name="exploration",
                episodes=int(total_episodes * 0.3),
                opponents=['random'],
                epsilon=1.0,
                description="High exploration phase - learn basic movement and food collection"
            ),
            CurriculumPhase(
                name="refinement",
                episodes=int(total_episodes * 0.4),
                opponents=['random', 'smart_random'],
                description="Mix easy opponents - refine basic strategies"
            ),
            CurriculumPhase(
                name="challenge",
                episodes=int(total_episodes * 0.3),
                opponents=['smart_random', 'greedy'],
                description="Harder opponents - learn defensive play"
            ),
        ],
        initial_learning_rate=0.3,
        final_learning_rate=0.01,
        initial_epsilon=1.0,
        final_epsilon=0.05
    )


def get_standard_curriculum(total_episodes: int = 20000) -> TrainingCurriculum:
    """
    Standard curriculum for balanced training.

    Focus: Comprehensive skill development with progressive difficulty
    Time: ~20-40 minutes for 20000 episodes
    Use: Default recommended training for most agents
    """
    return TrainingCurriculum(
        name="standard",
        description="Comprehensive balanced training with progressive difficulty",
        phases=[
            CurriculumPhase(
                name="fundamentals",
                episodes=int(total_episodes * 0.2),
                opponents=['random'],
                epsilon=1.0,
                epsilon_decay=0.9995,
                description="Master basic individual ant behaviors: movement, food collection, survival"
            ),
            CurriculumPhase(
                name="basic_tactics",
                episodes=int(total_episodes * 0.25),
                opponents=['smart_random'],
                epsilon_decay=0.9995,
                description="Learn tactical awareness: threat assessment, risk management"
            ),
            CurriculumPhase(
                name="intermediate_strategy",
                episodes=int(total_episodes * 0.25),
                opponents=['greedy', 'smart_random'],
                epsilon_decay=0.9994,
                description="Develop strategies against balanced opponents"
            ),
            CurriculumPhase(
                name="advanced_play",
                episodes=int(total_episodes * 0.2),
                opponents=['greedy', 'greedy_aggressive', 'greedy_defensive'],
                epsilon_decay=0.9993,
                description="Handle diverse opponent styles and strategies"
            ),
            CurriculumPhase(
                name="generalization",
                episodes=int(total_episodes * 0.1),
                opponents=['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive'],
                epsilon_decay=0.999,
                description="Generalize across all opponent types for robust play"
            ),
        ],
        initial_learning_rate=0.3,
        final_learning_rate=0.001,
        learning_rate_decay_type="polynomial",
        initial_epsilon=1.0,
        final_epsilon=0.05
    )


def get_intensive_curriculum(total_episodes: int = 50000) -> TrainingCurriculum:
    """
    Intensive curriculum for maximum performance.

    Focus: Deep learning with extensive experience across all opponent types
    Time: ~1-2 hours for 50000 episodes
    Use: Competition-level training, maximum win rate optimization
    """
    return TrainingCurriculum(
        name="intensive",
        description="Extensive training for maximum performance and generalization",
        phases=[
            CurriculumPhase(
                name="foundation",
                episodes=int(total_episodes * 0.15),
                opponents=['random'],
                epsilon=1.0,
                epsilon_decay=0.9997,
                eval_freq=500,
                description="Build solid foundation of individual ant control"
            ),
            CurriculumPhase(
                name="core_tactics",
                episodes=int(total_episodes * 0.15),
                opponents=['smart_random'],
                epsilon_decay=0.9996,
                eval_freq=500,
                description="Develop core tactical skills"
            ),
            CurriculumPhase(
                name="strategic_depth",
                episodes=int(total_episodes * 0.2),
                opponents=['greedy', 'smart_random'],
                epsilon_decay=0.9995,
                eval_freq=500,
                description="Build strategic depth through varied opposition"
            ),
            CurriculumPhase(
                name="style_adaptation",
                episodes=int(total_episodes * 0.2),
                opponents=['greedy_aggressive', 'greedy_defensive', 'greedy'],
                epsilon_decay=0.9994,
                eval_freq=500,
                description="Learn to adapt to aggressive and defensive playstyles"
            ),
            CurriculumPhase(
                name="mixed_mastery",
                episodes=int(total_episodes * 0.2),
                opponents=['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive'],
                epsilon_decay=0.9993,
                eval_freq=500,
                description="Master all scenarios through extensive mixed training"
            ),
            CurriculumPhase(
                name="refinement",
                episodes=int(total_episodes * 0.1),
                opponents=['greedy', 'greedy_aggressive', 'greedy_defensive'],
                epsilon_decay=0.999,
                eval_freq=1000,
                description="Final refinement against hardest opponents with low exploration"
            ),
        ],
        initial_learning_rate=0.4,
        final_learning_rate=0.0005,
        learning_rate_decay_type="polynomial",
        initial_epsilon=1.0,
        final_epsilon=0.03
    )


def get_rapid_curriculum(total_episodes: int = 2000) -> TrainingCurriculum:
    """
    Rapid curriculum for fast experimentation.

    Focus: Minimal viable training for quick results
    Time: ~2-5 minutes for 2000 episodes
    Use: Fast iteration, parameter tuning, proof of concept
    """
    return TrainingCurriculum(
        name="rapid",
        description="Ultra-fast training for quick experimentation",
        phases=[
            CurriculumPhase(
                name="quick_learn",
                episodes=int(total_episodes * 0.5),
                opponents=['random'],
                epsilon=1.0,
                epsilon_decay=0.998,
                eval_freq=100,
                description="Rapid learning against simple opponent"
            ),
            CurriculumPhase(
                name="quick_test",
                episodes=int(total_episodes * 0.5),
                opponents=['smart_random', 'greedy'],
                epsilon_decay=0.995,
                eval_freq=100,
                description="Quick test against harder opponents"
            ),
        ],
        initial_learning_rate=0.5,
        final_learning_rate=0.05,
        learning_rate_decay_type="polynomial",
        initial_epsilon=1.0,
        final_epsilon=0.1
    )


def get_specialized_curriculum(
    focus: str,
    total_episodes: int = 15000
) -> TrainingCurriculum:
    """
    Create a specialized curriculum focused on a specific aspect.

    Args:
        focus: One of 'aggressive', 'defensive', 'adaptive'
        total_episodes: Total training episodes

    Returns:
        Specialized training curriculum
    """
    if focus == 'aggressive':
        return TrainingCurriculum(
            name="aggressive_specialist",
            description="Specialized training for aggressive playstyle",
            phases=[
                CurriculumPhase(
                    name="basics",
                    episodes=int(total_episodes * 0.2),
                    opponents=['random', 'smart_random'],
                    description="Learn basics"
                ),
                CurriculumPhase(
                    name="aggression_training",
                    episodes=int(total_episodes * 0.5),
                    opponents=['greedy_aggressive', 'greedy'],
                    description="Focus on aggressive tactics"
                ),
                CurriculumPhase(
                    name="counter_defense",
                    episodes=int(total_episodes * 0.3),
                    opponents=['greedy_defensive', 'greedy'],
                    description="Learn to break defensive play"
                ),
            ],
            initial_learning_rate=0.35,
            final_learning_rate=0.001
        )

    elif focus == 'defensive':
        return TrainingCurriculum(
            name="defensive_specialist",
            description="Specialized training for defensive playstyle",
            phases=[
                CurriculumPhase(
                    name="basics",
                    episodes=int(total_episodes * 0.2),
                    opponents=['random', 'smart_random'],
                    description="Learn basics"
                ),
                CurriculumPhase(
                    name="defense_training",
                    episodes=int(total_episodes * 0.5),
                    opponents=['greedy_defensive', 'greedy'],
                    description="Master defensive tactics"
                ),
                CurriculumPhase(
                    name="counter_aggression",
                    episodes=int(total_episodes * 0.3),
                    opponents=['greedy_aggressive', 'greedy'],
                    description="Learn to counter aggressive play"
                ),
            ],
            initial_learning_rate=0.3,
            final_learning_rate=0.001
        )

    else:  # adaptive
        return TrainingCurriculum(
            name="adaptive_specialist",
            description="Specialized training for adaptive playstyle",
            phases=[
                CurriculumPhase(
                    name="broad_foundation",
                    episodes=int(total_episodes * 0.3),
                    opponents=['random', 'smart_random', 'greedy'],
                    description="Build broad foundation"
                ),
                CurriculumPhase(
                    name="style_mixing",
                    episodes=int(total_episodes * 0.5),
                    opponents=['greedy', 'greedy_aggressive', 'greedy_defensive'],
                    description="Learn all playstyles"
                ),
                CurriculumPhase(
                    name="full_adaptation",
                    episodes=int(total_episodes * 0.2),
                    opponents=['random', 'smart_random', 'greedy', 'greedy_aggressive', 'greedy_defensive'],
                    description="Master adaptation to any opponent"
                ),
            ],
            initial_learning_rate=0.3,
            final_learning_rate=0.001
        )


def get_scenario_curriculum(total_episodes: int = 10000) -> TrainingCurriculum:
    """
    Scenario-based curriculum focusing on isolated skill development.

    This curriculum trains agents in simplified scenarios before full games:
    - Food collection (no enemies)
    - Combat basics (isolated combat)
    - Anthill attacks (no defenders)
    - Then progresses to full opponent training

    Focus: Master fundamentals in isolation before complex situations
    Time: ~10-20 minutes for 10000 episodes
    Use: Agents that need strong fundamental skills
    """
    return TrainingCurriculum(
        name="scenario_based",
        description="Scenario-based training with isolated skill development",
        phases=[
            CurriculumPhase(
                name="food_mastery",
                episodes=int(total_episodes * 0.2),
                scenarios=['food_easy', 'food_medium', 'food_hard'],
                training_type="scenario",
                epsilon=1.0,
                epsilon_decay=0.9995,
                eval_freq=200,
                description="Master food collection in isolation"
            ),
            CurriculumPhase(
                name="combat_basics",
                episodes=int(total_episodes * 0.15),
                scenarios=['combat_1v1', 'combat_1v2'],
                training_type="scenario",
                epsilon_decay=0.9995,
                description="Learn basic combat and survival"
            ),
            CurriculumPhase(
                name="anthill_tactics",
                episodes=int(total_episodes * 0.15),
                scenarios=['anthill_close', 'anthill_medium', 'anthill_far'],
                training_type="scenario",
                epsilon_decay=0.9994,
                description="Learn to find and attack enemy anthills"
            ),
            CurriculumPhase(
                name="multi_ant_coordination",
                episodes=int(total_episodes * 0.15),
                scenarios=['efficient_2ants', 'efficient_3ants', 'contested_2v2'],
                training_type="scenario",
                epsilon_decay=0.9994,
                description="Learn independent multi-ant efficiency"
            ),
            CurriculumPhase(
                name="transition_to_full_game",
                episodes=int(total_episodes * 0.2),
                opponents=['random', 'smart_random'],
                training_type="opponent",
                epsilon_decay=0.9993,
                description="Apply learned skills in full games"
            ),
            CurriculumPhase(
                name="advanced_opponents",
                episodes=int(total_episodes * 0.15),
                opponents=['greedy', 'greedy_aggressive', 'greedy_defensive'],
                training_type="opponent",
                epsilon_decay=0.999,
                description="Challenge against heuristic opponents"
            ),
        ],
        initial_learning_rate=0.3,
        final_learning_rate=0.001,
        learning_rate_decay_type="polynomial",
        initial_epsilon=1.0,
        final_epsilon=0.05
    )


def get_hybrid_curriculum(total_episodes: int = 20000) -> TrainingCurriculum:
    """
    Hybrid curriculum mixing scenarios and opponent training throughout.

    Alternates between isolated skill training and full game experience
    for balanced development.

    Focus: Continuous skill development with real-world application
    Time: ~20-40 minutes for 20000 episodes
    Use: Best of both worlds - scenarios + opponents
    """
    return TrainingCurriculum(
        name="hybrid",
        description="Hybrid training mixing scenarios and full opponent games",
        phases=[
            CurriculumPhase(
                name="fundamentals_scenarios",
                episodes=int(total_episodes * 0.15),
                scenarios=['food_easy', 'food_medium', 'combat_1v1'],
                training_type="scenario",
                epsilon=1.0,
                description="Learn basics in isolation"
            ),
            CurriculumPhase(
                name="fundamentals_opponents",
                episodes=int(total_episodes * 0.15),
                opponents=['random', 'smart_random'],
                training_type="opponent",
                description="Apply basics in full games"
            ),
            CurriculumPhase(
                name="intermediate_scenarios",
                episodes=int(total_episodes * 0.15),
                scenarios=['anthill_medium', 'combat_1v2', 'contested_2v2'],
                training_type="scenario",
                description="Learn intermediate tactics"
            ),
            CurriculumPhase(
                name="intermediate_opponents",
                episodes=int(total_episodes * 0.20),
                opponents=['smart_random', 'greedy'],
                training_type="opponent",
                description="Apply tactics in full games"
            ),
            CurriculumPhase(
                name="advanced_scenarios",
                episodes=int(total_episodes * 0.10),
                scenarios=['survival_1v3', 'defense_3attackers', 'maze'],
                training_type="scenario",
                description="Master difficult challenges"
            ),
            CurriculumPhase(
                name="advanced_opponents",
                episodes=int(total_episodes * 0.25),
                opponents=['greedy', 'greedy_aggressive', 'greedy_defensive', 'smart_random'],
                training_type="opponent",
                description="Face diverse opponent strategies"
            ),
        ],
        initial_learning_rate=0.3,
        final_learning_rate=0.001,
        initial_epsilon=1.0,
        final_epsilon=0.05
    )


def get_curriculum(name: str, total_episodes: Optional[int] = None) -> TrainingCurriculum:
    """
    Get a curriculum by name.

    Args:
        name: Curriculum name ('basic', 'standard', 'intensive', 'rapid',
              'aggressive', 'defensive', 'adaptive', 'scenario', 'hybrid')
        total_episodes: Override total episodes (None = use default)

    Returns:
        Training curriculum

    Raises:
        ValueError: If curriculum name is unknown
    """
    curricula_map = {
        'rapid': get_rapid_curriculum,
        'basic': get_basic_curriculum,
        'standard': get_standard_curriculum,
        'intensive': get_intensive_curriculum,
        'scenario': get_scenario_curriculum,
        'hybrid': get_hybrid_curriculum,
    }

    specialized_map = {
        'aggressive': lambda eps: get_specialized_curriculum('aggressive', eps),
        'defensive': lambda eps: get_specialized_curriculum('defensive', eps),
        'adaptive': lambda eps: get_specialized_curriculum('adaptive', eps),
    }

    if name in curricula_map:
        if total_episodes:
            return curricula_map[name](total_episodes)
        else:
            return curricula_map[name]()

    elif name in specialized_map:
        if total_episodes:
            return specialized_map[name](total_episodes)
        else:
            return specialized_map[name](15000)

    else:
        raise ValueError(
            f"Unknown curriculum: {name}\n"
            f"Available: {', '.join(sorted(list(curricula_map.keys()) + list(specialized_map.keys())))}"
        )


def parse_custom_curriculum(spec: str, name: str = "custom") -> TrainingCurriculum:
    """
    Parse a custom curriculum specification.

    Format: "phase_name:opponents:episodes,phase_name:opponents:episodes,..."
    Opponents can be a single type or '+' separated list

    Examples:
        "explore:random:1000,refine:smart_random+greedy:2000"
        "basics:random:500,mid:greedy:1000,hard:greedy_aggressive+greedy_defensive:1500"

    Args:
        spec: Curriculum specification string
        name: Name for the curriculum

    Returns:
        Custom training curriculum
    """
    phases = []

    for phase_spec in spec.split(','):
        parts = phase_spec.strip().split(':')
        if len(parts) != 3:
            raise ValueError(
                f"Invalid phase spec: {phase_spec}\n"
                f"Expected format: phase_name:opponents:episodes"
            )

        phase_name, opponents_str, episodes_str = parts
        opponents = opponents_str.split('+')
        episodes = int(episodes_str)

        phases.append(CurriculumPhase(
            name=phase_name,
            episodes=episodes,
            opponents=opponents,
            description=f"Custom phase: {phase_name}"
        ))

    return TrainingCurriculum(
        name=name,
        description=f"Custom curriculum: {spec}",
        phases=phases
    )
