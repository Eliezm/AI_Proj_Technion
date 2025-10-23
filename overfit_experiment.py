#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERFIT EXPERIMENT INFRASTRUCTURE
==================================
Trains GNN policy on a small set of problems and tests overfitting.

Features:
  ✓ Train on K problems from a domain
  ✓ Test on the same K problems to measure overfitting
  ✓ Track learning curves (reward, plan cost, expansions)
  ✓ Per-problem learning curves
  ✓ Comparison: initial vs final performance
  ✓ Statistical analysis of overfitting

Usage:
    python overfit_experiment.py \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --num-problems 5 \
        --num-train-episodes 100 \
        --output overfit_results/

Output:
    overfit_results/
    ├── training_log.jsonl          # Per-episode metrics
    ├── overfit_summary.json         # Final statistics
    ├── per_problem_learning.json    # Per-problem curves
    └── plots/
        ├── learning_curve.png       # Overall reward curve
        ├── per_problem_curves.png   # Individual problem curves
        ├── convergence_analysis.png # Convergence rate
        └── overfitting_metrics.png  # Specialization analysis
"""

import sys
import os
import json
import glob
import logging
import argparse
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import numpy as np
from datetime import datetime

sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("overfit_experiment.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    episode: int
    problem_idx: int
    problem_name: str
    reward: float
    plan_cost: int = 0
    num_expansions: int = 0
    solved: bool = False
    total_reward: float = 0.0  # Cumulative reward
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProblemStats:
    """Statistics for a single problem across training."""
    problem_idx: int
    problem_name: str
    num_episodes: int
    avg_reward: float
    best_reward: float
    worst_reward: float
    final_reward: float
    improvement_ratio: float  # (final - worst) / (best - worst)
    avg_plan_cost: float
    solve_rate: float
    episodes_to_convergence: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# FILE: overfit_experiment.py
# REPLACE THIS DATACLASS DEFINITION

@dataclass
class OverfitExperimentSummary:
    """Overall statistics for the overfit experiment."""
    # --- Fields WITHOUT defaults ---
    num_problems: int
    num_train_episodes: int
    total_timesteps: int
    start_time: str
    end_time: str
    duration_seconds: float
    avg_reward_over_all: float
    best_reward_over_all: float
    worst_reward_over_all: float
    per_problem_stats: List[Dict]
    reward_variance: float
    plan_cost_improvement_ratio: float
    solve_rate_improvement: float
    early_convergence_episodes: int
    learning_rate_estimate: float  # Moved up

    # --- Fields WITH defaults (must come AFTER non-defaults) ---
    convergence_threshold: float = 0.05
    plateau_episode: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# PROBLEM SELECTION
# ============================================================================

def select_training_problems(
        domain_file: str,
        problem_pattern: str,
        num_problems: int,
        seed: int = 42
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Select K problems for overfitting experiment.

    Returns:
        (domain_path, [(domain, problem), ...])
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: SELECT TRAINING PROBLEMS")
    logger.info("=" * 80 + "\n")

    domain_path = os.path.abspath(domain_file)

    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Domain not found: {domain_path}")

    logger.info(f"Domain: {domain_path}\n")

    # Load all problems matching pattern
    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    logger.info(f"Found {len(all_problems)} total problems matching pattern")
    logger.info(f"Selecting {num_problems} for overfitting experiment\n")

    # Random selection
    random.seed(seed)
    selected = random.sample(all_problems, min(num_problems, len(all_problems)))
    selected = sorted(selected)

    # Convert to absolute paths
    benchmarks = [(domain_path, os.path.abspath(p)) for p in selected]

    logger.info("Selected problems:")
    for i, (domain, problem) in enumerate(benchmarks, 1):
        logger.info(f"  [{i}] {os.path.basename(problem)}")

    return domain_path, benchmarks


# ============================================================================
# TRAINING PHASE
# ============================================================================

class OverfitTrainer:
    """Trains GNN on a fixed set of problems."""

    def __init__(self, benchmarks: List[Tuple[str, str]], output_dir: str):
        self.benchmarks = benchmarks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episode_log: List[EpisodeMetrics] = []
        self.start_time = datetime.now()

        # Import here to avoid issues
        from stable_baselines3 import PPO
        from gnn_policy import GNNPolicy
        from merge_env import MergeEnv
        from stable_baselines3.common.monitor import Monitor

        self.PPO = PPO
        self.GNNPolicy = GNNPolicy
        self.MergeEnv = MergeEnv
        self.Monitor = Monitor

    def run_training(
            self,
            num_episodes: int,
            timesteps_per_episode: int = 50,
            save_interval: int = 10
    ) -> str:
        """
        Train GNN on problems with overfitting measurement.

        Returns:
            Path to trained model
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: TRAINING PHASE")
        logger.info("=" * 80 + "\n")

        model = None
        episode = 0

        try:
            # Create problem iterator (cycle through problems for multiple episodes)
            problem_cycle = [self.benchmarks[i % len(self.benchmarks)]
                             for i in range(num_episodes)]

            for episode in range(num_episodes):
                domain_file, problem_file = problem_cycle[episode]
                problem_idx = episode % len(self.benchmarks)
                problem_name = os.path.basename(problem_file)

                logger.info(f"\nEpisode {episode + 1}/{num_episodes}: {problem_name}")

                try:
                    # Create environment
                    env = self.MergeEnv(
                        domain_file=domain_file,
                        problem_file=problem_file,
                        max_merges=50,
                        debug=False,
                        reward_variant='astar_search',
                        w_search_efficiency=0.30,
                        w_solution_quality=0.20,
                        w_f_stability=0.35,
                        w_state_control=0.15,
                    )

                    env = self.Monitor(env)

                    # Create or continue model
                    if model is None:
                        logger.info("  Creating new PPO model...")
                        model = self.PPO(
                            policy=self.GNNPolicy,
                            env=env,
                            learning_rate=0.0003,
                            n_steps=64,
                            batch_size=32,
                            ent_coef=0.01,
                            verbose=0,
                            tensorboard_log=str(self.output_dir / "tb_logs"),
                            policy_kwargs={"hidden_dim": 64},
                        )
                    else:
                        model.set_env(env)

                    # Train for this episode
                    logger.info(f"  Training for {timesteps_per_episode} timesteps...")
                    model.learn(
                        total_timesteps=timesteps_per_episode,
                        tb_log_name=f"overfit_episode_{episode}",
                        reset_num_timesteps=False,
                    )

                    # Extract metrics
                    obs, _ = env.reset()
                    episode_reward = 0.0
                    steps = 0

                    for step in range(20):  # Quick test run
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(int(action))
                        episode_reward += reward
                        steps += 1

                        if done or truncated:
                            break

                    # Log metrics
                    metrics = EpisodeMetrics(
                        episode=episode,
                        problem_idx=problem_idx,
                        problem_name=problem_name,
                        reward=episode_reward,
                        plan_cost=info.get('plan_cost', 0),
                        num_expansions=info.get('num_expansions', 0),
                        solved=info.get('plan_cost', 0) > 0,
                        total_reward=sum(m.reward for m in self.episode_log) + episode_reward
                    )

                    self.episode_log.append(metrics)

                    logger.info(f"  Reward: {episode_reward:.4f}")
                    logger.info(f"  Cumulative: {metrics.total_reward:.4f}")

                    env.close()

                    # Save checkpoint
                    if (episode + 1) % save_interval == 0:
                        model_path = self.output_dir / f"model_ep{episode+1}.zip"
                        model.save(model_path)
                        logger.info(f"  ✓ Saved checkpoint: {model_path}")

                except Exception as e:
                    logger.error(f"  ✗ Episode {episode} failed: {e}")
                    logger.error(traceback.format_exc())
                    continue

            # Save final model
            if model is not None:
                final_model_path = self.output_dir / "model_final.zip"
                model.save(final_model_path)
                logger.info(f"\n✅ Training complete! Final model: {final_model_path}")
                return str(final_model_path)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def save_training_log(self):
        """Save episode metrics to JSONL file."""
        log_path = self.output_dir / "training_log.jsonl"

        with open(log_path, 'w') as f:
            for metrics in self.episode_log:
                f.write(json.dumps(metrics.to_dict()) + '\n')

        logger.info(f"Saved training log: {log_path}")
        return log_path


# ============================================================================
# EVALUATION PHASE
# ============================================================================

class OverfitEvaluator:
    """Tests overfitting on the training problems."""

    def __init__(self, model_path: str, benchmarks: List[Tuple[str, str]], output_dir: str):
        self.model_path = model_path
        self.benchmarks = benchmarks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from stable_baselines3 import PPO
        from merge_env import MergeEnv

        self.PPO = PPO
        self.MergeEnv = MergeEnv

    def evaluate(self, num_runs_per_problem: int = 5) -> Dict:
        """
        Test on training problems.

        Returns:
            Per-problem statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: EVALUATION PHASE")
        logger.info("=" * 80 + "\n")

        # Load model
        logger.info(f"Loading model: {self.model_path}")
        model = self.PPO.load(self.model_path)
        logger.info("✓ Model loaded\n")

        results = {}

        for prob_idx, (domain_file, problem_file) in enumerate(self.benchmarks):
            problem_name = os.path.basename(problem_file)
            logger.info(f"Problem {prob_idx + 1}/{len(self.benchmarks)}: {problem_name}")

            rewards = []
            plan_costs = []
            expansions = []

            for run in range(num_runs_per_problem):
                try:
                    env = self.MergeEnv(
                        domain_file=domain_file,
                        problem_file=problem_file,
                        max_merges=50,
                        debug=False,
                        reward_variant='astar_search',
                    )

                    obs, _ = env.reset()
                    total_reward = 0.0

                    for step in range(20):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(int(action))
                        total_reward += reward

                        if done or truncated:
                            break

                    rewards.append(total_reward)
                    plan_costs.append(info.get('plan_cost', 0))
                    expansions.append(info.get('num_expansions', 0))

                    logger.info(f"  Run {run + 1}: reward={total_reward:.4f}")
                    env.close()

                except Exception as e:
                    logger.error(f"  Run {run + 1} failed: {e}")
                    continue

            # Aggregate stats
            if rewards:
                results[problem_name] = {
                    'problem_idx': prob_idx,
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'min_reward': np.min(rewards),
                    'avg_plan_cost': np.mean(plan_costs) if plan_costs else 0,
                    'avg_expansions': int(np.mean(expansions)) if expansions else 0,
                }

                logger.info(f"  Average reward: {results[problem_name]['avg_reward']:.4f} "
                            f"(±{results[problem_name]['std_reward']:.4f})")

        return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_overfitting(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        benchmarks: List[Tuple[str, str]]
) -> OverfitExperimentSummary:
    """
    Analyze overfitting indicators.

    Returns:
        Comprehensive summary with overfitting metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ANALYSIS")
    logger.info("=" * 80 + "\n")

    # Group by problem
    by_problem = defaultdict(list)
    for metrics in training_log:
        by_problem[metrics.problem_name].append(metrics)

    # Per-problem statistics
    per_problem_stats = []

    for problem_idx, (domain, problem_file) in enumerate(benchmarks):
        problem_name = os.path.basename(problem_file)

        if problem_name in by_problem:
            episodes = by_problem[problem_name]
            rewards = [e.reward for e in episodes]

            # Improvement analysis
            initial_reward = rewards[0] if rewards else 0
            final_reward = rewards[-1] if rewards else 0
            best_reward = max(rewards) if rewards else 0
            worst_reward = min(rewards) if rewards else 0

            if best_reward != worst_reward:
                improvement_ratio = (final_reward - worst_reward) / (best_reward - worst_reward)
            else:
                improvement_ratio = 0.0

            # Convergence detection
            episodes_to_convergence = None
            if len(rewards) > 10:
                for i in range(10, len(rewards)):
                    recent_avg = np.mean(rewards[i-10:i])
                    older_avg = np.mean(rewards[max(0, i-20):i-10])
                    if older_avg > 0 and abs(recent_avg - older_avg) / abs(older_avg) < 0.05:
                        episodes_to_convergence = i
                        break

            stats = ProblemStats(
                problem_idx=problem_idx,
                problem_name=problem_name,
                num_episodes=len(episodes),
                avg_reward=np.mean(rewards),
                best_reward=best_reward,
                worst_reward=worst_reward,
                final_reward=final_reward,
                improvement_ratio=improvement_ratio,
                avg_plan_cost=np.mean([e.plan_cost for e in episodes]) if episodes else 0,
                solve_rate=sum(1 for e in episodes if e.solved) / len(episodes) if episodes else 0,
                episodes_to_convergence=episodes_to_convergence
            )

            per_problem_stats.append(stats)

            logger.info(f"\n{problem_name}:")
            logger.info(f"  Episodes: {len(episodes)}")
            logger.info(f"  Reward: {stats.avg_reward:.4f} (best: {best_reward:.4f})")
            logger.info(f"  Improvement: {improvement_ratio*100:.1f}%")
            logger.info(f"  Convergence: {episodes_to_convergence} episodes")

    # Overall statistics
    all_rewards = [m.reward for m in training_log]

    summary = OverfitExperimentSummary(
        num_problems=len(benchmarks),
        num_train_episodes=len(training_log),
        total_timesteps=len(training_log) * 50,  # Assuming 50 timesteps per episode
        start_time=training_log[0].timestamp if training_log else datetime.now().isoformat(),
        end_time=training_log[-1].timestamp if training_log else datetime.now().isoformat(),
        duration_seconds=training_log[-1].timestamp - training_log[0].timestamp if len(training_log) > 1 else 0,
        avg_reward_over_all=np.mean(all_rewards) if all_rewards else 0,
        best_reward_over_all=np.max(all_rewards) if all_rewards else 0,
        worst_reward_over_all=np.min(all_rewards) if all_rewards else 0,
        per_problem_stats=[s.to_dict() for s in per_problem_stats],
        reward_variance=np.var(all_rewards) if all_rewards else 0,
        plan_cost_improvement_ratio=0.0,  # Can be computed from eval_results
        solve_rate_improvement=0.0,
        early_convergence_episodes=min([s.episodes_to_convergence for s in per_problem_stats
                                        if s.episodes_to_convergence], default=None) or 0,
        learning_rate_estimate=0.0,
    )

    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_plots(
        training_log: List[EpisodeMetrics],
        eval_results: Dict,
        output_dir: Path
):
    """Generate learning curve plots."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib not available - skipping plots")
        return

    logger.info("\nGenerating plots...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Overall learning curve
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = [m.episode for m in training_log]
    rewards = [m.reward for m in training_log]

    ax.plot(episodes, rewards, alpha=0.3, label='Per-episode')

    # Rolling average
    window = min(10, len(rewards) // 4)
    if window > 1:
        rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), rolling_avg, linewidth=2, label=f'Rolling avg (window={window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curve - Overall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "learning_curve.png", dpi=150)
    logger.info(f"  ✓ {plots_dir / 'learning_curve.png'}")
    plt.close()

    # Plot 2: Per-problem curves
    by_problem = defaultdict(list)
    for m in training_log:
        by_problem[m.problem_name].append(m)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (problem_name, metrics) in enumerate(sorted(by_problem.items())[:4]):
        ax = axes[idx]

        episodes = [m.episode for m in metrics]
        rewards = [m.reward for m in metrics]

        ax.plot(episodes, rewards, marker='o', label=problem_name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(problem_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "per_problem_curves.png", dpi=150)
    logger.info(f"  ✓ {plots_dir / 'per_problem_curves.png'}")
    plt.close()

    logger.info("✅ Plots generated")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overfit experiment infrastructure")
    parser.add_argument("--domain", required=True, help="Domain PDDL file")
    parser.add_argument("--problems", required=True, help="Problem glob pattern")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems to train on")
    parser.add_argument("--num-train-episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--output", default="overfit_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        # Step 1: Select problems
        domain_file, benchmarks = select_training_problems(
            args.domain,
            args.problems,
            args.num_problems,
            args.seed
        )

        # Step 2: Train
        trainer = OverfitTrainer(benchmarks, args.output)
        model_path = trainer.run_training(args.num_train_episodes)
        trainer.save_training_log()

        if not model_path:
            logger.error("Training failed")
            return 1

        # Step 3: Evaluate
        evaluator = OverfitEvaluator(model_path, benchmarks, args.output)
        eval_results = evaluator.evaluate()

        # Step 4: Analyze
        summary = analyze_overfitting(trainer.episode_log, eval_results, benchmarks)

        # Save summary
        summary_path = Path(args.output) / "overfit_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")

        # Generate plots
        generate_plots(trainer.episode_log, eval_results, Path(args.output))

        logger.info("\n" + "=" * 80)
        logger.info("✅ OVERFIT EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {os.path.abspath(args.output)}")
        logger.info(f"  - Training log: training_log.jsonl")
        logger.info(f"  - Summary: overfit_summary.json")
        logger.info(f"  - Plots: plots/")

        return 0

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())