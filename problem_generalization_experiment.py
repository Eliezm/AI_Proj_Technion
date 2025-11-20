#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SINGLE-DOMAIN TRAIN/TEST SPLIT EXPERIMENT
==========================================
Trains a GNN policy on a fixed subset of problems from ONE domain
and evaluates performance on a separate test set.

Features:
  ✓ Single domain focus (e.g., blocks_world, gripper, logistics)
  ✓ Automatic train/test splitting (configurable ratio)
  ✓ Cross-validation support (multiple splits)
  ✓ Baseline comparison (optional)
  ✓ Detailed metrics and visualizations
  ✓ Reproducible experiments (seed-based)

Usage:
    python train_test_split_experiment.py \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --domain-name blocks_world \
        --train-ratio 0.8 \
        --output experiment_results/

    # For cross-validation (multiple splits):
    python train_test_split_experiment.py \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --domain-name blocks_world \
        --cross-validation 5 \
        --output experiment_results_cv/

Environment Variables:
    REWARD_VARIANT: Reward function to use (default: astar_search)
    TRAIN_RATIO: Train/test split ratio (default: 0.8)
    RANDOM_SEED: Seed for reproducibility (default: 42)
    NUM_TRAIN_EPOCHS: Training epochs per problem (default: 1)
"""

import sys
import os
import logging
import glob
import json
import random
import traceback
import argparse
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from dataclasses import dataclass

# Setup paths
sys.path.insert(0, os.getcwd())
os.makedirs("downward/gnn_output", exist_ok=True)
os.makedirs("downward/fd_output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_test_split_experiment.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    domain_file: str
    domain_name: str
    train_problems: List[str]
    test_problems: List[str]
    train_ratio: float
    random_seed: int
    output_dir: str
    reward_variant: str = 'astar_search'
    total_timesteps: int = 5000
    timesteps_per_problem: int = 500
    max_merges: int = 50
    timeout_per_eval: int = 300
    include_baselines: bool = True
    cv_fold: Optional[int] = None
    cv_total: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'domain_name': self.domain_name,
            'train_problems_count': len(self.train_problems),
            'test_problems_count': len(self.test_problems),
            'train_ratio': self.train_ratio,
            'random_seed': self.random_seed,
            'reward_variant': self.reward_variant,
            'total_timesteps': self.total_timesteps,
            'timeout_per_eval': self.timeout_per_eval,
            'include_baselines': self.include_baselines,
            'cv_fold': self.cv_fold,
            'cv_total': self.cv_total,
            'timestamp': datetime.now().isoformat(),
        }


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    config: ExperimentConfig
    training_time_sec: float
    training_timesteps: int
    training_problems_used: int

    # Test set metrics (GNN only initially)
    gnn_solve_rate: float
    gnn_avg_time: float
    gnn_avg_expansions: int

    # Baseline metrics (if enabled)
    baseline_solve_rates: Dict[str, float] = None  # planner_name -> rate
    baseline_avg_times: Dict[str, float] = None

    # Derived metrics
    improvement_over_baseline: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'config': self.config.to_dict(),
            'training_time_sec': self.training_time_sec,
            'training_timesteps': self.training_timesteps,
            'training_problems_used': self.training_problems_used,
            'gnn_solve_rate': self.gnn_solve_rate,
            'gnn_avg_time': self.gnn_avg_time,
            'gnn_avg_expansions': self.gnn_avg_expansions,
        }

        if self.baseline_solve_rates:
            d['baseline_solve_rates'] = self.baseline_solve_rates
            d['baseline_avg_times'] = self.baseline_avg_times
            d['improvement_over_baseline'] = self.improvement_over_baseline

        return d


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 90):
    """Print formatted section header."""
    logger.info("\n" + "=" * width)
    logger.info(f"// {title.upper()}")
    logger.info("=" * width + "\n")


def print_subsection(title: str):
    """Print formatted subsection header."""
    logger.info("\n" + "-" * 80)
    logger.info(f">>> {title}")
    logger.info("-" * 80 + "\n")


# ============================================================================
# PHASE 0: DATASET PREPARATION
# ============================================================================

def load_and_split_problems(
        domain_file: str,
        problem_pattern: str,
        train_ratio: float = 0.8,
        random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load all problems matching pattern and split into train/test.

    Args:
        domain_file: Path to domain PDDL
        problem_pattern: Glob pattern for problems
        train_ratio: Fraction for training
        random_seed: Seed for reproducibility

    Returns:
        (train_problems, test_problems)
    """
    print_subsection("Loading and Splitting Problems")

    # Verify domain exists
    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain file not found: {domain_file}")

    logger.info(f"Domain: {domain_file}")

    # Load problems
    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    logger.info(f"Found {len(all_problems)} total problem(s)")

    if len(all_problems) < 2:
        raise ValueError("Need at least 2 problems for train/test split")

    # Set seed for reproducibility
    random.seed(random_seed)
    problems_shuffled = all_problems.copy()
    random.shuffle(problems_shuffled)

    # Split
    split_idx = int(len(problems_shuffled) * train_ratio)
    train_problems = sorted(problems_shuffled[:split_idx])
    test_problems = sorted(problems_shuffled[split_idx:])

    logger.info(f"\nTrain/test split (ratio={train_ratio:.1%}, seed={random_seed}):")
    logger.info(f"  Train: {len(train_problems)} problems")
    for i, p in enumerate(train_problems[:3], 1):
        logger.info(f"    {i}. {os.path.basename(p)}")
    if len(train_problems) > 3:
        logger.info(f"    ... and {len(train_problems) - 3} more")

    logger.info(f"  Test:  {len(test_problems)} problems")
    for i, p in enumerate(test_problems[:3], 1):
        logger.info(f"    {i}. {os.path.basename(p)}")
    if len(test_problems) > 3:
        logger.info(f"    ... and {len(test_problems) - 3} more")

    return train_problems, test_problems


# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

def train_gnn_on_split(
        config: ExperimentConfig
) -> Tuple[str, float]:
    """
    Train GNN on the training set.

    Returns:
        (model_path, training_time_sec)
    """
    print_section("PHASE 1: TRAINING GNN ON TRAINING SET")

    try:
        from common_utils import train_model

        logger.info(f"Configuration:")
        logger.info(f"  Domain:              {config.domain_name}")
        logger.info(f"  Training problems:   {len(config.train_problems)}")
        logger.info(f"  Test problems:       {len(config.test_problems)}")
        logger.info(f"  Total timesteps:     {config.total_timesteps}")
        logger.info(f"  Timesteps/problem:   {config.timesteps_per_problem}")
        logger.info(f"  Reward variant:      {config.reward_variant}")
        logger.info(f"  Random seed:         {config.random_seed}")

        if config.cv_fold is not None:
            logger.info(f"  CV fold:             {config.cv_fold}/{config.cv_total}")

        # Create benchmark list (train set)
        benchmarks = [(config.domain_file, p) for p in config.train_problems]

        # Prepare model path
        os_makedirs(config.output_dir, exist_ok=True)

        if config.cv_fold is not None:
            model_name = f"gnn_model_fold_{config.cv_fold}.zip"
        else:
            model_name = "gnn_model_trained.zip"

        model_path = os.path.join(config.output_dir, model_name)

        # Define hyperparameters
        hyperparams = {
            'learning_rate': 0.0003,
            'n_steps': 64,
            'batch_size': 32,
            'ent_coef': 0.01,
            'reward_variant': config.reward_variant,
            'w_search_efficiency': 0.30,
            'w_solution_quality': 0.20,
            'w_f_stability': 0.35,
            'w_state_control': 0.15,
        }

        logger.info(f"\nHyperparameters:")
        for k, v in hyperparams.items():
            logger.info(f"  {k:<30} {v}")

        print_subsection("Starting Training")
        logger.info(f"Training on {len(benchmarks)} problem(s)...\n")

        import time
        start_time = time.time()

        model = train_model(
            model_save_path=model_path,
            benchmarks=benchmarks,
            hyperparams=hyperparams,
            total_timesteps=config.total_timesteps,
            tb_log_dir="tb_logs/",
            tb_log_name=f"train_test_{config.domain_name}" + (f"_fold{config.cv_fold}" if config.cv_fold else ""),
            debug_mode=False,  # REAL FD
        )

        elapsed = time.time() - start_time

        if model is None:
            logger.error("Training failed - model is None")
            return None, 0.0

        if not os.path.exists(model_path):
            logger.error(f"Model not saved: {model_path}")
            return None, 0.0

        logger.info(f"\n✅ Training complete!")
        logger.info(f"  Time: {elapsed:.1f}s")
        logger.info(f"  Model: {model_path}")

        return model_path, elapsed

    except Exception as e:
        logger.error(f"\n❌ Training FAILED: {e}")
        logger.error(traceback.format_exc())
        return None, 0.0


# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================

def evaluate_on_test_set(
        config: ExperimentConfig,
        model_path: str
) -> Dict[str, Any]:
    """
    Evaluate trained model on test set.
    Uses evaluation_comprehensive framework.
    """
    print_section("PHASE 2: EVALUATION ON TEST SET")

    try:
        from misc.evaluation.evaluation_comprehensive import (
            EvaluationFramework,
            GNNPolicyRunner,
            BaselineRunner,
            BenchmarkConfig
        )

        eval_output_dir = os.path.join(config.output_dir, "evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)

        framework = EvaluationFramework(eval_output_dir)

        # Prepare test problem pattern
        test_problems_str = "|".join(config.test_problems)

        logger.info(f"Evaluation configuration:")
        logger.info(f"  Model:              {model_path}")
        logger.info(f"  Test problems:      {len(config.test_problems)}")
        logger.info(f"  Include baselines:  {config.include_baselines}")
        logger.info(f"  Timeout per eval:   {config.timeout_per_eval}s")

        # Run evaluation
        results = framework.run_comprehensive_evaluation(
            domain_file=config.domain_file,
            problem_pattern="|".join(config.test_problems),  # Use alternation for glob
            model_path=model_path,
            timeout_sec=config.timeout_per_eval,
            include_baselines=config.include_baselines
        )

        # Extract and summarize results
        summaries = results.get("summaries", {})

        gnn_results = summaries.get("GNN", {})
        baseline_results = {name: summary for name, summary in summaries.items()
                            if name != "GNN"}

        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"\nGNN Results:")
        logger.info(f"  Solve rate:     {gnn_results.get('solve_rate_pct', 0):.1f}%")
        logger.info(f"  Avg time:       {gnn_results.get('avg_time_sec', 0):.2f}s")
        logger.info(f"  Avg expansions: {gnn_results.get('avg_expansions', 0)}")

        if baseline_results:
            logger.info(f"\nBaseline Results:")
            for name, summary in baseline_results.items():
                logger.info(f"  {name}:")
                logger.info(f"    Solve rate: {summary.get('solve_rate_pct', 0):.1f}%")
                logger.info(f"    Avg time:   {summary.get('avg_time_sec', 0):.2f}s")

        return {
            'gnn': gnn_results,
            'baselines': baseline_results,
            'all_summaries': summaries,
        }

    except Exception as e:
        logger.error(f"\n❌ Evaluation FAILED: {e}")
        logger.error(traceback.format_exc())
        return {}


# ============================================================================
# PHASE 3: ANALYSIS AND REPORTING
# ============================================================================

def generate_experiment_report(
        config: ExperimentConfig,
        training_time: float,
        eval_results: Dict[str, Any]
) -> ExperimentResults:
    """
    Generate comprehensive report and create result object.
    """
    print_section("PHASE 3: ANALYSIS AND REPORTING")

    gnn_results = eval_results.get('gnn', {})
    baseline_results = eval_results.get('baselines', {})

    # Extract GNN metrics
    gnn_solve_rate = gnn_results.get('solve_rate_pct', 0) / 100.0
    gnn_avg_time = gnn_results.get('avg_time_sec', 0)
    gnn_avg_expansions = int(gnn_results.get('avg_expansions', 0))

    # Extract baseline metrics
    baseline_solve_rates = {}
    baseline_avg_times = {}
    best_baseline_rate = 0.0

    for name, summary in baseline_results.items():
        rate = summary.get('solve_rate_pct', 0) / 100.0
        baseline_solve_rates[name] = rate
        baseline_avg_times[name] = summary.get('avg_time_sec', 0)
        best_baseline_rate = max(best_baseline_rate, rate)

    # Compute improvement
    improvement = None
    if best_baseline_rate > 0:
        improvement = (gnn_solve_rate - best_baseline_rate) / best_baseline_rate * 100

    # Create result object
    results = ExperimentResults(
        config=config,
        training_time_sec=training_time,
        training_timesteps=config.total_timesteps,
        training_problems_used=len(config.train_problems),
        gnn_solve_rate=gnn_solve_rate,
        gnn_avg_time=gnn_avg_time,
        gnn_avg_expansions=gnn_avg_expansions,
        baseline_solve_rates=baseline_solve_rates if baseline_solve_rates else None,
        baseline_avg_times=baseline_avg_times if baseline_avg_times else None,
        improvement_over_baseline=improvement,
    )

    # Write report
    report_path = os.path.join(config.output_dir, "experiment_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("TRAIN/TEST SPLIT EXPERIMENT REPORT\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Domain: {config.domain_name}\n\n")

        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 90 + "\n")
        f.write(f"Training problems:  {len(config.train_problems)}\n")
        f.write(f"Test problems:      {len(config.test_problems)}\n")
        f.write(f"Train ratio:        {config.train_ratio:.1%}\n")
        f.write(f"Random seed:        {config.random_seed}\n")
        f.write(f"Reward variant:     {config.reward_variant}\n")
        f.write(f"Total timesteps:    {config.total_timesteps}\n\n")

        if config.cv_fold is not None:
            f.write(f"Cross-validation:   Fold {config.cv_fold}/{config.cv_total}\n\n")

        f.write("TRAINING RESULTS\n")
        f.write("-" * 90 + "\n")
        f.write(f"Training time:      {results.training_time_sec:.1f}s\n")
        f.write(f"Problems used:      {results.training_problems_used}\n\n")

        f.write("TEST SET RESULTS\n")
        f.write("-" * 90 + "\n")
        f.write(f"GNN Solve Rate:     {results.gnn_solve_rate * 100:.1f}%\n")
        f.write(f"GNN Avg Time:       {results.gnn_avg_time:.2f}s\n")
        f.write(f"GNN Avg Expansions: {results.gnn_avg_expansions}\n\n")

        if results.baseline_solve_rates:
            f.write("BASELINE COMPARISON\n")
            f.write("-" * 90 + "\n")

            for name in sorted(results.baseline_solve_rates.keys()):
                rate = results.baseline_solve_rates[name]
                time = results.baseline_avg_times[name]
                f.write(f"{name:<30} Solve: {rate * 100:5.1f}%  Time: {time:7.2f}s\n")

            f.write("\n")

            if results.improvement_over_baseline is not None:
                f.write(f"GNN Improvement over best baseline: {results.improvement_over_baseline:+.1f}%\n")

        f.write("\n" + "=" * 90 + "\n")

    logger.info(f"✅ Report written: {report_path}")

    # Save results as JSON
    json_path = os.path.join(config.output_dir, "experiment_results.json")

    with open(json_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"✅ Results saved: {json_path}")

    return results


# ============================================================================
# CROSS-VALIDATION SUPPORT
# ============================================================================

def run_cross_validation(
        domain_file: str,
        problem_pattern: str,
        domain_name: str,
        num_folds: int = 5,
        output_dir: str = "experiment_results_cv",
        **kwargs
) -> List[ExperimentResults]:
    """
    Run multiple train/test splits (cross-validation).
    """
    print_section(f"CROSS-VALIDATION: {num_folds}-FOLD")

    all_problems = sorted(glob.glob(problem_pattern))

    if len(all_problems) < num_folds * 2:
        raise ValueError(f"Need at least {num_folds * 2} problems for {num_folds}-fold CV")

    logger.info(f"Total problems: {len(all_problems)}")
    logger.info(f"Number of folds: {num_folds}\n")

    fold_size = len(all_problems) // num_folds
    results_list = []

    for fold_idx in range(num_folds):
        print_section(f"FOLD {fold_idx + 1}/{num_folds}")

        # Create fold-specific splits
        fold_start = fold_idx * fold_size
        fold_end = fold_start + fold_size if fold_idx < num_folds - 1 else len(all_problems)

        test_problems = all_problems[fold_start:fold_end]
        train_problems = all_problems[:fold_start] + all_problems[fold_end:]

        logger.info(f"Fold {fold_idx + 1}:")
        logger.info(f"  Train: problems 0-{fold_start - 1}, {fold_end}-{len(all_problems) - 1} "
                    f"({len(train_problems)} total)")
        logger.info(f"  Test:  problems {fold_start}-{fold_end - 1} ({len(test_problems)} total)\n")

        # Create fold output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Create config
        config = ExperimentConfig(
            domain_file=domain_file,
            domain_name=domain_name,
            train_problems=train_problems,
            test_problems=test_problems,
            output_dir=fold_output_dir,
            cv_fold=fold_idx + 1,
            cv_total=num_folds,
            **kwargs
        )

        # Run single fold
        try:
            # Train
            model_path, training_time = train_gnn_on_split(config)

            if model_path is None:
                logger.error(f"Fold {fold_idx + 1} training failed - skipping")
                continue

            # Evaluate
            eval_results = evaluate_on_test_set(config, model_path)

            # Report
            fold_results = generate_experiment_report(config, training_time, eval_results)
            results_list.append(fold_results)

        except Exception as e:
            logger.error(f"Fold {fold_idx + 1} FAILED: {e}")
            logger.error(traceback.format_exc())
            continue

    # Generate cross-validation summary
    print_section("CROSS-VALIDATION SUMMARY")

    if not results_list:
        logger.error("No folds completed successfully!")
        return []

    logger.info(f"Completed: {len(results_list)}/{num_folds} folds\n")

    # Aggregate metrics
    gnn_solve_rates = [r.gnn_solve_rate for r in results_list]
    gnn_avg_times = [r.gnn_avg_time for r in results_list]
    training_times = [r.training_time_sec for r in results_list]

    logger.info("GNN Performance Across Folds:")
    logger.info(f"  Solve rate:  {np.mean(gnn_solve_rates) * 100:.1f}% ± {np.std(gnn_solve_rates) * 100:.1f}%")
    logger.info(f"  Avg time:    {np.mean(gnn_avg_times):.2f}s ± {np.std(gnn_avg_times):.2f}s")
    logger.info(f"  Training:    {np.mean(training_times):.1f}s ± {np.std(training_times):.1f}s")

    # Write CV summary
    cv_summary_path = os.path.join(output_dir, "cv_summary.json")

    cv_summary = {
        'num_folds': num_folds,
        'completed_folds': len(results_list),
        'domain': domain_name,
        'gnn_solve_rate_mean': float(np.mean(gnn_solve_rates)),
        'gnn_solve_rate_std': float(np.std(gnn_solve_rates)),
        'gnn_avg_time_mean': float(np.mean(gnn_avg_times)),
        'gnn_avg_time_std': float(np.std(gnn_avg_times)),
        'training_time_mean': float(np.mean(training_times)),
        'fold_results': [r.to_dict() for r in results_list],
        'timestamp': datetime.now().isoformat(),
    }

    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)

    logger.info(f"\n✅ CV summary saved: {cv_summary_path}")

    return results_list


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-domain train/test split experiment"
    )

    parser.add_argument("--domain", required=True, help="Path to domain PDDL")
    parser.add_argument("--problems", required=True, help="Glob pattern for problems")
    parser.add_argument("--domain-name", required=True, help="Name of domain (blocks_world, etc)")

    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of problems for training")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed for reproducibility")

    parser.add_argument("--cross-validation", type=int, default=0,
                        help="Number of folds for CV (0=no CV)")

    parser.add_argument("--reward-variant", default="astar_search",
                        help="Reward function variant")
    parser.add_argument("--total-timesteps", type=int, default=5000,
                        help="Total training timesteps")
    parser.add_argument("--timeout-eval", type=int, default=300,
                        help="Timeout per evaluation (seconds)")

    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline evaluation")

    parser.add_argument("--output", default="experiment_results",
                        help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print_section("SINGLE-DOMAIN TRAIN/TEST SPLIT EXPERIMENT")

    try:
        # Load and split problems
        train_problems, test_problems = load_and_split_problems(
            domain_file=args.domain,
            problem_pattern=args.problems,
            train_ratio=args.train_ratio,
            random_seed=args.random_seed
        )

        if args.cross_validation > 0:
            # Run cross-validation
            results = run_cross_validation(
                domain_file=args.domain,
                problem_pattern=args.problems,
                domain_name=args.domain_name,
                num_folds=args.cross_validation,
                output_dir=args.output,
                train_ratio=args.train_ratio,
                random_seed=args.random_seed,
                reward_variant=args.reward_variant,
                total_timesteps=args.total_timesteps,
                timeout_per_eval=args.timeout_eval,
                include_baselines=not args.skip_baselines,
            )
        else:
            # Run single train/test split
            config = ExperimentConfig(
                domain_file=args.domain,
                domain_name=args.domain_name,
                train_problems=train_problems,
                test_problems=test_problems,
                train_ratio=args.train_ratio,
                random_seed=args.random_seed,
                output_dir=args.output,
                reward_variant=args.reward_variant,
                total_timesteps=args.total_timesteps,
                timeout_per_eval=args.timeout_eval,
                include_baselines=not args.skip_baselines,
            )

            # Train
            model_path, training_time = train_gnn_on_split(config)

            if model_path is None:
                logger.error("Training failed!")
                return 1

            # Evaluate
            eval_results = evaluate_on_test_set(config, model_path)

            if not eval_results:
                logger.error("Evaluation failed!")
                return 1

            # Report
            results = generate_experiment_report(config, training_time, eval_results)

        print_section("EXPERIMENT COMPLETE")
        logger.info(f"✅ Results saved to: {os.path.abspath(args.output)}")
        logger.info(f"\nKey files:")
        logger.info(f"  - experiment_report.txt")
        logger.info(f"  - experiment_results.json")
        logger.info(f"  - evaluation/evaluation_results.csv")
        logger.info(f"  - evaluation/comparison_report.txt")

        return 0

    except Exception as e:
        logger.error(f"\n❌ EXPERIMENT FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


# Helper to avoid conflicts with os.makedirs
def os_makedirs(*args, **kwargs):
    return os.makedirs(*args, **kwargs)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)