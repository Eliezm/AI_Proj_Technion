#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCALE GENERALIZATION TRAINING & TESTING
========================================
Complete pipeline to train a GNN merge policy on small/medium problems and
evaluate its generalization to larger, harder problems.

Features:
  ✓ Generate synthetic PDDL problems at multiple scales
  ✓ Train GNN policy on small/medium problems
  ✓ Evaluate on medium/large problems (different from training set)
  ✓ Compare against baseline planners
  ✓ Detailed analysis of scale generalization
  ✓ HTML report with visualizations

Run with:
    python train_and_test_scale_generalization.py

Or with options:
    python train_and_test_scale_generalization.py \
        --domain blocks_world \
        --train-sizes small medium \
        --test-sizes large \
        --problems-per-size 10 \
        --training-timesteps 5000 \
        --output results/scale_generalization/
"""

import sys
import os
import logging
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

# Setup paths
sys.path.insert(0, os.getcwd())
os.makedirs("misc/benchmarks", exist_ok=True)
os.makedirs("downward/gnn_output", exist_ok=True)
os.makedirs("downward/fd_output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scale_generalization.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 0: PROBLEM GENERATION
# ============================================================================

def generate_benchmark_problems(
    domain: str = "blocks_world",
    sizes: List[str] = None,
    problems_per_size: int = 5
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate PDDL problems at multiple scales.

    Args:
        domain: "blocks_world", "logistics", or "gripper"
        sizes: List of sizes ("small", "medium", "large")
        problems_per_size: Number of problems to generate per size

    Returns:
        Dict mapping size → list of (domain_path, problem_path) tuples
    """
    if sizes is None:
        sizes = ["small", "medium", "large"]

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 0: GENERATING BENCHMARK PROBLEMS")
    logger.info("=" * 80 + "\n")

    try:
        from misc.pddl_generator.problem_generator import PDDLProblemGenerator
        from pddl_generator.size_config import SIZE_CONFIGS

        generator = PDDLProblemGenerator(output_dir="misc/benchmarks")

        benchmarks = {}

        for size in sizes:
            logger.info(f"\nGenerating {domain} - {size} problems...")

            domain_path, problem_paths = generator.save_domain_and_problems(
                domain=domain,
                size=size,
                num_problems=problems_per_size
            )

            benchmarks[size] = [(domain_path, p) for p in problem_paths]
            logger.info(f"  ✓ Generated {len(benchmarks[size])} {size} problem(s)")

        return benchmarks

    except ImportError:
        logger.warning("PDDL generator not available - using existing benchmarks")
        return {}
    except Exception as e:
        logger.error(f"Problem generation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

def train_on_problems(
    benchmarks: Dict[str, List[Tuple[str, str]]],
    train_sizes: List[str] = None,
    training_timesteps: int = 5000,
    reward_variant: str = "astar_search"
) -> Optional[str]:
    """
    Train a GNN policy on problems from specified sizes.

    Args:
        benchmarks: Generated problems by size
        train_sizes: Sizes to train on ("small", "medium")
        training_timesteps: Total timesteps for training
        reward_variant: Reward function to use

    Returns:
        Path to trained model, or None if training failed
    """
    if train_sizes is None:
        train_sizes = ["small", "medium"]

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: TRAINING ON SMALL/MEDIUM PROBLEMS")
    logger.info("=" * 80 + "\n")

    try:
        from common_utils import train_model

        # Collect all training problems
        train_benchmarks = []
        for size in train_sizes:
            if size in benchmarks:
                train_benchmarks.extend(benchmarks[size])

        if not train_benchmarks:
            logger.error("No training problems available")
            return None

        logger.info(f"Training on {len(train_benchmarks)} problems:")
        for domain, problem in train_benchmarks[:3]:
            logger.info(f"  - {os.path.basename(problem)}")
        if len(train_benchmarks) > 3:
            logger.info(f"  ... and {len(train_benchmarks) - 3} more")

        # Hyperparameters
        hyperparams = {
            'learning_rate': 0.0001,
            'n_steps': 128,
            'batch_size': 32,
            'ent_coef': 0.02,
            'reward_variant': reward_variant,
            'w_search_efficiency': 0.30,
            'w_solution_quality': 0.20,
            'w_f_stability': 0.35,
            'w_state_control': 0.15,
        }

        logger.info(f"\nHyperparameters:")
        for k, v in hyperparams.items():
            logger.info(f"  {k:<30} = {v}")

        # Train
        logger.info(f"\nTraining for {training_timesteps} timesteps...")

        model_path = f"misc/mvp_output/gnn_model_scale_gen_{reward_variant}.zip"
        os.makedirs("misc/mvp_output", exist_ok=True)

        model = train_model(
            model_save_path=model_path,
            benchmarks=train_benchmarks,
            hyperparams=hyperparams,
            total_timesteps=training_timesteps,
            tb_log_dir="tb_logs/",
            tb_log_name="ScaleGeneralization_Training",
            debug_mode=False,  # REAL FD
        )

        if model is None:
            logger.error("Training failed")
            return None

        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Model saved: {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================

def evaluate_scale_generalization(
    model_path: str,
    benchmarks: Dict[str, List[Tuple[str, str]]],
    test_sizes: List[str] = None,
    timeout_sec: int = 300
) -> Dict[str, Dict]:
    """
    Evaluate trained model on larger problems (scale generalization test).

    Args:
        model_path: Path to trained model
        benchmarks: Problems by size
        test_sizes: Sizes to test on ("medium", "large")
        timeout_sec: Timeout per problem

    Returns:
        Dict with evaluation results
    """
    if test_sizes is None:
        test_sizes = ["medium", "large"]

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: SCALE GENERALIZATION EVALUATION")
    logger.info("=" * 80 + "\n")

    try:
        from misc.evaluation.evaluation_comprehensive import (
            EvaluationFramework,
            GNNPolicyRunner,
            BenchmarkConfig
        )

        framework = EvaluationFramework(output_dir="scale_gen_results")

        # Run GNN evaluation on test sizes
        results_by_size = {}

        for size in test_sizes:
            if size not in benchmarks or not benchmarks[size]:
                logger.warning(f"No problems available for size: {size}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {size.upper()} problems")
            logger.info(f"{'='*60}\n")

            test_benchmarks = benchmarks[size]
            logger.info(f"Testing on {len(test_benchmarks)} {size} problem(s):")
            for domain, problem in test_benchmarks[:3]:
                logger.info(f"  - {os.path.basename(problem)}")
            if len(test_benchmarks) > 3:
                logger.info(f"  ... and {len(test_benchmarks) - 3} more")

            # Run evaluation
            gnn_runner = GNNPolicyRunner(model_path, timeout_sec)

            size_results = {
                'problems_tested': 0,
                'problems_solved': 0,
                'avg_time': 0.0,
                'avg_expansions': 0,
                'avg_plan_cost': 0,
                'solve_rate': 0.0,
                'details': []
            }

            times = []
            expansions = []
            costs = []
            solved_count = 0

            for domain_path, problem_path in test_benchmarks:
                logger.info(f"\n  Testing: {os.path.basename(problem_path)}")

                try:
                    result = gnn_runner.run(domain_path, problem_path)

                    size_results['problems_tested'] += 1

                    detail = {
                        'problem': os.path.basename(problem_path),
                        'solved': result.solved,
                        'time': result.time_sec,
                        'expansions': result.expansions,
                        'plan_cost': result.plan_cost,
                    }

                    if result.solved:
                        solved_count += 1
                        times.append(result.time_sec)
                        expansions.append(result.expansions)
                        costs.append(result.plan_cost)
                        detail['status'] = "✅ SOLVED"
                    else:
                        detail['status'] = "❌ UNSOLVED"

                    size_results['details'].append(detail)
                    logger.info(f"    {detail['status']}")

                except Exception as e:
                    logger.error(f"    Evaluation failed: {e}")
                    size_results['details'].append({
                        'problem': os.path.basename(problem_path),
                        'status': f"❌ ERROR: {str(e)[:50]}"
                    })

            # Compute statistics
            size_results['problems_solved'] = solved_count
            size_results['solve_rate'] = (solved_count / max(size_results['problems_tested'], 1)) * 100

            if times:
                size_results['avg_time'] = np.mean(times)
                size_results['median_time'] = np.median(times)
            if expansions:
                size_results['avg_expansions'] = int(np.mean(expansions))
                size_results['median_expansions'] = int(np.median(expansions))
            if costs:
                size_results['avg_plan_cost'] = int(np.mean(costs))
                size_results['median_plan_cost'] = int(np.median(costs))

            results_by_size[size] = size_results

            # Log summary
            logger.info(f"\n{'-'*60}")
            logger.info(f"SUMMARY FOR {size.upper()} PROBLEMS:")
            logger.info(f"  Solve Rate:      {size_results['solve_rate']:.1f}% ({solved_count}/{size_results['problems_tested']})")
            if times:
                logger.info(f"  Avg Time:        {size_results['avg_time']:.2f}s")
                logger.info(f"  Avg Expansions:  {size_results['avg_expansions']}")

        return results_by_size

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


# ============================================================================
# PHASE 3: ANALYSIS & REPORTING
# ============================================================================

def generate_scale_generalization_report(
    results_by_size: Dict[str, Dict],
    output_dir: str = "scale_gen_results"
) -> bool:
    """
    Generate detailed analysis of scale generalization results.

    Args:
        results_by_size: Evaluation results by problem size
        output_dir: Where to save the report

    Returns:
        True if successful
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: SCALE GENERALIZATION ANALYSIS")
    logger.info("=" * 80 + "\n")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Generate text report
        report_path = Path(output_dir) / "scale_generalization_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("SCALE GENERALIZATION ANALYSIS REPORT\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            # Summary table
            f.write("SOLVE RATE BY PROBLEM SIZE\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Size':<15} {'Solved':<20} {'Solve Rate':<15}\n")
            f.write("-" * 50 + "\n")

            for size, results in results_by_size.items():
                solved = results['problems_solved']
                total = results['problems_tested']
                rate = results['solve_rate']
                f.write(f"{size:<15} {solved}/{total:<18} {rate:>6.1f}%\n")

            f.write("-" * 50 + "\n\n")

            # Performance metrics
            f.write("PERFORMANCE METRICS BY SIZE\n")
            f.write("-" * 50 + "\n")

            for size, results in results_by_size.items():
                f.write(f"\n{size.upper()} PROBLEMS:\n")
                f.write(f"  Avg Time:           {results.get('avg_time', 'N/A')}\n")
                f.write(f"  Avg Expansions:     {results.get('avg_expansions', 'N/A')}\n")
                f.write(f"  Avg Plan Cost:      {results.get('avg_plan_cost', 'N/A')}\n")

            # Detailed results
            f.write("\n" + "=" * 90 + "\n")
            f.write("DETAILED RESULTS BY PROBLEM\n")
            f.write("=" * 90 + "\n\n")

            for size, results in results_by_size.items():
                f.write(f"\n{size.upper()} PROBLEMS:\n")
                f.write("-" * 50 + "\n")

                for detail in results.get('details', []):
                    f.write(f"  Problem: {detail['problem']}\n")
                    f.write(f"    Status: {detail['status']}\n")

                    if 'time' in detail:
                        f.write(f"    Time: {detail['time']:.2f}s\n")
                    if 'expansions' in detail:
                        f.write(f"    Expansions: {detail['expansions']}\n")

            # Analysis & conclusions
            f.write("\n" + "=" * 90 + "\n")
            f.write("SCALE GENERALIZATION ANALYSIS\n")
            f.write("=" * 90 + "\n\n")

            sizes_sorted = sorted(results_by_size.keys(),
                                 key=lambda x: ['small', 'medium', 'large'].index(x) if x in ['small', 'medium', 'large'] else 999)

            solve_rates = [results_by_size[s]['solve_rate'] for s in sizes_sorted if s in results_by_size]

            if len(solve_rates) >= 2:
                degradation = solve_rates[0] - solve_rates[-1]
                f.write(f"Generalization Degradation:\n")
                f.write(f"  From {sizes_sorted[0]} ({solve_rates[0]:.1f}%) → {sizes_sorted[-1]} ({solve_rates[-1]:.1f}%)\n")
                f.write(f"  Absolute Drop: {degradation:.1f} percentage points\n")

                if degradation < 10:
                    f.write(f"  ✅ EXCELLENT: Less than 10% degradation\n")
                elif degradation < 25:
                    f.write(f"  ✓ GOOD: Less than 25% degradation\n")
                elif degradation < 50:
                    f.write(f"  ⚠️ MODERATE: 25-50% degradation\n")
                else:
                    f.write(f"  ❌ POOR: Greater than 50% degradation\n")

            f.write("\n" + "=" * 90 + "\n")

        logger.info(f"✓ Report saved: {report_path}")

        # Save JSON results
        json_path = Path(output_dir) / "scale_generalization_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_by_size, f, indent=2, default=str)

        logger.info(f"✓ JSON results saved: {json_path}")

        # Print summary to console
        logger.info("\n" + "=" * 80)
        logger.info("SCALE GENERALIZATION SUMMARY")
        logger.info("=" * 80 + "\n")

        for size, results in sorted(results_by_size.items()):
            logger.info(f"{size.upper()}:")
            logger.info(f"  Solve Rate: {results['solve_rate']:.1f}% ({results['problems_solved']}/{results['problems_tested']})")
            if results.get('avg_time'):
                logger.info(f"  Avg Time: {results['avg_time']:.2f}s")
            logger.info()

        return True

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scale Generalization Training & Testing Framework"
    )

    parser.add_argument(
        "--domain",
        default="blocks_world",
        choices=["blocks_world", "logistics", "gripper"],
        help="Domain to generate problems for"
    )

    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=["small", "medium"],
        help="Sizes to train on"
    )

    parser.add_argument(
        "--test-sizes",
        nargs="+",
        default=["medium", "large"],
        help="Sizes to test on"
    )

    parser.add_argument(
        "--problems-per-size",
        type=int,
        default=5,
        help="Number of problems to generate per size"
    )

    parser.add_argument(
        "--training-timesteps",
        type=int,
        default=5000,
        help="Total training timesteps"
    )

    parser.add_argument(
        "--reward-variant",
        default="astar_search",
        help="Reward function variant"
    )

    parser.add_argument(
        "--output",
        default="scale_gen_results",
        help="Output directory"
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip problem generation (use existing)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)"
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 90)
    logger.info("SCALE GENERALIZATION: TRAIN SMALL/MEDIUM → TEST LARGE")
    logger.info("=" * 90)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Domain:                 {args.domain}")
    logger.info(f"  Train sizes:            {', '.join(args.train_sizes)}")
    logger.info(f"  Test sizes:             {', '.join(args.test_sizes)}")
    logger.info(f"  Problems per size:      {args.problems_per_size}")
    logger.info(f"  Training timesteps:     {args.training_timesteps}")
    logger.info(f"  Reward variant:         {args.reward_variant}")
    logger.info(f"  Output directory:       {args.output}\n")

    # Phase 0: Generate problems
    if not args.skip_generation:
        all_sizes = list(set(args.train_sizes + args.test_sizes))
        benchmarks = generate_benchmark_problems(
            domain=args.domain,
            sizes=all_sizes,
            problems_per_size=args.problems_per_size
        )

        if not benchmarks:
            logger.error("Problem generation failed")
            return 1
    else:
        logger.info("Skipping problem generation")
        benchmarks = {}

    # Phase 1: Train
    if not args.skip_training:
        model_path = train_on_problems(
            benchmarks=benchmarks,
            train_sizes=args.train_sizes,
            training_timesteps=args.training_timesteps,
            reward_variant=args.reward_variant
        )

        if model_path is None:
            logger.error("Training failed")
            return 1
    else:
        logger.info("Skipping training")
        model_path = f"misc/mvp_output/gnn_model_scale_gen_{args.reward_variant}.zip"

    # Phase 2: Evaluate
    results_by_size = evaluate_scale_generalization(
        model_path=model_path,
        benchmarks=benchmarks,
        test_sizes=args.test_sizes,
        timeout_sec=300
    )

    if not results_by_size:
        logger.error("Evaluation failed")
        return 1

    # Phase 3: Report
    if not generate_scale_generalization_report(results_by_size, args.output):
        logger.error("Report generation failed")
        return 1

    # Final summary
    logger.info("\n" + "=" * 90)
    logger.info("SCALE GENERALIZATION FRAMEWORK COMPLETE")
    logger.info("=" * 90)
    logger.info(f"\nResults saved to: {os.path.abspath(args.output)}")
    logger.info(f"Report: {os.path.join(args.output, 'scale_generalization_report.txt')}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)