#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
END-TO-END EVALUATION SCRIPT
============================
Runs complete evaluation pipeline: baselines + GNN + analysis.

Usage:
    python run_full_evaluation.py \
        --model mvp_output/gnn_model.zip \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --output evaluation_results/
"""

import sys
import os
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and log progress."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f">>> {description}")
    logger.info(f"{'=' * 80}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        logger.error(f"❌ {description} FAILED")
        return False

    logger.info(f"✅ {description} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation pipeline")

    # --- MODIFIED ARGUMENTS ---

    # Removed required=True and added a default
    parser.add_argument(
        "--model",
        default="mvp_output/gnn_model_multi_problem_astar_search.zip"
    )

    # Removed required=True and added a default
    parser.add_argument(
        "--domain",
        default="domain.pddl"
    )

    # Removed required=True and added a default
    parser.add_argument(
        "--problems",
        default="problem_small_*.pddl"
    )

    # --- UNCHANGED ARGUMENTS ---

    # This one already had a default
    parser.add_argument(
        "--output",
        default="evaluation_results"
    )

    # action="store_true" implicitly defaults to False
    # You don't need to add default=False, but it's good to know.
    parser.add_argument(
        "--skip-baselines",
        action="store_true"
    )

    args = parser.parse_args()

    # Step 1: Comprehensive evaluation
    eval_cmd = (
        f"python evaluation_comprehensive.py "
        f"--model {args.model} "
        f"--domain {args.domain} "
        f"--problems {args.problems} "
        f"--output {args.output}"
    )

    if args.skip_baselines:
        eval_cmd += " --skip-baselines"

    if not run_command(eval_cmd, "COMPREHENSIVE EVALUATION"):
        return 1

    # Step 2: Analysis and visualization
    results_csv = f"{args.output}/evaluation_results.csv"
    plots_dir = f"{args.output}/plots"

    analysis_cmd = (
        f"python analysis_and_visualization.py "
        f"--results {results_csv} "
        f"--output {plots_dir}"
    )

    if not run_command(analysis_cmd, "ANALYSIS AND VISUALIZATION"):
        logger.warning("Analysis step failed, but evaluation is complete")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {os.path.abspath(args.output)}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review comparison report:")
    logger.info(f"     cat {args.output}/comparison_report.txt")
    logger.info(f"  2. View plots:")
    logger.info(f"     - {plots_dir}/solve_rate_comparison.png")
    logger.info(f"     - {plots_dir}/time_comparison.png")
    logger.info(f"     - {plots_dir}/efficiency_frontier.png")
    logger.info(f"  3. Detailed results CSV:")
    logger.info(f"     {results_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())