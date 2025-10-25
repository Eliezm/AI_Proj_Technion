#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERFIT EXPERIMENT - PRODUCTION VERSION
=======================================
Train on a fixed set of problems from ONE domain.
Test on the SAME problems to measure overfitting.

This experiment verifies that the GNN can learn to specialize on a problem set.

Usage:
    python overfit_experiment_final.py
"""

import sys
import os
import json
import glob
import random
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Import shared utilities
from shared_experiment_utils import (
    setup_logging, print_section, print_subsection,
    ExperimentCheckpoint, train_gnn_model, evaluate_model_on_problems,
    save_results_to_json, save_results_to_txt, ensure_directories_exist,
    get_timestamp_str, format_duration
)

import time


# ============================================================================
# EXPERIMENT CONFIG
# ============================================================================

class OverfitExperimentConfig:
    """Configuration for overfit experiment."""

    # Experiment name
    EXPERIMENT_NAME = "overfit_experiment"

    # Output directory
    OUTPUT_DIR = "overfit_experiment_results"

    # Dataset
    DOMAIN_FILE = "benchmarks/small/domain.pddl"
    PROBLEM_PATTERN = "benchmarks/small/problem_*.pddl"
    NUM_PROBLEMS = 5  # Use only 5 problems for overfitting

    # Training
    REWARD_VARIANT = "astar_search"
    TOTAL_TIMESTEPS = 2000
    TIMESTEPS_PER_PROBLEM = 500

    # Evaluation
    NUM_EVAL_RUNS_PER_PROBLEM = 3

    # Seeding
    RANDOM_SEED = 42


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_overfit_experiment():
    """Run the complete overfit experiment with checkpoint/resume support."""

    # Setup
    ensure_directories_exist()
    os.makedirs(OverfitExperimentConfig.OUTPUT_DIR, exist_ok=True)

    logger = setup_logging(
        OverfitExperimentConfig.EXPERIMENT_NAME,
        OverfitExperimentConfig.OUTPUT_DIR
    )

    checkpoint_manager = ExperimentCheckpoint(OverfitExperimentConfig.OUTPUT_DIR)

    print_section("OVERFIT EXPERIMENT", logger)

    logger.info("Configuration:")
    logger.info(f"  Domain: {OverfitExperimentConfig.DOMAIN_FILE}")
    logger.info(f"  Problems: {OverfitExperimentConfig.NUM_PROBLEMS}")
    logger.info(f"  Total timesteps: {OverfitExperimentConfig.TOTAL_TIMESTEPS}")
    logger.info(f"  Reward variant: {OverfitExperimentConfig.REWARD_VARIANT}\n")

    try:
        # ====================================================================
        # PHASE 1: SELECT PROBLEMS
        # ====================================================================

        print_subsection("PHASE 1: SELECT TRAINING PROBLEMS", logger)

        # Load all problems
        all_problems = sorted(glob.glob(OverfitExperimentConfig.PROBLEM_PATTERN))

        if not all_problems:
            raise ValueError(f"No problems found: {OverfitExperimentConfig.PROBLEM_PATTERN}")

        # Select subset
        random.seed(OverfitExperimentConfig.RANDOM_SEED)
        selected = random.sample(all_problems, min(OverfitExperimentConfig.NUM_PROBLEMS, len(all_problems)))
        selected = sorted(selected)

        domain_file = os.path.abspath(OverfitExperimentConfig.DOMAIN_FILE)
        benchmarks = [(domain_file, os.path.abspath(p)) for p in selected]

        logger.info(f"Selected {len(benchmarks)} problems:")
        for i, (_, prob) in enumerate(benchmarks, 1):
            logger.info(f"  {i}. {os.path.basename(prob)}")

        # ====================================================================
        # PHASE 2: TRAIN MODEL
        # ====================================================================

        print_subsection("PHASE 2: TRAINING", logger)

        checkpoint = checkpoint_manager.load()

        if checkpoint and 'model_path' in checkpoint and os.path.exists(checkpoint['model_path']):
            logger.info("Resuming from checkpoint...")
            model_path = checkpoint['model_path']
            logger.info(f"Using model: {model_path}")
        else:
            logger.info("Starting fresh training...")

            train_start = time.time()

            model_path = train_gnn_model(
                benchmarks=benchmarks,
                reward_variant=OverfitExperimentConfig.REWARD_VARIANT,
                total_timesteps=OverfitExperimentConfig.TOTAL_TIMESTEPS,
                timesteps_per_problem=OverfitExperimentConfig.TIMESTEPS_PER_PROBLEM,
                model_output_path=os.path.join(
                    OverfitExperimentConfig.OUTPUT_DIR,
                    "gnn_model_trained.zip"
                ),
                logger=logger,
                tb_log_name="overfit_training"
            )

            train_elapsed = time.time() - train_start

            if model_path is None:
                logger.error("Training failed!")
                return 1

            logger.info(f"\n✅ Training complete ({format_duration(train_elapsed)})")

            # Save checkpoint
            checkpoint_manager.save({
                'model_path': model_path,
                'phase': 'training_complete',
            })

        # ====================================================================
        # PHASE 3: EVALUATE ON TRAINING PROBLEMS
        # ====================================================================

        print_subsection("PHASE 3: EVALUATION ON TRAINING SET", logger)

        eval_start = time.time()

        eval_results = evaluate_model_on_problems(
            model_path=model_path,
            benchmarks=benchmarks,
            reward_variant=OverfitExperimentConfig.REWARD_VARIANT,
            logger=logger
        )

        eval_elapsed = time.time() - eval_start

        logger.info(f"\n✅ Evaluation complete ({format_duration(eval_elapsed)})")

        # ====================================================================
        # PHASE 4: COMPILE RESULTS
        # ====================================================================

        print_subsection("PHASE 4: RESULTS", logger)

        results = {
            'experiment': OverfitExperimentConfig.EXPERIMENT_NAME,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'domain': OverfitExperimentConfig.DOMAIN_FILE,
                'num_problems': len(benchmarks),
                'total_timesteps': OverfitExperimentConfig.TOTAL_TIMESTEPS,
                'reward_variant': OverfitExperimentConfig.REWARD_VARIANT,
            },
            'training': {
                'model_path': model_path,
                'duration_seconds': train_elapsed,
                'duration_str': format_duration(train_elapsed),
            },
            'evaluation': eval_results,
            'summary': {
                'solve_rate': eval_results.get('solve_rate', 0),
                'avg_reward': eval_results.get('avg_reward', 0),
                'avg_time': eval_results.get('avg_time', 0),
                'problems_solved': eval_results.get('solved_count', 0),
            }
        }

        # Log summary
        logger.info(f"\nExperiment Results:")
        logger.info(f"  Solve Rate: {results['summary']['solve_rate']:.1f}%")
        logger.info(f"  Avg Reward: {results['summary']['avg_reward']:.4f}")
        logger.info(f"  Avg Time: {results['summary']['avg_time']:.2f}s")
        logger.info(f"  Problems Solved: {results['summary']['problems_solved']}/{len(benchmarks)}")

        # Save results
        json_path = os.path.join(OverfitExperimentConfig.OUTPUT_DIR, "results.json")
        txt_path = os.path.join(OverfitExperimentConfig.OUTPUT_DIR, "results.txt")

        save_results_to_json(results, json_path, logger)
        save_results_to_txt(results, txt_path, OverfitExperimentConfig.EXPERIMENT_NAME, logger)

        # Clear checkpoint on success
        checkpoint_manager.clear()

        # Final summary
        print_section("EXPERIMENT COMPLETE", logger)
        logger.info(f"✅ Overfit experiment completed successfully!")
        logger.info(f"   Results: {OverfitExperimentConfig.OUTPUT_DIR}/")
        logger.info(f"   Model: {model_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Experiment interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        logger.error(traceback.format_exc())

        # Save checkpoint on error (for resume)
        checkpoint_manager.save({
            'phase': 'failed',
            'error': str(e)
        })

        return 1


if __name__ == "__main__":
    exit_code = run_overfit_experiment()
    sys.exit(exit_code)