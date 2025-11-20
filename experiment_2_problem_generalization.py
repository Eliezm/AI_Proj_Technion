#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROBLEM GENERALIZATION EXPERIMENT - PRODUCTION VERSION (REFACTORED)
==================================================================
Train on a set of problems from a domain.
Test on DIFFERENT problems from the same domain.

Compatible with new benchmark format.

Usage:
    python experiment_2_problem_generalization.py
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

from shared_experiment_utils import (
    setup_logging, print_section, print_subsection,
    ExperimentCheckpoint, train_gnn_model, evaluate_model_on_problems,
    save_results_to_json, save_results_to_txt,
    ensure_directories_exist, format_duration,
    load_and_validate_benchmarks,  # ✅ NEW
    get_benchmarks_for_sizes  # ✅ NEW
)

import time


class ProblemGeneralizationConfig:
    """Configuration for problem generalization experiment."""

    EXPERIMENT_NAME = "problem_generalization_experiment"
    OUTPUT_DIR = "problem_generalization_results"

    # ✅ REFACTORED: Use new benchmark format
    BENCHMARK_DIR = "misc/benchmarks"
    SIZES = ["small"]  # Use small problems
    TRAIN_RATIO = 0.8

    REWARD_VARIANT = "astar_search"
    TOTAL_TIMESTEPS = 5000
    TIMESTEPS_PER_PROBLEM = 500
    RANDOM_SEED = 42


def run_problem_generalization_experiment():
    """Run the problem generalization experiment."""

    ensure_directories_exist()
    os.makedirs(ProblemGeneralizationConfig.OUTPUT_DIR, exist_ok=True)

    logger = setup_logging(
        ProblemGeneralizationConfig.EXPERIMENT_NAME,
        ProblemGeneralizationConfig.OUTPUT_DIR
    )

    checkpoint_manager = ExperimentCheckpoint(ProblemGeneralizationConfig.OUTPUT_DIR)

    print_section("PROBLEM GENERALIZATION EXPERIMENT", logger)

    logger.info("Configuration:")
    logger.info(f"  Benchmark directory: {ProblemGeneralizationConfig.BENCHMARK_DIR}")
    logger.info(f"  Sizes: {ProblemGeneralizationConfig.SIZES}")
    logger.info(f"  Train/test ratio: {ProblemGeneralizationConfig.TRAIN_RATIO:.0%}")
    logger.info(f"  Total timesteps: {ProblemGeneralizationConfig.TOTAL_TIMESTEPS}")
    logger.info(f"  Reward variant: {ProblemGeneralizationConfig.REWARD_VARIANT}\n")

    try:
        # ====================================================================
        # PHASE 1: LOAD BENCHMARKS
        # ====================================================================

        print_subsection("PHASE 1: LOAD BENCHMARKS", logger)

        all_benchmarks = load_and_validate_benchmarks(
            benchmark_dir=ProblemGeneralizationConfig.BENCHMARK_DIR,
            logger=logger
        )

        if not all_benchmarks:
            logger.error("No benchmarks loaded!")
            return 1

        # ====================================================================
        # PHASE 2: GET PROBLEMS AND SPLIT
        # ====================================================================

        print_subsection("PHASE 2: SELECT AND SPLIT PROBLEMS", logger)

        all_problems = get_benchmarks_for_sizes(
            all_benchmarks,
            sizes=ProblemGeneralizationConfig.SIZES
        )

        if not all_problems:
            logger.error("No problems found for selected sizes!")
            return 1

        # Split into train/test
        random.seed(ProblemGeneralizationConfig.RANDOM_SEED)
        shuffled = all_problems.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * ProblemGeneralizationConfig.TRAIN_RATIO)
        train_benchmarks = sorted(shuffled[:split_idx])
        test_benchmarks = sorted(shuffled[split_idx:])

        logger.info(f"Train set ({len(train_benchmarks)} problems):")
        for i, (_, prob) in enumerate(train_benchmarks[:3], 1):
            logger.info(f"  {i}. {os.path.basename(prob)}")
        if len(train_benchmarks) > 3:
            logger.info(f"  ... and {len(train_benchmarks) - 3} more")

        logger.info(f"\nTest set ({len(test_benchmarks)} problems):")
        for i, (_, prob) in enumerate(test_benchmarks[:3], 1):
            logger.info(f"  {i}. {os.path.basename(prob)}")
        if len(test_benchmarks) > 3:
            logger.info(f"  ... and {len(test_benchmarks) - 3} more")

        # ====================================================================
        # PHASE 3: TRAIN MODEL
        # ====================================================================

        print_subsection("PHASE 3: TRAINING ON TRAIN SET", logger)

        checkpoint = checkpoint_manager.load()

        if checkpoint and 'model_path' in checkpoint and os.path.exists(checkpoint['model_path']):
            logger.info("Resuming from checkpoint...")
            model_path = checkpoint['model_path']
            train_elapsed = checkpoint.get('train_elapsed', 0)
        else:
            logger.info("Starting fresh training...")

            train_start = time.time()

            model_path = train_gnn_model(
                benchmarks=train_benchmarks,
                reward_variant=ProblemGeneralizationConfig.REWARD_VARIANT,
                total_timesteps=ProblemGeneralizationConfig.TOTAL_TIMESTEPS,
                timesteps_per_problem=ProblemGeneralizationConfig.TIMESTEPS_PER_PROBLEM,
                model_output_path=os.path.join(
                    ProblemGeneralizationConfig.OUTPUT_DIR,
                    "gnn_model_trained.zip"
                ),
                logger=logger,
                tb_log_name="problem_gen_training"
            )

            train_elapsed = time.time() - train_start

            if model_path is None:
                logger.error("Training failed!")
                return 1

            logger.info(f"\n✅ Training complete ({format_duration(train_elapsed)})")

            checkpoint_manager.save({
                'model_path': model_path,
                'train_elapsed': train_elapsed,
                'phase': 'training_complete',
            })

        # ====================================================================
        # PHASE 4: EVALUATE ON TEST SET
        # ====================================================================

        print_subsection("PHASE 4: EVALUATION ON TEST SET", logger)

        eval_start = time.time()

        eval_results = evaluate_model_on_problems(
            model_path=model_path,
            benchmarks=test_benchmarks,
            reward_variant=ProblemGeneralizationConfig.REWARD_VARIANT,
            logger=logger
        )

        eval_elapsed = time.time() - eval_start

        logger.info(f"\n✅ Evaluation complete ({format_duration(eval_elapsed)})")

        # ====================================================================
        # PHASE 5: COMPILE RESULTS
        # ====================================================================

        print_subsection("PHASE 5: RESULTS", logger)

        results = {
            'experiment': ProblemGeneralizationConfig.EXPERIMENT_NAME,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'benchmark_dir': ProblemGeneralizationConfig.BENCHMARK_DIR,
                'sizes': ProblemGeneralizationConfig.SIZES,
                'train_problems': len(train_benchmarks),
                'test_problems': len(test_benchmarks),
                'train_ratio': ProblemGeneralizationConfig.TRAIN_RATIO,
                'total_timesteps': ProblemGeneralizationConfig.TOTAL_TIMESTEPS,
                'reward_variant': ProblemGeneralizationConfig.REWARD_VARIANT,
            },
            'training': {
                'model_path': model_path,
                'problems_used': len(train_benchmarks),
                'duration_seconds': train_elapsed,
                'duration_str': format_duration(train_elapsed),
            },
            'evaluation': eval_results,
            'summary': {
                'generalization_solve_rate': eval_results.get('solve_rate', 0),
                'avg_reward_on_test': eval_results.get('avg_reward', 0),
                'avg_time_on_test': eval_results.get('avg_time', 0),
                'test_problems_solved': eval_results.get('solved_count', 0),
            }
        }

        logger.info(f"\nExperiment Results:")
        logger.info(f"  Generalization Solve Rate: {results['summary']['generalization_solve_rate']:.1f}%")
        logger.info(f"  Avg Reward (test): {results['summary']['avg_reward_on_test']:.4f}")
        logger.info(f"  Avg Time (test): {results['summary']['avg_time_on_test']:.2f}s")
        logger.info(f"  Test Problems Solved: {results['summary']['test_problems_solved']}/{len(test_benchmarks)}")

        json_path = os.path.join(ProblemGeneralizationConfig.OUTPUT_DIR, "results.json")
        txt_path = os.path.join(ProblemGeneralizationConfig.OUTPUT_DIR, "results.txt")

        save_results_to_json(results, json_path, logger)
        save_results_to_txt(results, txt_path, ProblemGeneralizationConfig.EXPERIMENT_NAME, logger)

        checkpoint_manager.clear()

        print_section("EXPERIMENT COMPLETE", logger)
        logger.info(f"✅ Problem generalization experiment completed successfully!")
        logger.info(f"   Results: {ProblemGeneralizationConfig.OUTPUT_DIR}/")
        logger.info(f"   Model: {model_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Experiment interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        logger.error(traceback.format_exc())

        checkpoint_manager.save({
            'phase': 'failed',
            'error': str(e)
        })

        return 1


if __name__ == "__main__":
    exit_code = run_problem_generalization_experiment()
    sys.exit(exit_code)