#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCALE GENERALIZATION EXPERIMENT - PRODUCTION VERSION
===================================================
Train on small/medium problems.
Test on medium/large problems from the same domain.

This measures how well the model generalizes to larger problems.

Usage:
    python scale_generalization_final.py
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
    load_benchmarks_by_difficulty, save_results_to_json, save_results_to_txt,
    ensure_directories_exist, get_timestamp_str, format_duration
)

import time


# ============================================================================
# EXPERIMENT CONFIG
# ============================================================================

class ScaleGeneralizationConfig:
    """Configuration for scale generalization experiment."""

    # Experiment name
    EXPERIMENT_NAME = "scale_generalization_experiment"

    # Output directory
    OUTPUT_DIR = "scale_generalization_results"

    # Training on: small + medium
    TRAIN_SIZES = ["small", "medium"]

    # Testing on: medium + large
    TEST_SIZES = ["medium", "large"]

    # Max problems per size
    MAX_PROBLEMS_PER_SIZE = 5

    # Training
    REWARD_VARIANT = "astar_search"
    TOTAL_TIMESTEPS = 5000
    TIMESTEPS_PER_PROBLEM = 500

    # Seeding
    RANDOM_SEED = 42


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_scale_generalization_experiment():
    """Run the scale generalization experiment."""

    # Setup
    ensure_directories_exist()
    os.makedirs(ScaleGeneralizationConfig.OUTPUT_DIR, exist_ok=True)

    logger = setup_logging(
        ScaleGeneralizationConfig.EXPERIMENT_NAME,
        ScaleGeneralizationConfig.OUTPUT_DIR
    )

    checkpoint_manager = ExperimentCheckpoint(ScaleGeneralizationConfig.OUTPUT_DIR)

    print_section("SCALE GENERALIZATION EXPERIMENT", logger)

    logger.info("Configuration:")
    logger.info(f"  Train sizes: {', '.join(ScaleGeneralizationConfig.TRAIN_SIZES)}")
    logger.info(f"  Test sizes: {', '.join(ScaleGeneralizationConfig.TEST_SIZES)}")
    logger.info(f"  Max problems per size: {ScaleGeneralizationConfig.MAX_PROBLEMS_PER_SIZE}")
    logger.info(f"  Total timesteps: {ScaleGeneralizationConfig.TOTAL_TIMESTEPS}")
    logger.info(f"  Reward variant: {ScaleGeneralizationConfig.REWARD_VARIANT}\n")

    try:
        # ====================================================================
        # PHASE 1: LOAD BENCHMARKS BY DIFFICULTY
        # ====================================================================

        print_subsection("PHASE 1: LOAD BENCHMARKS", logger)

        all_benchmarks = load_benchmarks_by_difficulty(logger=logger)

        if not all_benchmarks:
            raise RuntimeError("No benchmarks loaded")

        # ====================================================================
        # PHASE 2: SELECT TRAINING SET
        # ====================================================================

        print_subsection("PHASE 2: SELECT TRAINING PROBLEMS", logger)

        random.seed(ScaleGeneralizationConfig.RANDOM_SEED)

        train_benchmarks = []
        for size in ScaleGeneralizationConfig.TRAIN_SIZES:
            if size not in all_benchmarks or not all_benchmarks[size]:
                logger.warning(f"No problems available for size: {size}")
                continue

            problems = all_benchmarks[size]
            sampled = random.sample(
                problems,
                min(ScaleGeneralizationConfig.MAX_PROBLEMS_PER_SIZE, len(problems))
            )

            train_benchmarks.extend(sampled)
            logger.info(f"  {size}: {len(sampled)} problems added")

        if not train_benchmarks:
            raise RuntimeError("No training problems selected")

        logger.info(f"\nTotal training problems: {len(train_benchmarks)}")

        # ====================================================================
        # PHASE 3: SELECT TEST SET
        # ====================================================================

        print_subsection("PHASE 3: SELECT TEST PROBLEMS", logger)

        test_benchmarks = []
        for size in ScaleGeneralizationConfig.TEST_SIZES:
            if size not in all_benchmarks or not all_benchmarks[size]:
                logger.warning(f"No problems available for size: {size}")
                continue

            problems = all_benchmarks[size]
            sampled = random.sample(
                problems,
                min(ScaleGeneralizationConfig.MAX_PROBLEMS_PER_SIZE, len(problems))
            )

            test_benchmarks.extend(sampled)
            logger.info(f"  {size}: {len(sampled)} problems added")

        if not test_benchmarks:
            raise RuntimeError("No test problems selected")

        logger.info(f"\nTotal test problems: {len(test_benchmarks)}")

        # ====================================================================
        # PHASE 4: TRAIN MODEL
        # ====================================================================

        print_subsection("PHASE 4: TRAINING ON SMALL/MEDIUM PROBLEMS", logger)

        checkpoint = checkpoint_manager.load()

        if checkpoint and 'model_path' in checkpoint and os.path.exists(checkpoint['model_path']):
            logger.info("Resuming from checkpoint...")
            model_path = checkpoint['model_path']
            logger.info(f"Using model: {model_path}")
        else:
            logger.info("Starting fresh training...")

            train_start = time.time()

            model_path = train_gnn_model(
                benchmarks=train_benchmarks,
                reward_variant=ScaleGeneralizationConfig.REWARD_VARIANT,
                total_timesteps=ScaleGeneralizationConfig.TOTAL_TIMESTEPS,
                timesteps_per_problem=ScaleGeneralizationConfig.TIMESTEPS_PER_PROBLEM,
                model_output_path=os.path.join(
                    ScaleGeneralizationConfig.OUTPUT_DIR,
                    "gnn_model_trained.zip"
                ),
                logger=logger,
                tb_log_name="scale_gen_training"
            )

            train_elapsed = time.time() - train_start

            if model_path is None:
                logger.error("Training failed!")
                return 1

            logger.info(f"\n✅ Training complete ({format_duration(train_elapsed)})")

            checkpoint_manager.save({
                'model_path': model_path,
                'phase': 'training_complete',
            })

        # ====================================================================
        # PHASE 5: EVALUATE ON LARGER PROBLEMS
        # ====================================================================

        print_subsection("PHASE 5: EVALUATION ON LARGER PROBLEMS", logger)

        eval_start = time.time()

        eval_results = evaluate_model_on_problems(
            model_path=model_path,
            benchmarks=test_benchmarks,
            reward_variant=ScaleGeneralizationConfig.REWARD_VARIANT,
            logger=logger
        )

        eval_elapsed = time.time() - eval_start

        logger.info(f"\n✅ Evaluation complete ({format_duration(eval_elapsed)})")

        # ====================================================================
        # PHASE 6: COMPILE RESULTS
        # ====================================================================

        print_subsection("PHASE 6: RESULTS", logger)

        results = {
            'experiment': ScaleGeneralizationConfig.EXPERIMENT_NAME,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'train_sizes': ScaleGeneralizationConfig.TRAIN_SIZES,
                'test_sizes': ScaleGeneralizationConfig.TEST_SIZES,
                'max_problems_per_size': ScaleGeneralizationConfig.MAX_PROBLEMS_PER_SIZE,
                'total_timesteps': ScaleGeneralizationConfig.TOTAL_TIMESTEPS,
                'reward_variant': ScaleGeneralizationConfig.REWARD_VARIANT,
            },
            'training': {
                'model_path': model_path,
                'problems_used': len(train_benchmarks),
                'duration_seconds': train_elapsed,
                'duration_str': format_duration(train_elapsed),
            },
            'evaluation': eval_results,
            'summary': {
                'scale_generalization_solve_rate': eval_results.get('solve_rate', 0),
                'avg_reward_on_larger': eval_results.get('avg_reward', 0),
                'avg_time_on_larger': eval_results.get('avg_time', 0),
                'larger_problems_solved': eval_results.get('solved_count', 0),
            }
        }

        # Log summary
        logger.info(f"\nExperiment Results:")
        logger.info(f"  Scale Generalization Solve Rate: {results['summary']['scale_generalization_solve_rate']:.1f}%")
        logger.info(f"  Avg Reward (larger): {results['summary']['avg_reward_on_larger']:.4f}")
        logger.info(f"  Avg Time (larger): {results['summary']['avg_time_on_larger']:.2f}s")
        logger.info(f"  Larger Problems Solved: {results['summary']['larger_problems_solved']}/{len(test_benchmarks)}")

        # Save results
        json_path = os.path.join(ScaleGeneralizationConfig.OUTPUT_DIR, "results.json")
        txt_path = os.path.join(ScaleGeneralizationConfig.OUTPUT_DIR, "results.txt")

        save_results_to_json(results, json_path, logger)
        save_results_to_txt(results, txt_path, ScaleGeneralizationConfig.EXPERIMENT_NAME, logger)

        checkpoint_manager.clear()

        # Final summary
        print_section("EXPERIMENT COMPLETE", logger)
        logger.info(f"✅ Scale generalization experiment completed successfully!")
        logger.info(f"   Results: {ScaleGeneralizationConfig.OUTPUT_DIR}/")
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
    exit_code = run_scale_generalization_experiment()
    sys.exit(exit_code)