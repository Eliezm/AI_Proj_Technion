#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRICULUM LEARNING EXPERIMENT - PRODUCTION VERSION
===================================================
Train progressively: small → medium → large problems.
Test on all sizes to see benefit of curriculum learning.

Usage:
    python curriculum_learning_final.py
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

class CurriculumLearningConfig:
    """Configuration for curriculum learning experiment."""

    # Experiment name
    EXPERIMENT_NAME = "curriculum_learning_experiment"

    # Output directory
    OUTPUT_DIR = "curriculum_learning_results"

    # Curriculum: train on these sizes IN ORDER
    CURRICULUM_SEQUENCE = ["small", "medium", "large"]

    # Test on all sizes
    TEST_SIZES = ["small", "medium", "large"]

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

def run_curriculum_learning_experiment():
    """Run the curriculum learning experiment."""

    # Setup
    ensure_directories_exist()
    os.makedirs(CurriculumLearningConfig.OUTPUT_DIR, exist_ok=True)

    logger = setup_logging(
        CurriculumLearningConfig.EXPERIMENT_NAME,
        CurriculumLearningConfig.OUTPUT_DIR
    )

    checkpoint_manager = ExperimentCheckpoint(CurriculumLearningConfig.OUTPUT_DIR)

    print_section("CURRICULUM LEARNING EXPERIMENT", logger)

    logger.info("Configuration:")
    logger.info(f"  Curriculum: {' → '.join(CurriculumLearningConfig.CURRICULUM_SEQUENCE)}")
    logger.info(f"  Test sizes: {', '.join(CurriculumLearningConfig.TEST_SIZES)}")
    logger.info(f"  Max problems per size: {CurriculumLearningConfig.MAX_PROBLEMS_PER_SIZE}")
    logger.info(f"  Total timesteps: {CurriculumLearningConfig.TOTAL_TIMESTEPS}")
    logger.info(f"  Reward variant: {CurriculumLearningConfig.REWARD_VARIANT}\n")

    try:
        # ====================================================================
        # PHASE 1: LOAD BENCHMARKS BY DIFFICULTY
        # ====================================================================

        print_subsection("PHASE 1: LOAD BENCHMARKS", logger)

        all_benchmarks = load_benchmarks_by_difficulty(logger=logger)

        if not all_benchmarks:
            raise RuntimeError("No benchmarks loaded")

        # ====================================================================
        # PHASE 2: CREATE CURRICULUM SEQUENCE
        # ====================================================================

        print_subsection("PHASE 2: CREATE CURRICULUM SEQUENCE", logger)

        random.seed(CurriculumLearningConfig.RANDOM_SEED)

        curriculum_problems = []
        for size in CurriculumLearningConfig.CURRICULUM_SEQUENCE:
            if size not in all_benchmarks or not all_benchmarks[size]:
                logger.warning(f"No problems available for size: {size}")
                continue

            problems = all_benchmarks[size]
            sampled = random.sample(
                problems,
                min(CurriculumLearningConfig.MAX_PROBLEMS_PER_SIZE, len(problems))
            )

            curriculum_problems.extend(sampled)
            logger.info(f"  {size}: {len(sampled)} problems (total in curriculum: {len(curriculum_problems)})")

        if not curriculum_problems:
            raise RuntimeError("No curriculum problems selected")

        logger.info(f"\nTotal curriculum problems: {len(curriculum_problems)}")

        # ====================================================================
        # PHASE 3: TRAIN WITH CURRICULUM
        # ====================================================================

        print_subsection("PHASE 3: CURRICULUM TRAINING", logger)

        checkpoint = checkpoint_manager.load()

        if checkpoint and 'model_path' in checkpoint and os.path.exists(checkpoint['model_path']):
            logger.info("Resuming from checkpoint...")
            model_path = checkpoint['model_path']
            logger.info(f"Using model: {model_path}")
            training_time = checkpoint.get('training_time', 0)
        else:
            logger.info("Starting fresh curriculum training...")

            train_start = time.time()

            model_path = train_gnn_model(
                benchmarks=curriculum_problems,
                reward_variant=CurriculumLearningConfig.REWARD_VARIANT,
                total_timesteps=CurriculumLearningConfig.TOTAL_TIMESTEPS,
                timesteps_per_problem=CurriculumLearningConfig.TIMESTEPS_PER_PROBLEM,
                model_output_path=os.path.join(
                    CurriculumLearningConfig.OUTPUT_DIR,
                    "gnn_model_curriculum.zip"
                ),
                logger=logger,
                tb_log_name="curriculum_training"
            )

            training_time = time.time() - train_start

            if model_path is None:
                logger.error("Training failed!")
                return 1

            logger.info(f"\n✅ Curriculum training complete ({format_duration(training_time)})")

            checkpoint_manager.save({
                'model_path': model_path,
                'training_time': training_time,
                'phase': 'training_complete',
            })

        # ====================================================================
        # PHASE 4: EVALUATE ON ALL SIZES
        # ====================================================================

        print_subsection("PHASE 4: EVALUATION ON ALL SIZES", logger)

        eval_start = time.time()

        # Collect all test problems
        test_benchmarks = []
        for size in CurriculumLearningConfig.TEST_SIZES:
            if size not in all_benchmarks or not all_benchmarks[size]:
                continue

            problems = all_benchmarks[size]
            sampled = random.sample(
                problems,
                min(CurriculumLearningConfig.MAX_PROBLEMS_PER_SIZE, len(problems))
            )

            test_benchmarks.extend(sampled)

        logger.info(f"Testing on {len(test_benchmarks)} problems across all sizes...")

        eval_results = evaluate_model_on_problems(
            model_path=model_path,
            benchmarks=test_benchmarks,
            reward_variant=CurriculumLearningConfig.REWARD_VARIANT,
            logger=logger
        )

        eval_elapsed = time.time() - eval_start

        logger.info(f"\n✅ Evaluation complete ({format_duration(eval_elapsed)})")

        # ====================================================================
        # PHASE 5: COMPILE RESULTS
        # ====================================================================

        print_subsection("PHASE 5: RESULTS", logger)

        results = {
            'experiment': CurriculumLearningConfig.EXPERIMENT_NAME,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'curriculum': CurriculumLearningConfig.CURRICULUM_SEQUENCE,
                'test_sizes': CurriculumLearningConfig.TEST_SIZES,
                'max_problems_per_size': CurriculumLearningConfig.MAX_PROBLEMS_PER_SIZE,
                'total_timesteps': CurriculumLearningConfig.TOTAL_TIMESTEPS,
                'reward_variant': CurriculumLearningConfig.REWARD_VARIANT,
            },
            'training': {
                'model_path': model_path,
                'curriculum_problems': len(curriculum_problems),
                'duration_seconds': training_time,
                'duration_str': format_duration(training_time),
            },
            'evaluation': eval_results,
            'summary': {
                'curriculum_solve_rate': eval_results.get('solve_rate', 0),
                'avg_reward_curriculum': eval_results.get('avg_reward', 0),
                'avg_time_curriculum': eval_results.get('avg_time', 0),
                'all_sizes_problems_solved': eval_results.get('solved_count', 0),
            }
        }

        # Log summary
        logger.info(f"\nExperiment Results:")
        logger.info(f"  Curriculum Solve Rate: {results['summary']['curriculum_solve_rate']:.1f}%")
        logger.info(f"  Avg Reward: {results['summary']['avg_reward_curriculum']:.4f}")
        logger.info(f"  Avg Time: {results['summary']['avg_time_curriculum']:.2f}s")
        logger.info(f"  Problems Solved: {results['summary']['all_sizes_problems_solved']}/{len(test_benchmarks)}")

        # Save results
        json_path = os.path.join(CurriculumLearningConfig.OUTPUT_DIR, "results.json")
        txt_path = os.path.join(CurriculumLearningConfig.OUTPUT_DIR, "results.txt")

        save_results_to_json(results, json_path, logger)
        save_results_to_txt(results, txt_path, CurriculumLearningConfig.EXPERIMENT_NAME, logger)

        checkpoint_manager.clear()

        # Final summary
        print_section("EXPERIMENT COMPLETE", logger)
        logger.info(f"✅ Curriculum learning experiment completed successfully!")
        logger.info(f"   Results: {CurriculumLearningConfig.OUTPUT_DIR}/")
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
    exit_code = run_curriculum_learning_experiment()
    sys.exit(exit_code)