# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTI-PROBLEM REAL TRAINING - GNN Merge Strategy with Dataset Support
======================================================================

Trains a GNN policy using actual Fast Downward with multiple problems across
difficulty levels (small, medium, hard).

Features:
  ✓ Load problems from benchmarks/small|medium|hard/ structure
  ✓ Train on single or multiple difficulty levels
  ✓ Curriculum learning (easy→medium→hard)
  ✓ Mixed training (random difficulty per episode)
  ✓ Full compatibility with existing framework
  ✓ Comprehensive logging and diagnostics

Run with:
    python train_real_working.py

Environment Variables:
    REWARD_VARIANT: Which reward function to use
      Options: simple_stability, information_preservation, hybrid, conservative,
               progressive, rich, astar_search (default: astar_search)

    DIFFICULTY: Which difficulty to train on
      Options: small, medium, hard, mixed (default: mixed)

    MAX_PROBLEMS_PER_DIFFICULTY: Max problems per difficulty (default: 5)

    CURRICULUM_LEARNING: Enable curriculum learning (default: false)
      When true: train on small→medium→hard sequentially
      When false: random sampling from selected difficulties

Example:
    REWARD_VARIANT=astar_search DIFFICULTY=mixed python train_real_working.py
    CURRICULUM_LEARNING=true DIFFICULTY=mixed python train_real_working.py
"""

import sys
import os
import logging
import glob
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

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
        logging.FileHandler("training_multi_problem.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


def print_section(title: str, symbol: str = "=", width: int = 90):
    """Print a formatted section header."""
    logger.info("\n" + symbol * width)
    logger.info(f"// {title.upper()}")
    logger.info(symbol * width + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    logger.info("\n" + "-" * 80)
    logger.info(f">>> {title}")
    logger.info("-" * 80)


# ============================================================================
# PHASE 0: BENCHMARK LOADING
# ============================================================================

def load_benchmarks_from_folders() -> Dict[str, List[Tuple[str, str]]]:
    """
    Load benchmarks from benchmarks/small|medium|hard/ directory structure.

    Expected structure:
        benchmarks/
        ├── small/
        │   ├── domain.pddl
        │   ├── problem_small_00.pddl
        │   └── ...
        ├── medium/
        │   ├── domain.pddl
        │   ├── problem_medium_00.pddl
        │   └── ...
        └── hard/
            ├── domain.pddl
            ├── problem_hard_00.pddl
            └── ...

    Returns:
        Dict mapping difficulty → list of (domain_file, problem_file) tuples
    """
    print_section("PHASE 0: LOAD BENCHMARKS FROM FOLDER STRUCTURE")

    benchmarks_dir = os.path.abspath("benchmarks")

    if not os.path.isdir(benchmarks_dir):
        logger.error(f"Benchmarks directory not found: {benchmarks_dir}")
        logger.error("Expected structure:")
        logger.error("  benchmarks/")
        logger.error("  ├── small/")
        logger.error("  ├── medium/")
        logger.error("  └── hard/")
        return {}

    logger.info(f"Benchmarks directory: {benchmarks_dir}\n")

    difficulties = ["small", "medium", "hard"]
    all_benchmarks = {}

    for difficulty in difficulties:
        difficulty_dir = os.path.join(benchmarks_dir, difficulty)

        logger.info(f"Loading {difficulty.upper()} difficulty problems...")

        if not os.path.isdir(difficulty_dir):
            logger.warning(f"  ⚠️ Directory not found: {difficulty_dir}")
            all_benchmarks[difficulty] = []
            continue

        # Find domain file
        domain_file = os.path.join(difficulty_dir, "domain.pddl")
        if not os.path.exists(domain_file):
            logger.warning(f"  ⚠️ Domain file not found: {domain_file}")
            all_benchmarks[difficulty] = []
            continue

        logger.info(f"  ✓ Domain: {domain_file}")

        # Find problem files (look for problem_*.pddl patterns)
        problem_patterns = [
            f"problem_{difficulty}_*.pddl",
            f"problem_*.pddl"
        ]

        problems = []
        for pattern in problem_patterns:
            problems = sorted(glob.glob(os.path.join(difficulty_dir, pattern)))
            if problems:
                break

        if not problems:
            logger.warning(f"  ⚠️ No problem files found in {difficulty_dir}")
            all_benchmarks[difficulty] = []
            continue

        logger.info(f"  ✓ Found {len(problems)} problem(s)")
        for i, prob in enumerate(problems[:3], 1):  # Show first 3
            logger.info(f"    {i}. {os.path.basename(prob)}")
        if len(problems) > 3:
            logger.info(f"    ... and {len(problems) - 3} more")

        # Create benchmark list (absolute paths)
        benchmarks_list = [
            (os.path.abspath(domain_file), os.path.abspath(prob))
            for prob in problems
        ]

        all_benchmarks[difficulty] = benchmarks_list
        logger.info(f"  ✓ Loaded {len(benchmarks_list)} benchmark(s) for {difficulty}\n")

    # Summary
    total = sum(len(b) for b in all_benchmarks.values())
    if total == 0:
        logger.error("No benchmarks loaded!")
        return {}

    logger.info(f"✅ Loaded {total} total benchmarks across all difficulties:")
    for difficulty in difficulties:
        count = len(all_benchmarks[difficulty])
        logger.info(f"   {difficulty:<10} {count:>3} problems")

    return all_benchmarks


def get_benchmark_sequence(
        all_benchmarks: Dict[str, List[Tuple[str, str]]],
        difficulty: str = "mixed",
        max_problems_per_difficulty: int = 5,
        curriculum_learning: bool = False
) -> List[Tuple[str, str]]:
    """
    Create a sequence of benchmark problems for training.

    Args:
        all_benchmarks: All loaded benchmarks by difficulty
        difficulty: "small", "medium", "hard", or "mixed"
        max_problems_per_difficulty: Max problems to use per difficulty
        curriculum_learning: If True, train small→medium→hard sequentially

    Returns:
        List of (domain_file, problem_file) tuples for training
    """
    print_subsection("Create Benchmark Sequence")

    sequence = []

    if difficulty == "mixed":
        difficulties = ["small", "medium", "hard"]
    else:
        difficulties = [difficulty]

    logger.info(f"Difficulty mode: {difficulty}")
    logger.info(f"Max problems per difficulty: {max_problems_per_difficulty}")
    logger.info(f"Curriculum learning: {curriculum_learning}\n")

    if curriculum_learning and difficulty == "mixed":
        # Curriculum: small → medium → hard
        logger.info("Using CURRICULUM LEARNING: small → medium → hard\n")

        for diff in difficulties:
            if diff not in all_benchmarks or not all_benchmarks[diff]:
                continue

            problems = all_benchmarks[diff][:max_problems_per_difficulty]
            sequence.extend(problems)
            logger.info(f"  Added {len(problems)} {diff} problems (total: {len(sequence)})")

    else:
        # Random mixing or single difficulty
        for diff in difficulties:
            if diff not in all_benchmarks or not all_benchmarks[diff]:
                continue

            problems = all_benchmarks[diff][:max_problems_per_difficulty]
            sequence.extend(problems)
            logger.info(f"  Added {len(problems)} {diff} problems")

        # Shuffle if mixed
        if difficulty == "mixed":
            random.shuffle(sequence)
            logger.info(f"\nShuffled sequence randomly")

    logger.info(f"\n✅ Total problems in sequence: {len(sequence)}")

    return sequence


def create_problem_iterator(
        benchmark_sequence: List[Tuple[str, str]],
        epochs: int = 1
):
    """
    Create an iterator that cycles through problems for multiple epochs.

    Args:
        benchmark_sequence: List of (domain, problem) tuples
        epochs: Number of times to cycle through the sequence

    Yields:
        (domain_file, problem_file, problem_name, epoch, step)
    """
    for epoch in range(epochs):
        for step, (domain_file, problem_file) in enumerate(benchmark_sequence):
            problem_name = os.path.basename(problem_file)
            yield domain_file, problem_file, problem_name, epoch + 1, step + 1


# ============================================================================
# PHASE 1: ENVIRONMENT INITIALIZATION
# ============================================================================

# FILE: train_real_working.py
# REPLACE THIS FUNCTION

def init_training_environment(
        domain_file: str,
        problem_file: str,
        reward_variant: str = "astar_search",
        **reward_kwargs  # ✅ ADD THIS
) -> Optional[Tuple]:
    """Initialize a training environment for a single problem."""
    try:
        from merge_env import MergeEnv

        domain_path = os.path.abspath(domain_file)
        problem_path = os.path.abspath(problem_file)

        logger.info(f"Creating environment...")
        logger.info(f"  Domain:  {domain_path}")
        logger.info(f"  Problem: {problem_path}")

        # ✅ FIX: Pass the reward_kwargs dictionary instead of hard-coded values
        env = MergeEnv(
            domain_file=domain_path,
            problem_file=problem_path,
            max_merges=50,
            debug=False,  # REAL FD
            reward_variant=reward_variant,
            **reward_kwargs # ✅ USE THE PASSED ARGUMENTS
        )

        logger.info("✓ Environment created\n")
        return env, domain_path, problem_path

    except Exception as e:
        logger.error(f"✗ Environment creation failed: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# PHASE 2: MULTI-PROBLEM TRAINING
# ============================================================================

# FILE: train_real_working.py
# REPLACE THIS FUNCTION

def run_multi_problem_training(
        benchmark_sequence: List[Tuple[str, str]],
        reward_variant: str = "astar_search",
        total_timesteps: int = 10000,
        timesteps_per_problem: int = 500,
        **reward_kwargs  # ✅ This was already here and is correct
) -> bool:
    """
    Run training on multiple problems with real FD.

    Args:
        benchmark_sequence: List of (domain, problem) tuples to train on
        reward_variant: Reward function variant
        total_timesteps: Total training timesteps (if 0, use problems-based limit)
        timesteps_per_problem: Timesteps to train on each problem
        **reward_kwargs: Reward function parameters

    Returns:
        True if training succeeded, False otherwise
    """
    print_section("PHASE 2: MULTI-PROBLEM TRAINING")

    try:
        from merge_env import MergeEnv
        from gnn_policy import GNNPolicy
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        # ✅ NEW: Load or create model once for all problems
        model_path = "mvp_output/gnn_model.zip"
        model = None

        if os.path.exists(model_path):
            logger.info(f"Loading existing model: {model_path}")
            try:
                model = PPO.load(model_path)
                logger.info("✓ Model loaded, will continue training\n")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
                logger.warning("Will create new model\n")

        # Variables for tracking
        total_steps = 0
        problems_trained = 0

        # Iterate through problems
        for domain_file, problem_file, problem_name, epoch, problem_step in \
                create_problem_iterator(benchmark_sequence, epochs=1):

            print_subsection(f"Problem {problem_step}/{len(benchmark_sequence)}: {problem_name}")

            logger.info(f"Epoch {epoch}, Step {problem_step}")
            logger.info(f"Training timesteps for this problem: {timesteps_per_problem}\n")

            # ✅ FIX: Pass reward_kwargs to the init function
            result = init_training_environment(
                domain_file,
                problem_file,
                reward_variant,
                **reward_kwargs
            )

            if result is None:
                logger.error(f"Failed to initialize environment for {problem_name}")
                logger.error("Skipping this problem\n")
                continue

            env, domain_path, problem_path = result
            env = Monitor(env)

            # Create model if not loaded
            if model is None:
                logger.info("Creating new PPO model with GNN policy...")
                model = PPO(
                    policy=GNNPolicy,
                    env=env,
                    learning_rate=0.0003,
                    n_steps=64,
                    batch_size=32,
                    ent_coef=0.01,
                    verbose=0,
                    tensorboard_log="tb_logs/",
                    policy_kwargs={"hidden_dim": 64},
                )
                logger.info("✓ New model created\n")
            else:
                # Update model's environment
                model.set_env(env)

            # Train on this problem
            logger.info(f"Training for {timesteps_per_problem} timesteps...")
            try:
                model.learn(
                    total_timesteps=timesteps_per_problem,
                    tb_log_name=f"multi_problem_training_p{problem_step}",
                    reset_num_timesteps=False,
                )
                logger.info("✓ Training completed for this problem\n")

                total_steps += timesteps_per_problem
                problems_trained += 1

            except KeyboardInterrupt:
                logger.warning("\n⚠️ Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"✗ Training failed for {problem_name}: {e}")
                logger.error(traceback.format_exc())
                continue
            finally:
                env.close()

            # Check if we've reached total timesteps limit
            if total_timesteps > 0 and total_steps >= total_timesteps:
                logger.info(f"\n✓ Reached total timesteps limit: {total_steps}/{total_timesteps}")
                break

        # Save final model
        if model is not None:
            model_filename = f"gnn_model_multi_problem_{reward_variant}.zip"
            model_path_out = f"mvp_output/{model_filename}"
            model.save(model_path_out)
            logger.info(f"\n✓ Final model saved to: {model_path_out}")
            logger.info(f"  Problems trained on: {problems_trained}")
            logger.info(f"  Total timesteps: {total_steps}")

        return True

    except Exception as e:
        logger.error(f"\n❌ Multi-problem training failed: {e}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# PHASE 3: VALIDATION (OPTIONAL)
# ============================================================================

def validate_on_benchmark_sample(model, benchmarks_dict: Dict, n_problems: int = 3) -> bool:
    """
    Quick validation: test trained model on a sample of problems.

    Args:
        model: Trained PPO model
        benchmarks_dict: All loaded benchmarks
        n_problems: Number of problems to test on

    Returns:
        True if validation succeeded
    """
    print_section("PHASE 3: VALIDATION ON SAMPLE")

    try:
        from merge_env import MergeEnv

        if model is None:
            logger.warning("No model to validate")
            return False

        # Sample problems from each difficulty
        test_problems = []
        for difficulty in ["small", "medium", "hard"]:
            if difficulty in benchmarks_dict and benchmarks_dict[difficulty]:
                # Take first problem from each difficulty
                test_problems.append(benchmarks_dict[difficulty][0])

        logger.info(f"Testing on {len(test_problems)} sampled problems:\n")

        total_reward = 0.0
        episodes_completed = 0

        for i, (domain_file, problem_file) in enumerate(test_problems, 1):
            problem_name = os.path.basename(problem_file)
            logger.info(f"  [{i}] {problem_name}")

            try:
                env = MergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=10,
                    debug=False,
                    reward_variant="astar_search",
                )

                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0

                while steps < 10:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    episode_reward += reward
                    steps += 1

                    if done or truncated:
                        break

                total_reward += episode_reward
                episodes_completed += 1

                logger.info(f"      ✓ Reward: {episode_reward:+.4f} ({steps} steps)")
                env.close()

            except Exception as e:
                logger.warning(f"      ⚠️ Test failed: {e}")

        if episodes_completed > 0:
            avg_reward = total_reward / episodes_completed
            logger.info(f"\n✅ Average validation reward: {avg_reward:+.4f} ({episodes_completed} episodes)")
            return True
        else:
            logger.warning("No validation episodes completed")
            return False

    except Exception as e:
        logger.error(f"\n❌ Validation failed: {e}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    print_section("MULTI-PROBLEM REAL TRAINING - GNN MERGE STRATEGY", "=", 95)

    # ✅ NEW: Get environment variables for multi-problem training
    reward_variant = os.environ.get('REWARD_VARIANT', 'astar_search')
    difficulty = os.environ.get('DIFFICULTY', 'small').lower()
    max_problems = int(os.environ.get('MAX_PROBLEMS_PER_DIFFICULTY', '5'))
    curriculum_learning = os.environ.get('CURRICULUM_LEARNING', 'false').lower() == 'true'

    logger.info(f"Configuration:")
    logger.info(f"  Reward variant: {reward_variant}")
    logger.info(f"  Difficulty: {difficulty}")
    logger.info(f"  Max problems per difficulty: {max_problems}")
    logger.info(f"  Curriculum learning: {curriculum_learning}\n")

    # Phase 0: Load benchmarks from folder structure
    all_benchmarks = load_benchmarks_from_folders()
    if not all_benchmarks or sum(len(b) for b in all_benchmarks.values()) == 0:
        logger.error("No benchmarks loaded - aborting")
        return 1

    # Create benchmark sequence
    benchmark_sequence = get_benchmark_sequence(
        all_benchmarks,
        difficulty=difficulty,
        max_problems_per_difficulty=max_problems,
        curriculum_learning=curriculum_learning
    )

    if not benchmark_sequence:
        logger.error("No benchmarks in sequence - aborting")
        return 1

    # Phase 2: Run multi-problem training
    success = run_multi_problem_training(
        benchmark_sequence=benchmark_sequence,
        reward_variant=reward_variant,
        total_timesteps=10000,  # ✅ NEW: Can be adjusted
        timesteps_per_problem=500,  # ✅ NEW: Timesteps per problem
    )

    if not success:
        logger.error("Training failed")
        return 1

    # Summary
    print_section("TRAINING COMPLETE", "=", 95)
    logger.info("✅ Multi-problem training pipeline completed successfully!\n")
    logger.info("Next steps:")
    logger.info("  1. View TensorBoard logs:")
    logger.info(f"     tensorboard --logdir={os.path.abspath('tb_logs/')}\n")
    logger.info("  2. Review training log:")
    logger.info(f"     {os.path.abspath('training_multi_problem.log')}\n")
    logger.info("  3. Trained models:")
    logger.info(f"     {os.path.abspath('mvp_output/')}\n")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
