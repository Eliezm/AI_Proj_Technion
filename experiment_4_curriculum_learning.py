#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRICULUM LEARNING TRAINING - Progressive Difficulty
=======================================================
Trains a GNN policy using curriculum learning: small → medium → large problems.

Features:
  ✓ Progressive difficulty training (curriculum learning)
  ✓ Continuous monitoring and detailed logging
  ✓ Save checkpoints after each difficulty level
  ✓ Validation on large problems after training
  ✓ Performance tracking across difficulties
  ✓ TensorBoard visualization

Run with:
    python train_curriculum.py

Environment Variables:
    REWARD_VARIANT: Which reward function to use (default: astar_search)
    TOTAL_TIMESTEPS: Total training timesteps (default: 50000)
    SMALL_TIMESTEPS: Timesteps per small problem (default: 500)
    MEDIUM_TIMESTEPS: Timesteps per medium problem (default: 750)
    LARGE_TIMESTEPS: Timesteps per large problem (default: 1000)
    NUM_PROBLEMS_PER_DIFFICULTY: Problems per difficulty (default: 10)
"""

import sys
import os
import logging
import glob
import json
import traceback
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

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
        logging.FileHandler("training_curriculum.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class CurriculumConfig:
    """Configuration for curriculum learning."""

    REWARD_VARIANT = os.environ.get('REWARD_VARIANT', 'astar_search')

    SMALL_TIMESTEPS = int(os.environ.get('SMALL_TIMESTEPS', '500'))
    MEDIUM_TIMESTEPS = int(os.environ.get('MEDIUM_TIMESTEPS', '750'))
    LARGE_TIMESTEPS = int(os.environ.get('LARGE_TIMESTEPS', '1000'))
    TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', '50000'))

    NUM_PROBLEMS_PER_DIFFICULTY = int(os.environ.get('NUM_PROBLEMS_PER_DIFFICULTY', '10'))

    REWARD_KWARGS = {
        'w_search_efficiency': 0.30,
        'w_solution_quality': 0.20,
        'w_f_stability': 0.35,
        'w_state_control': 0.15,
    }


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
# PHASE 0: BENCHMARK LOADING
# ============================================================================

def load_benchmarks_by_difficulty() -> Dict[str, List[Tuple[str, str]]]:
    """Load benchmarks organized by difficulty (small/medium/large)."""
    print_section("PHASE 0: LOAD BENCHMARKS BY DIFFICULTY")

    benchmarks_dir = os.path.abspath("benchmarks")

    if not os.path.isdir(benchmarks_dir):
        logger.error(f"Benchmarks directory not found: {benchmarks_dir}")
        logger.error("Expected structure:")
        logger.error("  benchmarks/small/   (with domain.pddl and problem_*.pddl)")
        logger.error("  benchmarks/medium/  (with domain.pddl and problem_*.pddl)")
        logger.error("  benchmarks/large/   (with domain.pddl and problem_*.pddl)")
        return {}

    logger.info(f"Benchmarks directory: {benchmarks_dir}\n")

    difficulties = ["small", "medium", "large"]
    all_benchmarks = {}

    for difficulty in difficulties:
        difficulty_dir = os.path.join(benchmarks_dir, difficulty)
        logger.info(f"Loading {difficulty.upper()} difficulty problems...")

        if not os.path.isdir(difficulty_dir):
            logger.warning(f"  ⚠️ Directory not found: {difficulty_dir}")
            all_benchmarks[difficulty] = []
            continue

        # Find domain
        domain_file = os.path.join(difficulty_dir, "domain.pddl")
        if not os.path.exists(domain_file):
            logger.warning(f"  ⚠️ Domain file not found: {domain_file}")
            all_benchmarks[difficulty] = []
            continue

        logger.info(f"  ✓ Domain: {domain_file}")

        # Find problems
        problems = sorted(glob.glob(os.path.join(difficulty_dir, "problem_*.pddl")))

        if not problems:
            logger.warning(f"  ⚠️ No problem files found")
            all_benchmarks[difficulty] = []
            continue

        logger.info(f"  ✓ Found {len(problems)} problem(s)")
        for i, prob in enumerate(problems[:3], 1):
            logger.info(f"    {i}. {os.path.basename(prob)}")
        if len(problems) > 3:
            logger.info(f"    ... and {len(problems) - 3} more")

        # Create benchmark list
        benchmarks_list = [
            (os.path.abspath(domain_file), os.path.abspath(prob))
            for prob in problems
        ]

        all_benchmarks[difficulty] = benchmarks_list
        logger.info(f"  ✓ Loaded {len(benchmarks_list)} benchmark(s)\n")

    # Summary
    total = sum(len(b) for b in all_benchmarks.values())
    if total == 0:
        logger.error("❌ No benchmarks loaded!")
        return {}

    logger.info(f"✅ Loaded {total} total benchmarks:")
    for difficulty in difficulties:
        count = len(all_benchmarks[difficulty])
        logger.info(f"   {difficulty:<10} {count:>3} problems")

    return all_benchmarks


def create_curriculum_sequence(
        all_benchmarks: Dict[str, List[Tuple[str, str]]],
        num_per_difficulty: int = 10
) -> List[Tuple[str, str, str]]:
    """Create curriculum sequence: small → medium → large."""
    print_subsection("Create Curriculum Sequence")

    sequence = []
    difficulties = ["small", "medium", "large"]

    for difficulty in difficulties:
        if difficulty not in all_benchmarks or not all_benchmarks[difficulty]:
            logger.warning(f"No problems for difficulty: {difficulty}")
            continue

        problems_this_difficulty = all_benchmarks[difficulty]
        sampled = random.sample(
            problems_this_difficulty,
            min(num_per_difficulty, len(problems_this_difficulty))
        )

        for domain_file, problem_file in sampled:
            sequence.append((domain_file, problem_file, difficulty))

        logger.info(f"✓ Added {len(sampled)} {difficulty} problems (total: {len(sequence)})")

    logger.info(f"\n✅ Curriculum sequence ready:")
    logger.info(f"   Small:  {len([s for s in sequence if s[2] == 'small'])} problems")
    logger.info(f"   Medium: {len([s for s in sequence if s[2] == 'medium'])} problems")
    logger.info(f"   Large:  {len([s for s in sequence if s[2] == 'large'])} problems")

    return sequence


# ============================================================================
# PHASE 1: CURRICULUM TRAINING
# ============================================================================

def run_curriculum_training(
        curriculum_sequence: List[Tuple[str, str, str]],
        reward_variant: str = 'astar_search',
        **reward_kwargs
) -> Optional[str]:
    """
    Run curriculum learning training.

    Returns:
        Path to final model, or None if training failed
    """
    print_section("PHASE 1: CURRICULUM LEARNING TRAINING")

    try:
        from stable_baselines3 import PPO
        from merge_env import MergeEnv
        from gnn_policy import GNNPolicy
        from stable_baselines3.common.monitor import Monitor

        timesteps_config = {
            'small': CurriculumConfig.SMALL_TIMESTEPS,
            'medium': CurriculumConfig.MEDIUM_TIMESTEPS,
            'large': CurriculumConfig.LARGE_TIMESTEPS,
        }

        # Try to load existing model
        model_path = "mvp_output/gnn_model_curriculum_start.zip"
        model = None

        if os.path.exists(model_path):
            logger.info(f"Loading existing model: {model_path}")
            try:
                model = PPO.load(model_path)
                logger.info("✓ Model loaded, continuing training\n")
            except Exception as e:
                logger.warning(f"Could not load: {e}")

        total_steps = 0
        problems_trained = 0
        difficulty_counts = {'small': 0, 'medium': 0, 'large': 0}

        # Iterate through curriculum
        for step, (domain_file, problem_file, difficulty) in enumerate(curriculum_sequence, 1):
            problem_name = os.path.basename(problem_file)
            timesteps_this_problem = timesteps_config.get(difficulty, 500)

            print_subsection(f"Problem {step}/{len(curriculum_sequence)}: [{difficulty.upper()}] {problem_name}")

            logger.info(f"Difficulty: {difficulty}")
            logger.info(f"Timesteps: {timesteps_this_problem}\n")

            # Create environment
            try:
                env = MergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=50,
                    debug=False,
                    reward_variant=reward_variant,
                    **reward_kwargs
                )
                env = Monitor(env)
                logger.info("✓ Environment created\n")
            except Exception as e:
                logger.error(f"Failed to create environment: {e}")
                logger.error("Skipping this problem\n")
                continue

            # Create model if needed
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
                model.set_env(env)

            # Train
            logger.info(f"Training for {timesteps_this_problem} timesteps...")
            try:
                model.learn(
                    total_timesteps=timesteps_this_problem,
                    tb_log_name=f"curriculum_{difficulty}_{step}",
                    reset_num_timesteps=False,
                )
                logger.info("✓ Training completed\n")

                total_steps += timesteps_this_problem
                problems_trained += 1
                difficulty_counts[difficulty] += 1

            except KeyboardInterrupt:
                logger.warning("⚠️ Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Training failed: {e}")
                logger.error(traceback.format_exc())
                continue
            finally:
                try:
                    env.close()
                except:
                    pass

            # Save checkpoint
            checkpoint_path = f"mvp_output/gnn_model_curriculum_{difficulty}_{step}.zip"
            model.save(checkpoint_path)
            logger.info(f"✓ Checkpoint saved: {checkpoint_path}\n")

            # Check total timesteps limit
            if total_steps >= CurriculumConfig.TOTAL_TIMESTEPS:
                logger.info(f"✓ Reached total timesteps: {total_steps}/{CurriculumConfig.TOTAL_TIMESTEPS}")
                break

        # Save final model
        if model is not None:
            final_model_path = "mvp_output/gnn_model_curriculum_final.zip"
            model.save(final_model_path)
            logger.info(f"\n✅ Final model saved: {final_model_path}")
            logger.info(f"  Total problems trained: {problems_trained}")
            logger.info(f"  Total timesteps: {total_steps}")
            logger.info(
                f"  Small: {difficulty_counts['small']}, Medium: {difficulty_counts['medium']}, Large: {difficulty_counts['large']}")
            return final_model_path

        return None

    except Exception as e:
        logger.error(f"❌ Curriculum training failed: {e}")
        logger.error(traceback.format_exc())
        return None


# ============================================================================
# PHASE 2: VALIDATION ON LARGE PROBLEMS
# ============================================================================

def validate_on_large_problems(
        model,
        all_benchmarks: Dict[str, List[Tuple[str, str]]],
        num_problems: int = 5
) -> Dict:
    """Validate trained model on large problems."""
    print_section("PHASE 2: VALIDATION ON LARGE PROBLEMS")

    if model is None:
        logger.warning("No model to validate")
        return {}

    if 'large' not in all_benchmarks or not all_benchmarks['large']:
        logger.warning("No large problems available")
        return {}

    try:
        from merge_env import MergeEnv

        large_problems = all_benchmarks['large']
        test_problems = random.sample(
            large_problems,
            min(num_problems, len(large_problems))
        )

        logger.info(f"Testing on {len(test_problems)} large problems:\n")

        total_reward = 0.0
        episodes_completed = 0
        episode_rewards = []

        for i, (domain_file, problem_file) in enumerate(test_problems, 1):
            problem_name = os.path.basename(problem_file)
            logger.info(f"  [{i}] {problem_name}")

            try:
                env = MergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=50,
                    debug=False,
                    reward_variant=CurriculumConfig.REWARD_VARIANT,
                    **CurriculumConfig.REWARD_KWARGS
                )

                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0

                while steps < 20:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    episode_reward += reward
                    steps += 1
                    if done or truncated:
                        break

                total_reward += episode_reward
                episodes_completed += 1
                episode_rewards.append(episode_reward)

                logger.info(f"      ✓ Reward: {episode_reward:+.4f} ({steps} steps)")
                env.close()

            except Exception as e:
                logger.warning(f"      ⚠️ Failed: {e}")

        if episodes_completed > 0:
            avg_reward = total_reward / episodes_completed
            logger.info(f"\n✅ Validation Results:")
            logger.info(f"  Episodes: {episodes_completed}/{len(test_problems)}")
            logger.info(f"  Avg reward: {avg_reward:+.4f}")
            logger.info(f"  Max reward: {max(episode_rewards):+.4f}")
            logger.info(f"  Min reward: {min(episode_rewards):+.4f}")

            return {
                'avg_reward': avg_reward,
                'max_reward': max(episode_rewards),
                'min_reward': min(episode_rewards),
                'episodes': episodes_completed,
            }

        return {}

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print_section("CURRICULUM LEARNING TRAINING", "=", 95)

    logger.info(f"Configuration:")
    logger.info(f"  Reward variant: {CurriculumConfig.REWARD_VARIANT}")
    logger.info(f"  Total timesteps: {CurriculumConfig.TOTAL_TIMESTEPS}")
    logger.info(f"  Small: {CurriculumConfig.SMALL_TIMESTEPS} ts/problem")
    logger.info(f"  Medium: {CurriculumConfig.MEDIUM_TIMESTEPS} ts/problem")
    logger.info(f"  Large: {CurriculumConfig.LARGE_TIMESTEPS} ts/problem")
    logger.info(f"  Problems per difficulty: {CurriculumConfig.NUM_PROBLEMS_PER_DIFFICULTY}\n")

    # Phase 0: Load
    all_benchmarks = load_benchmarks_by_difficulty()
    if not all_benchmarks or sum(len(b) for b in all_benchmarks.values()) == 0:
        logger.error("No benchmarks loaded")
        return 1

    # Create curriculum
    curriculum_sequence = create_curriculum_sequence(
        all_benchmarks,
        num_per_difficulty=CurriculumConfig.NUM_PROBLEMS_PER_DIFFICULTY
    )

    if not curriculum_sequence:
        logger.error("No curriculum sequence created")
        return 1

    # Phase 1: Train
    model_path = run_curriculum_training(
        curriculum_sequence,
        reward_variant=CurriculumConfig.REWARD_VARIANT,
        **CurriculumConfig.REWARD_KWARGS
    )

    if not model_path:
        logger.error("Training failed")
        return 1

    # Phase 2: Validate
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        validate_on_large_problems(model, all_benchmarks, num_problems=5)
    except Exception as e:
        logger.warning(f"Validation skipped: {e}")

    # Summary
    print_section("TRAINING COMPLETE", "=", 95)
    logger.info("✅ Curriculum learning pipeline completed!\n")
    logger.info("Next steps:")
    logger.info(f"  1. TensorBoard: tensorboard --logdir={os.path.abspath('tb_logs/')}")
    logger.info(f"  2. Model: {model_path}")
    logger.info(f"  3. Log: {os.path.abspath('training_curriculum.log')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())