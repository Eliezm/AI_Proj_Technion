#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHARED EXPERIMENT UTILITIES - ENHANCED WITH EVALUATION INTEGRATION
===================================================================
Common code used across all 4 experiments + evaluation framework integration.

Provides:
  - Checkpoint/resume functionality
  - Standardized logging
  - Common training/evaluation patterns
  - Result persistence
  - ✅ NEW: Experiment-to-evaluation conversion
  - ✅ NEW: Multi-experiment result aggregation
"""

import sys
import os
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

sys.path.insert(0, os.getcwd())
os.makedirs("downward/gnn_output", exist_ok=True)
os.makedirs("downward/fd_output", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ============================================================================
# LOGGING & FORMATTING
# ============================================================================

def setup_logging(experiment_name: str, output_dir: str) -> logging.Logger:
    """Setup consistent logging for experiment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_dir, f"{experiment_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
        force=True
    )

    logger = logging.getLogger(experiment_name)
    return logger


def print_section(title: str, logger: logging.Logger, width: int = 90):
    """Print formatted section header."""
    logger.info("\n" + "=" * width)
    logger.info(f"// {title.upper()}")
    logger.info("=" * width + "\n")


def print_subsection(title: str, logger: logging.Logger):
    """Print formatted subsection header."""
    logger.info("\n" + "-" * 80)
    logger.info(f">>> {title}")
    logger.info("-" * 80 + "\n")


# ============================================================================
# CHECKPOINT & RESUME
# ============================================================================

class ExperimentCheckpoint:
    """Simple checkpoint manager for robust experiment recovery."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "experiment_checkpoint.json"

    def save(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        state['checkpoint_time'] = datetime.now().isoformat()

        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self) -> None:
        """Clear checkpoint (call after successful completion)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_gnn_model(
        benchmarks: List[Tuple[str, str]],
        reward_variant: str = "astar_search",
        total_timesteps: int = 5000,
        timesteps_per_problem: int = 500,
        model_output_path: str = "mvp_output/gnn_model.zip",
        logger: Optional[logging.Logger] = None,
        tb_log_name: str = "experiment_training",
) -> Optional[str]:
    """
    Train GNN model on benchmarks.

    Returns:
        Path to trained model, or None if failed
    """
    if logger is None:
        logger = logging.getLogger("train_gnn")

    try:
        from stable_baselines3 import PPO
        from gnn_policy import GNNPolicy
        from merge_env import MergeEnv
        from stable_baselines3.common.monitor import Monitor

        logger.info(f"Training on {len(benchmarks)} problem(s)")
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Timesteps per problem: {timesteps_per_problem}\n")

        os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)

        reward_kwargs = {
            'w_search_efficiency': 0.30,
            'w_solution_quality': 0.20,
            'w_f_stability': 0.35,
            'w_state_control': 0.15,
        }

        model = None
        total_steps = 0

        for step, (domain_file, problem_file) in enumerate(benchmarks):
            problem_name = os.path.basename(problem_file)
            logger.info(f"  [{step + 1}/{len(benchmarks)}] {problem_name}")

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

                if model is None:
                    logger.info("    Creating new PPO model...")
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
                else:
                    model.set_env(env)

                logger.info(f"    Training for {timesteps_per_problem} timesteps...")
                model.learn(
                    total_timesteps=timesteps_per_problem,
                    tb_log_name=f"{tb_log_name}_{step}",
                    reset_num_timesteps=False,
                )

                total_steps += timesteps_per_problem
                logger.info(f"    ✓ Training complete (total: {total_steps} steps)")

                env.close()

                if total_steps >= total_timesteps:
                    break

            except Exception as e:
                logger.error(f"    ⚠️ Problem failed: {e}")
                continue

        if model is None:
            logger.error("Training failed - no model created")
            return None

        model.save(model_output_path)
        logger.info(f"\n✅ Model saved: {model_output_path}")

        return model_output_path

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return None


def evaluate_model_on_problems(
        model_path: str,
        benchmarks: List[Tuple[str, str]],
        reward_variant: str = "astar_search",
        logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation of trained model on test problems.

    Returns:
        Dict with evaluation results
    """
    if logger is None:
        logger = logging.getLogger("evaluate")

    try:
        from stable_baselines3 import PPO
        from merge_env import MergeEnv

        model = PPO.load(model_path)
        logger.info(f"Model loaded: {model_path}")

        reward_kwargs = {
            'w_search_efficiency': 0.30,
            'w_solution_quality': 0.20,
            'w_f_stability': 0.35,
            'w_state_control': 0.15,
        }

        results = {
            'total_problems': len(benchmarks),
            'solved_count': 0,
            'avg_reward': 0.0,
            'avg_time': 0.0,
            'details': []
        }

        rewards = []
        times = []

        for i, (domain_file, problem_file) in enumerate(benchmarks):
            problem_name = os.path.basename(problem_file)
            logger.info(f"  [{i + 1}/{len(benchmarks)}] {problem_name}")

            try:
                env = MergeEnv(
                    domain_file=os.path.abspath(domain_file),
                    problem_file=os.path.abspath(problem_file),
                    max_merges=50,
                    debug=False,
                    reward_variant=reward_variant,
                    **reward_kwargs
                )

                start_time = time.time()
                obs, _ = env.reset()
                episode_reward = 0.0
                steps = 0

                for step in range(20):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    episode_reward += reward
                    steps += 1

                    if done or truncated:
                        break

                elapsed = time.time() - start_time

                rewards.append(episode_reward)
                times.append(elapsed)
                results['solved_count'] += 1

                logger.info(f"    ✓ Reward: {episode_reward:.4f}, Time: {elapsed:.2f}s")

                results['details'].append({
                    'problem': problem_name,
                    'reward': float(episode_reward),
                    'time': float(elapsed),
                    'steps': steps,
                    'solved': True
                })

                env.close()

            except Exception as e:
                logger.warning(f"    ⚠️ Failed: {e}")
                results['details'].append({
                    'problem': problem_name,
                    'solved': False,
                    'error': str(e)[:100]
                })

        if rewards:
            results['avg_reward'] = float(np.mean(rewards))
            results['avg_time'] = float(np.mean(times))
            results['solve_rate'] = (results['solved_count'] / len(benchmarks)) * 100

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {}


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def load_and_split_problems(
        domain_file: str,
        problem_pattern: str,
        train_ratio: float = 0.8,
        random_seed: int = 42,
        logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str]]:
    """Load problems and split into train/test."""
    if logger is None:
        logger = logging.getLogger("dataset")

    import glob
    import random

    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain not found: {domain_file}")

    all_problems = sorted(glob.glob(problem_pattern))

    if not all_problems:
        raise ValueError(f"No problems found matching: {problem_pattern}")

    if len(all_problems) < 2:
        raise ValueError("Need at least 2 problems for train/test split")

    random.seed(random_seed)
    problems_shuffled = all_problems.copy()
    random.shuffle(problems_shuffled)

    split_idx = int(len(problems_shuffled) * train_ratio)
    train_problems = sorted(problems_shuffled[:split_idx])
    test_problems = sorted(problems_shuffled[split_idx:])

    logger.info(f"Loaded {len(all_problems)} problems")
    logger.info(f"Train/test split: {len(train_problems)} train, {len(test_problems)} test")

    return train_problems, test_problems


def load_benchmarks_by_difficulty(
        logger: Optional[logging.Logger] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """Load benchmarks organized by difficulty."""
    if logger is None:
        logger = logging.getLogger("dataset")

    import glob

    benchmarks_dir = os.path.abspath("benchmarks")

    if not os.path.isdir(benchmarks_dir):
        logger.warning(f"Benchmarks dir not found: {benchmarks_dir}")
        return {}

    all_benchmarks = {}

    for difficulty in ["small", "medium", "large"]:
        difficulty_dir = os.path.join(benchmarks_dir, difficulty)

        domain_file = os.path.join(difficulty_dir, "domain.pddl")
        if not os.path.exists(domain_file):
            logger.warning(f"Domain not found for {difficulty}")
            all_benchmarks[difficulty] = []
            continue

        problems = sorted(glob.glob(os.path.join(difficulty_dir, "problem_*.pddl")))

        if not problems:
            logger.warning(f"No problems found for {difficulty}")
            all_benchmarks[difficulty] = []
            continue

        benchmarks_list = [
            (os.path.abspath(domain_file), os.path.abspath(prob))
            for prob in problems
        ]

        all_benchmarks[difficulty] = benchmarks_list
        logger.info(f"  {difficulty:<10}: {len(benchmarks_list)} problems")

    return all_benchmarks


# ============================================================================
# RESULT PERSISTENCE
# ============================================================================

def save_results_to_json(
        results: Dict[str, Any],
        output_path: str,
        logger: Optional[logging.Logger] = None
) -> None:
    """Save results to JSON file."""
    if logger is None:
        logger = logging.getLogger("results")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✓ Results saved: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def save_results_to_txt(
        results: Dict[str, Any],
        output_path: str,
        experiment_name: str,
        logger: Optional[logging.Logger] = None
) -> None:
    """Save human-readable results to text file."""
    if logger is None:
        logger = logging.getLogger("results")

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write(f"{experiment_name.upper()} - RESULTS SUMMARY\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            for key, value in results.items():
                if key.startswith('_'):
                    continue

                if isinstance(value, dict):
                    f.write(f"\n{key.upper()}:\n")
                    f.write("-" * 50 + "\n")
                    for k, v in value.items():
                        f.write(f"  {k:<40} {v}\n")
                elif isinstance(value, list):
                    f.write(f"\n{key.upper()}:\n")
                    f.write("-" * 50 + "\n")
                    for i, item in enumerate(value[:5], 1):
                        f.write(f"  [{i}] {item}\n")
                    if len(value) > 5:
                        f.write(f"  ... and {len(value) - 5} more\n")
                else:
                    f.write(f"{key:<40} {value}\n")

            f.write("\n" + "=" * 90 + "\n")

        logger.info(f"✓ Results saved: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


# ============================================================================
# ✅ NEW: EXPERIMENT-TO-EVALUATION INTEGRATION
# ============================================================================

@dataclass
class ExperimentResultsMetrics:
    """Unified metrics format for evaluation."""
    experiment_type: str
    problem_name: str
    difficulty: str
    solved: bool
    solve_time: float
    reward: float
    merge_episodes: int
    num_problems_total: int
    num_problems_solved: int
    avg_reward: float
    avg_time: float
    solve_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment_results_from_directory(
        experiment_dir: str,
        experiment_type: str,
        logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Load results from a single experiment directory.

    Expected structure:
        experiment_dir/
        ├── results.json      # Contains evaluation data
        └── results.txt       # Human-readable summary
    """
    if logger is None:
        logger = logging.getLogger("experiment_loader")

    results_file = os.path.join(experiment_dir, "results.json")

    if not os.path.exists(results_file):
        logger.warning(f"Results file not found: {results_file}")
        return []

    try:
        with open(results_file, 'r') as f:
            experiment_results = json.load(f)

        logger.info(f"Loaded experiment results from {experiment_dir}")

        parsed_results = []

        if 'evaluation' in experiment_results and 'details' in experiment_results['evaluation']:
            for detail in experiment_results['evaluation']['details']:
                parsed_results.append({
                    'experiment_type': experiment_type,
                    'problem_name': detail.get('problem', ''),
                    'solved': detail.get('solved', False),
                    'time': detail.get('time', 0),
                    'reward': detail.get('reward', 0),
                    'difficulty': _extract_difficulty_from_problem_name(detail.get('problem', '')),
                })

        elif 'details' in experiment_results:
            for detail in experiment_results['details']:
                parsed_results.append({
                    'experiment_type': experiment_type,
                    'problem_name': detail.get('problem', ''),
                    'solved': detail.get('solved', False),
                    'time': detail.get('time', 0),
                    'reward': detail.get('reward', 0),
                    'difficulty': _extract_difficulty_from_problem_name(detail.get('problem', '')),
                })

        logger.info(f"Parsed {len(parsed_results)} problem results")

        return parsed_results

    except Exception as e:
        logger.error(f"Failed to load experiment results: {e}")
        logger.error(traceback.format_exc())
        return []


def aggregate_experiment_results(
        experiment_directories: Dict[str, str],
        logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple experiments.

    Args:
        experiment_directories: Dict mapping experiment_type -> directory_path
                               E.g., {
                                   'overfit': 'overfit_experiment_results',
                                   'problem_gen': 'problem_generalization_results',
                                   'scale_gen': 'scale_generalization_results',
                                   'curriculum': 'curriculum_learning_results'
                               }

    Returns:
        Aggregated results dictionary with statistics
    """
    if logger is None:
        logger = logging.getLogger("experiment_aggregator")

    logger.info("Aggregating experiment results...")

    all_results = []
    experiment_summaries = {}

    for exp_type, exp_dir in experiment_directories.items():
        if not os.path.exists(exp_dir):
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        logger.info(f"\nLoading {exp_type} from {exp_dir}...")

        results = load_experiment_results_from_directory(
            exp_dir,
            exp_type,
            logger
        )

        if results:
            all_results.extend(results)

            solved_count = sum(1 for r in results if r['solved'])
            total_count = len(results)
            avg_time = np.mean([r['time'] for r in results if r['solved']])
            avg_reward = np.mean([r['reward'] for r in results])

            experiment_summaries[exp_type] = {
                'total_problems': total_count,
                'solved': solved_count,
                'solve_rate': (solved_count / total_count * 100) if total_count > 0 else 0,
                'avg_time': float(avg_time) if not np.isnan(avg_time) else 0,
                'avg_reward': float(avg_reward) if not np.isnan(avg_reward) else 0,
            }

            logger.info(f"  ✓ {exp_type}: {solved_count}/{total_count} solved "
                        f"({experiment_summaries[exp_type]['solve_rate']:.1f}%)")

    logger.info(f"\n✅ Total results aggregated: {len(all_results)} problems")

    return {
        'all_results': all_results,
        'experiment_summaries': experiment_summaries,
        'timestamp': datetime.now().isoformat()
    }


def _extract_difficulty_from_problem_name(problem_name: str) -> str:
    """Extract difficulty level from problem name."""
    if 'small' in problem_name.lower():
        return 'small'
    elif 'medium' in problem_name.lower():
        return 'medium'
    elif 'large' in problem_name.lower() or 'hard' in problem_name.lower():
        return 'large'
    else:
        return 'unknown'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    for directory in [
        "downward/gnn_output",
        "downward/fd_output",
        "logs",
        "mvp_output",
        "benchmarks",
        "tb_logs"
    ]:
        os.makedirs(directory, exist_ok=True)


def get_timestamp_str() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"