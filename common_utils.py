# -*- coding: utf-8 -*-
"""
COMPREHENSIVE CENTRAL UTILITIES FILE
Single source of truth for ALL shared code across the project.
"""

import os
import glob
import random
import logging
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ============================================================================
# 1. FAST DOWNWARD COMMAND TEMPLATE (CENTRALIZED)
# ============================================================================

# DOWNWARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "downward"))
#
# # Windows version
# FD_COMMAND_TEMPLATE = (
#     f'python "{DOWNWARD_DIR}\\builds\\release\\bin\\translate\\translate.py" '
#     r'"{domain}" "{problem}" --sas-file output.sas && '
#     f'"{DOWNWARD_DIR}\\builds\\release\\bin\\downward.exe" '
#     r'--search "astar(merge_and_shrink('
#     r'merge_strategy=merge_gnn(),'
#     r'shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),'
#     r'label_reduction=exact(before_shrinking=true,before_merging=false),'
#     r'max_states=4000,threshold_before_merge=1'
#     r'))"'
# )

# FILE: common_utils.py (UPDATE THIS)

import os
import sys

# Get absolute path to downward folder
DOWNWARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "downward"))

# ✅ FIXED: Run translate and downward from the downward/ directory
# The key is to:
# 1. Use absolute paths for domain/problem (so they're found from downward/ cwd)
# 2. Run translate FIRST, verify output.sas exists
# 3. Then run downward to read that file
FD_COMMAND_TEMPLATE = (
    f'python "{DOWNWARD_DIR}\\builds\\release\\bin\\translate\\translate.py" '
    r'"{domain}" "{problem}" --sas-file output.sas && '
    f'"{DOWNWARD_DIR}\\builds\\release\\bin\\downward.exe" '
    r'--search "astar(merge_and_shrink('
    r'merge_strategy=merge_gnn(),'
    r'shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),'
    r'label_reduction=exact(before_shrinking=true,before_merging=false),'
    r'max_states=4000,threshold_before_merge=1'
    r'))" < output.sas'
)


# ============================================================================
# 2. SIMPLE SINGLE-PROBLEM ENVIRONMENT (NO MULTIENV)
# ============================================================================

# ============================================================================
# 2. SIMPLE SINGLE-PROBLEM ENVIRONMENT (FIXED)
# ============================================================================

class SimpleTrainingEnv(gym.Env):
    def __init__(self, domain_file: str, problem_file: str,
                 reward_variant: str = 'rich', debug: bool = False,
                 **reward_kwargs):
        super().__init__()

        from merge_env import MergeEnv

        # ✅ FIXED: Store environment as self.merge_env
        self.merge_env = MergeEnv(
            domain_file=domain_file,
            problem_file=problem_file,
            max_merges=50,
            debug=debug,
            reward_variant=reward_variant,
            **reward_kwargs
        )

        # ✅ CRITICAL: These MUST be set before training
        self.observation_space = self.merge_env.observation_space
        self.action_space = self.merge_env.action_space

        # ✅ ADD: Store reference to metadata
        self.metadata = {"render_modes": []}

    def reset(self, **kwargs):
        return self.merge_env.reset(**kwargs)
    def step(self, action):
        return self.merge_env.step(action)

    def close(self):
        try:
            self.merge_env.close()
        except:
            pass

    def render(self, mode='human'):
        pass


# ============================================================================
# 3. CALLBACKS (UNIFIED)
# ============================================================================

class SimpleProgressCallback(BaseCallback):
    """✅ FIXED: Simple progress tracking without issues."""

    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = total_steps
        self.pbar = None

    def _on_training_start(self):
        """Initialize progress bar."""
        self.pbar = tqdm(total=self.total_steps, desc="Training", unit="steps")

    def _on_step(self) -> bool:
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()


# ============================================================================
# 4. CHECKPOINT UTILITIES
# ============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the latest checkpoint by step count."""
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_steps.zip"))
    if not checkpoints:
        return None

    try:
        latest = max(checkpoints, key=lambda f: int(f.split('_')[-2]))
        logger.info(f"Found checkpoint: {latest}")
        return latest
    except (ValueError, IndexError):
        logger.warning(f"Could not parse checkpoint names in {checkpoint_dir}")
        return None


# ============================================================================
# 5. TRAINING WORKFLOW (COMPLETE & FIXED)
# ============================================================================

def train_model(
        model_save_path: str,
        benchmarks: List[Tuple[str, str]],
        hyperparams: Dict[str, Any],
        total_timesteps: int = 500,
        tb_log_dir: str = "tb_logs/",
        tb_log_name: str = "MVP_Training",
        debug_mode: bool = True,
        max_states: int = 4000,
        threshold_before_merge: int = 1,
        reward_variant: str = 'astar_search',
) -> Optional[PPO]:
    """
    Train a GNN policy using RL with REAL Fast Downward feedback.

    Args:
        model_save_path: Path to save the trained model
        benchmarks: List of (domain_file, problem_file) tuples
        hyperparams: Dictionary of PPO and reward function hyperparameters
        total_timesteps: Total training timesteps
        tb_log_dir: TensorBoard log directory
        tb_log_name: TensorBoard run name
        debug_mode: If True, use debug mode (no real FD)
        max_states: M&S max_states parameter
        threshold_before_merge: M&S threshold parameter
        reward_variant: Which reward function to use

    Returns:
        Trained PPO model, or None if training failed
    """

    # ✅ STEP 1: Validate and extract reward parameters
    valid_variants = [
        'simple_stability',
        'information_preservation',
        'hybrid',
        'conservative',
        'progressive',
        'rich',
        'astar_search'
    ]

    if reward_variant not in valid_variants:
        logger.error(f"Invalid reward variant: {reward_variant}")
        logger.error(f"Valid options: {', '.join(valid_variants)}")
        return None

    # ✅ STEP 2: Extract reward-specific kwargs from hyperparams
    reward_kwargs = {}

    # Define all possible keys for each variant
    reward_param_map = {
        'rich': ['w_f_stability', 'w_state_efficiency', 'w_transition_quality', 'w_reachability'],
        'astar_search': ['w_search_efficiency', 'w_solution_quality', 'w_f_stability', 'w_state_control'],
        'hybrid': ['w_f_stability', 'w_state_control', 'w_transition', 'w_search'],
        'simple_stability': ['alpha', 'beta', 'lambda_shrink', 'f_threshold'],
        'information_preservation': ['alpha', 'beta', 'lambda_density'],
        'conservative': ['stability_threshold'],
        'progressive': [],  # No special params, uses defaults
    }

    # Extract parameters for this variant
    if reward_variant in reward_param_map:
        for key in reward_param_map[reward_variant]:
            if key in hyperparams:
                reward_kwargs[key] = hyperparams[key]

    logger.info(f"\n{'=' * 80}")
    logger.info(f"REWARD VARIANT: {reward_variant}")
    logger.info(f"{'=' * 80}")
    if reward_kwargs:
        logger.info("Reward function parameters:")
        for k, v in reward_kwargs.items():
            logger.info(f"  {k:<30} = {v}")
    else:
        logger.info("(Using default parameters for reward function)")
    logger.info(f"{'=' * 80}\n")

    # ✅ STEP 3: Create environment with reward variant
    from merge_env import MergeEnv

    if not benchmarks or len(benchmarks) == 0:
        logger.error("No benchmarks provided!")
        return None

    domain_file, problem_file = benchmarks[0]

    logger.info(f"Creating environment with reward_variant={reward_variant}...")
    logger.info(f"  Domain:  {domain_file}")
    logger.info(f"  Problem: {problem_file}")

    env = MergeEnv(
        domain_file=domain_file,
        problem_file=problem_file,
        max_merges=50,
        debug=debug_mode,
        reward_variant=reward_variant,
        max_states=max_states,
        threshold_before_merge=threshold_before_merge,
        **reward_kwargs
    )

    env = Monitor(env)
    logger.info("✓ Environment created and wrapped with Monitor")

    # ✅ STEP 4: Create and train model
    from gnn_policy import GNNPolicy

    logger.info("Creating PPO model with GNN policy...")

    model = PPO(
        policy=GNNPolicy,
        env=env,
        learning_rate=hyperparams.get('learning_rate', 0.0003),
        n_steps=hyperparams.get('n_steps', 64),
        batch_size=hyperparams.get('batch_size', 32),
        ent_coef=hyperparams.get('ent_coef', 0.01),
        verbose=1,
        tensorboard_log=tb_log_dir,
        policy_kwargs={"hidden_dim": 64},
    )

    logger.info("✓ PPO model created")
    logger.info(f"\nStarting training for {total_timesteps} timesteps...")
    logger.info(f"Reward variant: {reward_variant}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True,
        )
        logger.info(f"✓ Training complete")
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        env.close()
        return None

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    logger.info(f"✓ Model saved: {model_save_path}")

    env.close()
    return model


# ============================================================================
# 6. BENCHMARK LOADING
# ============================================================================

def load_benchmarks_from_pattern(
        domain_file: str,
        problem_pattern: str,
        set_name: str = "Unknown"
) -> List[Tuple[str, str]]:
    """Loads benchmark problems matching a glob pattern."""
    if not os.path.exists(domain_file):
        logger.warning(f"Domain file not found: {domain_file}")
        return []

    problems = sorted(glob.glob(problem_pattern))
    if not problems:
        logger.warning(f"No problems found matching: {problem_pattern}")
        return []

    benchmarks = [(domain_file, p) for p in problems]
    logger.info(f"{set_name}: Loaded {len(benchmarks)} problems")
    return benchmarks


# ============================================================================
# 7. JSON UTILITIES
# ============================================================================

def write_json_atomic(obj: Any, path: str) -> None:
    """Atomically writes JSON to a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except:
        try:
            os.remove(tmp_path)
        except:
            pass
        raise