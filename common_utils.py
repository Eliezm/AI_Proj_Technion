# # -*- coding: utf-8 -*-
# """
# COMPREHENSIVE CENTRAL UTILITIES FILE
# Single source of truth for ALL shared code across the project.
# Includes: environment setup, training workflows, callbacks, file handling, commands.
# """
#
# import os
# import glob
# import random
# import logging
# import json
# import tempfile
# from typing import List, Dict, Any, Optional, Tuple, Callable
# from pathlib import Path
#
# import numpy as np
# from tqdm import tqdm
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
#
# import gymnasium as gym
# from gymnasium import spaces
#
# logger = logging.getLogger(__name__)
#
# # ============================================================================
# # 1. FAST DOWNWARD COMMAND TEMPLATE (CENTRALIZED)
# # ============================================================================
#
# # Windows version (adjust for Linux/macOS as needed)
# # FIXED: Added quotes around {domain} and {problem} and corrected search quotes.
# # FD_COMMAND_TEMPLATE = (
# #     r'python .\builds\release\bin\translate\translate.py '
# #     r'"{domain}" "{problem}" --sas-file output.sas && '
# #     r'.\builds\release\bin\downward.exe '
# #     r'--search "astar(merge_and_shrink('
# #     r'merge_strategy=merge_gnn(),'
# #     r'shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),'
# #     r'label_reduction=exact(before_shrinking=true,before_merging=false),'
# #     r'max_states=4000,threshold_before_merge=1'
# #     r'))"'
# # )
# # In common_utils.py, replace FD_COMMAND_TEMPLATE with:
#
# # Get the absolute path to downward folder (should be in project root)
# DOWNWARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "downward"))
#
# # Windows version - FIXED to use correct path
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
#
#
#
# # Alternative for Linux (uncomment if needed)
# # FD_COMMAND_TEMPLATE_LINUX = (
# #     r"python ./fast-downward.py {domain} {problem} "
# #     r"--search 'astar(merge_and_shrink(...))'"
# # )
#
#
# # ============================================================================
# # 2. ENVIRONMENT FACTORY (UNIFIED)
# # ============================================================================
#
# class MultiProblemEnv(gym.Env):
#     """
#     ✅ FIXED: Now properly inherits from gym.Env
#     Wraps MergeEnv to sample new problems on each reset.
#     """
#
#     def __init__(self, benchmarks: List[Tuple[str, str]], reward_variant: str = 'rich', **reward_kwargs):
#         """
#         Args:
#             benchmarks: List of (domain_file, problem_file) tuples
#             reward_variant: Reward function variant
#             **reward_kwargs: Additional parameters for reward function
#         """
#         super().__init__()  # ✅ CRITICAL: Initialize gym.Env
#
#         if not benchmarks:
#             raise ValueError("Benchmark list cannot be empty.")
#
#         self.benchmarks = benchmarks
#         self.reward_variant = reward_variant
#         self.reward_kwargs = reward_kwargs
#
#         # Import here to avoid circular dependencies
#         from merge_env import MergeEnv
#         self.MergeEnv = MergeEnv
#         self._env = None
#
#         # ✅ NEW: Define observation and action spaces (required by gym.Env)
#         # These will match MergeEnv's spaces
#         self.observation_space = spaces.Dict({
#             "x": spaces.Box(0.0, 1.0, shape=(100, 15), dtype=np.float32),
#             "edge_index": spaces.Box(0, 100, shape=(2, 1000), dtype=np.int64),
#             "num_nodes": spaces.Box(0, 100, shape=(), dtype=np.int32),
#             "num_edges": spaces.Box(0, 1000, shape=(), dtype=np.int32),
#         })
#         self.action_space = spaces.Discrete(1000)
#
#     def reset(self, **kwargs):
#         """Resets to a new random problem."""
#         domain, problem = random.choice(self.benchmarks)
#         cmd = FD_COMMAND_TEMPLATE.format(domain=domain, problem=problem)
#
#         # Close old environment if exists
#         if self._env is not None:
#             try:
#                 self._env.close()
#             except:
#                 pass
#
#         # Create new environment
#         self._env = self.MergeEnv(
#             fd_command=cmd,
#             max_merges=20,
#             debug=False,
#             reward_variant=self.reward_variant,
#             **self.reward_kwargs
#         )
#
#         # ✅ FIX: Update observation/action spaces to match current env
#         self.observation_space = self._env.observation_space
#         self.action_space = self._env.action_space
#
#         return self._env.reset(**kwargs)
#
#     def step(self, action):
#         if self._env is None:
#             raise RuntimeError("Environment not initialized. Call reset() first.")
#         return self._env.step(action)
#
#     def close(self):
#         if self._env is not None:
#             self._env.close()
#
#     def render(self, mode='human'):
#         """Optional: implement if needed."""
#         pass
#
#     def __getattr__(self, name):
#         """Delegate attribute access to wrapped environment."""
#         if self._env is not None:
#             return getattr(self._env, name)
#         raise AttributeError(f"Environment not initialized. Attribute '{name}' not available.")
#
#
# def make_env_factory(
#         benchmarks: List[Tuple[str, str]],
#         env_kwargs: Dict[str, Any]
# ) -> Callable:
#     """
#     Factory function for SB3 vectorized environments.
#     Returns a callable that creates Monitor-wrapped MultiProblemEnv instances.
#
#     Args:
#         benchmarks: List of (domain, problem) tuples
#         env_kwargs: Dict with reward_variant, alpha, beta, lambda_shrink, f_change_threshold
#
#     Returns:
#         Callable that SB3 can use
#     """
#
#     def _init():
#         env = MultiProblemEnv(benchmarks, **env_kwargs)
#         return Monitor(env)
#
#     return _init
#
#
# # ============================================================================
# # 3. CALLBACKS (UNIFIED)
# # ============================================================================
#
# class TqdmCallback(BaseCallback):
#     """✅ FIXED: Progress bar that doesn't rely on n_steps."""
#
#     def __init__(self):
#         super().__init__()
#         self.pbar = None
#
#     def _on_training_start(self):
#         """Initialize progress bar."""
#         # ✅ FIX: Get total_timesteps from the learn() call
#         total_steps = self.model.num_timesteps
#         self.pbar = tqdm(total=total_steps, desc="Training Progress", unit="steps", initial=0)
#
#     def _on_step(self) -> bool:
#         """Update progress bar."""
#         if self.pbar:
#             self.pbar.update(1)  # Increment by 1 step
#         return True
#
#     def _on_training_end(self):
#         """Close progress bar."""
#         if self.pbar:
#             self.pbar.close()
#             self.pbar = None
#
#
# # ============================================================================
# # 4. CHECKPOINT UTILITIES
# # ============================================================================
#
# def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
#     """
#     Finds the latest checkpoint by step count in the filename.
#     Essential for resuming interrupted training.
#
#     Args:
#         checkpoint_dir: Directory containing checkpoints
#
#     Returns:
#         Path to latest checkpoint, or None
#     """
#     if not os.path.isdir(checkpoint_dir):
#         return None
#
#     checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_steps.zip"))
#     if not checkpoints:
#         return None
#
#     try:
#         # Assumes filenames like "ppo_model_10000_steps.zip"
#         latest = max(checkpoints, key=lambda f: int(f.split('_')[-2]))
#         logger.info(f"Found checkpoint: {latest}")
#         return latest
#     except (ValueError, IndexError):
#         logger.warning(f"Could not parse checkpoint names in {checkpoint_dir}")
#         return None
#
#
# # ============================================================================
# # 5. HYPERPARAMETER EXTRACTION
# # ============================================================================
#
# def extract_env_params(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
#     """Extract environment-specific hyperparameters."""
#     env_keys = ['reward_variant', 'alpha', 'beta', 'lambda_shrink', 'f_change_threshold']
#     return {k: hyperparams[k] for k in env_keys if k in hyperparams}
#
#
# def extract_ppo_params(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
#     """Extract PPO-specific hyperparameters."""
#     ppo_keys = ['learning_rate', 'ent_coef', 'n_steps', 'batch_size']
#     return {k: hyperparams[k] for k in ppo_keys if k in hyperparams}
#
#
# # ============================================================================
# # 6. TRAINING WORKFLOW (UNIFIED)
# # ============================================================================
#
# # In common_utils.py, REPLACE the train_model() function with this:
#
# def train_model(
#         model_save_path: str,
#         vecnorm_save_path: str,
#         benchmarks: List[Tuple[str, str]],
#         hyperparams: Dict[str, Any],
#         total_timesteps: int,
#         checkpoint_prefix: str = "model",
#         initial_model_path: Optional[str] = None,
#         tb_log_dir: str = "./tb_logs/",
#         tb_log_name: Optional[str] = None,
# ) -> PPO:
#     """
#     ✅ FIXED: Complete training workflow with proper error handling.
#     """
#     from gnn_policy import GNNPolicy
#     from merge_env import MergeEnv
#
#     os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
#     os.makedirs(tb_log_dir, exist_ok=True)
#
#     env_params = extract_env_params(hyperparams)
#     ppo_params = extract_ppo_params(hyperparams)
#
#     logger.info(f"Environment params: {env_params}")
#     logger.info(f"PPO params: {ppo_params}")
#
#     # ✅ Create single environment (no VecNormalize with Dict spaces)
#     domain, problem = benchmarks[0]
#     fd_cmd = FD_COMMAND_TEMPLATE.format(domain=domain, problem=problem)
#
#     logger.info(f"Creating environment...")
#     logger.info(f"  Domain: {domain}")
#     logger.info(f"  Problem: {problem}")
#
#     env = MergeEnv(
#         fd_command=fd_cmd,
#         max_merges=5,
#         debug=True,  # ✅ DEBUG MODE
#         reward_variant=env_params.get('reward_variant', 'rich'),
#         w_f_stability=env_params.get('w_f_stability', 0.35),
#         w_state_efficiency=env_params.get('w_state_efficiency', 0.30),
#         w_transition_quality=env_params.get('w_transition_quality', 0.20),
#         w_reachability=env_params.get('w_reachability', 0.15),
#     )
#
#     # Wrap with Monitor ONLY (no VecNormalize)
#     env = Monitor(env)
#
#     logger.info("Creating PPO model...")
#     model = PPO(
#         policy=GNNPolicy,
#         env=env,
#         verbose=1,
#         tensorboard_log=tb_log_dir,
#         policy_kwargs={"hidden_dim": 64},
#         **ppo_params
#     )
#
#     logger.info(f"Starting training for {total_timesteps} timesteps...")
#     try:
#         model.learn(
#             total_timesteps=total_timesteps,
#             tb_log_name=tb_log_name or "model",
#             reset_num_timesteps=False
#         )
#     except KeyboardInterrupt:
#         logger.info("Training interrupted by user")
#     except Exception as e:
#         logger.error(f"Training error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
#
#     logger.info(f"Saving model to {model_save_path}...")
#     model.save(model_save_path)
#     logger.info(f"✅ Training complete!")
#
#     env.close()
#     return model
#
#
# # ============================================================================
# # 7. BENCHMARK LOADING
# # ============================================================================
#
# def load_benchmarks_from_pattern(
#         domain_file: str,
#         problem_pattern: str,
#         set_name: str = "Unknown"
# ) -> List[Tuple[str, str]]:
#     """
#     Loads benchmark problems matching a glob pattern.
#
#     Args:
#         domain_file: Path to domain PDDL
#         problem_pattern: Glob pattern for problems
#         set_name: Name for logging
#
#     Returns:
#         List of (domain, problem) tuples
#     """
#     if not os.path.exists(domain_file):
#         logger.warning(f"Domain file not found: {domain_file}")
#         return []
#
#     problems = sorted(glob.glob(problem_pattern))
#     if not problems:
#         logger.warning(f"No problems found matching: {problem_pattern}")
#         return []
#
#     benchmarks = [(domain_file, p) for p in problems]
#     logger.info(f"{set_name}: Loaded {len(benchmarks)} problems")
#     return benchmarks
#
#
# def prepare_train_test_split(
#         domain_file: str,
#         problem_pattern: str,
#         train_ratio: float = 0.8,
#         seed: int = 42
# ) -> Tuple[List, List]:
#     """
#     Splits problems into train/test sets.
#
#     Args:
#         domain_file: Path to domain
#         problem_pattern: Glob pattern
#         train_ratio: Fraction for training
#         seed: Random seed
#
#     Returns:
#         (train_benchmarks, test_benchmarks)
#     """
#     random.seed(seed)
#
#     problems = sorted(glob.glob(problem_pattern))
#     if len(problems) < 5:
#         raise ValueError(f"Need at least 5 problems, found {len(problems)}")
#
#     random.shuffle(problems)
#     split = int(len(problems) * train_ratio)
#
#     train = [(domain_file, p) for p in problems[:split]]
#     test = [(domain_file, p) for p in problems[split:]]
#
#     logger.info(f"Split: {len(train)} train, {len(test)} test")
#     return train, test
#
#
# # ============================================================================
# # 8. JSON FILE UTILITIES (ATOMIC WRITES)
# # ============================================================================
#
# def write_json_atomic(obj: Any, path: str) -> None:
#     """
#     Atomically writes JSON to a file (prevents partial writes).
#     """
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#     fd, tmp_path = tempfile.mkstemp(
#         dir=os.path.dirname(path) or ".",
#         suffix=".tmp"
#     )
#     try:
#         with os.fdopen(fd, "w") as f:
#             json.dump(obj, f)
#             f.flush()
#             os.fsync(f.fileno())
#         os.replace(tmp_path, path)
#     except:
#         try:
#             os.remove(tmp_path)
#         except:
#             pass
#         raise
#
#
# # ============================================================================
# # 9. LOGGING SETUP
# # ============================================================================
#
# def setup_logging(name: str, log_file: Optional[str] = None) -> logging.Logger:
#     """
#     Sets up consistent logging for a module.
#
#     Args:
#         name: Module name
#         log_file: Optional file to log to
#
#     Returns:
#         Logger instance
#     """
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#
#     # Console handler
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#
#     # File handler (if specified)
#     if log_file:
#         os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
#         fh = logging.FileHandler(log_file)
#         fh.setLevel(logging.INFO)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#
#     return logger
#
# # FILE: common_utils.py - ADD THESE FUNCTIONS
#
# # ============================================================================
# # 2.5a: Add make_sampler_env (for experiments)
# # ============================================================================
#
# def make_sampler_env(benchmarks: List[Tuple[str, str]], env_kwargs: Dict[str, Any]):
#     """
#     Creates an env that samples from benchmarks.
#     Used in experiments where we want to evaluate on a fixed set.
#     """
#     def _init():
#         env = MultiProblemEnv(benchmarks, **env_kwargs)
#         return Monitor(env)
#     return _init

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

        # ✅ NOW self.merge_env is available
        self.observation_space = self.merge_env.observation_space
        self.action_space = self.merge_env.action_space

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