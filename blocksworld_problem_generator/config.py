"""
Configuration and constants for the problem generation framework.
"""

from dataclasses import dataclass
from typing import Dict
import os


# @dataclass
# class TimeDifficultyTier:
#     """Definition of a difficulty tier based on solving time."""
#     name: str
#     min_time: float      # seconds
#     max_time: float      # seconds
#     target_time: float   # seconds (for reporting)
#     max_rejects: float   # Don't use config if it takes longer than this
#
#
# # Existing plan-length tiers
# DIFFICULTY_TIERS = {
#     'small': DifficultyTier(
#         name='small',
#         min_length=15,
#         max_length=17,
#         target_length=16
#     ),
#     'medium': DifficultyTier(
#         name='medium',
#         min_length=25,
#         max_length=30,
#         target_length=27
#     ),
#     'large': DifficultyTier(
#         name='large',
#         min_length=500,
#         max_length=1000,
#         target_length=858
#     ),
# }
#
# # NEW: Time-based difficulty tiers
# TIME_DIFFICULTY_TIERS = {
#     'small': TimeDifficultyTier(
#         name='small',
#         min_time=30,          # 30 seconds
#         max_time=90,          # 1.5 minutes
#         target_time=60,
#         max_rejects=120       # Stop trying config if > 2 min
#     ),
#     'medium': TimeDifficultyTier(
#         name='medium',
#         min_time=180,         # 3 minutes
#         max_time=240,         # 4 minutes
#         target_time=210,
#         max_rejects=270       # 4.5 minutes (your threshold)
#     ),
#     'large': TimeDifficultyTier(
#         name='large',
#         min_time=600,         # 10 minutes
#         max_time=900,         # 15 minutes
#         target_time=750,
#         max_rejects=1200      # 20 minutes
#     ),
# }
#
# # Baseline planner configuration
# BASELINE_PLANNER_CONFIG = {
#     'planner': 'downward',
#     'search': 'astar(lmcut())',
#     'timeout': 240,  # 10 minutes in seconds
# }
#
# # Generation parameters
# MAX_BLOCKS = 14
# MIN_BLOCKS = 10
#
# # Calibration parameters (NEW)
# CALIBRATION_CONFIG = {
#     'initial_samples_per_config': 20,  # Try each config twice during calibration
#     'min_configs_to_find': 20,          # Find at least 3 working configurations
#     'block_range': range(11, 13),        # Try 4-7 blocks
#     'plan_length_step': 500,             # Step size for plan length sampling
# }


@dataclass
class DifficultyTier:
    """Definition of a difficulty tier based on plan length."""
    name: str
    min_length: int
    max_length: int
    target_length: int


# Difficulty tier definitions (Requirement #16)
DIFFICULTY_TIERS = {
    'small': DifficultyTier(
        name='small',
        min_length=15,
        max_length=17,
        target_length=16
    ),
    'medium': DifficultyTier(
        name='medium',
        min_length=25,
        max_length=30,
        target_length=27
    ),
    'large': DifficultyTier(
        name='large',
        min_length=50,
        max_length=500,
        target_length=154
    ),
}



# Baseline planner configuration (Requirement #12)
BASELINE_PLANNER_CONFIG = {
    'planner': 'downward',
    'search': 'astar(lmcut())',
    'timeout': 260,  # 10 minutes in seconds
}

# Generation parameters
MAX_BLOCKS = 10
MIN_BLOCKS = 3

# Output directories
OUTPUT_DIR = 'generated_problems'
DOMAIN_DIR = os.path.join(OUTPUT_DIR, 'domains')
PROBLEMS_DIR = os.path.join(OUTPUT_DIR, 'problems')
METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata')

# Directories to create
REQUIRED_DIRS = [OUTPUT_DIR, DOMAIN_DIR, PROBLEMS_DIR, METADATA_DIR]


def ensure_output_dirs():
    """Create required output directories if they don't exist."""
    for dir_path in REQUIRED_DIRS:
        os.makedirs(dir_path, exist_ok=True)

