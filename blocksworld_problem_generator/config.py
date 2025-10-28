"""
Configuration and constants for the problem generation framework.
"""

from dataclasses import dataclass
from typing import Dict
import os

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
        min_length=500,
        max_length=1000,
        target_length=858
    ),
}



# Baseline planner configuration (Requirement #12)
BASELINE_PLANNER_CONFIG = {
    'planner': 'downward',
    'search': 'astar(lmcut())',
    'timeout': 80,  # 10 minutes in seconds
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