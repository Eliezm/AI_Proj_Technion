"""
Configuration and constants for the Logistics problem generation framework.
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
        min_length=6,
        max_length=8,
        target_length=7
    ),
    'medium': DifficultyTier(
        name='medium',
        min_length=11,
        max_length=13,
        target_length=12
    ),
    'large': DifficultyTier(
        name='large',
        min_length=14,
        max_length=16,
        target_length=15
    ),
}

# Baseline planner configuration (Requirement #12)
BASELINE_PLANNER_CONFIG = {
    'planner': 'downward',
    'search': 'astar(lmcut())',
    'timeout': 600,  # 10 minutes in seconds
}

# Logistics-specific parameters
@dataclass
class LogisticsGenerationParams:
    """Parameters controlling Logistics problem structure."""
    num_cities: int
    locations_per_city: int
    num_packages: int
    num_trucks: int
    num_airplanes: int
    prob_airport: float  # Probability a location is an airport


# Default generation parameters (can be overridden)
DEFAULT_LOGISTICS_PARAMS = {
    'small': LogisticsGenerationParams(
        num_cities=2,
        locations_per_city=2,
        num_packages=2,
        num_trucks=1,
        num_airplanes=1,
        prob_airport=0.5
    ),
    'medium': LogisticsGenerationParams(
        num_cities=3,
        locations_per_city=3,
        num_packages=4,
        num_trucks=2,
        num_airplanes=1,
        prob_airport=0.4
    ),
    'large': LogisticsGenerationParams(
        num_cities=4,
        locations_per_city=3,
        num_packages=6,
        num_trucks=2,
        num_airplanes=2,
        prob_airport=0.5
    ),
}

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