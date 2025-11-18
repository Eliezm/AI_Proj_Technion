"""
Configuration and constants for the Logistics domain.pddl problems files generation.
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


# FIXED: Reduced target lengths to be achievable with small world
DIFFICULTY_TIERS = {
    'small': DifficultyTier(
        name='small',
        min_length=500,
        max_length=1050,
        target_length=1000
    ),
    'medium': DifficultyTier(
        name='medium',
        min_length=20,
        max_length=35,
        target_length=27
    ),
    'large': DifficultyTier(
        name='large',
        min_length=50,
        max_length=100,
        target_length=75
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


# FIXED: Increased complexity to support longer plans
DEFAULT_LOGISTICS_PARAMS = {
    'small': LogisticsGenerationParams(
        num_cities=10,  # More cities
        locations_per_city=5,  # More locations
        num_packages=10,  # More packages
        num_trucks=2,  # More trucks
        num_airplanes=2,  # More airplanes
        prob_airport=0.6  # More airports
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
        locations_per_city=4,
        num_packages=6,
        num_trucks=3,
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


