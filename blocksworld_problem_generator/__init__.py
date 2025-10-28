"""
Blocksworld PDDL Problem Generation Framework

A modular framework for generating diverse, structurally complex Blocksworld
problems for training GNN-based planners.
"""

__version__ = "1.0.0"

from .state import BlocksWorldState, create_empty_state
from .actions import Action, ActionType, ActionExecutor
from .goal_archetypes import GoalArchetype, GoalArchetypeGenerator
from .backward_generator import BackwardProblemGenerator
from .pddl_writer import PDDLWriter
from .baseline_planner import FastDownwardRunner
from .metadata_store import MetadataStore, ProblemMetadata
from .validator import PDDLValidator
from .main import ProblemGenerationFramework

__all__ = [
    'BlocksWorldState',
    'create_empty_state',
    'Action',
    'ActionType',
    'ActionExecutor',
    'GoalArchetype',
    'GoalArchetypeGenerator',
    'BackwardProblemGenerator',
    'PDDLWriter',
    'FastDownwardRunner',
    'MetadataStore',
    'ProblemMetadata',
    'PDDLValidator',
    'ProblemGenerationFramework',
]