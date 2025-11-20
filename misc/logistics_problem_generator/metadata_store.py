"""
Metadata storage and retrieval for Logistics problems.

Stores metadata for each generated problem, including baseline planner metrics.
Requirement #6, #12: Metadata capture and storage.
"""

import json
import os
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ProblemMetadata:
    """Metadata for a single generated Logistics problem."""
    problem_name: str
    domain: str
    difficulty: str
    num_cities: int
    num_locations: int
    num_packages: int
    num_trucks: int
    num_airplanes: int
    goal_archetype: str
    plan_length: int
    optimal_plan_cost: int

    # Baseline planner metrics
    planner_time: float
    planner_success: bool
    nodes_expanded: int
    plan_cost: int

    # Files
    domain_file: str
    problem_file: str


class MetadataStore:
    """Manages problem metadata storage and retrieval."""

    def __init__(self, metadata_dir: str):
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        self.metadata_index: Dict[str, ProblemMetadata] = {}
        self.load_all_metadata()

    def save_metadata(self, metadata: ProblemMetadata) -> None:
        """Save metadata for a single problem."""
        self.metadata_index[metadata.problem_name] = metadata
        self._write_json_metadata()

    def load_all_metadata(self) -> None:
        """Load all metadata from disk."""
        index_file = os.path.join(self.metadata_dir, "index.json")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                data = json.load(f)
                for problem_name, meta_dict in data.items():
                    self.metadata_index[problem_name] = ProblemMetadata(**meta_dict)

    def _write_json_metadata(self) -> None:
        """Write metadata index to JSON."""
        index_file = os.path.join(self.metadata_dir, "index.json")
        data = {
            name: asdict(meta)
            for name, meta in self.metadata_index.items()
        }
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_by_difficulty(self, difficulty: str) -> List[ProblemMetadata]:
        """Get all problems of a specific difficulty."""
        return [
            meta for meta in self.metadata_index.values()
            if meta.difficulty == difficulty
        ]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metadata_index:
            return {}

        by_difficulty = {}
        for difficulty in ['small', 'medium', 'large']:
            problems = self.get_by_difficulty(difficulty)
            if problems:
                times = [p.planner_time for p in problems if p.planner_time]
                successful = sum(1 for p in problems if p.planner_success)
                by_difficulty[difficulty] = {
                    "count": len(problems),
                    "successful": successful,
                    "avg_time": sum(times) / len(times) if times else None,
                    "max_time": max(times) if times else None,
                    "min_time": min(times) if times else None,
                }

        return by_difficulty