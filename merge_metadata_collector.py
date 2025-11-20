#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERGE METADATA COLLECTOR - FIXED
================================
Collects merge metadata from C++ JSON exports.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# ✅ FIXED: Define INF constant
INF = 1000000000


@dataclass
class MergeDecision:
    """Metadata for a single merge decision."""
    iteration: int
    ts1_id: int
    ts2_id: int
    ts1_size: int
    ts2_size: int
    expected_product_size: int
    ts1_transitions: int
    ts2_transitions: int
    ts1_density: float
    ts2_density: float
    ts1_goal_states: int
    ts2_goal_states: int
    ts1_reachable_fraction: float
    ts2_reachable_fraction: float
    ts1_variables: List[int] = field(default_factory=list)
    ts2_variables: List[int] = field(default_factory=list)
    shrunk: bool = False
    reduced: bool = False
    merged_size: int = 0
    merged_goal_states: int = 0
    merged_transitions: int = 0
    merged_density: float = 0.0
    reachable_states: int = 0
    unreachable_states: int = 0
    reachability_ratio: float = 0.0
    shrinking_ratio: float = 0.0
    ts1_f_min: int = 0
    ts1_f_max: int = 0
    ts1_f_mean: float = 0.0
    ts1_f_std: float = 0.0
    ts2_f_min: int = 0
    ts2_f_max: int = 0
    ts2_f_mean: float = 0.0
    ts2_f_std: float = 0.0
    merged_f_min: int = 0
    merged_f_max: int = 0
    merged_f_mean: float = 0.0
    merged_f_std: float = 0.0
    branching_factor: float = 1.0
    search_depth: int = 0
    solution_found: bool = False
    solution_cost: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def quality_score(self) -> float:
        """Compute quality score for this merge (0-1, higher is better)."""
        # Component 1: Compression efficiency
        compression = self.shrinking_ratio
        compression_score = min(compression, 1.0)

        # Component 2: F-value stability
        if self.merged_f_mean > 0 and self.ts1_f_mean > 0 and self.ts2_f_mean > 0:
            avg_before = (self.ts1_f_mean + self.ts2_f_mean) / 2.0
            f_stability = 1.0 - abs(self.merged_f_mean - avg_before) / (avg_before + 1e-6)
            f_stability_score = max(0.0, min(f_stability, 1.0))
        else:
            f_stability_score = 0.5

        # Component 3: Reachability
        reachability_score = self.reachability_ratio

        # Component 4: Search efficiency
        if self.branching_factor >= 1.0:
            bf_score = 1.0 / (1.0 + (self.branching_factor - 1.0) / 3.0)
            bf_score = max(0.0, min(bf_score, 1.0))
        else:
            bf_score = 0.5

        # Weighted combination
        overall = (
            0.35 * compression_score +
            0.35 * f_stability_score +
            0.20 * reachability_score +
            0.10 * bf_score
        )

        return max(0.0, min(overall, 1.0))


class MergeMetadataCollector:
    """Collects merge metadata from C++ JSON exports."""

    def __init__(self, fd_output_dir: str = "downward/fd_output"):
        """Initialize collector."""
        self.fd_output_dir = Path(fd_output_dir)
        self.merges: List[MergeDecision] = []

    def load_all_metadata(self) -> int:
        """Load all merge metadata from JSON files."""
        logger.info(f"Loading merge metadata from: {self.fd_output_dir}")

        if not self.fd_output_dir.exists():
            logger.warning(f"fd_output directory not found: {self.fd_output_dir}")
            return 0

        iteration = 0
        loaded_count = 0

        while True:
            before_file = self.fd_output_dir / f"merge_before_{iteration}.json"
            after_file = self.fd_output_dir / f"merge_after_{iteration}.json"

            if not before_file.exists() or not after_file.exists():
                break

            try:
                merge_data = self._load_merge_pair(iteration, before_file, after_file)
                if merge_data:
                    self.merges.append(merge_data)
                    loaded_count += 1
                    logger.info(f"  ✓ Loaded merge {iteration}")
            except Exception as e:
                logger.warning(f"  Failed to load merge {iteration}: {e}")

            iteration += 1

        logger.info(f"✓ Loaded {loaded_count} merge decisions")
        return loaded_count

    def _load_merge_pair(
        self,
        iteration: int,
        before_file: Path,
        after_file: Path
    ) -> Optional[MergeDecision]:
        """Load a single merge pair."""
        with open(before_file) as f:
            before_data = json.load(f)

        with open(after_file) as f:
            after_data = json.load(f)

        # Extract F-value statistics
        ts1_f_stats = before_data.get("ts1_f_stats", {})
        ts2_f_stats = before_data.get("ts2_f_stats", {})
        merged_f_stats = after_data.get("f_stats", {})

        # Count reachable states
        ts1_f = before_data.get("ts1_f_values", [])
        ts2_f = before_data.get("ts2_f_values", [])
        ts1_reachable = sum(1 for f in ts1_f if f != INF and f >= 0) / max(len(ts1_f), 1)
        ts2_reachable = sum(1 for f in ts2_f if f != INF and f >= 0) / max(len(ts2_f), 1)

        merge_data = MergeDecision(
            iteration=iteration,
            ts1_id=before_data.get("ts1_id", -1),
            ts2_id=before_data.get("ts2_id", -1),
            ts1_size=before_data.get("ts1_size", 0),
            ts2_size=before_data.get("ts2_size", 0),
            expected_product_size=before_data.get("expected_product_size", 0),
            ts1_transitions=before_data.get("ts1_transitions", 0),
            ts2_transitions=before_data.get("ts2_transitions", 0),
            ts1_density=before_data.get("ts1_density", 0.0),
            ts2_density=before_data.get("ts2_density", 0.0),
            ts1_goal_states=before_data.get("ts1_goal_states", 0),
            ts2_goal_states=before_data.get("ts2_goal_states", 0),
            ts1_reachable_fraction=ts1_reachable,
            ts2_reachable_fraction=ts2_reachable,
            ts1_variables=before_data.get("ts1_variables", []),
            ts2_variables=before_data.get("ts2_variables", []),
            shrunk=before_data.get("shrunk", False),
            reduced=before_data.get("reduced", False),
            merged_size=after_data.get("merged_size", 0),
            merged_goal_states=after_data.get("merged_goal_states", 0),
            merged_transitions=after_data.get("merged_transitions", 0),
            merged_density=after_data.get("merged_density", 0.0),
            reachable_states=after_data.get("reachable_states", 0),
            unreachable_states=after_data.get("unreachable_states", 0),
            reachability_ratio=after_data.get("reachability_ratio", 0.0),
            shrinking_ratio=after_data.get("shrinking_ratio", 0.0),
            ts1_f_min=ts1_f_stats.get("min", 0),
            ts1_f_max=ts1_f_stats.get("max", 0),
            ts1_f_mean=ts1_f_stats.get("mean", 0.0),
            ts1_f_std=ts1_f_stats.get("std", 0.0),
            ts2_f_min=ts2_f_stats.get("min", 0),
            ts2_f_max=ts2_f_stats.get("max", 0),
            ts2_f_mean=ts2_f_stats.get("mean", 0.0),
            ts2_f_std=ts2_f_stats.get("std", 0.0),
            merged_f_min=merged_f_stats.get("min", 0),
            merged_f_max=merged_f_stats.get("max", 0),
            merged_f_mean=merged_f_stats.get("mean", 0.0),
            merged_f_std=merged_f_stats.get("std", 0.0),
            branching_factor=after_data.get("search_signals", {}).get("branching_factor", 1.0),
            search_depth=after_data.get("search_signals", {}).get("search_depth", 0),
            solution_found=after_data.get("search_signals", {}).get("solution_found", False),
            solution_cost=after_data.get("search_signals", {}).get("solution_cost", 0),
        )

        return merge_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all merges."""
        if not self.merges:
            return {}

        quality_scores = [m.quality_score() for m in self.merges]
        compression_ratios = [m.shrinking_ratio for m in self.merges]
        branching_factors = [m.branching_factor for m in self.merges]

        return {
            'total_merges': len(self.merges),
            'avg_quality_score': float(np.mean(quality_scores)),
            'min_quality_score': float(np.min(quality_scores)),
            'max_quality_score': float(np.max(quality_scores)),
            'avg_compression_ratio': float(np.mean(compression_ratios)),
            'avg_branching_factor': float(np.mean(branching_factors)),
            'avg_reachability': float(np.mean([m.reachability_ratio for m in self.merges])),
            'solution_found_rate': sum(1 for m in self.merges if m.solution_found) / len(self.merges) if self.merges else 0,
        }

    def export_to_json(self, output_path: str) -> None:
        """Export all metadata to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': [m.to_dict() for m in self.merges],
            'statistics': self.get_statistics(),
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"✓ Exported metadata to: {output_path}")