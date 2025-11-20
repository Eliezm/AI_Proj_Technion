#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERGE EXPLAINABILITY - ENHANCED
===============================
Comprehensive analysis of merge decisions from metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np

from merge_metadata_collector import MergeMetadataCollector, MergeDecision

logger = logging.getLogger(__name__)


@dataclass
class MergeExplanation:
    """Complete explanation for a merge decision."""
    iteration: int
    ts1_id: int
    ts2_id: int

    # Why were these chosen?
    rationale: str  # Human-readable explanation

    # Pre-merge properties that motivated selection
    ts1_properties: Dict[str, Any] = field(default_factory=dict)
    ts2_properties: Dict[str, Any] = field(default_factory=dict)
    relative_properties: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    outcome: Dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    quality_score: float = 0.0
    success_indicators: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MergeExplainabilityAnalyzer:
    """Analyzes and explains merge decisions."""

    def __init__(self, metadata_collector: Optional[MergeMetadataCollector] = None):
        """Initialize analyzer."""
        self.collector = metadata_collector or MergeMetadataCollector()
        self.explanations: List[MergeExplanation] = []
        self._analyze_all_merges()

    def _analyze_all_merges(self) -> None:
        """Generate explanations for all merges."""
        for merge in self.collector.merges:
            explanation = self._explain_merge(merge)
            self.explanations.append(explanation)

    def _explain_merge(self, merge: MergeDecision) -> MergeExplanation:
        """Generate explanation for a single merge."""

        # Extract properties
        ts1_props = {
            'size': merge.ts1_size,
            'transitions': merge.ts1_transitions,
            'density': merge.ts1_density,
            'goal_states': merge.ts1_goal_states,
            'variables': len(merge.ts1_variables),
            'f_value_quality': {
                'min': merge.ts1_f_min,
                'mean': merge.ts1_f_mean,
                'max': merge.ts1_f_max,
                'std': merge.ts1_f_std,
            }
        }

        ts2_props = {
            'size': merge.ts2_size,
            'transitions': merge.ts2_transitions,
            'density': merge.ts2_density,
            'goal_states': merge.ts2_goal_states,
            'variables': len(merge.ts2_variables),
            'f_value_quality': {
                'min': merge.ts2_f_min,
                'mean': merge.ts2_f_mean,
                'max': merge.ts2_f_max,
                'std': merge.ts2_f_std,
            }
        }

        relative_props = {
            'size_ratio': merge.ts2_size / max(merge.ts1_size, 1),
            'transition_ratio': merge.ts2_transitions / max(merge.ts1_transitions, 1),
            'density_similarity': 1.0 - abs(merge.ts1_density - merge.ts2_density),
        }

        outcome = {
            'merged_size': merge.merged_size,
            'expected_size': merge.expected_product_size,
            'compression_ratio': merge.shrinking_ratio,
            'reachability': merge.reachability_ratio,
            'branching_factor': merge.branching_factor,
            'solution_found': merge.solution_found,
        }

        # Generate rationale
        rationale = self._generate_rationale(merge, ts1_props, ts2_props, relative_props, outcome)

        # Identify success/risk indicators
        success, risks = self._identify_indicators(merge, outcome)

        return MergeExplanation(
            iteration=merge.iteration,
            ts1_id=merge.ts1_id,
            ts2_id=merge.ts2_id,
            rationale=rationale,
            ts1_properties=ts1_props,
            ts2_properties=ts2_props,
            relative_properties=relative_props,
            outcome=outcome,
            quality_score=merge.quality_score(),
            success_indicators=success,
            risk_indicators=risks,
        )

    def _generate_rationale(
            self,
            merge: MergeDecision,
            ts1_props: Dict,
            ts2_props: Dict,
            relative_props: Dict,
            outcome: Dict
    ) -> str:
        """Generate human-readable explanation."""

        points = []

        # Size analysis
        if merge.ts1_size > merge.ts2_size:
            points.append(
                f"Merging smaller TS{merge.ts2_id} ({merge.ts2_size} states) into larger TS{merge.ts1_id} ({merge.ts1_size} states)")
        elif merge.ts2_size > merge.ts1_size:
            points.append(
                f"Merging smaller TS{merge.ts1_id} ({merge.ts1_size} states) into larger TS{merge.ts2_id} ({merge.ts2_size} states)")
        else:
            points.append(f"Merging equally-sized transition systems ({merge.ts1_size} states each)")

        # Density analysis
        if merge.ts1_density > merge.ts2_density:
            points.append(
                f"TS{merge.ts1_id} is denser ({merge.ts1_density:.2f} edges/state) than TS{merge.ts2_id} ({merge.ts2_density:.2f})")
        else:
            points.append(
                f"TS{merge.ts2_id} is denser ({merge.ts2_density:.2f} edges/state) than TS{merge.ts1_id} ({merge.ts1_density:.2f})")

        # Outcome analysis
        if outcome['compression_ratio'] < 0.5:
            points.append(
                f"Successfully compressed merged product to {outcome['compression_ratio'] * 100:.1f}% of theoretical maximum")
        elif outcome['compression_ratio'] < 1.0:
            points.append(
                f"Moderate compression achieved ({outcome['compression_ratio'] * 100:.1f}% of theoretical maximum)")
        else:
            points.append(
                f"WARNING: Merged size ({outcome['merged_size']}) exceeds theoretical maximum ({outcome['expected_size']})")

        # Reachability
        if outcome['reachability'] > 0.9:
            points.append("Excellent reachability preserved")
        elif outcome['reachability'] > 0.7:
            points.append(f"Good reachability ({outcome['reachability'] * 100:.1f}%)")
        else:
            points.append(f"Warning: Low reachability ({outcome['reachability'] * 100:.1f}%)")

        # Search efficiency
        if outcome['branching_factor'] < 2.0:
            points.append(f"Excellent branching factor ({outcome['branching_factor']:.2f})")
        else:
            points.append(f"High branching factor ({outcome['branching_factor']:.2f}) may reduce search efficiency")

        return " | ".join(points)

    def _identify_indicators(self, merge: MergeDecision, outcome: Dict) -> tuple:
        """Identify success and risk indicators."""
        success = []
        risks = []

        # Success indicators
        if merge.shrinking_ratio < 0.5:
            success.append("Excellent compression (< 50%)")
        if merge.reachability_ratio > 0.85:
            success.append("High reachability preserved (> 85%)")
        if merge.branching_factor < 1.5:
            success.append("Low branching factor (< 1.5)")
        if merge.solution_found:
            success.append("Solution found")
        if merge.quality_score() > 0.7:
            success.append("High quality merge")

        # Risk indicators
        if merge.shrinking_ratio > 1.0:
            risks.append("State explosion after merge")
        if merge.reachability_ratio < 0.5:
            risks.append("Low reachability (> 50% unreachable)")
        if merge.branching_factor > 5.0:
            risks.append("Very high branching factor")
        if merge.unreachable_states > merge.merged_size * 0.5:
            risks.append("Most states became unreachable")
        if merge.quality_score() < 0.3:
            risks.append("Low quality merge")

        return success, risks

    def get_best_merges(self, n: int = 5) -> List[MergeExplanation]:
        """Get top N best merges."""
        sorted_explanations = sorted(
            self.explanations,
            key=lambda e: e.quality_score,
            reverse=True
        )
        return sorted_explanations[:n]

    def get_worst_merges(self, n: int = 5) -> List[MergeExplanation]:
        """Get top N worst merges."""
        sorted_explanations = sorted(
            self.explanations,
            key=lambda e: e.quality_score
        )
        return sorted_explanations[:n]

    def export_explanations(self, output_path: str) -> None:
        """Export all explanations to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'explanations': [e.to_dict() for e in self.explanations],
            'summary': {
                'total': len(self.explanations),
                'avg_quality': float(np.mean([e.quality_score for e in self.explanations])),
                'best_quality': float(np.max([e.quality_score for e in self.explanations])),
                'worst_quality': float(np.min([e.quality_score for e in self.explanations])),
            },
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported explanations to: {output_path}")

    def generate_human_report(self, output_path: str) -> None:
        """Generate human-readable report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MERGE EXPLAINABILITY REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Total merges analyzed: {len(self.explanations)}\n")
            f.write(f"Average quality: {np.mean([e.quality_score for e in self.explanations]):.3f}\n\n")

            # Best merges
            f.write("\n" + "=" * 100 + "\n")
            f.write("TOP 5 BEST MERGES\n")
            f.write("=" * 100 + "\n\n")

            for i, exp in enumerate(self.get_best_merges(5), 1):
                f.write(f"[{i}] Iteration {exp.iteration}: TS{exp.ts1_id} + TS{exp.ts2_id}\n")
                f.write(f"    Quality: {exp.quality_score:.3f}\n")
                f.write(f"    {exp.rationale}\n")
                if exp.success_indicators:
                    f.write(f"    ✓ {', '.join(exp.success_indicators)}\n")
                f.write("\n")

            # Worst merges
            f.write("\n" + "=" * 100 + "\n")
            f.write("TOP 5 WORST MERGES\n")
            f.write("=" * 100 + "\n\n")

            for i, exp in enumerate(self.get_worst_merges(5), 1):
                f.write(f"[{i}] Iteration {exp.iteration}: TS{exp.ts1_id} + TS{exp.ts2_id}\n")
                f.write(f"    Quality: {exp.quality_score:.3f}\n")
                f.write(f"    {exp.rationale}\n")
                if exp.risk_indicators:
                    f.write(f"    ⚠️  {', '.join(exp.risk_indicators)}\n")
                f.write("\n")

        logger.info(f"Generated human report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = MergeExplainabilityAnalyzer()
    analyzer.export_explanations("merge_explanations.json")
    analyzer.generate_human_report("merge_explainability_report.txt")

    print("\n✅ Explainability analysis complete!")