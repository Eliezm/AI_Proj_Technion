#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERGE CHOICE ANALYSIS - ENHANCED
================================
Learn patterns from good and bad merge choices.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

from merge_metadata_collector import MergeMetadataCollector, MergeDecision

logger = logging.getLogger(__name__)


class MergeChoiceAnalyzer:
    """Analyzes patterns in merge decisions to learn optimal strategies."""

    def __init__(self, metadata_collector: Optional[MergeMetadataCollector] = None):
        """Initialize analyzer."""
        self.collector = metadata_collector or MergeMetadataCollector()

    def extract_features(self, merge: MergeDecision) -> Dict[str, float]:
        """Extract numerical features from a merge."""
        return {
            'ts1_size': float(merge.ts1_size),
            'ts2_size': float(merge.ts2_size),
            'size_ratio': float(merge.ts2_size / max(merge.ts1_size, 1)),
            'size_product': float(merge.ts1_size * merge.ts2_size),
            'ts1_density': float(merge.ts1_density),
            'ts2_density': float(merge.ts2_density),
            'density_diff': float(abs(merge.ts1_density - merge.ts2_density)),
            'ts1_reachable': float(merge.ts1_reachable_fraction),
            'ts2_reachable': float(merge.ts2_reachable_fraction),
            'ts1_goal_ratio': float(merge.ts1_goal_states / max(merge.ts1_size, 1)),
            'ts2_goal_ratio': float(merge.ts2_goal_states / max(merge.ts2_size, 1)),
            'ts1_f_mean': float(merge.ts1_f_mean),
            'ts2_f_mean': float(merge.ts2_f_mean),
            'f_stability_delta': float(abs(merge.ts1_f_mean - merge.ts2_f_mean)),
        }

    def analyze_good_vs_bad(self, threshold: float = 0.6) -> Dict[str, Any]:
        """Compare features of good merges vs bad merges."""

        good_merges = [m for m in self.collector.merges if m.quality_score() >= threshold]
        bad_merges = [m for m in self.collector.merges if m.quality_score() < threshold]

        if not good_merges or not bad_merges:
            logger.warning("Insufficient data for good/bad comparison")
            return {}

        good_features = [self.extract_features(m) for m in good_merges]
        bad_features = [self.extract_features(m) for m in bad_merges]

        analysis = {}
        all_keys = set(good_features[0].keys())

        for key in all_keys:
            good_values = np.array([f[key] for f in good_features])
            bad_values = np.array([f[key] for f in bad_features])

            analysis[key] = {
                'good': {
                    'mean': float(np.mean(good_values)),
                    'std': float(np.std(good_values)),
                    'min': float(np.min(good_values)),
                    'max': float(np.max(good_values)),
                },
                'bad': {
                    'mean': float(np.mean(bad_values)),
                    'std': float(np.std(bad_values)),
                    'min': float(np.min(bad_values)),
                    'max': float(np.max(bad_values)),
                },
                'delta': float(np.mean(good_values) - np.mean(bad_values)),
            }

        return analysis

    def find_patterns_in_good_merges(self) -> Dict[str, Any]:
        """Identify common patterns in high-quality merges."""

        good_merges = [m for m in self.collector.merges if m.quality_score() >= 0.7]

        if not good_merges:
            return {}

        patterns = {
            'compression': [],
            'reachability': [],
            'branching': [],
            'f_stability': [],
        }

        # Compression pattern
        compressions = [m.shrinking_ratio for m in good_merges]
        patterns['compression'] = {
            'mean': float(np.mean(compressions)),
            'std': float(np.std(compressions)),
            'ideal_range': [float(np.percentile(compressions, 25)),
                            float(np.percentile(compressions, 75))],
        }

        # Reachability pattern
        reachabilities = [m.reachability_ratio for m in good_merges]
        patterns['reachability'] = {
            'mean': float(np.mean(reachabilities)),
            'std': float(np.std(reachabilities)),
            'ideal_range': [float(np.percentile(reachabilities, 25)),
                            float(np.percentile(reachabilities, 75))],
        }

        # Branching pattern
        branching = [m.branching_factor for m in good_merges]
        patterns['branching'] = {
            'mean': float(np.mean(branching)),
            'std': float(np.std(branching)),
            'ideal_range': [float(np.percentile(branching, 25)),
                            float(np.percentile(branching, 75))],
        }

        # F-stability pattern
        f_stability = [m.merged_f_std / max(m.merged_f_mean, 1) for m in good_merges]
        patterns['f_stability'] = {
            'mean': float(np.mean(f_stability)),
            'std': float(np.std(f_stability)),
            'ideal_range': [float(np.percentile(f_stability, 25)),
                            float(np.percentile(f_stability, 75))],
        }

        return patterns

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on learned patterns."""

        recommendations = []

        good_merges = [m for m in self.collector.merges if m.quality_score() >= 0.7]
        bad_merges = [m for m in self.collector.merges if m.quality_score() < 0.3]

        if not good_merges or not bad_merges:
            return recommendations

        # Recommendation 1: Size balance
        avg_good_ratio = np.mean([m.ts2_size / max(m.ts1_size, 1) for m in good_merges])
        avg_bad_ratio = np.mean([m.ts2_size / max(m.ts1_size, 1) for m in bad_merges])

        if avg_good_ratio < 0.3 or avg_good_ratio > 3.0:
            recommendations.append(
                f"Prefer merging similarly-sized transition systems (ratio ~1.0) rather than "
                f"highly imbalanced pairs (ratio {avg_good_ratio:.2f})"
            )

        # Recommendation 2: Density
        avg_good_density_diff = np.mean([abs(m.ts1_density - m.ts2_density) for m in good_merges])
        avg_bad_density_diff = np.mean([abs(m.ts1_density - m.ts2_density) for m in bad_merges])

        if avg_good_density_diff < avg_bad_density_diff:
            recommendations.append(
                f"Prefer merging TS with similar transition density "
                f"(good merges differ by {avg_good_density_diff:.3f}, bad by {avg_bad_density_diff:.3f})"
            )

        # Recommendation 3: Reachability
        avg_good_reachability = np.mean([m.reachability_ratio for m in good_merges])
        avg_bad_reachability = np.mean([m.reachability_ratio for m in bad_merges])

        if avg_good_reachability > avg_bad_reachability + 0.2:
            recommendations.append(
                f"Prioritize merges that preserve reachability "
                f"(good merges: {avg_good_reachability:.1%}, bad merges: {avg_bad_reachability:.1%})"
            )

        # Recommendation 4: Compression
        avg_good_compression = np.mean([m.shrinking_ratio for m in good_merges])
        avg_bad_compression = np.mean([m.shrinking_ratio for m in bad_merges])

        if avg_good_compression < 0.7:
            recommendations.append(
                f"Target merges with good compression ratios "
                f"(good: {avg_good_compression:.1%}, bad: {avg_bad_compression:.1%})"
            )

        return recommendations

    def export_analysis(self, output_dir: str) -> None:
        """Export full analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_data = {
            'good_vs_bad': self.analyze_good_vs_bad(),
            'good_merge_patterns': self.find_patterns_in_good_merges(),
            'recommendations': self.generate_recommendations(),
            'timestamp': str(datetime.now()),
        }

        with open(output_dir / "merge_choice_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        logger.info(f"Exported analysis to: {output_dir}/merge_choice_analysis.json")

    def generate_report(self, output_path: str) -> None:
        """Generate human-readable report."""
        from datetime import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MERGE CHOICE ANALYSIS REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Merges analyzed: {len(self.collector.merges)}\n\n")

            # Patterns
            f.write("\n" + "=" * 100 + "\n")
            f.write("PATTERNS IN GOOD MERGES\n")
            f.write("=" * 100 + "\n\n")

            patterns = self.find_patterns_in_good_merges()
            for key, data in patterns.items():
                f.write(f"{key.upper()}:\n")
                f.write(f"  Mean: {data['mean']:.3f}\n")
                f.write(f"  Range: {data['ideal_range'][0]:.3f} - {data['ideal_range'][1]:.3f}\n\n")

            # Recommendations
            f.write("\n" + "=" * 100 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")

            for i, rec in enumerate(self.generate_recommendations(), 1):
                f.write(f"[{i}] {rec}\n\n")

        logger.info(f"Generated report: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import datetime

    analyzer = MergeChoiceAnalyzer()
    analyzer.export_analysis("merge_choice_results/")
    analyzer.generate_report("merge_choice_report.txt")

    print("\n✅ Merge choice analysis complete!")