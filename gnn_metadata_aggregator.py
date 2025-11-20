#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN METADATA AGGREGATOR
=======================
Aggregates all GNN decision metadata from multiple episodes/problems.
Must be run AFTER training/evaluation is complete.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNMetadataAggregator:
    """Aggregates GNN decision metadata from all episodes."""

    def __init__(self, gnn_metadata_dir: str = "downward/gnn_metadata"):
        self.gnn_metadata_dir = Path(gnn_metadata_dir)
        self.episodes = []
        self.decisions = []
        self.problems_processed = set()

    def load_all_episodes(self) -> int:
        """Load all episode metadata files."""
        if not self.gnn_metadata_dir.exists():
            logger.warning(f"Metadata directory not found: {self.gnn_metadata_dir}")
            return 0

        logger.info(f"Loading episodes from: {self.gnn_metadata_dir}")

        count = 0
        for episode_file in sorted(self.gnn_metadata_dir.glob("episode_*.json")):
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)

                self.episodes.append(episode_data)

                # Flatten decisions
                if 'decisions' in episode_data:
                    for decision in episode_data['decisions']:
                        self.decisions.append({
                            **decision,
                            'episode_file': episode_file.name,
                        })

                    self.problems_processed.add(episode_data.get('problem', 'unknown'))

                count += 1

            except Exception as e:
                logger.warning(f"Failed to load {episode_file}: {e}")

        logger.info(f"✓ Loaded {count} episodes with {len(self.decisions)} total decisions")
        return count

    def compute_decision_statistics(self) -> Dict[str, Any]:
        """Compute statistics on GNN decisions."""

        if not self.decisions:
            logger.warning("No decisions loaded")
            return {}

        logger.info("\nComputing decision statistics...")

        rewards = [d['reward_received'] for d in self.decisions]
        deltas = [d['merge_info'].get('delta_states', 0) for d in self.decisions]
        expansions = [d['merge_info'].get('num_expansions', 0) for d in self.decisions]

        stats = {
            'total_decisions': len(self.decisions),
            'problems_processed': len(self.problems_processed),
            'problems': sorted(list(self.problems_processed)),

            'reward_statistics': {
                'mean': float(np.mean(rewards)),
                'median': float(np.median(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'q1': float(np.percentile(rewards, 25)),
                'q3': float(np.percentile(rewards, 75)),
            },

            'state_dynamics': {
                'mean_delta_states': float(np.mean(deltas)),
                'mean_expansions': float(np.mean(expansions)),
                'problems_with_expansion': sum(1 for d in deltas if d > 0),
                'problems_with_reduction': sum(1 for d in deltas if d < 0),
            },

            'decision_distribution': self._compute_decision_distribution(),
        }

        logger.info(f"\nDecision Statistics:")
        logger.info(f"  Total decisions: {stats['total_decisions']}")
        logger.info(f"  Avg reward: {stats['reward_statistics']['mean']:.4f}")
        logger.info(f"  Avg delta states: {stats['state_dynamics']['mean_delta_states']:.2f}")

        return stats

    def _compute_decision_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of decisions."""

        decisions_per_episode = defaultdict(int)
        rewards_per_episode = defaultdict(list)

        for decision in self.decisions:
            episode = decision['episode_file']
            decisions_per_episode[episode] += 1
            rewards_per_episode[episode].append(decision['reward_received'])

        return {
            'avg_decisions_per_episode': float(np.mean(list(decisions_per_episode.values()))),
            'max_decisions_in_episode': max(decisions_per_episode.values()),
            'min_decisions_in_episode': min(decisions_per_episode.values()),
        }

    def identify_patterns(self) -> Dict[str, Any]:
        """Identify patterns in GNN decisions."""

        logger.info("\nIdentifying decision patterns...")

        high_reward_decisions = [d for d in self.decisions if d['reward_received'] > 0.1]
        low_reward_decisions = [d for d in self.decisions if d['reward_received'] < -0.5]

        patterns = {
            'high_reward_decisions': {
                'count': len(high_reward_decisions),
                'avg_delta_states': float(np.mean([
                    d['merge_info'].get('delta_states', 0)
                    for d in high_reward_decisions
                ])) if high_reward_decisions else 0,
                'avg_node_count': float(np.mean([
                    d['observation_shape']['num_nodes']
                    for d in high_reward_decisions
                ])) if high_reward_decisions else 0,
            },

            'low_reward_decisions': {
                'count': len(low_reward_decisions),
                'avg_delta_states': float(np.mean([
                    d['merge_info'].get('delta_states', 0)
                    for d in low_reward_decisions
                ])) if low_reward_decisions else 0,
                'avg_node_count': float(np.mean([
                    d['observation_shape']['num_nodes']
                    for d in low_reward_decisions
                ])) if low_reward_decisions else 0,
            },
        }

        logger.info(f"  High reward decisions: {patterns['high_reward_decisions']['count']}")
        logger.info(f"  Low reward decisions: {patterns['low_reward_decisions']['count']}")

        return patterns

    def export_aggregated_report(self, output_dir: str = "gnn_metadata_results") -> str:
        """Export comprehensive aggregated report."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nExporting results to: {output_dir}")

        stats = self.compute_decision_statistics()
        patterns = self.identify_patterns()

        # Main report
        report_data = {
            'metadata_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_episodes': len(self.episodes),
                'total_decisions': len(self.decisions),
                'unique_problems': len(self.problems_processed),
            },
            'statistics': stats,
            'patterns': patterns,
            'episodes': self.episodes,
            'all_decisions': self.decisions,
        }

        report_path = output_dir / "gnn_metadata_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"✓ Saved report: {report_path}")

        # Summary TXT
        self._write_summary_report(output_dir, stats, patterns)

        return str(output_dir)

    def _write_summary_report(self, output_dir: Path, stats: Dict, patterns: Dict) -> None:
        """Write human-readable summary."""

        report_path = output_dir / "gnn_metadata_summary.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("GNN METADATA AGGREGATION REPORT\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total episodes: {len(self.episodes)}\n")
            f.write(f"Total decisions: {len(self.decisions)}\n")
            f.write(f"Unique problems: {len(self.problems_processed)}\n\n")

            f.write("REWARD STATISTICS\n")
            f.write("-" * 90 + "\n")
            f.write(f"  Mean reward:     {stats['reward_statistics']['mean']:>8.4f}\n")
            f.write(f"  Median reward:   {stats['reward_statistics']['median']:>8.4f}\n")
            f.write(f"  Std reward:      {stats['reward_statistics']['std']:>8.4f}\n")
            f.write(f"  Reward range:    [{stats['reward_statistics']['min']:.4f}, "
                    f"{stats['reward_statistics']['max']:.4f}]\n\n")

            f.write("STATE DYNAMICS\n")
            f.write("-" * 90 + "\n")
            f.write(f"  Mean delta states:        {stats['state_dynamics']['mean_delta_states']:>8.2f}\n")
            f.write(f"  Mean expansions:          {stats['state_dynamics']['mean_expansions']:>8.0f}\n")
            f.write(f"  Problems with expansion:  {stats['state_dynamics']['problems_with_expansion']}\n")
            f.write(f"  Problems with reduction:  {stats['state_dynamics']['problems_with_reduction']}\n\n")

            f.write("DECISION PATTERNS\n")
            f.write("-" * 90 + "\n")
            f.write(f"  High reward decisions:  {patterns['high_reward_decisions']['count']}\n")
            f.write(f"    Avg delta states:     {patterns['high_reward_decisions']['avg_delta_states']:.2f}\n")
            f.write(f"    Avg node count:       {patterns['high_reward_decisions']['avg_node_count']:.1f}\n\n")
            f.write(f"  Low reward decisions:   {patterns['low_reward_decisions']['count']}\n")
            f.write(f"    Avg delta states:     {patterns['low_reward_decisions']['avg_delta_states']:.2f}\n")
            f.write(f"    Avg node count:       {patterns['low_reward_decisions']['avg_node_count']:.1f}\n\n")

            f.write("=" * 90 + "\n")

        logger.info(f"✓ Saved summary: {report_path}")


def main():
    """Main execution."""

    import argparse

    parser = argparse.ArgumentParser(description="GNN Metadata Aggregator")
    parser.add_argument("--metadata-dir", default="downward/gnn_metadata",
                        help="GNN metadata directory")
    parser.add_argument("--output", default="gnn_metadata_results",
                        help="Output directory")

    args = parser.parse_args()

    aggregator = GNNMetadataAggregator(args.metadata_dir)

    if aggregator.load_all_episodes() == 0:
        logger.error("No episodes loaded!")
        return 1

    aggregator.export_aggregated_report(args.output)

    logger.info(f"\n✅ Complete! Results in: {args.output}/")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())