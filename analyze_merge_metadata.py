#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER MERGE METADATA ANALYSIS
=============================
Orchestrates complete analysis pipeline including:
  - FD merge metadata (C++ exports)
  - GNN decision metadata (Python collection)
  - Unified explainability and choice analysis
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Complete analysis pipeline."""

    logger.info("\n" + "=" * 100)
    logger.info("COMPLETE MERGE METADATA ANALYSIS PIPELINE")
    logger.info("=" * 100 + "\n")

    output_root = Path("merge_metadata_analysis_results")
    output_root.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PHASE 1: AGGREGATE FD MERGE METADATA (from C++)
    # ========================================================================

    logger.info("\n[PHASE 1] Aggregating Fast Downward merge metadata...")
    logger.info("-" * 100 + "\n")

    try:
        from merge_metadata_collector import MergeMetadataCollector

        fd_collector = MergeMetadataCollector()
        fd_stats = fd_collector.get_statistics()

        logger.info(f"✓ FD merges collected: {fd_stats.get('total_merges', 0)}")

        # Export FD analysis
        fd_collector.export_to_json(output_root / "fd_metadata.json")
        fd_collector.export_good_bad_analysis(output_root / "fd_analysis")

    except Exception as e:
        logger.warning(f"⚠️ FD metadata aggregation failed: {e}")
        fd_collector = None

    # ========================================================================
    # PHASE 2: AGGREGATE GNN DECISION METADATA (from Python)
    # ========================================================================

    logger.info("\n[PHASE 2] Aggregating GNN decision metadata...")
    logger.info("-" * 100 + "\n")

    try:
        # Import the new aggregator
        sys.path.insert(0, os.getcwd())
        from gnn_metadata_aggregator import GNNMetadataAggregator

        gnn_aggregator = GNNMetadataAggregator()
        episodes_loaded = gnn_aggregator.load_all_episodes()

        if episodes_loaded > 0:
            logger.info(f"✓ GNN episodes loaded: {episodes_loaded}")
            gnn_aggregator.export_aggregated_report(output_root / "gnn_metadata")
        else:
            logger.warning("⚠️ No GNN metadata episodes found")
            gnn_aggregator = None

    except Exception as e:
        logger.warning(f"⚠️ GNN metadata aggregation failed: {e}")
        gnn_aggregator = None

    # ========================================================================
    # PHASE 3: EXPLAINABILITY ANALYSIS (FD merges)
    # ========================================================================

    if fd_collector and fd_collector.merges:
        logger.info("\n[PHASE 3] Generating merge explainability...")
        logger.info("-" * 100 + "\n")

        try:
            from merge_explainability import MergeExplainabilityAnalyzer

            explainer = MergeExplainabilityAnalyzer(fd_collector)
            explainer.export_explanations(output_root / "fd_explanations.json")
            explainer.generate_human_report(output_root / "fd_explainability_report.txt")

            logger.info("✓ Explainability analysis complete")

        except Exception as e:
            logger.warning(f"⚠️ Explainability analysis failed: {e}")

    # ========================================================================
    # PHASE 4: CHOICE PATTERN ANALYSIS (FD merges)
    # ========================================================================

    if fd_collector and fd_collector.merges:
        logger.info("\n[PHASE 4] Analyzing merge choice patterns...")
        logger.info("-" * 100 + "\n")

        try:
            from merge_choice_analysis import MergeChoiceAnalyzer

            choice_analyzer = MergeChoiceAnalyzer(fd_collector)
            choice_analyzer.export_analysis(output_root / "fd_choice_analysis")
            choice_analyzer.generate_report(output_root / "fd_choice_report.txt")

            logger.info("✓ Choice analysis complete")

        except Exception as e:
            logger.warning(f"⚠️ Choice analysis failed: {e}")

    # ========================================================================
    # PHASE 5: GENERATE MASTER SUMMARY
    # ========================================================================

    logger.info("\n[PHASE 5] Generating master summary...")
    logger.info("-" * 100 + "\n")

    _generate_master_summary(output_root, fd_collector, gnn_aggregator)

    # ========================================================================
    # COMPLETION
    # ========================================================================

    logger.info("\n" + "=" * 100)
    logger.info("✅ COMPLETE MERGE METADATA ANALYSIS FINISHED")
    logger.info("=" * 100 + "\n")

    logger.info(f"Results saved to: {output_root.absolute()}/\n")
    logger.info("Key outputs:")
    logger.info(f"  - FD metadata:          {output_root / 'fd_metadata.json'}")
    logger.info(f"  - GNN metadata:         {output_root / 'gnn_metadata' / 'gnn_metadata_report.json'}")
    logger.info(f"  - Explanations:         {output_root / 'fd_explanations.json'}")
    logger.info(f"  - Choice patterns:      {output_root / 'fd_choice_analysis'}")
    logger.info(f"  - Master summary:       {output_root / 'METADATA_ANALYSIS_SUMMARY.txt'}\n")

    return 0


def _generate_master_summary(output_root: Path, fd_collector, gnn_aggregator) -> None:
    """Generate comprehensive master summary."""

    summary_file = output_root / "METADATA_ANALYSIS_SUMMARY.txt"

    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MASTER MERGE METADATA ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n\n")

        # FD Summary
        if fd_collector:
            f.write("FAST DOWNWARD MERGE ANALYSIS\n")
            f.write("-" * 100 + "\n")

            stats = fd_collector.get_statistics()
            f.write(f"  Total merges:                 {stats.get('total_merges', 0)}\n")
            f.write(f"  Average quality score:        {stats.get('avg_quality_score', 0):.3f}\n")
            f.write(f"  Average compression ratio:    {stats.get('avg_compression_ratio', 0):.3f}\n")
            f.write(f"  Average branching factor:     {stats.get('avg_branching_factor', 0):.3f}\n")
            f.write(f"  Average reachability:         {stats.get('avg_reachability', 0):.1%}\n\n")

        # GNN Summary
        if gnn_aggregator:
            f.write("GNN DECISION ANALYSIS\n")
            f.write("-" * 100 + "\n")

            f.write(f"  Total GNN decisions:          {len(gnn_aggregator.decisions)}\n")
            f.write(f"  Episodes processed:           {len(gnn_aggregator.episodes)}\n")
            f.write(f"  Unique problems:              {len(gnn_aggregator.problems_processed)}\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 100 + "\n")
        f.write("  See individual files for detailed analysis:\n")
        f.write(f"    - fd_metadata.json (FD merge metadata)\n")
        f.write(f"    - gnn_metadata/ (GNN decision statistics)\n")
        f.write(f"    - fd_explanations.json (merge explanations)\n")
        f.write(f"    - fd_choice_analysis/ (pattern analysis)\n\n")

        f.write("=" * 100 + "\n")

    logger.info(f"✓ Master summary: {summary_file}")


if __name__ == "__main__":
    sys.exit(main())