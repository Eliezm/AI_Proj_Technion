#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUN MERGE ANALYSIS - Post-Evaluation Metadata Analysis
======================================================
Called AFTER evaluation to analyze collected metadata.
"""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_merge_analysis(evaluation_output_dir: str) -> bool:
    """
    Run complete merge metadata analysis pipeline.

    This should be called AFTER evaluation is complete.
    By then, the following files should exist:
      - downward/fd_output/merge_before_*.json
      - downward/fd_output/merge_after_*.json
      - downward/gnn_metadata/episode_*.json

    Args:
        evaluation_output_dir: Directory where evaluation results are saved

    Returns:
        True if analysis successful
    """

    logger.info("\n" + "=" * 90)
    logger.info("STAGE 1: MERGE METADATA ANALYSIS")
    logger.info("=" * 90 + "\n")

    try:
        # Import analysis modules
        from merge_metadata_collector import MergeMetadataCollector
        from merge_explainability import MergeExplainabilityAnalyzer
        from merge_choice_analysis import MergeChoiceAnalyzer
        from gnn_metadata_aggregator import GNNMetadataAggregator

        output_dir = Path(evaluation_output_dir) / "merge_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ====================================================================
        # PHASE 1: FD MERGE METADATA COLLECTION
        # ====================================================================

        logger.info("[PHASE 1] Collecting FD merge metadata...")

        try:
            fd_collector = MergeMetadataCollector("downward/fd_output")
            num_merges = fd_collector.load_all_metadata()

            if num_merges == 0:
                logger.warning("⚠️ No FD merge metadata found")
                fd_collector = None
            else:
                logger.info(f"✓ Collected {num_merges} merge decisions from FD")

                # Export FD analysis
                fd_collector.export_to_json(output_dir / "fd_metadata.json")

        except Exception as e:
            logger.warning(f"⚠️ FD metadata collection failed: {e}")
            fd_collector = None

        # ====================================================================
        # PHASE 2: GNN METADATA AGGREGATION
        # ====================================================================

        logger.info("[PHASE 2] Aggregating GNN decision metadata...")

        try:
            gnn_aggregator = GNNMetadataAggregator("downward/gnn_metadata")
            episodes_loaded = gnn_aggregator.load_all_episodes()

            if episodes_loaded == 0:
                logger.warning("⚠️ No GNN metadata found")
                gnn_aggregator = None
            else:
                logger.info(f"✓ Loaded {episodes_loaded} GNN episodes")
                gnn_aggregator.export_aggregated_report(output_dir / "gnn_metadata")

        except Exception as e:
            logger.warning(f"⚠️ GNN metadata aggregation failed: {e}")
            gnn_aggregator = None

        # ====================================================================
        # PHASE 3: MERGE EXPLAINABILITY ANALYSIS (FD)
        # ====================================================================

        if fd_collector and fd_collector.merges:
            logger.info("[PHASE 3] Generating merge explainability...")

            try:
                explainer = MergeExplainabilityAnalyzer(fd_collector)
                explainer.export_explanations(output_dir / "fd_explanations.json")
                explainer.generate_human_report(output_dir / "fd_explainability_report.txt")
                logger.info("✓ Explainability analysis complete")

            except Exception as e:
                logger.warning(f"⚠️ Explainability analysis failed: {e}")

        # ====================================================================
        # PHASE 4: MERGE CHOICE PATTERN ANALYSIS (FD)
        # ====================================================================

        if fd_collector and fd_collector.merges:
            logger.info("[PHASE 4] Analyzing merge choice patterns...")

            try:
                choice_analyzer = MergeChoiceAnalyzer(fd_collector)
                choice_analyzer.export_analysis(output_dir / "fd_choice_analysis")
                choice_analyzer.generate_report(output_dir / "fd_choice_report.txt")
                logger.info("✓ Choice pattern analysis complete")

            except Exception as e:
                logger.warning(f"⚠️ Choice pattern analysis failed: {e}")

        # ====================================================================
        # PHASE 5: GENERATE SUMMARY
        # ====================================================================

        logger.info("[PHASE 5] Generating analysis summary...")

        summary_file = output_dir / "MERGE_ANALYSIS_SUMMARY.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("MERGE METADATA ANALYSIS SUMMARY\n")
            f.write("=" * 90 + "\n\n")

            # FD Summary
            if fd_collector:
                f.write("FAST DOWNWARD MERGE ANALYSIS\n")
                f.write("-" * 90 + "\n")

                stats = fd_collector.get_statistics()
                f.write(f"  Total merges:              {stats.get('total_merges', 0)}\n")
                f.write(f"  Avg quality score:         {stats.get('avg_quality_score', 0):.3f}\n")
                f.write(f"  Avg compression ratio:     {stats.get('avg_compression_ratio', 0):.3f}\n")
                f.write(f"  Avg branching factor:      {stats.get('avg_branching_factor', 0):.3f}\n")
                f.write(f"  Avg reachability:          {stats.get('avg_reachability', 0):.1%}\n\n")

            # GNN Summary
            if gnn_aggregator:
                f.write("GNN DECISION ANALYSIS\n")
                f.write("-" * 90 + "\n")

                f.write(f"  Total GNN decisions:       {len(gnn_aggregator.decisions)}\n")
                f.write(f"  Episodes processed:        {len(gnn_aggregator.episodes)}\n")
                f.write(f"  Unique problems:           {len(gnn_aggregator.problems_processed)}\n\n")

            f.write("OUTPUT FILES\n")
            f.write("-" * 90 + "\n")
            f.write("  FD metadata:               fd_metadata.json\n")
            f.write("  FD explanations:           fd_explanations.json\n")
            f.write("  FD choice analysis:        fd_choice_analysis/\n")
            f.write("  GNN metadata:              gnn_metadata/\n")
            f.write("  Explainability report:     fd_explainability_report.txt\n")
            f.write("  Choice patterns report:    fd_choice_report.txt\n\n")

            f.write("=" * 90 + "\n")

        logger.info(f"✓ Summary written to: {summary_file}")

        logger.info("\n" + "=" * 90)
        logger.info("✅ MERGE ANALYSIS PIPELINE COMPLETE")
        logger.info("=" * 90 + "\n")

        logger.info(f"Results saved to: {output_dir.absolute()}/\n")

        return True

    except Exception as e:
        logger.error(f"❌ MERGE ANALYSIS FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run merge metadata analysis")
    parser.add_argument("--eval-dir", default="evaluation_results",
                        help="Evaluation output directory")

    args = parser.parse_args()

    success = run_merge_analysis(args.eval_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())