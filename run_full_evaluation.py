# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
END-TO-END EVALUATION ORCHESTRATOR
==================================
Master script that runs comprehensive evaluation: baselines + GNN + analysis.

This is the primary interface for evaluating your GNN model against baselines.

Usage:
    python run_all_evaluation.py \
        --model mvp_output/gnn_model.zip \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --output evaluation_results/ \
        --timeout 300

Features:
  ✓ Single entry point for all evaluation
  ✓ Automatic input validation
  ✓ Progressive execution with checkpointing
  ✓ Comprehensive error handling
  ✓ Final report generation
  ✓ Result visualization

Output Structure:
    evaluation_results/
    ├── evaluation_results.csv        # Detailed per-problem results
    ├── evaluation_summary.json        # Aggregate statistics
    ├── comparison_report.txt          # Human-readable comparison
    ├── evaluation.log                 # Full debug log
    ├── plots/
    │   ├── solve_rate_comparison.png
    │   ├── time_comparison.png
    │   ├── expansions_comparison.png
    │   ├── efficiency_frontier.png
    │   ├── per_problem_comparison.png
    │   ├── statistical_analysis.png
    │   └── summary_dashboard.html
    └── checkpoints/
        └── evaluation_checkpoint.json # Resume point
"""

import sys
import os
import logging
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_orchestrator.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 1: INPUT VALIDATION
# ============================================================================

class EvaluationValidator:
    """Validates all inputs before starting evaluation."""

    @staticmethod
    def validate_model(model_path: str) -> bool:
        """Check that model file exists and is readable."""
        if not model_path:
            logger.error("Model path not provided")
            return False

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        if not model_path.endswith('.zip'):
            logger.warning(f"Model file does not end with .zip: {model_path}")

        try:
            # Try to read as ZIP to verify it's a valid model file
            import zipfile
            with zipfile.ZipFile(model_path, 'r') as z:
                if 'data' not in z.namelist() and 'policy.optimizer_states' not in z.namelist():
                    logger.warning(f"Model file may not be a valid PPO model: {model_path}")
        except Exception as e:
            logger.warning(f"Could not validate model ZIP: {e}")

        logger.info(f"✓ Model validated: {model_path}")
        return True

    @staticmethod
    def validate_domain(domain_path: str) -> bool:
        """Check that domain PDDL file exists."""
        if not domain_path:
            logger.error("Domain path not provided")
            return False

        if not os.path.exists(domain_path):
            logger.error(f"Domain file not found: {domain_path}")
            return False

        if not domain_path.endswith('.pddl'):
            logger.warning(f"Domain file does not end with .pddl: {domain_path}")

        # Minimal syntax check
        try:
            with open(domain_path, 'r') as f:
                content = f.read()
                if '(define (domain' not in content:
                    logger.warning(f"Domain file may not be valid PDDL: {domain_path}")
        except Exception as e:
            logger.error(f"Could not read domain file: {e}")
            return False

        logger.info(f"✓ Domain validated: {domain_path}")
        return True

    @staticmethod
    def validate_problems(problem_pattern: str) -> bool:
        """Check that problem files exist matching the pattern."""
        if not problem_pattern:
            logger.error("Problem pattern not provided")
            return False

        import glob
        problems = sorted(glob.glob(problem_pattern))

        if not problems:
            logger.error(f"No problems found matching pattern: {problem_pattern}")
            return False

        logger.info(f"✓ Found {len(problems)} problem(s) matching pattern")

        # Validate first few
        for prob in problems[:min(3, len(problems))]:
            if not prob.endswith('.pddl'):
                logger.warning(f"Problem file does not end with .pddl: {prob}")
            try:
                with open(prob, 'r') as f:
                    content = f.read()
                    if '(define (problem' not in content:
                        logger.warning(f"File may not be valid PDDL: {prob}")
            except Exception as e:
                logger.error(f"Could not read problem file {prob}: {e}")
                return False

        return True

    @staticmethod
    def validate_output_dir(output_dir: str) -> bool:
        """Check that output directory is writable."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Try to write a test file
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            logger.info(f"✓ Output directory ready: {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Cannot write to output directory: {e}")
            return False

    @staticmethod
    def validate_all(model_path: str, domain_path: str, problem_pattern: str, output_dir: str) -> bool:
        """Run all validations."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 0: INPUT VALIDATION")
        logger.info("=" * 80 + "\n")

        checks = [
            ("Model file", lambda: EvaluationValidator.validate_model(model_path)),
            ("Domain PDDL", lambda: EvaluationValidator.validate_domain(domain_path)),
            ("Problem files", lambda: EvaluationValidator.validate_problems(problem_pattern)),
            ("Output directory", lambda: EvaluationValidator.validate_output_dir(output_dir)),
        ]

        results = []
        for name, check in checks:
            try:
                result = check()
                results.append((name, result))
            except Exception as e:
                logger.error(f"✗ {name} validation failed: {e}")
                results.append((name, False))

        # Summary
        passed = sum(1 for _, r in results if r)
        total = len(results)

        logger.info(f"\nValidation Summary: {passed}/{total} passed")

        return all(r for _, r in results)


# ============================================================================
# STAGE 2: RUN COMPREHENSIVE EVALUATION
# ============================================================================

def run_comprehensive_evaluation(
        model_path: str,
        domain_path: str,
        problem_pattern: str,
        output_dir: str,
        timeout: int,
        skip_baselines: bool
) -> bool:
    """Execute comprehensive evaluation framework."""

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: COMPREHENSIVE EVALUATION")
    logger.info("=" * 80 + "\n")

    try:
        cmd = [
            "python", "evaluation_comprehensive.py",
            "--model", model_path,
            "--domain", domain_path,
            "--problems", problem_pattern,
            "--output", output_dir,
            "--timeout", str(timeout)
        ]

        if skip_baselines:
            cmd.append("--skip-baselines")

        logger.info(f"Running command: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Comprehensive evaluation failed with code {result.returncode}")
            return False

        logger.info("\n✅ Comprehensive evaluation completed")
        return True

    except Exception as e:
        logger.error(f"Failed to run comprehensive evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# STAGE 3: RUN ANALYSIS AND VISUALIZATION
# ============================================================================

def run_analysis_and_visualization(
        results_csv: str,
        output_dir: str
) -> bool:
    """Execute analysis and visualization framework."""

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: ANALYSIS AND VISUALIZATION")
    logger.info("=" * 80 + "\n")

    try:
        # Check if CSV exists
        if not os.path.exists(results_csv):
            logger.warning(f"Results CSV not found: {results_csv}")
            logger.warning("Skipping analysis and visualization")
            return False

        plots_dir = os.path.join(output_dir, "plots")

        cmd = [
            "python", "analysis_and_visualization.py",
            "--results", results_csv,
            "--output", plots_dir
        ]

        logger.info(f"Running command: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            logger.warning(f"Analysis and visualization exited with code {result.returncode}")
            logger.warning("Continuing with evaluation results anyway")
            return False

        logger.info("\n✅ Analysis and visualization completed")
        return True

    except Exception as e:
        logger.error(f"Failed to run analysis and visualization: {e}")
        logger.error("Continuing with evaluation results anyway")
        return False


# ============================================================================
# STAGE 4: GENERATE FINAL REPORT
# ============================================================================

def generate_final_report(output_dir: str) -> bool:
    """Create comprehensive final report."""

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: FINAL REPORT GENERATION")
    logger.info("=" * 80 + "\n")

    try:
        report_path = os.path.join(output_dir, "EVALUATION_REPORT.txt")

        # CORRECTED LINE
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("GNN MERGE STRATEGY - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            # Overview
            f.write("EVALUATION OVERVIEW\n")
            f.write("-" * 90 + "\n")
            f.write("This report contains the complete evaluation of the trained GNN policy\n")
            f.write("against baseline Fast Downward planners on a test set of problems.\n\n")

            # Key Files
            f.write("KEY FILES\n")
            f.write("-" * 90 + "\n")
            f.write(f"  Detailed results:     evaluation_results.csv\n")
            f.write(f"  Summary statistics:   evaluation_summary.json\n")
            f.write(f"  Comparison report:    comparison_report.txt\n")
            f.write(f"  Plots directory:      plots/\n\n")

            # How to View Results
            f.write("HOW TO INTERPRET RESULTS\n")
            f.write("-" * 90 + "\n")
            f.write("1. SOLVE RATE: Percentage of problems solved by each planner\n")
            f.write("   - Higher is better. Target: > 80% for GNN on test set.\n\n")

            f.write("2. AVERAGE TIME: Mean solution time for solved problems\n")
            f.write("   - Lower is better. Measures computational efficiency.\n\n")

            f.write("3. EXPANSIONS: Mean number of nodes expanded during search\n")
            f.write("   - Lower is better. Indicates heuristic quality.\n\n")

            f.write("4. PLAN COST: Quality of solutions found\n")
            f.write("   - Lower is better. Indicates solution optimality.\n\n")

            f.write("5. EFFICIENCY FRONTIER: Trade-off between time and expansions\n")
            f.write("   - Curves closer to origin are better.\n\n")

            # Next Steps
            f.write("NEXT STEPS\n")
            f.write("-" * 90 + "\n")
            f.write("1. Review the comparison_report.txt for baseline performance\n")
            f.write("2. Check plots/solve_rate_comparison.png for GNN vs baselines\n")
            f.write("3. Analyze per_problem_comparison.png for problem-specific insights\n")
            f.write("4. Review evaluation_summary.json for detailed statistics\n\n")

            # Recommendations
            f.write("INTERPRETATION GUIDELINES\n")
            f.write("-" * 90 + "\n")
            f.write("✅ EXCELLENT: GNN solves >= 90% of problems with time comparable to LM-Cut\n")
            f.write("✓  GOOD:      GNN solves >= 80% of problems with reasonable time\n")
            f.write("⚠️  FAIR:      GNN solves >= 60% of problems, needs optimization\n")
            f.write("❌ POOR:      GNN solves < 60% of problems, requires retraining\n\n")

            f.write("=" * 90 + "\n")

        logger.info(f"✅ Final report written: {report_path}")

        # Print summary
        logger.info("\n" + "=" * 90)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 90 + "\n")

        logger.info("Results saved to:")
        logger.info(f"  {os.path.abspath(output_dir)}\n")

        logger.info("Key files:")
        logger.info(f"  - {os.path.join(output_dir, 'evaluation_results.csv')}")
        logger.info(f"  - {os.path.join(output_dir, 'comparison_report.txt')}")
        logger.info(f"  - {os.path.join(output_dir, 'plots/')}\n")

        logger.info("To view results:")
        logger.info(f"  cat {os.path.join(output_dir, 'EVALUATION_REPORT.txt')}\n")

        return True

    except Exception as e:
        logger.error(f"Failed to generate final report: {e}")
        return False


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    """Main execution orchestrator."""
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation orchestrator for GNN merge strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic evaluation
  python run_all_evaluation.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_small_*.pddl" \\
      --output evaluation_results/
      

  # Skip baselines (faster)
  python run_all_evaluation.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_small_*.pddl" \\
      --output evaluation_results/ \\
      --skip-baselines

  # Longer timeout
  python run_all_evaluation.py \\
      --model mvp_output/gnn_model.zip \\
      --domain domain.pddl \\
      --problems "problem_*.pddl" \\
      --output evaluation_results/ \\
      --timeout 600
        """
    )

    parser.add_argument(
        "--model", required=True,
        help="Path to trained GNN model (ZIP file)"
    )
    parser.add_argument(
        "--domain", required=True,
        help="Path to domain PDDL file"
    )
    parser.add_argument(
        "--problems", required=True,
        help="Glob pattern for problem PDDL files"
    )
    parser.add_argument(
        "--output", default="evaluation_results",
        help="Output directory for results (default: evaluation_results)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout per problem in seconds (default: 300)"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline evaluation (faster, GNN only)"
    )

    args = parser.parse_args()

    # ====================================================================
    # STAGE 0: VALIDATION
    # ====================================================================

    if not EvaluationValidator.validate_all(
            args.model,
            args.domain,
            args.problems,
            args.output
    ):
        logger.error("\n❌ INPUT VALIDATION FAILED")
        logger.error("Please check your inputs and try again")
        return 1

    # ====================================================================
    # STAGE 1: COMPREHENSIVE EVALUATION
    # ====================================================================

    if not run_comprehensive_evaluation(
            args.model,
            args.domain,
            args.problems,
            args.output,
            args.timeout,
            args.skip_baselines
    ):
        logger.error("\n❌ COMPREHENSIVE EVALUATION FAILED")
        return 1

    # ====================================================================
    # STAGE 2: ANALYSIS AND VISUALIZATION
    # ====================================================================

    results_csv = os.path.join(args.output, "evaluation_results.csv")

    if not run_analysis_and_visualization(results_csv, args.output):
        logger.warning("\n⚠️ ANALYSIS AND VISUALIZATION FAILED (continuing anyway)")

    # ====================================================================
    # STAGE 3: FINAL REPORT
    # ====================================================================

    if not generate_final_report(args.output):
        logger.error("\n❌ FINAL REPORT GENERATION FAILED")
        return 1

    logger.info("✅ EVALUATION PIPELINE COMPLETE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())