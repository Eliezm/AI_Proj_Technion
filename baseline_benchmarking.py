# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# BASELINE BENCHMARKING HARNESS FOR GNN MERGE STRATEGY LEARNING
# ==============================================================
#
# This script benchmarks STANDARD Fast Downward algorithms on your problem set
# and produces a CSV report for comparison with your GNN policy.
#
# Usage:
#     python baseline_benchmarking.py
#
# Output:
#     baseline_performance_summary.csv
#
# This CSV can later be compared with your GNN policy performance metrics.
# """
#
# import sys
# import os
# import logging
# import glob
# import json
# import subprocess
# import time
# import re
# import tempfile
# import shutil
# from typing import List, Dict, Tuple, Optional, Any
# from pathlib import Path
# from datetime import datetime
# import csv
#
# # Setup
# sys.path.insert(0, os.getcwd())
#
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)-8s - [%(filename)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler("baseline_benchmarking.log", encoding='utf-8'),
#     ],
#     force=True
# )
# logger = logging.getLogger(__name__)
#
#
# # ============================================================================
# # CONFIGURATION: CRITICAL SETTINGS
# # ============================================================================
#
# class BenchmarkConfig:
#     """Central configuration for all benchmark settings."""
#
#     # Time limits
#     TIME_LIMIT_PER_RUN_S = 300  # 5 minutes per problem per planner
#
#     # Fast Downward locations (ABSOLUTE PATHS)
#     FD_TRANSLATE_BIN = os.path.abspath("downward/builds/release/bin/translate/translate.py")
#     FD_DOWNWARD_BIN = os.path.abspath("downward/builds/release/bin/downward.exe")
#
#     # Output CSV file
#     OUTPUT_CSV = "baseline_performance_summary.csv"
#
#     # Working directories
#     FD_TEMP_DIR = "baseline_temp"  # Temporary SAS files
#
#     # Define benchmark sets (problems grouped by domain)
#     # For your MVP: just one domain with one problem
#     BENCHMARK_SETS = {
#         "Small": {
#             "domain": "domain.pddl",
#             "problem_pattern": "problem_small_*.pddl",
#             "description": "Small 8-puzzle problems"
#         }
#     }
#
#     # Define baseline planners to test
#     # Each planner has: name + search config string
#     BASELINES = [
#         {
#             "name": "FD ASTAR LM-Cut",
#             "search_config": "astar(lmcut())"
#         },
#         {
#             "name": "FD ASTAR DFP",
#             "search_config": (
#                 "astar(merge_and_shrink("
#                 "merge_strategy=merge_dfp(),"
#                 "shrink_strategy=shrink_bisimulation(greedy=false),"
#                 "label_reduction=exact(before_shrinking=true,before_merging=false),"
#                 "max_states=50000,threshold_before_merge=1))"
#             )
#         },
#         {
#             "name": "FD ASTAR SCC-DFP",
#             "search_config": (
#                 "astar(merge_and_shrink("
#                 "merge_strategy=merge_sccs(order_of_sccs=topological,"
#                 "merge_selector=score_based_filtering("
#                 "scoring_functions=[goal_relevance,dfp,total_order])),"
#                 "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
#                 "label_reduction=exact(before_shrinking=true,before_merging=false),"
#                 "max_states=50000,threshold_before_merge=1))"
#             )
#         },
#         {
#             "name": "FD ASTAR MIASM",
#             "search_config": (
#                 "astar(merge_and_shrink("
#                 "merge_strategy=merge_miasm(scoring_function=miasm_utils("
#                 "shrink_strategy=shrink_bisimulation(greedy=true,at_limit=return))),"
#                 "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
#                 "label_reduction=exact(before_shrinking=true,before_merging=false),"
#                 "max_states=50000,threshold_before_merge=1))"
#             )
#         },
#     ]
#
#
# # ============================================================================
# # STEP 1: VERIFY SETUP
# # ============================================================================
#
# def verify_fd_setup() -> bool:
#     """Verify that Fast Downward is properly installed."""
#     logger.info("\n" + "=" * 80)
#     logger.info("STEP 1: VERIFY FAST DOWNWARD SETUP")
#     logger.info("=" * 80)
#
#     checks = [
#         ("FD Translate Script", BenchmarkConfig.FD_TRANSLATE_BIN),
#         ("FD Downward Binary", BenchmarkConfig.FD_DOWNWARD_BIN),
#     ]
#
#     all_ok = True
#     for name, path in checks:
#         if os.path.exists(path):
#             logger.info(f"  ✓ {name:<30} {path}")
#         else:
#             logger.error(f"  ✗ {name:<30} NOT FOUND: {path}")
#             all_ok = False
#
#     if not all_ok:
#         logger.error("\n❌ FD setup verification FAILED")
#         logger.error("   Please ensure Fast Downward is built in downward/builds/release/")
#         return False
#
#     logger.info("\n✅ FD setup verified")
#     return True
#
#
# # ============================================================================
# # STEP 2: LOAD BENCHMARKS
# # ============================================================================
#
# def load_all_benchmarks() -> Dict[str, List[Tuple[str, str]]]:
#     """
#     Load all benchmark problems from disk.
#
#     Returns:
#         Dictionary mapping set_name -> list of (domain_file, problem_file) tuples
#     """
#     logger.info("\n" + "=" * 80)
#     logger.info("STEP 2: LOAD BENCHMARK PROBLEMS")
#     logger.info("=" * 80)
#
#     all_benchmarks = {}
#
#     for set_name, set_config in BenchmarkConfig.BENCHMARK_SETS.items():
#         logger.info(f"\nLoading benchmark set: {set_name}")
#
#         domain_file = set_config["domain"]
#         problem_pattern = set_config["problem_pattern"]
#
#         # Check domain exists
#         if not os.path.exists(domain_file):
#             logger.error(f"  ✗ Domain file not found: {domain_file}")
#             return {}
#         logger.info(f"  ✓ Domain: {domain_file}")
#
#         # Find problems
#         problems = sorted(glob.glob(problem_pattern))
#         if not problems:
#             logger.error(f"  ✗ No problems found matching: {problem_pattern}")
#             return {}
#
#         logger.info(f"  ✓ Found {len(problems)} problem(s)")
#         for i, prob in enumerate(problems, 1):
#             logger.info(f"    {i}. {prob}")
#
#         # Create benchmark list (absolute paths)
#         benchmarks = [
#             (os.path.abspath(domain_file), os.path.abspath(prob))
#             for prob in problems
#         ]
#
#         all_benchmarks[set_name] = benchmarks
#
#     if not all_benchmarks:
#         logger.error("\n❌ No benchmarks loaded")
#         return {}
#
#     logger.info(f"\n✅ Loaded {len(all_benchmarks)} benchmark set(s)")
#     return all_benchmarks
#
#
# # ============================================================================
# # STEP 3: RUN SINGLE PROBLEM
# # ============================================================================
#
# def run_single_fd_problem(
#         domain_file: str,
#         problem_file: str,
#         search_config: str,
#         time_limit: int,
#         temp_dir: str
# ) -> Dict[str, Any]:
#     """
#     Run Fast Downward on a SINGLE problem with a SINGLE search configuration.
#
#     This is the CORE WORKER function. It performs 3 steps:
#     1. TRANSLATE: Convert PDDL to SAS
#     2. SEARCH: Run planner with search config
#     3. PARSE: Extract metrics from output
#
#     Args:
#         domain_file: Path to domain.pddl
#         problem_file: Path to problem_N.pddl
#         search_config: Search strategy string (e.g., "astar(lmcut())")
#         time_limit: Timeout in seconds
#         temp_dir: Directory for temporary SAS files
#
#     Returns:
#         Dictionary with metrics or error info
#     """
#     problem_name = os.path.basename(problem_file)
#
#     try:
#         # ====== STEP 3a: TRANSLATE ======
#         logger.debug(f"    [TRANSLATE] Starting for {problem_name}...")
#
#         sas_file = os.path.join(temp_dir, "output.sas")
#
#         translate_cmd = (
#             f'python "{BenchmarkConfig.FD_TRANSLATE_BIN}" '
#             f'"{domain_file}" "{problem_file}" '
#             f'--sas-file "{sas_file}"'
#         )
#
#         logger.debug(f"    [TRANSLATE] Command: {translate_cmd[:100]}...")
#
#         result = subprocess.run(
#             translate_cmd,
#             shell=True,
#             cwd=temp_dir,
#             capture_output=True,
#             text=True,
#             timeout=time_limit
#         )
#
#         if result.returncode != 0:
#             logger.debug(f"    [TRANSLATE] Failed: {result.stderr[:200]}")
#             return {
#                 "solved": False,
#                 "reason": "translate_error",
#                 "error": result.stderr[:500]
#             }
#
#         if not os.path.exists(sas_file):
#             logger.debug(f"    [TRANSLATE] output.sas not created")
#             return {
#                 "solved": False,
#                 "reason": "translate_no_output"
#             }
#
#         logger.debug(f"    [TRANSLATE] Success ({os.path.getsize(sas_file)} bytes)")
#
#         # ====== STEP 3b: SEARCH ======
#         logger.debug(f"    [SEARCH] Starting with config: {search_config[:50]}...")
#
#         search_cmd = (
#             f'"{BenchmarkConfig.FD_DOWNWARD_BIN}" '
#             f'--search "{search_config}" '
#             f'< "{sas_file}"'
#         )
#
#         search_start = time.time()
#
#         result = subprocess.run(
#             search_cmd,
#             shell=True,
#             cwd=temp_dir,
#             capture_output=True,
#             text=True,
#             timeout=time_limit
#         )
#
#         search_time = time.time() - search_start
#
#         output_text = result.stdout + result.stderr
#
#         logger.debug(f"    [SEARCH] Completed in {search_time:.2f}s")
#
#         # ====== STEP 3c: PARSE ======
#         logger.debug(f"    [PARSE] Extracting metrics...")
#
#         # Check for solution
#         if "Solution found" not in output_text and "Plan length:" not in output_text:
#             logger.debug(f"    [PARSE] No solution found")
#             return {
#                 "solved": False,
#                 "reason": "no_solution",
#                 "time": search_time
#             }
#
#         logger.debug(f"    [PARSE] Solution detected!")
#
#         # Extract metrics
#         metrics = _parse_fd_output(output_text)
#
#         if metrics is None:
#             logger.debug(f"    [PARSE] Could not extract metrics")
#             return {
#                 "solved": True,  # Solution found but parse failed
#                 "reason": "parse_error",
#                 "time": search_time
#             }
#
#         metrics["solved"] = True
#         metrics["time"] = search_time
#
#         logger.debug(f"    [PARSE] Extracted: cost={metrics.get('cost', '?')}, "
#                      f"expansions={metrics.get('expansions', '?')}")
#
#         return metrics
#
#     except subprocess.TimeoutExpired:
#         logger.debug(f"    [TIMEOUT] Exceeded {time_limit}s")
#         return {
#             "solved": False,
#             "reason": "timeout",
#             "time": time_limit
#         }
#
#     except Exception as e:
#         logger.debug(f"    [ERROR] {e}")
#         return {
#             "solved": False,
#             "reason": "exception",
#             "error": str(e)[:200]
#         }
#
#
# def _parse_fd_output(output_text: str) -> Optional[Dict[str, Any]]:
#     """
#     Parse Fast Downward text output to extract metrics.
#
#     Looks for patterns like:
#     - "Plan length: 42"
#     - "Expanded 1234 states"
#     - "Search time: 1.23s"
#
#     Args:
#         output_text: Combined stdout + stderr from FD
#
#     Returns:
#         Dictionary with extracted metrics, or None if parsing fails
#     """
#     metrics = {}
#
#     # Extract plan cost (length)
#     match_cost = re.search(r"Plan length:\s*(\d+)", output_text)
#     if match_cost:
#         metrics["cost"] = int(match_cost.group(1))
#
#     # Extract expansions (all occurrences, take last)
#     matches_exp = list(re.finditer(r"Expanded\s+(\d+)\s+states?", output_text))
#     if matches_exp:
#         metrics["expansions"] = int(matches_exp[-1].group(1))
#
#     # Extract search time
#     matches_time = list(re.finditer(r"Search time:\s+([\d.]+)s", output_text))
#     if matches_time:
#         metrics["search_time"] = float(matches_time[-1].group(1))
#
#     # Require at least cost and expansions
#     if "cost" not in metrics or "expansions" not in metrics:
#         return None
#
#     return metrics
#
#
# # ============================================================================
# # STEP 4: RUN BASELINE PLANNER (all problems)
# # ============================================================================
#
# def run_baseline_on_benchmark_set(
#         domain_file: str,
#         problem_files: List[str],
#         baseline_name: str,
#         search_config: str
# ) -> Dict[str, Any]:
#     """
#     Run ONE baseline planner on ONE benchmark set (all its problems).
#
#     This aggregates results across multiple problems.
#
#     Args:
#         domain_file: Absolute path to domain.pddl
#         problem_files: List of absolute paths to problem files
#         baseline_name: Human-readable name for report
#         search_config: FD search strategy string
#
#     Returns:
#         Dictionary with aggregate metrics for this baseline
#     """
#     logger.info(f"\n  Running baseline: {baseline_name}")
#     logger.info(f"  Problems: {len(problem_files)}")
#
#     # Create temp directory
#     temp_dir = os.path.join(BenchmarkConfig.FD_TEMP_DIR, baseline_name.replace(" ", "_"))
#     os.makedirs(temp_dir, exist_ok=True)
#
#     try:
#         solved_count = 0
#         times_on_solved = []
#         expansions_on_solved = []
#         costs_on_solved = []
#         errors = []
#
#         # Run each problem
#         # Run each problem
#         for i, problem_file in enumerate(problem_files, 1):
#             problem_name = os.path.basename(problem_file)
#
#             # (No log message here... run the problem first)
#
#             result = run_single_fd_problem(
#                 domain_file=domain_file,
#                 problem_file=problem_file,
#                 search_config=search_config,
#                 time_limit=BenchmarkConfig.TIME_LIMIT_PER_RUN_S,
#                 temp_dir=temp_dir
#             )
#
#             if result.get("solved"):
#                 solved_count += 1
#                 times_on_solved.append(result.get("time", 0))
#                 expansions_on_solved.append(result.get("expansions", 0))
#                 costs_on_solved.append(result.get("cost", 0))
#
#                 # Log the FULL line at once
#                 logger.info(
#                     f"    [{i}/{len(problem_files)}] {problem_name:<30} ✓ SOLVED (t={result.get('time', 0):.2f}s)")
#             else:
#                 reason = result.get("reason", "unknown")
#
#                 # Log the FULL line at once
#                 logger.warning(f"    [{i}/{len(problem_files)}] {problem_name:<30} ✗ {reason.upper()}")
#                 errors.append(reason)
#
#         # ====== AGGREGATE RESULTS ======
#         total_problems = len(problem_files)
#         solve_rate = (solved_count / total_problems * 100) if total_problems > 0 else 0
#
#         avg_time = sum(times_on_solved) / len(times_on_solved) if times_on_solved else 0
#         avg_expansions = sum(expansions_on_solved) / len(expansions_on_solved) if expansions_on_solved else 0
#         avg_cost = sum(costs_on_solved) / len(costs_on_solved) if costs_on_solved else 0
#
#         summary = {
#             "name": baseline_name,
#             "set_size": total_problems,
#             "solved": solved_count,
#             "solve_rate_%": solve_rate,
#             "avg_time_on_solved_s": avg_time,
#             "avg_expansions_on_solved": int(avg_expansions),
#             "avg_cost_on_solved": int(avg_cost),
#             "errors": ", ".join(set(errors)) if errors else "None"
#         }
#
#         logger.info(f"  → Solved: {solved_count}/{total_problems} ({solve_rate:.1f}%)")
#         logger.info(f"  → Avg time (solved): {avg_time:.2f}s")
#         logger.info(f"  → Avg expansions: {int(avg_expansions)}")
#
#         return summary
#
#     finally:
#         # Cleanup temp directory
#         try:
#             shutil.rmtree(temp_dir)
#         except:
#             pass
#
#
# # ============================================================================
# # STEP 5: GENERATE REPORT
# # ============================================================================
#
# def generate_report(all_results: List[Dict[str, Any]]) -> None:
#     """
#     Generate CSV report from all baseline results.
#
#     Saves to: baseline_performance_summary.csv
#
#     Args:
#         all_results: List of result dictionaries from all baselines
#     """
#     logger.info("\n" + "=" * 80)
#     logger.info("STEP 5: GENERATE REPORT")
#     logger.info("=" * 80)
#
#     if not all_results:
#         logger.error("No results to report")
#         return
#
#     output_csv = BenchmarkConfig.OUTPUT_CSV
#
#     # Define column order
#     fieldnames = [
#         "name",
#         "set_size",
#         "solved",
#         "solve_rate_%",
#         "avg_time_on_solved_s",
#         "avg_expansions_on_solved",
#         "avg_cost_on_solved",
#         "errors"
#     ]
#
#     logger.info(f"\nWriting report to: {output_csv}")
#
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_results)
#
#     logger.info(f"✓ Report written ({len(all_results)} baselines)")
#
#     # Print summary table
#     logger.info("\n" + "=" * 80)
#     logger.info("BASELINE PERFORMANCE SUMMARY")
#     logger.info("=" * 80)
#
#     logger.info(f"\n{'Planner':<35} {'Solved':<15} {'Avg Time (s)':<15} {'Avg Expansions':<15}")
#     logger.info("-" * 80)
#
#     for result in all_results:
#         name = result["name"][:33]
#         solved = f"{result['solved']}/{result['set_size']} ({result['solve_rate_%']:.0f}%)"
#         time_str = f"{result['avg_time_on_solved_s']:.2f}"
#         exp_str = f"{result['avg_expansions_on_solved']:,}"
#         logger.info(f"{name:<35} {solved:<15} {time_str:<15} {exp_str:<15}")
#
#     logger.info("-" * 80)
#     logger.info(f"\nFull report saved: {os.path.abspath(output_csv)}")
#
#
# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================
#
# def main():
#     """Main execution orchestration."""
#     logger.info("\n" + "=" * 80)
#     logger.info("BASELINE BENCHMARKING HARNESS FOR GNN MERGE STRATEGY LEARNING")
#     logger.info("=" * 80)
#
#     # Step 1: Verify FD setup
#     if not verify_fd_setup():
#         return 1
#
#     # Step 2: Load benchmarks
#     all_benchmarks = load_all_benchmarks()
#     if not all_benchmarks:
#         return 1
#
#     # Step 3: Run all baselines on all benchmark sets
#     logger.info("\n" + "=" * 80)
#     logger.info("STEP 3: RUN BASELINE PLANNERS")
#     logger.info("=" * 80)
#
#     all_results = []
#
#     for set_name, benchmarks in all_benchmarks.items():
#         domain_file = benchmarks[0][0]  # Same domain for all
#         problem_files = [b[1] for b in benchmarks]  # All problems
#
#         logger.info(f"\n{set_name} Benchmark Set")
#         logger.info("-" * 80)
#
#         for baseline_config in BenchmarkConfig.BASELINES:
#             result = run_baseline_on_benchmark_set(
#                 domain_file=domain_file,
#                 problem_files=problem_files,
#                 baseline_name=baseline_config["name"],
#                 search_config=baseline_config["search_config"]
#             )
#             all_results.append(result)
#
#     # Step 4: Generate report
#     generate_report(all_results)
#
#     # Summary
#     logger.info("\n" + "=" * 80)
#     logger.info("✅ BENCHMARKING COMPLETE")
#     logger.info("=" * 80)
#     logger.info(f"\nNext steps:")
#     logger.info(f"  1. Review CSV: {BenchmarkConfig.OUTPUT_CSV}")
#     logger.info(f"  2. Use these metrics as BASELINE for GNN policy comparison")
#     logger.info(f"  3. Train GNN with: python train_mvp_debug.py")
#     logger.info(f"  4. Evaluate GNN and compare against these baselines")
#
#     return 0
#
#
# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASELINE BENCHMARKING HARNESS FOR GNN MERGE STRATEGY LEARNING
==============================================================

This script benchmarks STANDARD Fast Downward algorithms on your problem set
and produces a CSV report for comparison with your GNN policy.

Usage:
    python baseline_benchmarking.py

Output:
    baseline_performance_summary.csv
"""

import sys
import os
import logging
import glob
import json
import subprocess
import time
import re
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import csv

# Setup
sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("baseline_benchmarking.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION: CRITICAL SETTINGS
# ============================================================================

class BenchmarkConfig:
    """Central configuration for all benchmark settings."""

    # Time limits
    TIME_LIMIT_PER_RUN_S = 300  # 5 minutes per problem per planner

    # ✅ FIXED: Use absolute path to downward directory
    DOWNWARD_DIR = os.path.abspath("downward")
    FD_TRANSLATE_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/translate/translate.py")
    FD_DOWNWARD_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/downward.exe")

    # Output CSV file
    OUTPUT_CSV = "evaluation_results/baseline_performance_summary.csv"

    # Working directories
    FD_TEMP_DIR = "baseline_temp"  # Temporary SAS files

    # Define benchmark sets (problems grouped by domain)
    BENCHMARK_SETS = {
        "Small": {
            "domain": "domain.pddl",
            "problem_pattern": "problem_small_*.pddl",
            "description": "Small 8-puzzle problems"
        }
    }

    # Define baseline planners to test
    BASELINES = [
        {
            "name": "FD ASTAR LM-Cut",
            "search_config": "astar(lmcut())"
        },
        {
            "name": "FD ASTAR DFP (Stateless)",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_stateless("
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"  # ✅ CORRECT: () added
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            )
        },
        {
            "name": "FD ASTAR SCC-DFP",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_sccs("  # ✅ Use merge_sccs for SCC-DFP
                "order_of_sccs=topological,"
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"  # ✅ () added
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            )
        },
        {
            "name": "FD ASTAR Bisimulation",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_stateless("
                "merge_selector=score_based_filtering("
                "scoring_functions=[total_order()])),"  # ✅ () added
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            )
        },
        {
            "name": "FD ASTAR Blind",
            "search_config": "astar(blind())"
        },
        {
            "name": "FD ASTAR Add Heuristic",
            "search_config": "astar(add())"
        },
        {
            "name": "FD ASTAR Max Heuristic",
            "search_config": "astar(max())"
        },
    ]


# ============================================================================
# STEP 1: VERIFY SETUP
# ============================================================================

def verify_fd_setup() -> bool:
    """Verify that Fast Downward is properly installed."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: VERIFY FAST DOWNWARD SETUP")
    logger.info("=" * 80)

    checks = [
        ("FD Translate Script", BenchmarkConfig.FD_TRANSLATE_BIN),
        ("FD Downward Binary", BenchmarkConfig.FD_DOWNWARD_BIN),
    ]

    all_ok = True
    for name, path in checks:
        if os.path.exists(path):
            logger.info(f"  ✓ {name:<30} {path}")
        else:
            logger.error(f"  ✗ {name:<30} NOT FOUND: {path}")
            all_ok = False

    if not all_ok:
        logger.error("\n❌ FD setup verification FAILED")
        logger.error("   Please ensure Fast Downward is built in downward/builds/release/")
        return False

    logger.info("\n✅ FD setup verified")
    logger.info("\nRunning simplified baseline planners:")
    logger.info("  - LM-Cut (landmark-based heuristic)")
    logger.info("  - Blind (uninformed A*)")
    logger.info("  - Add (additive heuristic)")
    logger.info("  - Max (max heuristic)")
    logger.info("\nNote: Complex M&S strategies require specific compilation flags.")
    return True


# ============================================================================
# STEP 2: LOAD BENCHMARKS
# ============================================================================

def load_all_benchmarks() -> Dict[str, List[Tuple[str, str]]]:
    """
    Load all benchmark problems from disk.

    Returns:
        Dictionary mapping set_name -> list of (domain_file, problem_file) tuples
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: LOAD BENCHMARK PROBLEMS")
    logger.info("=" * 80)

    all_benchmarks = {}

    for set_name, set_config in BenchmarkConfig.BENCHMARK_SETS.items():
        logger.info(f"\nLoading benchmark set: {set_name}")

        domain_file = set_config["domain"]
        problem_pattern = set_config["problem_pattern"]

        # Check domain exists
        if not os.path.exists(domain_file):
            logger.error(f"  ✗ Domain file not found: {domain_file}")
            return {}
        logger.info(f"  ✓ Domain: {domain_file}")

        # Find problems
        problems = sorted(glob.glob(problem_pattern))
        if not problems:
            logger.error(f"  ✗ No problems found matching: {problem_pattern}")
            return {}

        logger.info(f"  ✓ Found {len(problems)} problem(s)")
        for i, prob in enumerate(problems, 1):
            logger.info(f"    {i}. {prob}")

        # Create benchmark list (absolute paths)
        benchmarks = [
            (os.path.abspath(domain_file), os.path.abspath(prob))
            for prob in problems
        ]

        all_benchmarks[set_name] = benchmarks

    if not all_benchmarks:
        logger.error("\n❌ No benchmarks loaded")
        return {}

    logger.info(f"\n✅ Loaded {len(all_benchmarks)} benchmark set(s)")
    return all_benchmarks


# ============================================================================
# STEP 3: RUN SINGLE PROBLEM
# ============================================================================

def run_single_fd_problem(
        domain_file: str,
        problem_file: str,
        search_config: str,
        time_limit: int,
        temp_dir: str
) -> Dict[str, Any]:
    """
    Run Fast Downward on a SINGLE problem with a SINGLE search configuration.

    ✅ FIXED: Complete diagnostic logging and proper working directory handling.
    """
    problem_name = os.path.basename(problem_file)

    try:
        # ====== STEP 3a: TRANSLATE ======
        logger.info(f"    [TRANSLATE] Starting for {problem_name}...")
        logger.info(f"    [TRANSLATE] Domain:  {os.path.abspath(domain_file)}")
        logger.info(f"    [TRANSLATE] Problem: {os.path.abspath(problem_file)}")

        os.makedirs(temp_dir, exist_ok=True)
        sas_file = os.path.join(temp_dir, "output.sas")

        # ✅ FIX: Use absolute paths for all file arguments
        abs_domain = os.path.abspath(domain_file)
        abs_problem = os.path.abspath(problem_file)
        abs_sas = os.path.abspath(sas_file)
        abs_translate_bin = os.path.abspath(BenchmarkConfig.FD_TRANSLATE_BIN)

        translate_cmd = (
            f'python "{abs_translate_bin}" '
            f'"{abs_domain}" "{abs_problem}" '
            f'--sas-file "{abs_sas}"'
        )

        logger.info(f"    [TRANSLATE] Command: {translate_cmd[:150]}...")

        # ✅ FIX: Run from PROJECT ROOT, not downward/
        # This ensures translate.py finds all its dependencies
        result = subprocess.run(
            translate_cmd,
            shell=True,
            cwd=os.path.abspath("."),  # ✅ CRITICAL: Run from project root
            capture_output=True,
            text=True,
            timeout=time_limit
        )

        if result.returncode != 0:
            # ✅ FIX: Log at INFO level so errors are always visible
            logger.info(f"    [TRANSLATE] ❌ FAILED with return code {result.returncode}")
            logger.info(f"    [TRANSLATE] STDOUT:\n{result.stdout}")
            logger.info(f"    [TRANSLATE] STDERR:\n{result.stderr}")
            return {
                "solved": False,
                "reason": "translate_error",
                "error": (result.stderr if result.stderr else result.stdout)[:500]
            }

        if not os.path.exists(abs_sas):
            logger.error(f"    [TRANSLATE] ❌ output.sas not created")
            logger.error(f"    [TRANSLATE] Expected at: {abs_sas}")
            logger.error(f"    [TRANSLATE] Temp dir exists: {os.path.exists(temp_dir)}")
            logger.error(
                f"    [TRANSLATE] Files in temp dir: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'N/A'}")
            return {
                "solved": False,
                "reason": "translate_no_output"
            }

        # ✅ FIX: Validate SAS file has content
        sas_size = os.path.getsize(abs_sas)
        if sas_size == 0:
            logger.error(f"    [TRANSLATE] ❌ output.sas is EMPTY (0 bytes)")
            return {
                "solved": False,
                "reason": "translate_empty_output"
            }

        logger.info(f"    [TRANSLATE] ✅ Success ({sas_size} bytes)")

        # ====== STEP 3b: SEARCH ======
        logger.info(f"    [SEARCH] Starting with config: {search_config[:50]}...")

        abs_downward_bin = os.path.abspath(BenchmarkConfig.FD_DOWNWARD_BIN)

        search_cmd = (
            f'"{abs_downward_bin}" '
            f'--search "{search_config}" '
            f'< "{abs_sas}"'
        )

        search_start = time.time()

        # ✅ RUN from downward/ directory (FD's preferred working directory)
        result = subprocess.run(
            search_cmd,
            shell=True,
            cwd=os.path.dirname(abs_downward_bin),  # downward/builds/release/bin/
            capture_output=True,
            text=True,
            timeout=time_limit
        )

        search_time = time.time() - search_start

        output_text = result.stdout + result.stderr

        if result.returncode != 0:
            logger.info(f"    [SEARCH] ⚠️  Non-zero return code: {result.returncode}")
            logger.info(f"    [SEARCH] Output:\n{output_text[:500]}")

        logger.info(f"    [SEARCH] ✅ Completed in {search_time:.2f}s")

        # ====== STEP 3c: PARSE ======
        logger.info(f"    [PARSE] Extracting metrics...")

        # Check for solution
        if "Solution found" not in output_text and "Plan length:" not in output_text:
            logger.info(f"    [PARSE] ❌ No solution found in output")
            return {
                "solved": False,
                "reason": "no_solution",
                "time": search_time
            }

        logger.info(f"    [PARSE] ✅ Solution detected!")

        # Extract metrics
        metrics = _parse_fd_output(output_text)

        if metrics is None:
            logger.warning(f"    [PARSE] ⚠️  Could not extract metrics from solution")
            return {
                "solved": True,
                "reason": "parse_error",
                "time": search_time
            }

        metrics["solved"] = True
        metrics["time"] = search_time

        logger.info(f"    [PARSE] ✅ Extracted: cost={metrics.get('cost', '?')}, "
                    f"expansions={metrics.get('expansions', '?')}")

        return metrics

    except subprocess.TimeoutExpired:
        logger.warning(f"    [TIMEOUT] ❌ Exceeded {time_limit}s")
        return {
            "solved": False,
            "reason": "timeout",
            "time": time_limit
        }

    except Exception as e:
        logger.error(f"    [ERROR] ❌ {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "solved": False,
            "reason": "exception",
            "error": str(e)[:200]
        }


def _parse_fd_output(output_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse Fast Downward text output to extract metrics.

    Looks for patterns like:
    - "Plan length: 42"
    - "Expanded 1234 states"
    - "Search time: 1.23s"

    Args:
        output_text: Combined stdout + stderr from FD

    Returns:
        Dictionary with extracted metrics, or None if parsing fails
    """
    metrics = {}

    # Extract plan cost (length)
    match_cost = re.search(r"Plan length:\s*(\d+)", output_text)
    if match_cost:
        metrics["cost"] = int(match_cost.group(1))

    # Extract expansions (all occurrences, take last)
    matches_exp = list(re.finditer(r"Expanded\s+(\d+)\s+states?", output_text))
    if matches_exp:
        metrics["expansions"] = int(matches_exp[-1].group(1))

    # Extract search time
    matches_time = list(re.finditer(r"Search time:\s+([\d.]+)s", output_text))
    if matches_time:
        metrics["search_time"] = float(matches_time[-1].group(1))

    # Require at least cost and expansions
    if "cost" not in metrics or "expansions" not in metrics:
        return None

    return metrics


# ============================================================================
# STEP 4: RUN BASELINE PLANNER (all problems)
# ============================================================================

def run_baseline_on_benchmark_set(
        domain_file: str,
        problem_files: List[str],
        baseline_name: str,
        search_config: str
) -> Dict[str, Any]:
    """
    Run ONE baseline planner on ONE benchmark set (all its problems).

    This aggregates results across multiple problems.

    Args:
        domain_file: Absolute path to domain.pddl
        problem_files: List of absolute paths to problem files
        baseline_name: Human-readable name for report
        search_config: FD search strategy string

    Returns:
        Dictionary with aggregate metrics for this baseline
    """
    logger.info(f"\n  Running baseline: {baseline_name}")
    logger.info(f"  Problems: {len(problem_files)}")

    # Create temp directory
    temp_dir = os.path.join(BenchmarkConfig.FD_TEMP_DIR, baseline_name.replace(" ", "_"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        solved_count = 0
        times_on_solved = []
        expansions_on_solved = []
        costs_on_solved = []
        errors = []

        # Run each problem
        for i, problem_file in enumerate(problem_files, 1):
            problem_name = os.path.basename(problem_file)

            result = run_single_fd_problem(
                domain_file=domain_file,
                problem_file=problem_file,
                search_config=search_config,
                time_limit=BenchmarkConfig.TIME_LIMIT_PER_RUN_S,
                temp_dir=temp_dir
            )

            if result.get("solved"):
                solved_count += 1
                times_on_solved.append(result.get("time", 0))
                expansions_on_solved.append(result.get("expansions", 0))
                costs_on_solved.append(result.get("cost", 0))

                logger.info(
                    f"    [{i}/{len(problem_files)}] {problem_name:<30} ✓ SOLVED (t={result.get('time', 0):.2f}s)")
            else:
                reason = result.get("reason", "unknown")

                logger.warning(f"    [{i}/{len(problem_files)}] {problem_name:<30} ✗ {reason.upper()}")

                # ✅ FIXED: Log error details for debugging
                if "error" in result:
                    logger.debug(f"        Error details: {result['error']}")

                errors.append(reason)

        # ====== AGGREGATE RESULTS ======
        total_problems = len(problem_files)
        solve_rate = (solved_count / total_problems * 100) if total_problems > 0 else 0

        avg_time = sum(times_on_solved) / len(times_on_solved) if times_on_solved else 0
        avg_expansions = sum(expansions_on_solved) / len(expansions_on_solved) if expansions_on_solved else 0
        avg_cost = sum(costs_on_solved) / len(costs_on_solved) if costs_on_solved else 0

        summary = {
            "name": baseline_name,
            "set_size": total_problems,
            "solved": solved_count,
            "solve_rate_%": solve_rate,
            "avg_time_on_solved_s": avg_time,
            "avg_expansions_on_solved": int(avg_expansions),
            "avg_cost_on_solved": int(avg_cost),
            "errors": ", ".join(set(errors)) if errors else "None"
        }

        logger.info(f"  → Solved: {solved_count}/{total_problems} ({solve_rate:.1f}%)")
        logger.info(f"  → Avg time (solved): {avg_time:.2f}s")
        logger.info(f"  → Avg expansions: {int(avg_expansions)}")

        return summary

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ============================================================================
# STEP 5: GENERATE REPORT
# ============================================================================

def generate_report(all_results: List[Dict[str, Any]]) -> None:
    """
    Generate CSV report from all baseline results.

    Saves to: baseline_performance_summary.csv

    Args:
        all_results: List of result dictionaries from all baselines
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: GENERATE REPORT")
    logger.info("=" * 80)

    if not all_results:
        logger.error("No results to report")
        return

    output_csv = BenchmarkConfig.OUTPUT_CSV

    # Define column order
    fieldnames = [
        "name",
        "set_size",
        "solved",
        "solve_rate_%",
        "avg_time_on_solved_s",
        "avg_expansions_on_solved",
        "avg_cost_on_solved",
        "errors"
    ]

    logger.info(f"\nWriting report to: {output_csv}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"✓ Report written ({len(all_results)} baselines)")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Planner':<35} {'Solved':<15} {'Avg Time (s)':<15} {'Avg Expansions':<15}")
    logger.info("-" * 80)

    for result in all_results:
        name = result["name"][:33]
        solved = f"{result['solved']}/{result['set_size']} ({result['solve_rate_%']:.0f}%)"
        time_str = f"{result['avg_time_on_solved_s']:.2f}"
        exp_str = f"{result['avg_expansions_on_solved']:,}"
        logger.info(f"{name:<35} {solved:<15} {time_str:<15} {exp_str:<15}")

    logger.info("-" * 80)
    logger.info(f"\nFull report saved: {os.path.abspath(output_csv)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution orchestration."""
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE BENCHMARKING HARNESS FOR GNN MERGE STRATEGY LEARNING")
    logger.info("=" * 80)

    # Step 1: Verify FD setup
    if not verify_fd_setup():
        return 1

    # Step 2: Load benchmarks
    all_benchmarks = load_all_benchmarks()
    if not all_benchmarks:
        return 1

    # Step 3: Run all baselines on all benchmark sets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: RUN BASELINE PLANNERS")
    logger.info("=" * 80)

    all_results = []

    for set_name, benchmarks in all_benchmarks.items():
        domain_file = benchmarks[0][0]  # Same domain for all
        problem_files = [b[1] for b in benchmarks]  # All problems

        logger.info(f"\n{set_name} Benchmark Set")
        logger.info("-" * 80)

        for baseline_config in BenchmarkConfig.BASELINES:
            result = run_baseline_on_benchmark_set(
                domain_file=domain_file,
                problem_files=problem_files,
                baseline_name=baseline_config["name"],
                search_config=baseline_config["search_config"]
            )
            all_results.append(result)

    # Step 4: Generate report
    generate_report(all_results)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ BENCHMARKING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review CSV: {BenchmarkConfig.OUTPUT_CSV}")
    logger.info(f"  2. Use these metrics as BASELINE for GNN policy comparison")
    logger.info(f"  3. Train GNN with: python train_mvp_debug.py")
    logger.info(f"  4. Evaluate GNN and compare against these baselines")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)