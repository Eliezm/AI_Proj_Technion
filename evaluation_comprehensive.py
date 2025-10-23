#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE EVALUATION FRAMEWORK
===================================
Unified evaluation harness that runs both baselines and trained GNN policy
on the same benchmark set, with detailed comparison metrics.

Features:
  ✓ Baseline planner evaluation (FD with various search configs)
  ✓ Trained GNN policy evaluation (using MergeEnv with real FD)
  ✓ Side-by-side comparison with statistical analysis
  ✓ CSV and JSON output formats
  ✓ Performance visualizations
  ✓ Detailed logging and diagnostics

Usage:
    python evaluation_comprehensive.py \
        --model mvp_output/gnn_model.zip \
        --domain domain.pddl \
        --problems "problem_small_*.pddl" \
        --timeout 300 \
        --output results/

Output:
    results/
    ├── evaluation_results.csv
    ├── evaluation_summary.json
    ├── comparison_report.txt
    └── plots/
        ├── solve_rate_comparison.png
        ├── time_comparison.png
        ├── expansions_comparison.png
        └── efficiency_frontier.png
"""

import sys
import os
import logging
import glob
import json
import subprocess
import time
import re
import csv
import argparse
import numpy as np
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Setup paths
sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_comprehensive.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION: BASELINE SETTINGS
# ============================================================================

class BenchmarkConfig:
    """Central configuration for all benchmark settings."""

    # Time limits
    TIME_LIMIT_PER_RUN_S = 300  # 5 minutes per problem per planner

    # ✅ FIXED: Use absolute path to downward directory
    DOWNWARD_DIR = os.path.abspath("downward")
    FD_TRANSLATE_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/translate/translate.py")
    FD_DOWNWARD_BIN = os.path.join(DOWNWARD_DIR, "builds/release/bin/downward.exe")

    # Working directories
    FD_TEMP_DIR = "evaluation_temp"  # Temporary SAS files

    # ✅ CORRECT BASELINE CONFIGURATIONS
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
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"
                "shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),"
                "label_reduction=exact(before_shrinking=true,before_merging=false),"
                "max_states=50000,threshold_before_merge=1))"
            )
        },
        {
            "name": "FD ASTAR SCC-DFP",
            "search_config": (
                "astar(merge_and_shrink("
                "merge_strategy=merge_sccs("
                "order_of_sccs=topological,"
                "merge_selector=score_based_filtering("
                "scoring_functions=[goal_relevance(),dfp(),total_order()])),"
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
                "scoring_functions=[total_order()])),"
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
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProblemResult:
    """Results for a single problem-planner combination."""
    problem_name: str
    planner_name: str
    solved: bool
    time_sec: float
    plan_cost: int = 0
    expansions: int = 0
    nodes_expanded: int = 0
    search_depth: int = 0
    error_reason: Optional[str] = None

    @property
    def efficiency(self) -> float:
        """Compute efficiency metric: plan_cost / expansions (lower is better)."""
        if not self.solved or self.expansions == 0:
            return float('inf')
        return self.plan_cost / self.expansions

    @property
    def speed(self) -> float:
        """Compute speed metric: expansions / time (higher is better)."""
        if self.time_sec == 0:
            return 0.0
        return self.expansions / self.time_sec

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        d = asdict(self)
        d['efficiency'] = self.efficiency
        d['speed'] = self.speed
        return d


@dataclass
class BenchmarkSummary:
    """Aggregate statistics for a benchmark set."""
    planner_name: str
    num_problems: int
    num_solved: int
    solve_rate_pct: float
    avg_time_sec: float
    median_time_sec: float
    avg_expansions: int
    median_expansions: int
    avg_plan_cost: int
    median_plan_cost: int
    total_time_sec: float
    avg_efficiency: float
    avg_speed: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BASELINE RUNNER (from evaluation_comprehensive.py - unchanged)
# ============================================================================

class BaselineRunner:
    """Runs baseline FD planners with correct calling conventions."""

    def __init__(self, timeout_sec: int = 300):
        self.timeout_sec = timeout_sec
        self.fd_bin = os.path.abspath(BenchmarkConfig.FD_DOWNWARD_BIN)
        self.fd_translate = os.path.abspath(BenchmarkConfig.FD_TRANSLATE_BIN)

        if not os.path.exists(self.fd_bin):
            raise FileNotFoundError(f"FD binary not found: {self.fd_bin}")
        if not os.path.exists(self.fd_translate):
            raise FileNotFoundError(f"FD translator not found: {self.fd_translate}")

    def run(self, domain_file: str, problem_file: str, search_config: str) -> ProblemResult:
        """Run FD with given search configuration."""
        problem_name = os.path.basename(problem_file)

        try:
            # ====== STEP 1: TRANSLATE ======
            logger.info(f"    [TRANSLATE] Starting for {problem_name}...")
            logger.debug(f"    [TRANSLATE] Domain:  {os.path.abspath(domain_file)}")
            logger.debug(f"    [TRANSLATE] Problem: {os.path.abspath(problem_file)}")

            work_dir = os.path.abspath(BenchmarkConfig.FD_TEMP_DIR)
            os.makedirs(work_dir, exist_ok=True)
            sas_file = os.path.join(work_dir, "output.sas")

            # ✅ FIX: Use absolute paths for all file arguments
            abs_domain = os.path.abspath(domain_file)
            abs_problem = os.path.abspath(problem_file)
            abs_sas = os.path.abspath(sas_file)
            abs_translate_bin = os.path.abspath(self.fd_translate)

            translate_cmd = (
                f'python "{abs_translate_bin}" '
                f'"{abs_domain}" "{abs_problem}" '
                f'--sas-file "{abs_sas}"'
            )

            logger.debug(f"    [TRANSLATE] Command: {translate_cmd[:150]}...")

            result = subprocess.run(
                translate_cmd,
                shell=True,
                cwd=os.path.abspath("."),  # ✅ CRITICAL: Run from project root
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            if result.returncode != 0:
                logger.debug(f"    [TRANSLATE] ❌ FAILED with return code {result.returncode}")
                return ProblemResult(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    time_sec=0,
                    error_reason="translate_error"
                )

            if not os.path.exists(abs_sas):
                logger.debug(f"    [TRANSLATE] ❌ output.sas not created")
                return ProblemResult(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    time_sec=0,
                    error_reason="translate_no_output"
                )

            sas_size = os.path.getsize(abs_sas)
            if sas_size == 0:
                logger.debug(f"    [TRANSLATE] ❌ output.sas is EMPTY (0 bytes)")
                return ProblemResult(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    time_sec=0,
                    error_reason="translate_empty_output"
                )

            logger.debug(f"    [TRANSLATE] ✅ Success ({sas_size} bytes)")

            # ====== STEP 2: SEARCH ======
            logger.debug(f"    [SEARCH] Starting with config: {search_config[:50]}...")

            abs_downward_bin = os.path.abspath(self.fd_bin)

            search_cmd = (
                f'"{abs_downward_bin}" '
                f'--search "{search_config}" '
                f'< "{abs_sas}"'
            )

            search_start = time.time()

            result = subprocess.run(
                search_cmd,
                shell=True,
                cwd=os.path.dirname(abs_downward_bin),  # downward/builds/release/bin/
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            elapsed = time.time() - search_start

            output_text = result.stdout + result.stderr

            if result.returncode != 0:
                logger.debug(f"    [SEARCH] ⚠️  Non-zero return code: {result.returncode}")

            logger.debug(f"    [SEARCH] ✅ Completed in {elapsed:.2f}s")

            # ====== STEP 3: PARSE ======
            logger.debug(f"    [PARSE] Extracting metrics...")

            # Check for solution
            if "Solution found" not in output_text and "Plan length:" not in output_text:
                logger.debug(f"    [PARSE] ❌ No solution found")
                return ProblemResult(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    time_sec=elapsed,
                    error_reason="no_solution"
                )

            logger.debug(f"    [PARSE] ✅ Solution detected!")

            # Extract metrics
            metrics = self._parse_fd_output(output_text)

            if metrics is None:
                logger.debug(f"    [PARSE] ⚠️  Could not extract metrics")
                return ProblemResult(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=True,
                    time_sec=elapsed,
                    error_reason="parse_error"
                )

            metrics["solved"] = True
            metrics["time"] = elapsed

            logger.debug(f"    [PARSE] ✅ Extracted: cost={metrics.get('cost', '?')}, "
                         f"expansions={metrics.get('expansions', '?')}")

            return ProblemResult(
                problem_name=problem_name,
                planner_name="FD",
                solved=True,
                time_sec=elapsed,
                plan_cost=metrics.get('cost', 0),
                expansions=metrics.get('expansions', 0)
            )

        except subprocess.TimeoutExpired:
            logger.debug(f"    [TIMEOUT] ❌ Exceeded {self.timeout_sec}s")
            return ProblemResult(
                problem_name=problem_name,
                planner_name="FD",
                solved=False,
                time_sec=self.timeout_sec,
                error_reason="timeout"
            )
        except Exception as e:
            logger.error(f"    [ERROR] ❌ {e}")
            return ProblemResult(
                problem_name=problem_name,
                planner_name="FD",
                solved=False,
                time_sec=0,
                error_reason=str(e)[:50]
            )

    @staticmethod
    def _parse_fd_output(output_text: str) -> Optional[Dict[str, Any]]:
        """Parse FD output."""
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
# GNN POLICY RUNNER - ✅ COMPLETELY REFACTORED
# ============================================================================

class GNNPolicyRunner:
    """
    ✅ FIXED: Runs trained GNN policy using MergeEnv with REAL FD.
    Uses the exact same logic as load_and_solve_with_gnn_complete.py
    """

    def __init__(self, model_path: str, timeout_sec: int = 300):
        self.model_path = model_path
        self.timeout_sec = timeout_sec

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Import here to avoid dependency errors
        from stable_baselines3 import PPO
        from merge_env import MergeEnv

        self.PPO = PPO
        self.MergeEnv = MergeEnv

    def run(self, domain_file: str, problem_file: str) -> ProblemResult:
        """
        ✅ FIXED: Run GNN policy on a problem using MergeEnv with real FD.

        This method:
        1. Loads the trained GNN model
        2. Creates a MergeEnv (with REAL FD, not debug mode)
        3. Runs the inference loop with the GNN policy
        4. Extracts metrics from FD output files
        5. Returns a ProblemResult with all metrics
        """
        problem_name = os.path.basename(problem_file)

        try:
            # ====================================================================
            # PHASE 1: LOAD MODEL
            # ====================================================================

            logger.info(f"\n{'=' * 80}")
            logger.info(f"GNN POLICY: {problem_name}")
            logger.info(f"{'=' * 80}\n")

            logger.info(f"Loading model from {self.model_path}...")
            model = self.PPO.load(self.model_path)
            logger.info("✓ Model loaded successfully\n")

            # ====================================================================
            # PHASE 2: CREATE ENVIRONMENT (REAL FD MODE)
            # ====================================================================

            logger.info("Creating environment (REAL FD MODE)...")
            logger.info(f"  Domain:  {os.path.abspath(domain_file)}")
            logger.info(f"  Problem: {os.path.abspath(problem_file)}")

            env = self.MergeEnv(
                domain_file=os.path.abspath(domain_file),
                problem_file=os.path.abspath(problem_file),
                max_merges=50,
                debug=False,  # ✅ REAL MODE: actual FD interaction
                reward_variant='astar_search',
                w_search_efficiency=0.30,
                w_solution_quality=0.20,
                w_f_stability=0.35,
                w_state_control=0.15,
            )

            logger.info("✓ Environment created\n")

            # ====================================================================
            # PHASE 3: RESET ENVIRONMENT (launches FD)
            # ====================================================================

            logger.info("Resetting environment (launching FD)...")
            solve_start_time = time.time()
            obs, info = env.reset()
            logger.info("✓ FD launched and initialized\n")

            # ====================================================================
            # PHASE 4: INFERENCE LOOP
            # ====================================================================

            total_reward = 0.0
            steps = 0
            max_steps = 50

            logger.info("Running GNN decision loop...\n")

            while steps < max_steps:
                try:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    total_reward += reward
                    steps += 1

                    logger.info(
                        f"Step {steps}: action={int(action)}, reward={reward:.4f}, "
                        f"total={total_reward:.4f}, done={done or truncated}"
                    )

                    if done or truncated:
                        logger.info("Episode finished\n")
                        break

                except KeyboardInterrupt:
                    logger.warning("⚠️ Interrupted by user")
                    break

                except Exception as e:
                    logger.error(f"Step {steps} failed: {e}")
                    logger.error(traceback.format_exc())
                    break

            elapsed = time.time() - solve_start_time

            # ====================================================================
            # PHASE 5: EXTRACT METRICS FROM FD OUTPUT
            # ====================================================================

            logger.info("-" * 80)
            logger.info("EXTRACTING METRICS FROM FD OUTPUT")
            logger.info("-" * 80 + "\n")

            plan_cost, expansions, nodes_expanded, search_depth, solution_found = \
                self._extract_fd_metrics()

            logger.info(f"Problem:        {problem_name}")
            logger.info(f"Solved:         {'✅ YES' if solution_found else '❌ NO'}")
            logger.info(f"Time:           {elapsed:.2f}s")
            logger.info(f"Plan Cost:      {plan_cost if solution_found else 'N/A'}")
            logger.info(f"Expansions:     {expansions if solution_found else 'N/A'}")
            logger.info(f"Nodes Expanded: {nodes_expanded if solution_found else 'N/A'}")
            logger.info(f"Search Depth:   {search_depth if solution_found else 'N/A'}")
            logger.info(f"Total Reward:   {total_reward:.4f}\n")

            # ====================================================================
            # PHASE 6: CLEANUP AND RETURN
            # ====================================================================

            try:
                env.close()
            except:
                pass

            return ProblemResult(
                problem_name=problem_name,
                planner_name="GNN",
                solved=solution_found,
                time_sec=elapsed,
                plan_cost=plan_cost,
                expansions=expansions,
                nodes_expanded=nodes_expanded,
                search_depth=search_depth
            )

        except Exception as e:
            logger.error(f"❌ FATAL: {e}")
            logger.error(traceback.format_exc())

            try:
                env.close()
            except:
                pass

            return ProblemResult(
                problem_name=problem_name,
                planner_name="GNN",
                solved=False,
                time_sec=0,
                error_reason=str(e)[:50]
            )

    @staticmethod
    def _extract_fd_metrics() -> Tuple[int, int, int, int, bool]:
        """
        ✅ FIXED: Extract metrics from Fast Downward output files.

        Returns:
            (plan_cost, expansions, nodes_expanded, search_depth, solution_found)
        """
        plan_cost = 0
        expansions = 0
        nodes_expanded = 0
        search_depth = 0
        solution_found = False

        log_file = os.path.join("downward", "fd_output", "log.txt")

        if not os.path.exists(log_file):
            logger.warning(f"FD log file not found: {log_file}")
            return 0, 0, 0, 0, False

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract plan cost
            match = re.search(r'Plan length:\s*(\d+)', content)
            if match:
                plan_cost = int(match.group(1))
                solution_found = True

            # Extract expansions
            matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', content))
            if matches:
                expansions = int(matches[-1].group(1))

            # Extract search depth
            match = re.search(r'Search depth:\s*(\d+)', content)
            if match:
                search_depth = int(match.group(1))

            # Estimate nodes_expanded from expansions
            nodes_expanded = expansions

        except Exception as e:
            logger.warning(f"Error extracting FD metrics: {e}")

        return plan_cost, expansions, nodes_expanded, search_depth, solution_found


# ============================================================================
# EVALUATION ORCHESTRATOR
# ============================================================================

class EvaluationFramework:
    """Main evaluation orchestrator."""

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ProblemResult] = []

    def run_comprehensive_evaluation(
            self,
            domain_file: str,
            problem_pattern: str,
            model_path: str,
            timeout_sec: int = 300,
            include_baselines: bool = True
    ) -> Dict[str, Any]:
        """Run complete evaluation: baselines + GNN on all problems."""

        print_section("COMPREHENSIVE EVALUATION FRAMEWORK")

        # Load problems
        logger.info(f"\nLoading problems matching: {problem_pattern}")
        problems = sorted(glob.glob(problem_pattern))

        if not problems:
            logger.error("No problems found!")
            return {}

        logger.info(f"Found {len(problems)} problem(s)")

        # Run baselines
        if include_baselines:
            self._run_baselines(domain_file, problems, timeout_sec)

        # Run GNN
        self._run_gnn(domain_file, problems, model_path, timeout_sec)

        # Analyze and report
        return self._generate_report()

    def _run_baselines(self, domain_file: str, problems: List[str], timeout_sec: int):
        """Run all baseline configurations."""
        print_subsection("RUNNING BASELINES")

        baseline_runner = BaselineRunner(timeout_sec)

        for baseline_config in BenchmarkConfig.BASELINES:
            logger.info(f"\nRunning baseline: {baseline_config['name']}")

            for i, problem in enumerate(problems, 1):
                logger.info(f"  [{i}/{len(problems)}] {os.path.basename(problem)}")

                result = baseline_runner.run(
                    domain_file,
                    problem,
                    baseline_config['search_config']
                )
                result.planner_name = baseline_config['name']
                self.results.append(result)

    def _run_gnn(self, domain_file: str, problems: List[str], model_path: str, timeout_sec: int):
        """Run GNN policy."""
        print_subsection("RUNNING GNN POLICY")

        gnn_runner = GNNPolicyRunner(model_path, timeout_sec)

        for i, problem in enumerate(problems, 1):
            logger.info(f"  [{i}/{len(problems)}] {os.path.basename(problem)}")

            result = gnn_runner.run(domain_file, problem)
            self.results.append(result)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        print_section("GENERATING REPORT")

        # Group by planner
        by_planner = {}
        for result in self.results:
            if result.planner_name not in by_planner:
                by_planner[result.planner_name] = []
            by_planner[result.planner_name].append(result)

        # Compute summaries
        summaries = {}
        for planner_name, results in by_planner.items():
            logger.info(f"\nSummary for {planner_name}:")

            solved_results = [r for r in results if r.solved]
            num_solved = len(solved_results)
            num_total = len(results)
            solve_rate = (num_solved / num_total * 100) if num_total > 0 else 0

            times = [r.time_sec for r in solved_results]
            expansions = [r.expansions for r in solved_results]
            costs = [r.plan_cost for r in solved_results]
            efficiencies = [r.efficiency for r in solved_results if r.efficiency != float('inf')]
            speeds = [r.speed for r in solved_results]

            summary = BenchmarkSummary(
                planner_name=planner_name,
                num_problems=num_total,
                num_solved=num_solved,
                solve_rate_pct=solve_rate,
                avg_time_sec=np.mean(times) if times else 0,
                median_time_sec=np.median(times) if times else 0,
                avg_expansions=int(np.mean(expansions)) if expansions else 0,
                median_expansions=int(np.median(expansions)) if expansions else 0,
                avg_plan_cost=int(np.mean(costs)) if costs else 0,
                median_plan_cost=int(np.median(costs)) if costs else 0,
                total_time_sec=sum(times) if times else 0,
                avg_efficiency=np.mean(efficiencies) if efficiencies else float('inf'),
                avg_speed=np.mean(speeds) if speeds else 0
            )

            summaries[planner_name] = summary

            logger.info(f"  Solved: {num_solved}/{num_total} ({solve_rate:.1f}%)")
            logger.info(f"  Avg time: {summary.avg_time_sec:.2f}s")
            logger.info(f"  Avg expansions: {summary.avg_expansions}")

        # Export results
        self._export_results(summaries)

        return {
            "summaries": {name: summary.to_dict() for name, summary in summaries.items()},
            "timestamp": datetime.now().isoformat()
        }

    def _export_results(self, summaries: Dict[str, BenchmarkSummary]):
        """Export results to CSV and JSON."""

        # CSV: detailed results
        csv_path = self.output_dir / "evaluation_results.csv"

        fieldnames = [
            'problem_name', 'planner_name', 'solved', 'time_sec',
            'plan_cost', 'expansions', 'nodes_expanded', 'search_depth',
            'error_reason', 'efficiency', 'speed'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

        logger.info(f"✓ Detailed results: {csv_path}")

        # JSON: summary
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(
                {name: summary.to_dict() for name, summary in summaries.items()},
                f,
                indent=2
            )
        logger.info(f"✓ Summary: {json_path}")

        # Text report
        self._write_text_report(summaries)

    def _write_text_report(self, summaries: Dict[str, BenchmarkSummary]):
        """Write formatted text report."""

        report_path = self.output_dir / "comparison_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 90 + "\n\n")

            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total problems: {len(set(r.problem_name for r in self.results))}\n")
            f.write(f"Total evaluations: {len(self.results)}\n\n")

            # Summary table
            f.write("SUMMARY TABLE\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Planner':<35} {'Solved':<15} {'Avg Time (s)':<15} {'Avg Exp.':<15}\n")
            f.write("-" * 90 + "\n")

            for planner_name, summary in summaries.items():
                solved_str = f"{summary.num_solved}/{summary.num_problems} ({summary.solve_rate_pct:.0f}%)"
                f.write(
                    f"{planner_name:<35} {solved_str:<15} {summary.avg_time_sec:<15.2f} {summary.avg_expansions:<15}\n")

            f.write("-" * 90 + "\n\n")

            # Detailed stats
            for planner_name, summary in summaries.items():
                f.write(f"\n{planner_name} DETAILED STATISTICS\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Solve Rate:         {summary.solve_rate_pct:.1f}%\n")
                f.write(f"  Avg Time (solved):  {summary.avg_time_sec:.2f}s\n")
                f.write(f"  Median Time:        {summary.median_time_sec:.2f}s\n")
                f.write(f"  Avg Expansions:     {summary.avg_expansions}\n")
                f.write(f"  Median Expansions:  {summary.median_expansions}\n")
                f.write(f"  Avg Plan Cost:      {summary.avg_plan_cost}\n")
                f.write(f"  Total Time:         {summary.total_time_sec:.1f}s\n")
                f.write(f"  Avg Efficiency:     {summary.avg_efficiency:.4f}\n")
                f.write(f"  Avg Speed:          {summary.avg_speed:.2f} exp/s\n")

        logger.info(f"✓ Text report: {report_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str, width: int = 90):
    print("\n" + "=" * width)
    print(f"// {title.upper()}")
    print("=" * width + "\n")


def print_subsection(title: str):
    print("\n" + "-" * 80)
    print(f">>> {title}")
    print("-" * 80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation framework for GNN merge strategy"
    )
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--domain", required=True, help="Path to domain PDDL")
    parser.add_argument("--problems", required=True, help="Glob pattern for problems")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem (seconds)")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline runs")

    args = parser.parse_args()

    framework = EvaluationFramework(args.output)

    results = framework.run_comprehensive_evaluation(
        domain_file=args.domain,
        problem_pattern=args.problems,
        model_path=args.model,
        timeout_sec=args.timeout,
        include_baselines=not args.skip_baselines
    )

    print_section("EVALUATION COMPLETE")
    logger.info("✅ All evaluations complete!")
    logger.info(f"Results saved to: {os.path.abspath(args.output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())