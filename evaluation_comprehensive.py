# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE EVALUATION FRAMEWORK - ENHANCED
==============================================
Complete rewrite with:
  ✓ Robust baseline runner with all major FD planners
  ✓ GNN policy runner using MergeEnv with real FD
  ✓ Detailed metric extraction (20+ metrics per run)
  ✓ Statistical analysis (mean, median, std, IQR)
  ✓ Multiple output formats (CSV, JSON, TXT)
  ✓ Per-problem and aggregate reporting
  ✓ Error tracking and diagnostics
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
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_comprehensive.log", encoding='utf-8'),
    ],
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES - COMPREHENSIVE METRICS
# ============================================================================

@dataclass
class DetailedMetrics:
    """Complete set of metrics for a single run."""
    problem_name: str
    planner_name: str

    # Solve status
    solved: bool
    wall_clock_time: float

    # Planning quality
    plan_cost: int = 0
    plan_length: int = 0

    # Search metrics
    nodes_expanded: int = 0
    nodes_generated: int = 0
    search_depth: int = 0
    branching_factor: float = 1.0

    # Memory metrics (if available)
    peak_memory_kb: int = 0

    # Time breakdown
    search_time: float = 0.0
    translate_time: float = 0.0
    preprocess_time: float = 0.0

    # Solution quality
    solution_length: int = 0
    plan_optimality: float = 1.0  # 1.0 if optimal

    # Heuristic quality (if evaluator)
    initial_heuristic: int = 0
    average_heuristic: float = 0.0

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # GNN-specific
    gnn_decisions: int = 0  # Number of merge decisions made
    merge_episodes: int = 0

    def efficiency_score(self) -> float:
        """Score: lower is better. 0 is best."""
        if not self.solved:
            return float('inf')

        # Normalize by problem size (approximated by plan cost)
        if self.nodes_expanded > 0 and self.plan_cost > 0:
            return (self.nodes_expanded / (self.plan_cost * 100.0)) + (self.wall_clock_time / 10.0)
        return self.wall_clock_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON."""
        d = asdict(self)
        d['efficiency_score'] = self.efficiency_score()
        return d


@dataclass
class AggregateStatistics:
    """Summary statistics across all problems."""
    planner_name: str
    num_problems_total: int
    num_problems_solved: int
    solve_rate_pct: float

    # Time stats (solved only)
    mean_time_sec: float
    median_time_sec: float
    std_time_sec: float
    min_time_sec: float
    max_time_sec: float
    q1_time_sec: float  # 25th percentile
    q3_time_sec: float  # 75th percentile

    # Expansions stats
    mean_expansions: int
    median_expansions: int
    std_expansions: int

    # Plan quality stats
    mean_plan_cost: int
    median_plan_cost: int
    std_plan_cost: int

    # Efficiency
    mean_efficiency_score: float

    # Coverage metrics
    unsolved_count: int
    error_count: int
    timeout_count: int

    # Aggregate times
    total_wall_clock_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BASELINE RUNNER - COMPLETE REFACTORING
# ============================================================================

class BaselineRunner:
    """Enhanced baseline runner with comprehensive metrics extraction."""

    # Baseline configurations with ALL important FD variants
    BASELINES = [
        {
            "name": "FD_LM-Cut",
            "search": "astar(lmcut())"
        },
        {
            "name": "FD_Blind",
            "search": "astar(blind())"
        },
        {
            "name": "FD_Add",
            "search": "astar(add())"
        },
        {
            "name": "FD_Max",
            "search": "astar(max())"
        },
        {
            "name": "FD_M&S_DFP",
            "search": (
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
            "name": "FD_M&S_SCC",
            "search": (
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
    ]

    def __init__(self, timeout_sec: int = 300):
        self.timeout_sec = timeout_sec
        self.fd_bin = os.path.abspath("downward/builds/release/bin/downward.exe")
        self.fd_translate = os.path.abspath("downward/builds/release/bin/translate/translate.py")

        if not os.path.exists(self.fd_bin) or not os.path.exists(self.fd_translate):
            raise FileNotFoundError("Fast Downward binary not found")

    def run(self, domain_file: str, problem_file: str, search_config: str) -> DetailedMetrics:
        """Run FD with comprehensive metrics extraction."""
        problem_name = os.path.basename(problem_file)

        try:
            # TRANSLATE PHASE
            logger.info(f"    [TRANSLATE] {problem_name}...")

            work_dir = os.path.abspath("evaluation_temp")
            os.makedirs(work_dir, exist_ok=True)
            sas_file = os.path.join(work_dir, "output.sas")

            translate_start = time.time()

            result = subprocess.run(
                f'python "{self.fd_translate}" "{os.path.abspath(domain_file)}" '
                f'"{os.path.abspath(problem_file)}" --sas-file "{sas_file}"',
                shell=True,
                cwd=os.path.abspath("."),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            translate_time = time.time() - translate_start

            if result.returncode != 0 or not os.path.exists(sas_file):
                logger.debug(f"    [TRANSLATE] Failed: {result.stderr[:200]}")
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    wall_clock_time=translate_time,
                    error_type="translate_error",
                    error_message=result.stderr[:500]
                )

            logger.debug(f"    [TRANSLATE] Success ({os.path.getsize(sas_file)} bytes)")

            # SEARCH PHASE
            logger.debug(f"    [SEARCH] Starting...")

            search_start = time.time()

            result = subprocess.run(
                f'"{self.fd_bin}" --search "{search_config}" < "{sas_file}"',
                shell=True,
                cwd=os.path.dirname(self.fd_bin),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            search_time = time.time() - search_start
            total_time = translate_time + search_time

            output_text = result.stdout + result.stderr

            logger.debug(f"    [SEARCH] Completed in {search_time:.2f}s")

            # CHECK FOR SOLUTION
            if "Solution found" not in output_text and "Plan length:" not in output_text:
                logger.debug(f"    [PARSE] No solution found")
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=False,
                    wall_clock_time=total_time,
                    translate_time=translate_time,
                    search_time=search_time,
                    error_type="no_solution"
                )

            # EXTRACT METRICS
            metrics_dict = self._parse_fd_output(output_text)

            if metrics_dict is None:
                logger.debug(f"    [PARSE] Could not extract metrics")
                return DetailedMetrics(
                    problem_name=problem_name,
                    planner_name="FD",
                    solved=True,
                    wall_clock_time=total_time,
                    translate_time=translate_time,
                    search_time=search_time,
                    error_type="parse_error"
                )

            # BUILD RESULT
            result_metrics = DetailedMetrics(
                problem_name=problem_name,
                planner_name="FD",
                solved=True,
                wall_clock_time=total_time,
                translate_time=translate_time,
                search_time=search_time,
                plan_cost=metrics_dict.get('cost', 0),
                plan_length=metrics_dict.get('cost', 0),
                nodes_expanded=metrics_dict.get('expansions', 0),
                search_depth=metrics_dict.get('search_depth', 0),
                branching_factor=metrics_dict.get('branching_factor', 1.0),
                peak_memory_kb=metrics_dict.get('memory', 0),
            )

            logger.debug(f"    [SUCCESS] cost={result_metrics.plan_cost}, "
                         f"exp={result_metrics.nodes_expanded}")

            return result_metrics

        except subprocess.TimeoutExpired:
            logger.debug(f"    [TIMEOUT] Exceeded {self.timeout_sec}s")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="FD",
                solved=False,
                wall_clock_time=self.timeout_sec,
                error_type="timeout"
            )

        except Exception as e:
            logger.error(f"    [ERROR] {e}")
            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="FD",
                solved=False,
                wall_clock_time=0,
                error_type="exception",
                error_message=str(e)[:500]
            )

    @staticmethod
    def _parse_fd_output(output_text: str) -> Optional[Dict[str, Any]]:
        """Extract comprehensive metrics from FD output."""
        metrics = {}

        # Plan cost
        match = re.search(r'Plan length:\s*(\d+)', output_text)
        if match:
            metrics['cost'] = int(match.group(1))

        # Expansions (take last)
        matches = list(re.finditer(r'Expanded\s+(\d+)\s+state', output_text))
        if matches:
            metrics['expansions'] = int(matches[-1].group(1))

        # Generated states
        matches = list(re.finditer(r'Generated\s+(\d+)\s+state', output_text))
        if matches:
            metrics['generated'] = int(matches[-1].group(1))

        # Search depth
        match = re.search(r'Search depth:\s*(\d+)', output_text)
        if match:
            metrics['search_depth'] = int(match.group(1))

        # Branching factor
        match = re.search(r'Branching factor:\s*([\d.]+)', output_text)
        if match:
            metrics['branching_factor'] = float(match.group(1))

        # Memory
        match = re.search(r'Peak memory:\s*(\d+)\s*KB', output_text)
        if match:
            metrics['memory'] = int(match.group(1))

        # Require at least cost and expansions
        if 'cost' not in metrics or 'expansions' not in metrics:
            return None

        return metrics


# ============================================================================
# GNN POLICY RUNNER - USING MergeEnv
# ============================================================================

class GNNPolicyRunner:
    """GNN policy runner using real MergeEnv with FD feedback."""

    def __init__(self, model_path: str, timeout_sec: int = 300):
        self.model_path = model_path
        self.timeout_sec = timeout_sec

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        from stable_baselines3 import PPO
        from merge_env import MergeEnv
        self.PPO = PPO
        self.MergeEnv = MergeEnv

    def run(self, domain_file: str, problem_file: str) -> DetailedMetrics:
        """Run GNN policy on a problem."""
        problem_name = os.path.basename(problem_file)

        try:
            start_time = time.time()

            logger.debug(f"    [LOAD] Loading model...")
            model = self.PPO.load(self.model_path)
            logger.debug(f"    [LOAD] Model loaded")

            logger.debug(f"    [ENV] Creating environment...")
            env = self.MergeEnv(
                domain_file=os.path.abspath(domain_file),
                problem_file=os.path.abspath(problem_file),
                max_merges=50,
                debug=False,
                reward_variant='astar_search',
                w_search_efficiency=0.30,
                w_solution_quality=0.20,
                w_f_stability=0.35,
                w_state_control=0.15,
            )
            logger.debug(f"    [ENV] Environment created")

            logger.debug(f"    [RESET] Resetting environment...")
            solve_start = time.time()
            obs, info = env.reset()
            logger.debug(f"    [RESET] Environment reset")

            total_reward = 0.0
            steps = 0
            max_steps = 50
            gnn_decisions = 0

            logger.debug(f"    [INFERENCE] Starting...")

            while steps < max_steps:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(int(action))
                    total_reward += reward
                    steps += 1
                    gnn_decisions += 1

                    if done or truncated:
                        break

                except KeyboardInterrupt:
                    logger.warning("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Step failed: {e}")
                    break

            solve_time = time.time() - solve_start
            total_time = time.time() - start_time

            logger.debug(f"    [INFERENCE] Completed in {steps} steps")

            # Extract metrics from FD output
            plan_cost, expansions, nodes_expanded, search_depth, solution_found = \
                self._extract_fd_metrics()

            logger.debug(f"    [EXTRACT] FD metrics extracted")

            try:
                env.close()
            except:
                pass

            result = DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=solution_found,
                wall_clock_time=total_time,
                plan_cost=plan_cost,
                nodes_expanded=nodes_expanded,
                search_depth=search_depth,
                gnn_decisions=gnn_decisions,
                merge_episodes=steps,
            )

            return result

        except Exception as e:
            logger.error(f"GNN run failed: {e}")
            logger.error(traceback.format_exc())

            try:
                env.close()
            except:
                pass

            return DetailedMetrics(
                problem_name=problem_name,
                planner_name="GNN",
                solved=False,
                wall_clock_time=0,
                error_type="exception",
                error_message=str(e)[:500]
            )

    @staticmethod
    def _extract_fd_metrics() -> Tuple[int, int, int, int, bool]:
        """Extract metrics from FD output files."""
        plan_cost = 0
        expansions = 0
        nodes_expanded = 0
        search_depth = 0
        solution_found = False

        log_file = os.path.join("downward", "fd_output", "log.txt")

        if not os.path.exists(log_file):
            return plan_cost, expansions, nodes_expanded, search_depth, solution_found

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
        self.results: List[DetailedMetrics] = []

    def run_comprehensive_evaluation(
            self,
            domain_file: str,
            problem_pattern: str,
            model_path: str,
            timeout_sec: int = 300,
            include_baselines: bool = True
    ) -> Dict[str, Any]:
        """Run complete evaluation."""

        print_section("COMPREHENSIVE EVALUATION FRAMEWORK")

        # Load problems
        logger.info(f"\nLoading problems matching: {problem_pattern}")
        problems = sorted(glob.glob(problem_pattern))

        if not problems:
            logger.error("No problems found!")
            return {}

        logger.info(f"Found {len(problems)} problem(s)\n")

        # Run baselines
        if include_baselines:
            self._run_all_baselines(domain_file, problems, timeout_sec)

        # Run GNN
        self._run_gnn(domain_file, problems, model_path, timeout_sec)

        # Generate reports
        return self._generate_comprehensive_report()

    def _run_all_baselines(self, domain_file: str, problems: List[str], timeout_sec: int):
        """Run all baseline configurations."""
        print_subsection("RUNNING BASELINE PLANNERS")

        baseline_runner = BaselineRunner(timeout_sec)

        for baseline_config in baseline_runner.BASELINES:
            logger.info(f"\n{baseline_config['name']}")
            logger.info("-" * 60)

            for i, problem in enumerate(problems, 1):
                logger.info(f"  [{i}/{len(problems)}] {os.path.basename(problem)}")

                result = baseline_runner.run(
                    domain_file,
                    problem,
                    baseline_config['search']
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

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate reports and statistics."""
        print_section("GENERATING COMPREHENSIVE REPORT")

        # Group by planner
        by_planner = defaultdict(list)
        for result in self.results:
            by_planner[result.planner_name].append(result)

        # Compute statistics
        summaries = {}
        for planner_name, results_list in by_planner.items():
            summary = self._compute_statistics(planner_name, results_list)
            summaries[planner_name] = summary

            logger.info(f"\n{planner_name}:")
            logger.info(f"  Solve rate: {summary.solve_rate_pct:.1f}%")
            logger.info(f"  Avg time: {summary.mean_time_sec:.2f}s")
            logger.info(f"  Avg expansions: {summary.mean_expansions}")

        # Export results
        self._export_all_results(summaries)

        return {
            "summaries": {name: summary.to_dict() for name, summary in summaries.items()},
            "timestamp": datetime.now().isoformat()
        }

    def _compute_statistics(self, planner_name: str, results_list: List[DetailedMetrics]) -> AggregateStatistics:
        """Compute comprehensive statistics."""

        solved = [r for r in results_list if r.solved]
        num_solved = len(solved)
        num_total = len(results_list)

        times = [r.wall_clock_time for r in solved]
        expansions = [r.nodes_expanded for r in solved]
        costs = [r.plan_cost for r in solved]
        efficiency_scores = [r.efficiency_score() for r in results_list if r.solved]

        unsolved = [r for r in results_list if not r.solved]
        errors = [r for r in unsolved if r.error_type]
        timeouts = [r for r in unsolved if r.error_type == "timeout"]

        return AggregateStatistics(
            planner_name=planner_name,
            num_problems_total=num_total,
            num_problems_solved=num_solved,
            solve_rate_pct=(num_solved / max(num_total, 1)) * 100,
            mean_time_sec=np.mean(times) if times else 0,
            median_time_sec=np.median(times) if times else 0,
            std_time_sec=np.std(times) if times else 0,
            min_time_sec=np.min(times) if times else 0,
            max_time_sec=np.max(times) if times else 0,
            q1_time_sec=np.percentile(times, 25) if times else 0,
            q3_time_sec=np.percentile(times, 75) if times else 0,
            mean_expansions=int(np.mean(expansions)) if expansions else 0,
            median_expansions=int(np.median(expansions)) if expansions else 0,
            std_expansions=int(np.std(expansions)) if expansions else 0,
            mean_plan_cost=int(np.mean(costs)) if costs else 0,
            median_plan_cost=int(np.median(costs)) if costs else 0,
            std_plan_cost=int(np.std(costs)) if costs else 0,
            mean_efficiency_score=np.mean(efficiency_scores) if efficiency_scores else float('inf'),
            unsolved_count=len(unsolved),
            error_count=len(errors),
            timeout_count=len(timeouts),
            total_wall_clock_time_sec=sum(times) if times else 0,
        )

    def _export_all_results(self, summaries: Dict[str, AggregateStatistics]):
        """Export results in all formats."""

        # CSV: detailed
        csv_path = self.output_dir / "evaluation_results.csv"
        fieldnames = list(self.results[0].to_dict().keys()) if self.results else []

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

        logger.info(f"✓ CSV: {csv_path}")

        # JSON: summary
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(
                {name: summary.to_dict() for name, summary in summaries.items()},
                f,
                indent=2
            )

        logger.info(f"✓ JSON: {json_path}")

        # TXT: comparison report
        self._write_text_report(summaries)

    def _write_text_report(self, summaries: Dict[str, AggregateStatistics]):
        """Write formatted text comparison report."""

        report_path = self.output_dir / "comparison_report.txt"

        try:
            with open(report_path, 'w') as f:
                f.write("=" * 90 + "\n")
                f.write("COMPREHENSIVE EVALUATION - COMPARISON REPORT\n")
                f.write("=" * 90 + "\n\n")

                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                num_problems = summaries[list(summaries.keys())[0]].num_problems_total if summaries else 0
                f.write(f"Total problems evaluated: {num_problems}\n\n")

                # Summary table
                f.write("SUMMARY TABLE\n")
                f.write("-" * 90 + "\n")
                header = (
                    f"{'Planner':<25} {'Solved':<18} {'Avg Time (s)':<15} "
                    f"{'Med Time (s)':<15} {'Avg Exp.':<15}"
                )
                f.write(header + "\n")
                f.write("-" * 90 + "\n")

                # Sort planners, maybe put 'GNN' first if present
                planner_names = sorted(summaries.keys())
                if "GNN" in planner_names:
                    planner_names.insert(0, planner_names.pop(planner_names.index("GNN")))

                for planner_name in planner_names:
                    if planner_name not in summaries: continue
                    summary = summaries[planner_name]
                    solved_str = (
                        f"{summary.num_problems_solved}/{summary.num_problems_total} "
                        f"({summary.solve_rate_pct:.1f}%)"
                    )
                    avg_exp_str = f"{summary.mean_expans:,}" if summary.num_problems_solved > 0 else "N/A" # Added comma formatting

                    line = (
                        f"{planner_name:<25} {solved_str:<18} {summary.mean_time_sec:<15.2f} "
                        f"{summary.median_time_sec:<15.2f} {avg_exp_str:<15}"
                    )
                    f.write(line + "\n")

                f.write("-" * 90 + "\n\n")

                # Detailed stats per planner (optional, could make report long)
                # You can add more details here if needed, similar to the JSON summary

            logger.info(f"✓ Text report: {report_path}")

        except Exception as e:
            logger.error(f"Failed to write text report: {e}")
            logger.error(traceback.format_exc())

def print_section(title: str, width: int = 90):
    logger.info("\n" + "=" * width)
    logger.info(f"// {title.upper()}")
    logger.info("=" * width + "\n")


def print_subsection(title: str):
    logger.info("\n" + "-" * 80)
    logger.info(f">>> {title}")
    logger.info("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation framework")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--domain", required=True, help="Path to domain PDDL")
    parser.add_argument("--problems", required=True, help="Glob pattern for problems")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per problem")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baselines")

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