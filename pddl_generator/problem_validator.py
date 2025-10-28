# -*- coding: utf-8 -*-
"""
PROBLEM VALIDATOR & CLASSIFIER
==============================

Validates PDDL problems and classifies by difficulty.
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProblemValidator:
    """Validates and classifies PDDL problems."""

    # Time thresholds for difficulty categories
    SMALL_MAX = 60  # seconds
    MEDIUM_MAX = 180  # seconds (3 minutes)
    LARGE_MAX = 480  # seconds (8 minutes)

    def __init__(
            self,
            fd_bin: str = "downward/builds/release/bin/downward.exe",
            fd_translate: str = "downward/builds/release/bin/translate/translate.py",
            timeout: int = 600  # 10 minutes max per problem
    ):
        self.fd_bin = fd_bin
        self.fd_translate = fd_translate
        self.timeout = timeout

    def validate_and_classify(
            self,
            domain_file: str,
            problem_file: str,
            metadata_file: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate problem and classify by difficulty.

        Returns:
            (is_valid, difficulty_category, error_message)
        """

        try:
            # Step 1: Validate syntax
            is_valid_syntax, syntax_error = self._validate_syntax(domain_file, problem_file)
            if not is_valid_syntax:
                return False, None, f"Syntax error: {syntax_error}"

            logger.info("✓ Syntax valid")

            # Step 2: Check solvability and measure difficulty
            plan_cost, solve_time, nodes_expanded, is_solvable = self._check_solvability(
                domain_file, problem_file
            )

            if not is_solvable:
                return False, None, "Problem is unsolvable"

            logger.info(f"✓ Problem solved in {solve_time:.2f}s")

            # Step 3: Classify by difficulty
            difficulty = self._classify_difficulty(solve_time)
            logger.info(f"✓ Classified as: {difficulty}")

            # Step 4: Update metadata
            self._update_metadata(metadata_file, {
                "plan_cost": plan_cost,
                "solve_time_sec": solve_time,
                "nodes_expanded": nodes_expanded,
                "is_solvable": True,
                "size_category": difficulty,
            })

            return True, difficulty, None

        except Exception as e:
            return False, None, str(e)

    def _validate_syntax(self, domain_file: str, problem_file: str) -> Tuple[bool, Optional[str]]:
        """Validate PDDL syntax using FD translate."""
        try:
            # Create temp output file
            temp_sas = "/tmp/test.sas"

            result = subprocess.run(
                [
                    "python",
                    self.fd_translate,
                    domain_file,
                    problem_file,
                    "--sas-file", temp_sas
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return False, result.stderr

            # Clean up
            if os.path.exists(temp_sas):
                os.remove(temp_sas)

            return True, None

        except subprocess.TimeoutExpired:
            return False, "Translation timeout"
        except Exception as e:
            return False, str(e)

    def _check_solvability(
            self,
            domain_file: str,
            problem_file: str
    ) -> Tuple[int, float, int, bool]:
        """
        Check if problem is solvable and measure difficulty.

        Returns:
            (plan_cost, solve_time, nodes_expanded, is_solvable)
        """
        try:
            start_time = time.time()

            result = subprocess.run(
                [
                    self.fd_bin,
                    "--search", "astar(lmcut())",
                    domain_file,
                    problem_file
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd="downward"
            )

            elapsed = time.time() - start_time
            output = result.stdout + result.stderr

            # Check if solved
            if "Solution found" not in output and "Plan length" not in output:
                return 0, elapsed, 0, False

            # Extract metrics
            plan_cost = self._extract_metric(output, r"Plan length:\s*(\d+)")
            nodes_expanded = self._extract_metric(output, r"Expanded\s+(\d+)\s+state")

            return plan_cost, elapsed, nodes_expanded, True

        except subprocess.TimeoutExpired:
            return 0, self.timeout, 0, False
        except Exception as e:
            logger.error(f"Error checking solvability: {e}")
            return 0, 0, 0, False

    def _extract_metric(self, text: str, pattern: str) -> int:
        """Extract numeric metric from text."""
        import re
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    def _classify_difficulty(self, solve_time: float) -> str:
        """Classify problem by solve time."""
        if solve_time <= self.SMALL_MAX:
            return "small"
        elif solve_time <= self.MEDIUM_MAX:
            return "medium"
        elif solve_time <= self.LARGE_MAX:
            return "large"
        else:
            return None  # Too hard

    def _update_metadata(self, metadata_file: str, updates: Dict[str, Any]):
        """Update problem metadata with runtime metrics."""
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata.update(updates)

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not update metadata: {e}")


def validate_all_benchmarks(benchmark_dir: str = "benchmarks"):
    """Validate all generated benchmarks."""

    validator = ProblemValidator()
    benchmark_path = Path(benchmark_dir)

    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "by_difficulty": {"small": 0, "medium": 0, "large": 0}
    }

    for domain_dir in benchmark_path.iterdir():
        if not domain_dir.is_dir():
            continue

        for size_dir in domain_dir.iterdir():
            if not size_dir.is_dir():
                continue

            logger.info(f"\nValidating {domain_dir.name}/{size_dir.name}")

            domain_file = size_dir / "domain.pddl"
            problem_files = sorted(size_dir.glob("problem_*.pddl"))

            for problem_file in problem_files:
                metadata_file = problem_file.with_suffix('.json')

                results["total"] += 1
                is_valid, difficulty, error = validator.validate_and_classify(
                    str(domain_file),
                    str(problem_file),
                    str(metadata_file)
                )

                if is_valid:
                    results["valid"] += 1
                    if difficulty:
                        results["by_difficulty"][difficulty] += 1
                    logger.info(f"✓ {problem_file.name} → {difficulty}")
                else:
                    results["invalid"] += 1
                    logger.warning(f"✗ {problem_file.name}: {error}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total: {results['total']}")
    logger.info(f"Valid: {results['valid']}")
    logger.info(f"Invalid: {results['invalid']}")
    logger.info(f"Distribution:")
    logger.info(f"  Small: {results['by_difficulty']['small']}")
    logger.info(f"  Medium: {results['by_difficulty']['medium']}")
    logger.info(f"  Large: {results['by_difficulty']['large']}")


if __name__ == "__main__":
    validate_all_benchmarks()