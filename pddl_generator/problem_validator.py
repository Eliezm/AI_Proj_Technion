# -*- coding: utf-8 -*-
"""
Validates that generated PDDL problems are well-formed and solvable.
"""

import subprocess
import os
from pathlib import Path
from typing import Tuple, Optional


class PDDLValidator:
    """Validates PDDL problems using Fast Downward or VAL."""

    def __init__(self, fd_path: str = "./downward/builds/release/bin/downward.exe"):
        self.fd_path = fd_path

    def validate_syntax(self, domain_file: str, problem_file: str) -> Tuple[bool, str]:
        """
        Check PDDL syntax using Fast Downward's translate.

        Returns:
            (is_valid, message)
        """
        try:
            result = subprocess.run(
                [
                    "python",
                    "./downward/builds/release/bin/translate/translate.py",
                    domain_file,
                    problem_file,
                    "--sas-file", "/tmp/test.sas"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, "Syntax valid"
            else:
                return False, result.stderr

        except Exception as e:
            return False, str(e)

    def validate_solvability(
            self,
            domain_file: str,
            problem_file: str,
            timeout: int = 60
    ) -> Tuple[bool, str]:
        """
        Check if problem is solvable using a simple search.

        Returns:
            (is_solvable, message)
        """
        try:
            result = subprocess.run(
                [
                    self.fd_path,
                    "--search", "astar(lmcut())",
                    domain_file,
                    problem_file
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            if "Solution found" in output or "Plan length" in output:
                return True, "Problem is solvable"
            else:
                return False, "No solution found"

        except subprocess.TimeoutExpired:
            return False, "Timeout (problem too hard or unsolvable)"
        except Exception as e:
            return False, str(e)

    def validate_problem(
            self,
            domain_file: str,
            problem_file: str,
            check_solvability: bool = True
    ) -> Tuple[bool, str]:
        """Full validation: syntax + optionally solvability."""

        # Check syntax
        is_valid, msg = self.validate_syntax(domain_file, problem_file)
        if not is_valid:
            return False, f"Syntax error: {msg}"

        # Check solvability
        if check_solvability:
            is_solvable, msg = self.validate_solvability(domain_file, problem_file)
            if not is_solvable:
                return False, f"Solvability check failed: {msg}"

        return True, "Problem is valid and solvable"


if __name__ == "__main__":
    # Test validation
    validator = PDDLValidator()

    domain = "benchmarks/blocks_world/small/domain_new.pddl"
    problem = "benchmarks/blocks_world/small/problem_small_00.pddl"

    if os.path.exists(domain) and os.path.exists(problem):
        is_valid, msg = validator.validate_problem(domain, problem)
        print(f"{'✓' if is_valid else '✗'} {msg}")
    else:
        print("Test files not found. Run problem_generator.py first.")