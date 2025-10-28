"""
Integration with Fast Downward baseline planner.

Requirement #12: Run baseline planner and collect metadata.

This version uses the robust two-step (Translate + Search) process
adapted from the comprehensive evaluation framework.
"""

import subprocess
import json
import re
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class FastDownwardRunner:
    """Runs Fast Downward and extracts comprehensive metrics."""

    def __init__(self, timeout: int = 600):
        """
        Initialize Fast Downward runner.

        Args:
            timeout: Maximum time in seconds for the entire run
        """
        self.timeout = timeout

        # --- MODIFICATION START ---
        # Get the directory where this file (baseline_planner.py) is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Go one step up to the project root (where 'downward' and the code dir live)
        project_root = os.path.abspath(os.path.join(script_dir, ".."))

        # Define the path to the 'downward/builds/release/bin' directory
        fd_bin_dir = os.path.join(project_root, "downward", "builds", "release", "bin")

        # Detect OS and set binary path
        if os.name == 'nt':  # Windows
            self.fd_bin = os.path.join(fd_bin_dir, "downward.exe")
        else:  # Linux/macOS
            self.fd_bin = os.path.join(fd_bin_dir, "downward")

        self.fd_translate = os.path.join(fd_bin_dir, "translate", "translate.py")

        # Place temp_dir in the project root as well
        self.temp_dir = os.path.join(project_root, "generation_temp")
        # --- MODIFICATION END ---

        os.makedirs(self.temp_dir, exist_ok=True)

        # Check if FD is available
        self.fd_available = self._check_fd_available()

    def _check_fd_available(self) -> bool:
        """Check if Fast Downward binaries are available."""
        fd_exists = os.path.exists(self.fd_bin)
        translate_exists = os.path.exists(self.fd_translate)

        if not fd_exists or not translate_exists:
            logger.warning("Fast Downward not fully available:")
            if not fd_exists:
                logger.warning(f"  ✗ Binary not found: {self.fd_bin}")
            if not translate_exists:
                logger.warning(f"  ✗ Translator not found: {self.fd_translate}")
            return False

        logger.debug(f"✓ FD binary: {self.fd_bin}")
        logger.debug(f"✓ FD translator: {self.fd_translate}")
        return True

    def run_problem(
            self,
            domain_file: str,
            problem_file: str,
            search_config: str = "astar(lmcut())",
            timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Fast Downward on a problem using the 2-step translate/search process.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            search_config: Fast Downward search configuration string
            timeout: Override default timeout

        Returns:
            Dict with:
                - success: bool (whether solution was found)
                - time: float (total wall clock time in seconds)
                - plan_cost: int or None (plan length)
                - nodes_expanded: int or None (search nodes)
                - nodes_generated: int or None (generated states)
                - search_depth: int or None (search depth)
                - plan: list or None (action strings if extracted)
                - error: str or None (error message if failed)
        """
        if not self.fd_available:
            return {
                "success": False,
                "time": 0,
                "plan_cost": None,
                "nodes_expanded": None,
                "nodes_generated": None,
                "search_depth": None,
                "plan": None,
                "error": "Fast Downward not installed or not found at expected paths"
            }

        timeout_to_use = timeout if timeout is not None else self.timeout
        problem_name = os.path.basename(problem_file)
        sas_file = os.path.join(self.temp_dir, "output.sas")

        try:
            # ==========================================================
            # PHASE 1: TRANSLATE (PDDL -> SAS)
            # ==========================================================
            logger.debug(f"[TRANSLATE] {problem_name}")
            translate_start = time.time()

            # Use absolute paths
            abs_domain = os.path.abspath(domain_file)
            abs_problem = os.path.abspath(problem_file)

            translate_cmd = (
                f'python "{self.fd_translate}" "{abs_domain}" '
                f'"{abs_problem}" --sas-file "{sas_file}"'
            )

            translate_result = subprocess.run(
                translate_cmd,
                shell=True,
                cwd=os.path.abspath(".."),
                capture_output=True,
                text=True,
                timeout=timeout_to_use
            )

            translate_time = time.time() - translate_start

            if translate_result.returncode != 0:
                error_msg = translate_result.stderr if translate_result.stderr else translate_result.stdout
                logger.debug(f"[TRANSLATE] Failed: {error_msg[:200]}")
                return {
                    "success": False,
                    "time": translate_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": f"Translate error: {error_msg[:300]}"
                }

            if not os.path.exists(sas_file):
                logger.debug(f"[TRANSLATE] Failed: SAS file not created")
                return {
                    "success": False,
                    "time": translate_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": "Translate: SAS file not created"
                }

            logger.debug(f"[TRANSLATE] Success ({os.path.getsize(sas_file)} bytes)")

            # ==========================================================
            # PHASE 2: SEARCH (SAS -> Plan)
            # ==========================================================
            logger.debug(f"[SEARCH] Starting with config: {search_config}")
            search_start = time.time()

            search_cmd = f'"{self.fd_bin}" --search "{search_config}" < "{sas_file}"'

            search_result = subprocess.run(
                search_cmd,
                shell=True,
                cwd=os.path.dirname(self.fd_bin),
                capture_output=True,
                text=True,
                timeout=timeout_to_use
            )

            search_time = time.time() - search_start
            total_time = translate_time + search_time

            output_text = search_result.stdout + search_result.stderr

            logger.debug(f"[SEARCH] Completed in {search_time:.2f}s")

            # Clean up SAS file
            try:
                if os.path.exists(sas_file):
                    os.remove(sas_file)
            except Exception as e:
                logger.debug(f"Could not remove SAS file: {e}")

            # ==========================================================
            # PHASE 3: PARSE OUTPUT
            # ==========================================================

            # Check if solution was found
            solution_found = (
                "Solution found" in output_text or
                "Plan length:" in output_text
            )

            if not solution_found:
                logger.debug(f"[PARSE] No solution found")
                return {
                    "success": False,
                    "time": total_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": "No solution found"
                }

            # Extract metrics
            metrics = self._parse_fd_output(output_text)

            logger.debug(
                f"[SUCCESS] cost={metrics['plan_cost']}, "
                f"exp={metrics['nodes_expanded']}, "
                f"time={total_time:.2f}s"
            )

            return {
                "success": True,
                "time": total_time,
                "plan_cost": metrics['plan_cost'],
                "nodes_expanded": metrics['nodes_expanded'],
                "nodes_generated": metrics['nodes_generated'],
                "search_depth": metrics['search_depth'],
                "plan": metrics['plan'],
                "error": None
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"[TIMEOUT] Exceeded {timeout_to_use}s for {problem_name}")
            return {
                "success": False,
                "time": timeout_to_use,
                "plan_cost": None,
                "nodes_expanded": None,
                "nodes_generated": None,
                "search_depth": None,
                "plan": None,
                "error": f"Timeout (>{timeout_to_use}s)"
            }

        except FileNotFoundError as e:
            logger.error(f"[ERROR] File not found: {e}")
            return {
                "success": False,
                "time": 0,
                "plan_cost": None,
                "nodes_expanded": None,
                "nodes_generated": None,
                "search_depth": None,
                "plan": None,
                "error": f"File not found: {str(e)[:200]}"
            }

        except Exception as e:
            logger.error(f"[ERROR] Exception: {e}")
            return {
                "success": False,
                "time": 0,
                "plan_cost": None,
                "nodes_expanded": None,
                "nodes_generated": None,
                "search_depth": None,
                "plan": None,
                "error": f"Exception: {str(e)[:200]}"
            }

    @staticmethod
    def _parse_fd_output(output_text: str) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from Fast Downward output.

        Args:
            output_text: Combined stdout + stderr from FD

        Returns:
            Dict with extracted metrics
        """
        result = {
            "plan_cost": None,
            "nodes_expanded": None,
            "nodes_generated": None,
            "search_depth": None,
            "plan": None
        }

        # Plan cost (use "Plan length")
        cost_match = re.search(r"Plan length:\s*(\d+)", output_text)
        if cost_match:
            result["plan_cost"] = int(cost_match.group(1))

        # Nodes expanded (take last occurrence)
        nodes_expanded_matches = list(re.finditer(r"Expanded\s+(\d+)\s+state", output_text))
        if nodes_expanded_matches:
            result["nodes_expanded"] = int(nodes_expanded_matches[-1].group(1))

        # Nodes generated (take last occurrence)
        nodes_generated_matches = list(re.finditer(r"Generated\s+(\d+)\s+state", output_text))
        if nodes_generated_matches:
            result["nodes_generated"] = int(nodes_generated_matches[-1].group(1))

        # Search depth
        depth_match = re.search(r"Search depth:\s*(\d+)", output_text)
        if depth_match:
            result["search_depth"] = int(depth_match.group(1))

        # Extract plan actions
        plan_section = re.search(
            r"Solution found\.\n(.*?)(?:Plan length:|$)",
            output_text,
            re.DOTALL
        )
        if plan_section:
            actions = []
            for line in plan_section.group(1).strip().split('\n'):
                line = line.strip()
                if line and line.startswith('('):
                    actions.append(line)
            if actions:
                result["plan"] = actions

        return result