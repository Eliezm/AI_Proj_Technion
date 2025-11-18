"""
Integration with Fast Downward baseline planner.

FIXED VERSION: Proper timeout enforcement with process termination guarantee.
"""

import subprocess
import json
import re
import os
import time
import logging
import signal
import sys
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class FastDownwardRunner:
    """Runs Fast Downward with guaranteed timeout enforcement."""

    def __init__(self, timeout: int = 260):  # ← FIXED: Changed from 600 to 260
        """
        Initialize Fast Downward runner.

        Args:
            timeout: Maximum time in seconds for SEARCH PHASE ONLY (default: 260)
        """
        self.timeout = timeout

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        fd_bin_dir = os.path.join(project_root, "downward", "builds", "release", "bin")

        if os.name == 'nt':  # Windows
            self.fd_bin = os.path.join(fd_bin_dir, "downward.exe")
        else:  # Linux/macOS
            self.fd_bin = os.path.join(fd_bin_dir, "downward")

        self.fd_translate = os.path.join(fd_bin_dir, "translate", "translate.py")
        self.temp_dir = os.path.join(project_root, "generation_temp")
        os.makedirs(self.temp_dir, exist_ok=True)

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
        Run Fast Downward on a problem using 2-step translate/search process.

        FIXED: Proper timeout enforcement for search phase.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            search_config: Fast Downward search configuration string
            timeout: Maximum time in seconds for SEARCH PHASE ONLY

        Returns:
            Dict with search results and metrics
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

        # FIXED: Use provided timeout or fall back to self.timeout
        search_timeout = timeout if timeout is not None else self.timeout
        translate_timeout = 300

        logger.info(f"[TIMEOUT CONFIG] Translate: {translate_timeout}s, Search: {search_timeout}s")

        overall_start_time = time.time()
        problem_name = os.path.basename(problem_file)
        sas_file = os.path.join(self.temp_dir, "output.sas")

        try:
            # ==========================================================
            # PHASE 1: TRANSLATE (PDDL -> SAS)
            # ==========================================================
            logger.debug(f"[TRANSLATE] {problem_name}")
            translate_start = time.time()

            abs_domain = os.path.abspath(domain_file)
            abs_problem = os.path.abspath(problem_file)

            translate_cmd = (
                f'python "{self.fd_translate}" "{abs_domain}" '
                f'"{abs_problem}" --sas-file "{sas_file}"'
            )

            try:
                translate_result = subprocess.run(
                    translate_cmd,
                    shell=True,
                    cwd=os.path.abspath(".."),
                    capture_output=True,
                    text=True,
                    timeout=translate_timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"[TRANSLATE TIMEOUT] Exceeded {translate_timeout}s for {problem_name}")
                return {
                    "success": False,
                    "time": time.time() - overall_start_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": f"Translate timeout (>{translate_timeout}s)"
                }

            translate_time = time.time() - translate_start

            if translate_result.returncode != 0:
                error_msg = translate_result.stderr if translate_result.stderr else translate_result.stdout
                logger.debug(f"[TRANSLATE] Failed: {error_msg[:200]}")
                return {
                    "success": False,
                    "time": time.time() - overall_start_time,
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
                    "time": time.time() - overall_start_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": "Translate: SAS file not created"
                }

            logger.debug(f"[TRANSLATE] Success ({os.path.getsize(sas_file)} bytes) in {translate_time:.2f}s")

            # ==========================================================
            # PHASE 2: SEARCH (SAS -> Plan)
            # FIXED: Use robust subprocess handling to enforce timeout
            # ==========================================================
            logger.debug(f"[SEARCH] Starting with config: {search_config}, timeout: {search_timeout}s")
            search_start = time.time()

            # FIXED: Use Popen + communicate for better timeout control
            search_cmd_list = [
                self.fd_bin,
                "--search", search_config
            ]

            try:
                # Open SAS file for stdin
                with open(sas_file, 'r') as sas_input:
                    search_process = subprocess.Popen(
                        search_cmd_list,
                        stdin=sas_input,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.path.dirname(self.fd_bin)
                    )

                    try:
                        stdout_text, stderr_text = search_process.communicate(timeout=search_timeout)
                        search_returncode = search_process.returncode
                    except subprocess.TimeoutExpired:
                        # FIXED: Kill the process tree
                        logger.warning(f"[SEARCH TIMEOUT] Killing process after {search_timeout}s")
                        try:
                            if sys.platform == 'win32':
                                # Windows: use SIGKILL equivalent
                                search_process.kill()
                            else:
                                # Unix: first try SIGTERM, then SIGKILL
                                search_process.terminate()
                                try:
                                    search_process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    search_process.kill()
                            search_process.wait()
                        except Exception as e:
                            logger.debug(f"Error terminating process: {e}")

                        return {
                            "success": False,
                            "time": time.time() - overall_start_time,
                            "plan_cost": None,
                            "nodes_expanded": None,
                            "nodes_generated": None,
                            "search_depth": None,
                            "plan": None,
                            "error": f"Search timeout (>{search_timeout}s)"
                        }

                search_time = time.time() - search_start
                total_time = time.time() - overall_start_time

                output_text = stdout_text + stderr_text

                logger.debug(f"[SEARCH] Completed in {search_time:.2f}s (total: {total_time:.2f}s)")

            except Exception as e:
                logger.error(f"[SEARCH] Exception: {e}")
                return {
                    "success": False,
                    "time": time.time() - overall_start_time,
                    "plan_cost": None,
                    "nodes_expanded": None,
                    "nodes_generated": None,
                    "search_depth": None,
                    "plan": None,
                    "error": f"Search error: {str(e)[:200]}"
                }

            # Clean up SAS file
            try:
                if os.path.exists(sas_file):
                    os.remove(sas_file)
            except Exception as e:
                logger.debug(f"Could not remove SAS file: {e}")

            # ==========================================================
            # PHASE 3: PARSE OUTPUT
            # ==========================================================

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

        except Exception as e:
            logger.error(f"[ERROR] Unexpected exception: {e}")
            return {
                "success": False,
                "time": time.time() - overall_start_time,
                "plan_cost": None,
                "nodes_expanded": None,
                "nodes_generated": None,
                "search_depth": None,
                "plan": None,
                "error": f"Exception: {str(e)[:200]}"
            }

    @staticmethod
    def _parse_fd_output(output_text: str) -> Dict[str, Any]:
        """Extract comprehensive metrics from Fast Downward output."""
        result = {
            "plan_cost": None,
            "nodes_expanded": None,
            "nodes_generated": None,
            "search_depth": None,
            "plan": None
        }

        cost_match = re.search(r"Plan length:\s*(\d+)", output_text)
        if cost_match:
            result["plan_cost"] = int(cost_match.group(1))

        nodes_expanded_matches = list(re.finditer(r"Expanded\s+(\d+)\s+state", output_text))
        if nodes_expanded_matches:
            result["nodes_expanded"] = int(nodes_expanded_matches[-1].group(1))

        nodes_generated_matches = list(re.finditer(r"Generated\s+(\d+)\s+state", output_text))
        if nodes_generated_matches:
            result["nodes_generated"] = int(nodes_generated_matches[-1].group(1))

        depth_match = re.search(r"Search depth:\s*(\d+)", output_text)
        if depth_match:
            result["search_depth"] = int(depth_match.group(1))

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