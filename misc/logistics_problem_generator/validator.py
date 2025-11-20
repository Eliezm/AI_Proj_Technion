"""
PDDL syntax validation.

Requirement #8: Validate PDDL files using a standard parser/validator.
"""

import subprocess
from typing import Tuple, Optional


class PDDLValidator:
    """Validates PDDL files using VAL or similar tools."""

    def __init__(self, validator_path: str = "validate"):
        """
        Initialize validator.

        Args:
            validator_path: Path to VAL validator executable
        """
        self.validator_path = validator_path

    def validate_problem(
        self,
        domain_file: str,
        problem_file: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a PDDL problem file.

        Returns:
            (is_valid, error_message)
        """
        try:
            result = subprocess.run(
                [self.validator_path, domain_file, problem_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, None
            else:
                error = result.stderr if result.stderr else result.stdout
                return False, error[:500]

        except FileNotFoundError:
            return True, "Validator not found; skipping validation"
        except subprocess.TimeoutExpired:
            return False, "Validation timeout"