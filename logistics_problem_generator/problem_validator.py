"""Comprehensive problem validation for 100% solvability guarantee."""

from typing import Tuple, Optional, List
from state import LogisticsState
from actions import Action, ActionExecutor


class ProblemValidator:
    """Validates generated problems meet all requirements."""

    @staticmethod
    def validate_complete_problem(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            backward_plan: List[Action]
    ) -> Tuple[bool, str]:
        """
        Complete validation pipeline.

        Returns: (is_valid, reason)
        """

        # Check 1: States are valid
        is_valid, error = initial_state.is_valid()
        if not is_valid:
            return False, f"Initial state invalid: {error}"

        is_valid, error = goal_state.is_valid()
        if not is_valid:
            return False, f"Goal state invalid: {error}"

        # Check 2: Problem is non-trivial
        if initial_state == goal_state:
            return False, "Trivial problem"

        # Check 3: Backward plan is valid
        if not ProblemValidator._verify_backward_plan(initial_state, goal_state, backward_plan):
            return False, "Backward plan does not connect initial to goal"

        # Check 4: Problem is solvable (BFS)
        if not ProblemValidator._is_forward_solvable(initial_state, goal_state):
            return False, "Problem not solvable via forward search"

        # Check 5: World is well-formed
        if not ProblemValidator._validate_world_structure(initial_state):
            return False, "World structure is malformed"

        return True, "Problem passes all validations"

    @staticmethod
    def _verify_backward_plan(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            plan: List[Action]
    ) -> bool:
        """Verify backward plan connects initial to goal."""
        if not plan:
            return initial_state == goal_state

        # Execute plan forward from initial
        current = initial_state.copy()
        for action in plan:
            current = ActionExecutor.execute_forward(current, action)
            if current is None:
                return False

        # Check goal
        return all(
            goal_state.at.get(pkg) == current.at.get(pkg)
            for pkg in goal_state.packages
        )

    @staticmethod
    def _is_forward_solvable(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            max_depth: int = 100
    ) -> bool:
        """Check if solvable via BFS (expensive but guaranteed)."""
        from collections import deque

        queue = deque([(initial_state, 0)])
        visited = {hash(initial_state)}

        while queue:
            current, depth = queue.popleft()

            if depth > max_depth:
                return False  # Search too deep

            # Check goal
            if all(
                    goal_state.at.get(pkg) == current.at.get(pkg)
                    for pkg in goal_state.packages
            ):
                return True

            # Expand
            for action in ActionExecutor.get_applicable_actions(current):
                next_state = ActionExecutor.execute_forward(current, action)
                if next_state:
                    h = hash(next_state)
                    if h not in visited:
                        visited.add(h)
                        queue.append((next_state, depth + 1))

        return False

    @staticmethod
    def _validate_world_structure(state: LogisticsState) -> bool:
        """Validate world has required structure."""

        # At least 1 city
        if len(state.cities) == 0:
            return False

        # At least 1 location
        if len(state.locations) == 0:
            return False

        # At least 1 package
        if len(state.packages) == 0:
            return False

        # At least 1 truck
        if len(state.trucks) == 0:
            return False

        # At least 1 airplane
        if len(state.airplanes) == 0:
            return False

        # At least 1 airport (for inter-city)
        if len(state.airports) == 0:
            return False

        return True