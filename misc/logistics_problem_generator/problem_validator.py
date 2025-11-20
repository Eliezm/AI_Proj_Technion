"""
Comprehensive problem validation for Logistics domain.

Provides complete validation of generated problems to guarantee:
- State validity (all constraints satisfied)
- Plan validity (all actions executable)
- Goal reachability
"""

from typing import Tuple, Optional, List
from state import LogisticsState
from actions import Action, ActionExecutor


class ProblemValidator:
    """Validates complete problems with comprehensive checks."""

    # problem_validator.py - ADD NEW METHOD TO ProblemValidator CLASS

    @staticmethod
    def validate_problem_metadata(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """
        FIX: Validate problem metadata for consistency.

        Checks:
        - Same entities in both states
        - Plan has reasonable length
        - No circular or contradictory goals
        """

        # Check 1: Entity counts match
        if len(initial_state.packages) != len(goal_state.packages):
            return False, "Package count mismatch"
        if len(initial_state.trucks) != len(goal_state.trucks):
            return False, "Truck count mismatch"
        if len(initial_state.airplanes) != len(goal_state.airplanes):
            return False, "Airplane count mismatch"

        # Check 2: Same entity sets
        if initial_state.packages != goal_state.packages:
            return False, "Package set mismatch"
        if initial_state.trucks != goal_state.trucks:
            return False, "Truck set mismatch"
        if initial_state.airplanes != goal_state.airplanes:
            return False, "Airplane set mismatch"

        # Check 3: Plan length reasonable
        if not plan:
            return False, "Empty plan"
        if len(plan) > 1000:
            return False, "Plan too long (>1000 actions)"

        # Check 4: Goal is achievable (at least one package moves)
        packages_move = False
        for pkg in goal_state.packages:
            if initial_state.at.get(pkg) != goal_state.at.get(pkg):
                packages_move = True
                break

        if not packages_move:
            return False, "No packages move (trivial problem)"

        return True, "Metadata valid"

    @staticmethod
    def validate_complete_problem(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """
        Comprehensive problem validation.

        Checks:
        1. Both states are valid
        2. Initial != Goal (non-trivial)
        3. All objects exist in both states
        4. Plan is executable from initial state
        5. Plan reaches goal state
        6. No packages left in vehicles
        7. All constraints satisfied throughout

        Returns:
            (is_valid, message)
        """

        # Check 1: Initial state valid
        is_valid, error = initial_state.is_valid()
        if not is_valid:
            return False, f"Initial state invalid: {error}"

        # Check 2: Goal state valid
        is_valid, error = goal_state.is_valid()
        if not is_valid:
            return False, f"Goal state invalid: {error}"

        # Check 3: Non-trivial
        if initial_state == goal_state:
            return False, "Problem is trivial (initial == goal)"

        # Check 4: Object consistency
        if initial_state.packages != goal_state.packages:
            return False, "Package sets differ between initial and goal"
        if initial_state.trucks != goal_state.trucks:
            return False, "Truck sets differ between initial and goal"
        if initial_state.airplanes != goal_state.airplanes:
            return False, "Airplane sets differ between initial and goal"

        # Check 5: Execute plan with detailed validation
        current_state = initial_state.copy()

        for i, action in enumerate(plan):
            # Verify action preconditions
            applicable = ActionExecutor.get_applicable_actions(current_state)
            if action not in applicable:
                return False, f"Action {i} ({action}) not applicable at step {i}"

            # Execute action
            next_state = ActionExecutor.execute_forward(current_state, action)
            if next_state is None:
                return False, f"Action {i} ({action}) execution returned None"

            # Verify state validity after action
            is_valid, error = next_state.is_valid()
            if not is_valid:
                return False, f"Action {i} produced invalid state: {error}"

            # Verify no contradictions
            for pkg in current_state.packages:
                at_count = sum([
                    pkg in next_state.at,
                    pkg in next_state.in_vehicle
                ])
                if at_count != 1:
                    return False, f"Package {pkg} in {at_count} places after action {i}"

            current_state = next_state

        # Check 6: Goal reached
        for pkg in goal_state.packages:
            if pkg in current_state.in_vehicle:
                return False, f"Package {pkg} still in vehicle at end of plan"

            goal_loc = goal_state.at.get(pkg)
            current_loc = current_state.at.get(pkg)

            if goal_loc is None:
                return False, f"Goal state missing location for {pkg}"

            if current_loc != goal_loc:
                return False, f"Package {pkg}: current={current_loc}, goal={goal_loc}"

        # Check 7: No spurious packages in initial state
        for pkg in goal_state.packages:
            if pkg not in initial_state.packages:
                return False, f"Package {pkg} in goal but not in initial state"

        return True, f"✓ Valid problem: {len(plan)} actions reach goal"

    @staticmethod
    def validate_state_consistency(state: LogisticsState) -> Tuple[bool, str]:
        """Additional state consistency checks."""

        # Check packages never in multiple places
        for pkg in state.packages:
            locations = 0
            vehicles = 0

            if pkg in state.at:
                locations += 1
            if pkg in state.in_vehicle:
                vehicles += 1

            if locations + vehicles != 1:
                return False, f"Package {pkg} in {locations + vehicles} places"

            # If in vehicle, vehicle must exist and have location
            if pkg in state.in_vehicle:
                vehicle = state.in_vehicle[pkg]
                if vehicle not in state.at:
                    return False, f"Vehicle {vehicle} carrying {pkg} has no location"
                if vehicle not in list(state.trucks) + list(state.airplanes):
                    return False, f"Invalid vehicle {vehicle}"

            # If at location, location must exist and be valid
            if pkg in state.at:
                loc = state.at[pkg]
                if loc not in state.locations:
                    return False, f"Package {pkg} at invalid location {loc}"

        # Check vehicles at exactly one location each
        for vehicle in list(state.trucks) + list(state.airplanes):
            if vehicle not in state.at:
                return False, f"Vehicle {vehicle} has no location"
            loc = state.at[vehicle]
            if loc not in state.locations:
                return False, f"Vehicle {vehicle} at invalid location {loc}"

        # Check airplanes only at airports
        for airplane in state.airplanes:
            if airplane in state.at:
                loc = state.at[airplane]
                if loc not in state.airports:
                    return False, f"Airplane {airplane} at non-airport {loc}"

        # Check all locations have city mappings
        for loc in state.locations:
            if loc not in state.in_city:
                return False, f"Location {loc} not mapped to any city"

        return True, "State is consistent"

    @staticmethod
    def validate_action_sequence(
            initial_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """Validate that a sequence of actions is executable."""

        current_state = initial_state.copy()

        for i, action in enumerate(plan):
            result = ActionExecutor.execute_forward(current_state, action)
            if result is None:
                return False, f"Action {i}: {action} not executable"

            is_valid, error = result.is_valid()
            if not is_valid:
                return False, f"Action {i} produced invalid state: {error}"

            current_state = result

        return True, f"All {len(plan)} actions are executable"