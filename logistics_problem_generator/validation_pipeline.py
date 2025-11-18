"""
Complete validation pipeline for 100% guarantee of valid, solvable problems.

This is the critical FIX #8 - comprehensive validation.
"""

import logging
from typing import Tuple, List, Dict, Any
from state import LogisticsState
from actions import Action, ActionExecutor
from problem_validator import ProblemValidator

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Complete 8-stage validation pipeline to guarantee problem validity.
    """

    @staticmethod
    def validate_all_stages(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete validation pipeline.

        Returns:
            (is_valid, detailed_report)
        """

        report = {
            "valid": False,
            "stages": {},
            "errors": []
        }

        # STAGE 1: State validity
        stage1, msg1 = ValidationPipeline._stage1_state_validity(initial_state, goal_state)
        report["stages"]["1_state_validity"] = stage1
        if not stage1:
            report["errors"].append(f"Stage 1: {msg1}")
            return False, report

        # STAGE 2: Non-triviality
        stage2, msg2 = ValidationPipeline._stage2_non_triviality(initial_state, goal_state)
        report["stages"]["2_non_triviality"] = stage2
        if not stage2:
            report["errors"].append(f"Stage 2: {msg2}")
            return False, report

        # STAGE 3: Entity consistency
        stage3, msg3 = ValidationPipeline._stage3_entity_consistency(initial_state, goal_state)
        report["stages"]["3_entity_consistency"] = stage3
        if not stage3:
            report["errors"].append(f"Stage 3: {msg3}")
            return False, report

        # STAGE 4: Plan executability
        stage4, msg4 = ValidationPipeline._stage4_plan_executability(initial_state, plan)
        report["stages"]["4_plan_executability"] = stage4
        if not stage4:
            report["errors"].append(f"Stage 4: {msg4}")
            return False, report

        # STAGE 5: Goal reachability
        final_state, stage5, msg5 = ValidationPipeline._stage5_goal_reachability(
            initial_state, plan, goal_state
        )
        report["stages"]["5_goal_reachability"] = stage5
        if not stage5:
            report["errors"].append(f"Stage 5: {msg5}")
            return False, report

        # STAGE 6: Constraint preservation
        stage6, msg6 = ValidationPipeline._stage6_constraint_preservation(initial_state, plan)
        report["stages"]["6_constraint_preservation"] = stage6
        if not stage6:
            report["errors"].append(f"Stage 6: {msg6}")
            return False, report

        # STAGE 7: Package accounting
        stage7, msg7 = ValidationPipeline._stage7_package_accounting(
            initial_state, goal_state, final_state
        )
        report["stages"]["7_package_accounting"] = stage7
        if not stage7:
            report["errors"].append(f"Stage 7: {msg7}")
            return False, report

        # STAGE 8: Plan length reasonableness
        stage8, msg8 = ValidationPipeline._stage8_plan_length(plan)
        report["stages"]["8_plan_length"] = stage8
        if not stage8:
            report["errors"].append(f"Stage 8: {msg8}")
            return False, report

        report["valid"] = True
        return True, report

    @staticmethod
    def _stage1_state_validity(
            initial_state: LogisticsState,
            goal_state: LogisticsState
    ) -> Tuple[bool, str]:
        """STAGE 1: Both states must be valid."""
        is_valid, error = initial_state.is_valid()
        if not is_valid:
            return False, f"Initial state invalid: {error}"

        is_valid, error = goal_state.is_valid()
        if not is_valid:
            return False, f"Goal state invalid: {error}"

        return True, "Both states valid"

    @staticmethod
    def _stage2_non_triviality(
            initial_state: LogisticsState,
            goal_state: LogisticsState
    ) -> Tuple[bool, str]:
        """STAGE 2: Problem must be non-trivial."""
        if initial_state == goal_state:
            return False, "States are identical (trivial problem)"

        # At least one package must move
        packages_move = False
        for pkg in initial_state.packages:
            if initial_state.at.get(pkg) != goal_state.at.get(pkg):
                packages_move = True
                break

        if not packages_move:
            return False, "No packages actually move"

        return True, "Problem is non-trivial"

    @staticmethod
    def _stage3_entity_consistency(
            initial_state: LogisticsState,
            goal_state: LogisticsState
    ) -> Tuple[bool, str]:
        """STAGE 3: Entity sets must match."""
        if initial_state.packages != goal_state.packages:
            return False, "Package sets differ"
        if initial_state.trucks != goal_state.trucks:
            return False, "Truck sets differ"
        if initial_state.airplanes != goal_state.airplanes:
            return False, "Airplane sets differ"
        if initial_state.locations != goal_state.locations:
            return False, "Location sets differ"
        if initial_state.cities != goal_state.cities:
            return False, "City sets differ"

        return True, "Entity sets consistent"

    @staticmethod
    def _stage4_plan_executability(
            initial_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """STAGE 4: All plan actions must be executable."""
        if not plan:
            return False, "Empty plan"

        current = initial_state.copy()
        for i, action in enumerate(plan):
            applicable = ActionExecutor.get_applicable_actions(current)
            if action not in applicable:
                return False, f"Action {i} not applicable: {action}"

            next_state = ActionExecutor.execute_forward(current, action)
            if next_state is None:
                return False, f"Action {i} execution failed: {action}"

            is_valid, error = next_state.is_valid()
            if not is_valid:
                return False, f"Action {i} produced invalid state: {error}"

            current = next_state

        return True, f"All {len(plan)} actions executable"

    @staticmethod
    def _stage5_goal_reachability(
            initial_state: LogisticsState,
            plan: List[Action],
            goal_state: LogisticsState
    ) -> Tuple[LogisticsState, bool, str]:
        """STAGE 5: Plan must reach goal."""
        current = initial_state.copy()
        for action in plan:
            current = ActionExecutor.execute_forward(current, action)

        for pkg in goal_state.packages:
            goal_loc = goal_state.at.get(pkg)
            current_loc = current.at.get(pkg)

            if pkg in current.in_vehicle:
                return current, False, f"Package {pkg} still in vehicle"

            if current_loc != goal_loc:
                return current, False, f"Package {pkg} not at goal: {current_loc} ≠ {goal_loc}"

        return current, True, "Goal reached"

    @staticmethod
    def _stage6_constraint_preservation(
            initial_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """STAGE 6: Logistics constraints must be preserved throughout."""
        current = initial_state.copy()

        for i, action in enumerate(plan):
            current = ActionExecutor.execute_forward(current, action)

            # Each package in exactly one place
            for pkg in current.packages:
                count = 0
                if pkg in current.at:
                    count += 1
                if pkg in current.in_vehicle:
                    count += 1
                if count != 1:
                    return False, f"After action {i}: package {pkg} in {count} places"

            # Trucks never in vehicles
            for truck in current.trucks:
                if truck in current.in_vehicle:
                    return False, f"After action {i}: truck {truck} in vehicle"

            # Airplanes only at airports
            for airplane in current.airplanes:
                loc = current.at.get(airplane)
                if loc and loc not in current.airports:
                    return False, f"After action {i}: airplane {airplane} at non-airport"

        return True, "Constraints preserved"

    @staticmethod
    def _stage7_package_accounting(
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            final_state: LogisticsState
    ) -> Tuple[bool, str]:
        """STAGE 7: Package accounting - no loss or creation."""
        if initial_state.packages != goal_state.packages:
            return False, "Package set changed"

        if final_state.packages != goal_state.packages:
            return False, "Final state has different packages"

        for pkg in initial_state.packages:
            if pkg not in final_state.packages:
                return False, f"Package {pkg} disappeared"

        return True, "Package accounting correct"

    @staticmethod
    def _stage8_plan_length(plan: List[Action]) -> Tuple[bool, str]:
        """STAGE 8: Plan length must be reasonable."""
        if len(plan) == 0:
            return False, "Empty plan"

        if len(plan) > 1000:
            return False, f"Plan too long: {len(plan)} actions"

        return True, f"Plan length reasonable: {len(plan)} actions"

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print validation report."""
        print("\n" + "=" * 70)
        if report["valid"]:
            print("✓ VALIDATION PASSED - Problem is 100% valid")
        else:
            print("✗ VALIDATION FAILED")
        print("=" * 70)

        print("\nStage Results:")
        for stage_name, stage_passed in report["stages"].items():
            status = "✓" if stage_passed else "✗"
            print(f"  {status} {stage_name}")

        if report["errors"]:
            print("\nErrors:")
            for error in report["errors"]:
                print(f"  • {error}")

        print()