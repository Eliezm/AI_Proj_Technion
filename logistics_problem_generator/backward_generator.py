"""
Core backward search generator for Logistics problem generation.

Requirement #14: Generate problems using backward state-space search.
"""

import random
from typing import List, Tuple, Optional
from state import LogisticsState
from actions import Action, ActionExecutor, ActionType
from goal_archetypes import GoalArchetypeGenerator, GoalArchetype
from logistics_problem_builder import LogisticsProblemBuilder
from config import LogisticsGenerationParams, DEFAULT_LOGISTICS_PARAMS


class ReverseActionExecutor:
    """
    Executes actions in reverse to generate problems via backward search.
    """

    @staticmethod
    def undo_load_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """Undo load-truck: restore package to location."""
        if obj not in state.in_vehicle or state.in_vehicle[obj] != truck:
            return None
        if truck not in state.at:
            return None

        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_unload_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """Undo unload-truck: restore package to vehicle."""

        # CRITICAL: Check all preconditions before reversing
        if obj in state.in_vehicle:
            return None  # Already in vehicle

        if obj not in state.at or state.at[obj] != loc:
            return None  # Package not at this location

        if truck not in state.trucks:
            return None  # Invalid truck

        if truck not in state.at or state.at[truck] != loc:
            return None  # Truck not at this location

        if truck in state.in_vehicle:
            return None  # Truck cannot be in a vehicle

        # FIX: Verify this is the reverse of a valid forward action
        # In forward: UNLOAD requires (in ?obj ?truck) and (at ?truck ?loc)
        # So reverse must produce that state

        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = truck

        # VALIDATE before returning
        is_valid, error = new_state.is_valid()
        if not is_valid:
            return None

        return new_state

    @staticmethod
    def undo_load_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """Undo load-airplane: restore package to location."""
        if obj not in state.in_vehicle or state.in_vehicle[obj] != airplane:
            return None
        if loc not in state.airports:
            return None
        if airplane not in state.at or state.at[airplane] != loc:
            return None

        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_unload_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """Undo unload-airplane: restore package to vehicle."""
        if obj in state.in_vehicle or obj not in state.at:
            return None
        if state.at[obj] != loc:
            return None
        if loc not in state.airports:
            return None
        if airplane not in state.airplanes or airplane not in state.at:
            return None
        if state.at[airplane] != loc:
            return None

        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = airplane

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_drive_truck(state: LogisticsState, truck: str, origin_loc: str) -> Optional[LogisticsState]:
        """Undo drive-truck: move truck back to origin."""
        if truck not in state.at or state.at[truck] == origin_loc:
            return None
        current_loc = state.at[truck]
        if current_loc not in state.locations or origin_loc not in state.locations:
            return None
        current_city = state.in_city.get(current_loc)
        origin_city = state.in_city.get(origin_loc)
        if current_city != origin_city or not current_city:
            return None

        new_state = state.copy()
        new_state.at[truck] = origin_loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_fly_airplane(state: LogisticsState, airplane: str, origin_loc: str) -> Optional[LogisticsState]:
        """Undo fly-airplane: move airplane back to origin."""
        if airplane not in state.at or state.at[airplane] == origin_loc:
            return None
        current_loc = state.at[airplane]
        if current_loc not in state.airports or origin_loc not in state.airports:
            return None

        new_state = state.copy()
        new_state.at[airplane] = origin_loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def get_applicable_reverse_actions(state: LogisticsState) -> List[Tuple[Action, LogisticsState]]:
        """Get all applicable reverse actions and their resulting states."""
        results = []

        # Undo load-truck actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                truck = state.in_vehicle[pkg]
                if truck in state.at:
                    loc = state.at[truck]
                    new_state = ReverseActionExecutor.undo_load_truck(state, pkg, truck, loc)
                    if new_state is not None:
                        action = Action(ActionType.LOAD_TRUCK, [pkg, truck, loc])
                        results.append((action, new_state))

        # Undo unload-truck actions
        for pkg in state.packages:
            if pkg in state.at and pkg not in state.in_vehicle:
                pkg_loc = state.at[pkg]
                for truck in state.trucks:
                    if truck in state.at and state.at[truck] == pkg_loc:
                        new_state = ReverseActionExecutor.undo_unload_truck(state, pkg, truck, pkg_loc)
                        if new_state is not None:
                            action = Action(ActionType.UNLOAD_TRUCK, [pkg, truck, pkg_loc])
                            results.append((action, new_state))

        # Undo load-airplane actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                vehicle = state.in_vehicle[pkg]
                if vehicle in state.airplanes and vehicle in state.at:
                    loc = state.at[vehicle]
                    if loc in state.airports:
                        new_state = ReverseActionExecutor.undo_load_airplane(state, pkg, vehicle, loc)
                        if new_state is not None:
                            action = Action(ActionType.LOAD_AIRPLANE, [pkg, vehicle, loc])
                            results.append((action, new_state))

        # Undo unload-airplane actions
        for pkg in state.packages:
            if pkg in state.at and pkg not in state.in_vehicle:
                pkg_loc = state.at[pkg]
                if pkg_loc in state.airports:
                    for airplane in state.airplanes:
                        if airplane in state.at and state.at[airplane] == pkg_loc:
                            new_state = ReverseActionExecutor.undo_unload_airplane(state, pkg, airplane, pkg_loc)
                            if new_state is not None:
                                action = Action(ActionType.UNLOAD_AIRPLANE, [pkg, airplane, pkg_loc])
                                results.append((action, new_state))

        # Undo drive-truck actions
        for truck in state.trucks:
            if truck in state.at:
                current_loc = state.at[truck]
                current_city = state.in_city.get(current_loc)
                if current_city:
                    for other_loc in state.locations:
                        if state.in_city.get(other_loc) == current_city and other_loc != current_loc:
                            new_state = ReverseActionExecutor.undo_drive_truck(state, truck, other_loc)
                            if new_state is not None:
                                action = Action(ActionType.DRIVE_TRUCK, [truck, other_loc, current_loc, current_city])
                                results.append((action, new_state))

        # Undo fly-airplane actions
        for airplane in state.airplanes:
            if airplane in state.at:
                current_loc = state.at[airplane]
                if current_loc in state.airports:
                    for other_airport in state.airports:
                        if other_airport != current_loc:
                            new_state = ReverseActionExecutor.undo_fly_airplane(state, airplane, other_airport)
                            if new_state is not None:
                                action = Action(ActionType.FLY_AIRPLANE, [airplane, other_airport, current_loc])
                                results.append((action, new_state))

        # Deduplicate by state
        seen_states = set()
        unique_results = []
        for action, new_state in results:
            state_hash = hash(new_state)
            if state_hash not in seen_states:
                seen_states.add(state_hash)
                unique_results.append((action, new_state))

        return unique_results


class BackwardProblemGenerator:
    """
    Generate Logistics problems using backward state-space search.
    """

    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
        self.archetype_gen = GoalArchetypeGenerator(random_seed)
        if random_seed is not None:
            random.seed(random_seed)

    def _verify_plan(
            self,
            initial_state: LogisticsState,
            goal_state: LogisticsState,
            plan: List[Action]
    ) -> Tuple[bool, str]:
        """Strict plan verification with complete state checking."""

        # Check 1: Initial state is valid
        is_valid, error = initial_state.is_valid()
        if not is_valid:
            return False, f"Initial state invalid: {error}"

        # Check 2: Goal state is valid
        is_valid, error = goal_state.is_valid()
        if not is_valid:
            return False, f"Goal state invalid: {error}"

        # Check 3: Initial != Goal (non-trivial)
        if initial_state == goal_state:
            return False, "Trivial problem (initial == goal)"

        # Check 4: Execute plan step by step
        current = initial_state.copy()
        for i, action in enumerate(plan):
            # Verify action can be executed
            next_state = ActionExecutor.execute_forward(current, action)
            if next_state is None:
                return False, f"Action {i} ({action}) cannot be executed at state: {current}"

            # Verify resulting state is valid
            is_valid, error = next_state.is_valid()
            if not is_valid:
                return False, f"Action {i} produced invalid state: {error}"

            current = next_state

        # Check 5: All goal packages at goal locations
        for pkg in goal_state.packages:
            goal_loc = goal_state.at.get(pkg)
            current_loc = current.at.get(pkg)

            if pkg in current.in_vehicle:
                return False, f"Package {pkg} still in vehicle at end of plan"

            if goal_loc is None:
                return False, f"Goal state missing location for {pkg}"

            if current_loc != goal_loc:
                return False, f"Package {pkg}: current={current_loc}, goal={goal_loc}"

        # Check 6: No spurious package movements
        for pkg in initial_state.packages:
            if pkg not in goal_state.packages:
                return False, f"Package {pkg} in initial but not in goal"

        return True, f"Plan valid: {len(plan)} actions reach goal"

    def _ensure_goal_is_different(self, initial_state: LogisticsState, goal_dict: dict) -> bool:
        """Check if goal dict creates a state different from initial."""
        for pkg, dest_loc in goal_dict.items():
            current_loc = initial_state.at.get(pkg)
            if current_loc != dest_loc:
                return True
        return False

    def generate_goal_dict_robust(
            self,
            initial_state: LogisticsState,
            packages: List[str],
            num_packages: int,
            max_attempts: int = 50
    ) -> dict:
        """
        Generate a robust goal dict that ensures:
        1. It's non-empty
        2. It creates a state different from initial
        3. It only uses valid locations
        """
        for attempt in range(max_attempts):
            # Try a random archetype
            archetype = random.choice(list(GoalArchetype))
            goal_dict = self.archetype_gen.generate_archetype(
                archetype,
                initial_state,
                packages,
                num_packages
            )

            # Validate goal dict
            if goal_dict and self._ensure_goal_is_different(initial_state, goal_dict):
                # Verify all destinations are valid locations
                all_valid = all(
                    dest_loc in initial_state.locations
                    for dest_loc in goal_dict.values()
                )
                if all_valid:
                    return goal_dict

        # Fallback: brute force a valid goal dict
        for pkg in packages:
            current_loc = initial_state.at.get(pkg)
            other_locs = [loc for loc in initial_state.locations if loc != current_loc]
            if other_locs:
                return {pkg: random.choice(other_locs)}

        # Last resort: use all packages
        goal_dict = {}
        for pkg in packages[:num_packages]:
            current_loc = initial_state.at.get(pkg)
            other_locs = [loc for loc in initial_state.locations if loc != current_loc]
            if other_locs:
                goal_dict[pkg] = random.choice(other_locs)
                if len(goal_dict) >= num_packages:
                    break

        return goal_dict

    def generate_problem(
            self,
            difficulty: str,
            generation_params: Optional[LogisticsGenerationParams] = None,
            target_plan_length: Optional[int] = None,
            archetype: Optional[GoalArchetype] = None,
            tolerance: int = 1,
            max_retries: int = 10
    ) -> Tuple[LogisticsState, LogisticsState, List[Action], GoalArchetype]:
        """
        Generate problem with 100% validity guarantee.

        Raises exception if problem cannot be generated.
        """
        from config import DIFFICULTY_TIERS
        from problem_validator import ProblemValidator

        if generation_params is None:
            generation_params = DEFAULT_LOGISTICS_PARAMS.get(difficulty)
        if target_plan_length is None:
            tier = DIFFICULTY_TIERS.get(difficulty)
            target_plan_length = tier.target_length if tier else 10

        # Retry loop
        for retry in range(max_retries):
            try:
                # Step 1: Build valid world
                initial_world, packages, trucks, airplanes = LogisticsProblemBuilder.build_world(
                    generation_params,
                    random_seed=self.random_seed + retry if self.random_seed else None
                )

                # Step 2: Generate goal with archetype tracking
                goal_dict, used_archetype = self.generate_goal_dict_robust_with_archetype(
                    initial_world,
                    packages,
                    len(packages),
                    max_attempts=50
                )

                if not goal_dict:
                    continue  # Retry

                # Step 3: Create goal state and validate
                goal_state = initial_world.copy()
                for pkg, dest_loc in goal_dict.items():
                    if pkg in goal_state.in_vehicle:
                        del goal_state.in_vehicle[pkg]
                    goal_state.at[pkg] = dest_loc

                is_valid, error = goal_state.is_valid()
                if not is_valid:
                    continue  # Retry

                if goal_state == initial_world:
                    continue  # Retry: trivial

                # Step 4: Backward search
                current_state = goal_state.copy()
                plan = []
                iteration = 0
                max_iterations = max(target_plan_length * 3, 150)

                while len(plan) < target_plan_length and iteration < max_iterations:
                    iteration += 1

                    reverse_actions = ReverseActionExecutor.get_applicable_reverse_actions(current_state)
                    if not reverse_actions:
                        break

                    action, new_state = random.choice(reverse_actions)

                    is_valid, _ = new_state.is_valid()
                    if not is_valid:
                        continue

                    if new_state != current_state:
                        plan.insert(0, action)
                        current_state = new_state

                initial_state = current_state

                # Step 5: Comprehensive validation
                is_valid, reason = ProblemValidator.validate_complete_problem(
                    initial_state,
                    goal_state,
                    plan
                )

                if is_valid:
                    return initial_state, goal_state, plan, used_archetype

            except Exception as e:
                continue  # Retry

        raise ValueError(f"Failed to generate valid problem after {max_retries} attempts")

    def generate_goal_dict_robust_with_archetype(
            self,
            initial_state: LogisticsState,
            packages: List[str],
            num_packages: int,
            max_attempts: int = 50
    ) -> Tuple[dict, GoalArchetype]:
        """
        Generate a robust goal dict with archetype tracking.

        Returns:
            (goal_dict, used_archetype)
        """
        archetypes_tried = []

        # Try each archetype at least once
        all_archetypes = list(GoalArchetype)
        random.shuffle(all_archetypes)

        for archetype in all_archetypes:
            goal_dict = self.archetype_gen.generate_archetype(
                archetype,
                initial_state,
                packages,
                num_packages
            )

            # Validate goal dict
            if goal_dict and self._ensure_goal_is_different(initial_state, goal_dict):
                # Verify all destinations are valid locations
                all_valid = all(
                    dest_loc in initial_state.locations
                    for dest_loc in goal_dict.values()
                )
                if all_valid:
                    return goal_dict, archetype

            archetypes_tried.append(archetype)

        # If all archetypes failed, try random selection multiple times
        for attempt in range(max_attempts - len(all_archetypes)):
            archetype = random.choice(all_archetypes)
            goal_dict = self.archetype_gen.generate_archetype(
                archetype,
                initial_state,
                packages,
                num_packages
            )

            if goal_dict and self._ensure_goal_is_different(initial_state, goal_dict):
                all_valid = all(
                    dest_loc in initial_state.locations
                    for dest_loc in goal_dict.values()
                )
                if all_valid:
                    return goal_dict, archetype

        # Fallback: brute force a valid goal dict
        for pkg in packages:
            current_loc = initial_state.at.get(pkg)
            other_locs = [loc for loc in initial_state.locations if loc != current_loc]
            if other_locs:
                return {pkg: random.choice(other_locs)}, GoalArchetype.MANY_TO_MANY

        # Last resort
        goal_dict = {}
        for pkg in packages[:num_packages]:
            current_loc = initial_state.at.get(pkg)
            other_locs = [loc for loc in initial_state.locations if loc != current_loc]
            if other_locs:
                goal_dict[pkg] = random.choice(other_locs)
                if len(goal_dict) >= num_packages:
                    break

        return goal_dict, GoalArchetype.MANY_TO_MANY

    def _generate_simple_forward_plan(
            self,
            initial_state: LogisticsState,
            goal_state: LogisticsState
    ) -> List[Action]:
        """
        Fallback: Generate a simple forward plan using greedy approach.
        """
        plan = []
        current_state = initial_state.copy()
        max_steps = 50
        steps = 0

        # Get packages that need to move
        packages_to_move = []
        for pkg in goal_state.packages:
            goal_loc = goal_state.at.get(pkg)
            current_loc = current_state.at.get(pkg)
            if goal_loc and current_loc and goal_loc != current_loc:
                packages_to_move.append((pkg, goal_loc))

        # Try to move each package
        for pkg, goal_loc in packages_to_move:
            while steps < max_steps:
                steps += 1
                if current_state.at.get(pkg) == goal_loc:
                    break

                # Get all applicable actions
                applicable = ActionExecutor.get_applicable_actions(current_state)

                # Filter for actions that move this package closer
                good_actions = []
                for action in applicable:
                    if action.params[0] == pkg:  # Action involves our package
                        next_state = ActionExecutor.execute_forward(current_state, action)
                        if next_state:
                            good_actions.append((action, next_state))

                if not good_actions:
                    break

                # Pick first applicable action
                action, next_state = good_actions[0]
                plan.append(action)
                current_state = next_state

        return plan

    def generate_problem_with_debug(
            self,
            difficulty: str,
            generation_params: Optional[LogisticsGenerationParams] = None,
            target_plan_length: Optional[int] = None,
            archetype: Optional[GoalArchetype] = None,
            tolerance: int = 1,
            debug: bool = False
    ) -> Tuple[LogisticsState, LogisticsState, List[Action], GoalArchetype]:
        """
        Generate a Logistics problem with optional debugging output.
        """
        if debug:
            print(f"[DEBUG] Starting problem generation for difficulty={difficulty}")

        initial_state, goal_state, plan, used_archetype = self.generate_problem(
            difficulty, generation_params, target_plan_length, archetype, tolerance
        )

        if debug:
            print(f"[DEBUG] Generated problem:")
            print(f"       Archetype: {used_archetype.value}")
            print(f"       Plan length: {len(plan)}")
            print(f"       World: {len(initial_state.cities)} cities, "
                  f"{len(initial_state.locations)} locs, "
                  f"{len(initial_state.packages)} pkgs")

        return initial_state, goal_state, plan, used_archetype