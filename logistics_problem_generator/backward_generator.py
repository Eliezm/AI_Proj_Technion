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
from problem_validator import ProblemValidator


import logging
logger = logging.getLogger(__name__)



# backward_generator.py - ADD NEW CLASS and MODIFY generate_problem

# In backward_generator.py, REPLACE the StateDeduplicator class with this:

class StateDeduplicator:
    """Tracks visited states in backward search to prevent cycles."""

    def __init__(self, max_states: int = 5000):  # INCREASED from 500
        self.visited_hashes = set()
        self.visited_states = []
        self.max_states = max_states
        self.cycle_detected = False

    def is_visited(self, state: LogisticsState) -> bool:
        """Check if state has been visited."""
        return hash(state) in self.visited_hashes

    def mark_visited(self, state: LogisticsState) -> bool:
        """
        Mark state as visited.

        Returns True if successfully added, False if would exceed max_states.
        """
        state_hash = hash(state)
        if state_hash not in self.visited_hashes:
            if len(self.visited_hashes) >= self.max_states:
                self.cycle_detected = True
                return False
            self.visited_hashes.add(state_hash)
            self.visited_states.append(state)
            return True
        return True  # Already visited

    def reset(self):
        """Clear visited states."""
        self.visited_hashes.clear()
        self.visited_states.clear()
        self.cycle_detected = False

    def get_visited_count(self) -> int:
        """Get number of visited states."""
        return len(self.visited_hashes)



class ReverseActionExecutor:
    """
    Executes actions in reverse with STRICT precondition validation.

    CRITICAL: Every reverse action must validate that:
    1. It exactly undoes a valid forward action
    2. The preconditions of the forward action are satisfied in the previous state
    3. The new state is valid according to domain rules
    """

    @staticmethod
    def _validate_reverse_preconditions(
            state: LogisticsState,
            action_type: ActionType,
            params: List[str]
    ) -> Tuple[bool, str]:
        """
        FIX: Validate that the reverse action makes sense.

        Before executing reverse action, check that forward action
        preconditions would have been satisfied.
        """

        if action_type == ActionType.LOAD_TRUCK:
            obj, truck, loc = params
            # Forward precondition: at(truck, loc) AND at(obj, loc)
            # Current state must have: in(obj, truck) AND at(truck, loc)
            if obj not in state.in_vehicle:
                return False, f"Package {obj} not in vehicle (cannot undo LOAD)"
            if state.in_vehicle[obj] != truck:
                return False, f"Package {obj} not in truck {truck}"
            if state.at.get(truck) != loc:
                return False, f"Truck {truck} not at location {loc}"
            if obj in state.at:
                return False, f"Package {obj} cannot be both in truck and at location"

        elif action_type == ActionType.UNLOAD_TRUCK:
            obj, truck, loc = params
            # Forward precondition: at(truck, loc) AND in(obj, truck)
            # Current state must have: at(obj, loc) AND at(truck, loc)
            if obj not in state.at:
                return False, f"Package {obj} not at location (cannot undo UNLOAD)"
            if state.at[obj] != loc:
                return False, f"Package {obj} not at location {loc}"
            if state.at.get(truck) != loc:
                return False, f"Truck {truck} not at location {loc}"
            if obj in state.in_vehicle:
                return False, f"Package {obj} cannot be both in truck and at location"

        elif action_type == ActionType.LOAD_AIRPLANE:
            obj, airplane, loc = params
            if obj not in state.in_vehicle:
                return False, f"Package {obj} not in vehicle"
            if state.in_vehicle[obj] != airplane:
                return False, f"Package {obj} not in airplane {airplane}"
            if loc not in state.airports:
                return False, f"Location {loc} not an airport"
            if state.at.get(airplane) != loc:
                return False, f"Airplane {airplane} not at airport {loc}"

        elif action_type == ActionType.UNLOAD_AIRPLANE:
            obj, airplane, loc = params
            if obj not in state.at:
                return False, f"Package {obj} not at location"
            if state.at[obj] != loc:
                return False, f"Package {obj} not at location {loc}"
            if loc not in state.airports:
                return False, f"Location {loc} not an airport"
            if state.at.get(airplane) != loc:
                return False, f"Airplane {airplane} not at airport {loc}"

        elif action_type == ActionType.DRIVE_TRUCK:
            truck, loc_from, loc_to, city = params
            if state.at.get(truck) != loc_to:
                return False, f"Truck not at destination location"
            if loc_from not in state.locations:
                return False, f"Origin location invalid"
            if state.in_city.get(loc_from) != city:
                return False, f"Origin location not in city"
            if state.in_city.get(loc_to) != city:
                return False, f"Destination location not in city"

        elif action_type == ActionType.FLY_AIRPLANE:
            airplane, loc_from, loc_to = params
            if state.at.get(airplane) != loc_to:
                return False, f"Airplane not at destination"
            if loc_from not in state.airports:
                return False, f"Origin location not an airport"
            if loc_to not in state.airports:
                return False, f"Destination location not an airport"

        return True, ""

    # FIX: Wrap all undo methods with this validation
    @staticmethod
    def undo_action_safe(
            state: LogisticsState,
            action_type: ActionType,
            params: List[str]
    ) -> Optional[LogisticsState]:
        """
        FIX: Safe reverse action execution with validation.
        """
        # First validate preconditions
        valid, reason = ReverseActionExecutor._validate_reverse_preconditions(
            state, action_type, params
        )
        if not valid:
            return None

        # Then execute
        if action_type == ActionType.LOAD_TRUCK:
            return ReverseActionExecutor.undo_load_truck(state, params[0], params[1], params[2])
        elif action_type == ActionType.UNLOAD_TRUCK:
            return ReverseActionExecutor.undo_unload_truck(state, params[0], params[1], params[2])
        elif action_type == ActionType.LOAD_AIRPLANE:
            return ReverseActionExecutor.undo_load_airplane(state, params[0], params[1], params[2])
        elif action_type == ActionType.UNLOAD_AIRPLANE:
            return ReverseActionExecutor.undo_unload_airplane(state, params[0], params[1], params[2])
        elif action_type == ActionType.DRIVE_TRUCK:
            return ReverseActionExecutor.undo_drive_truck(state, params[0], params[1])
        elif action_type == ActionType.FLY_AIRPLANE:
            return ReverseActionExecutor.undo_fly_airplane(state, params[0], params[1])

        return None


    @staticmethod
    def undo_load_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """
        Undo load-truck: move package from truck back to location.

        Forward preconditions were:
        - at(truck, loc), at(obj, loc)

        Current state must have:
        - in(obj, truck) - package is in truck
        - at(truck, loc) - truck is still at location

        Result state should have:
        - at(obj, loc) - package back at location
        - NOT in(obj, truck)
        """
        # VALIDATION 1: Package must be in this truck
        if obj not in state.in_vehicle:
            return None
        if state.in_vehicle[obj] != truck:
            return None

        # VALIDATION 2: Truck must exist and be at a location
        if truck not in state.trucks:
            return None
        if truck not in state.at:
            return None

        # VALIDATION 3: Location must be valid
        if loc not in state.locations:
            return None

        # VALIDATION 4: Truck must be at the specified location
        if state.at[truck] != loc:
            return None

        # VALIDATION 5: Package cannot already be at a location
        if obj in state.at:
            return None

        # VALIDATION 6: Truck cannot be in any vehicle
        if truck in state.in_vehicle:
            return None

        # Execute reverse action
        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        # Final validation: result must be valid
        is_valid, _ = new_state.is_valid()
        if not is_valid:
            return None

        return new_state

    @staticmethod
    def undo_unload_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """
        Undo unload-truck: move package from location back into truck.

        Forward preconditions were:
        - at(truck, loc), in(obj, truck)

        Current state must have:
        - at(obj, loc) - package is at location
        - at(truck, loc) - truck is at same location
        - NOT in(obj, truck)

        Result state should have:
        - in(obj, truck) - package back in truck
        - NOT at(obj, loc)
        """
        # VALIDATION 1: Package must be at this location
        if obj not in state.at:
            return None
        if state.at[obj] != loc:
            return None

        # VALIDATION 2: Package must NOT be in any vehicle
        if obj in state.in_vehicle:
            return None

        # VALIDATION 3: Truck must exist
        if truck not in state.trucks:
            return None

        # VALIDATION 4: Truck must be at a location
        if truck not in state.at:
            return None

        # VALIDATION 5: Truck must be at the specified location
        if state.at[truck] != loc:
            return None

        # VALIDATION 6: Location must be valid
        if loc not in state.locations:
            return None

        # VALIDATION 7: Truck cannot be in any vehicle
        if truck in state.in_vehicle:
            return None

        # VALIDATION 8: Package cannot already be in a vehicle
        for vehicle in list(state.trucks) + list(state.airplanes):
            if state.in_vehicle.get(obj) == vehicle:
                return None

        # Execute reverse action
        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = truck

        # Final validation
        is_valid, error = new_state.is_valid()
        if not is_valid:
            return None

        return new_state

    @staticmethod
    def undo_load_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """
        Undo load-airplane: move package from airplane back to airport.

        Forward preconditions were:
        - AIRPORT(loc), at(airplane, loc), at(obj, loc)

        Current state must have:
        - in(obj, airplane) - package in airplane
        - at(airplane, loc) - airplane at airport
        - AIRPORT(loc)
        """
        # VALIDATION 1: Package must be in this airplane
        if obj not in state.in_vehicle:
            return None
        if state.in_vehicle[obj] != airplane:
            return None

        # VALIDATION 2: Airplane must exist
        if airplane not in state.airplanes:
            return None

        # VALIDATION 3: Airplane must be at a location
        if airplane not in state.at:
            return None

        # VALIDATION 4: Location must be an airport
        if loc not in state.airports:
            return None

        # VALIDATION 5: Airplane must be at the specified airport
        if state.at[airplane] != loc:
            return None

        # VALIDATION 6: Package cannot be at location
        if obj in state.at:
            return None

        # Execute reverse action
        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_unload_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """
        Undo unload-airplane: move package from airport back into airplane.

        Forward preconditions were:
        - AIRPORT(loc), at(airplane, loc), in(obj, airplane)

        Current state must have:
        - at(obj, loc) - package at airport
        - at(airplane, loc) - airplane at airport
        - NOT in(obj, airplane)
        """
        # VALIDATION 1: Package must be at this location
        if obj not in state.at:
            return None
        if state.at[obj] != loc:
            return None

        # VALIDATION 2: Location must be an airport
        if loc not in state.airports:
            return None

        # VALIDATION 3: Airplane must exist
        if airplane not in state.airplanes:
            return None

        # VALIDATION 4: Airplane must be at the airport
        if airplane not in state.at:
            return None
        if state.at[airplane] != loc:
            return None

        # VALIDATION 5: Package must NOT be in any vehicle
        if obj in state.in_vehicle:
            return None

        # Execute reverse action
        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = airplane

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_drive_truck(state: LogisticsState, truck: str, origin_loc: str) -> Optional[LogisticsState]:
        """
        Undo drive-truck: move truck from current location back to origin.

        Forward preconditions were:
        - in-city(origin, city), in-city(dest, city), at(truck, origin)

        Current state must have:
        - at(truck, dest_loc) where dest != origin
        - in-city(dest_loc, city)
        - in-city(origin, city)
        """
        # VALIDATION 1: Truck must exist
        if truck not in state.trucks:
            return None

        # VALIDATION 2: Truck must be at a location
        if truck not in state.at:
            return None

        current_loc = state.at[truck]

        # VALIDATION 3: Cannot undo if already at origin
        if current_loc == origin_loc:
            return None

        # VALIDATION 4: Both locations must be valid
        if current_loc not in state.locations:
            return None
        if origin_loc not in state.locations:
            return None

        # VALIDATION 5: Both locations must be in same city
        current_city = state.in_city.get(current_loc)
        origin_city = state.in_city.get(origin_loc)
        if not current_city or not origin_city:
            return None
        if current_city != origin_city:
            return None

        # VALIDATION 6: Truck cannot be in any vehicle
        if truck in state.in_vehicle:
            return None

        # Execute reverse action
        new_state = state.copy()
        new_state.at[truck] = origin_loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def undo_fly_airplane(state: LogisticsState, airplane: str, origin_loc: str) -> Optional[LogisticsState]:
        """
        Undo fly-airplane: move airplane from current airport back to origin airport.

        Forward preconditions were:
        - AIRPORT(origin), AIRPORT(dest), at(airplane, origin)

        Current state must have:
        - at(airplane, dest_loc) where dest != origin
        - AIRPORT(dest_loc)
        - AIRPORT(origin_loc)
        """
        # VALIDATION 1: Airplane must exist
        if airplane not in state.airplanes:
            return None

        # VALIDATION 2: Airplane must be at a location
        if airplane not in state.at:
            return None

        current_loc = state.at[airplane]

        # VALIDATION 3: Cannot undo if already at origin
        if current_loc == origin_loc:
            return None

        # VALIDATION 4: Both locations must be airports
        if current_loc not in state.airports:
            return None
        if origin_loc not in state.airports:
            return None

        # VALIDATION 5: Airplane cannot be in any vehicle (n/a but check anyway)
        if airplane in state.in_vehicle:
            return None

        # Execute reverse action
        new_state = state.copy()
        new_state.at[airplane] = origin_loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

        # In backward_generator.py, REPLACE get_applicable_reverse_actions with this:

    @staticmethod
    def get_applicable_reverse_actions(state: LogisticsState) -> List[Tuple[Action, LogisticsState]]:
        """
        FIX #6: Get all applicable reverse actions with STRICT validation.

        Each action must:
        1. Pass precondition validation
        2. Execute successfully
        3. Produce a valid state
        4. Not recreate the same state
        """
        results = []
        seen_state_hashes = set()

        # Undo load-truck actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                truck = state.in_vehicle[pkg]
                if truck in state.trucks and truck in state.at:
                    loc = state.at[truck]

                    action = Action(ActionType.LOAD_TRUCK, [pkg, truck, loc])
                    new_state = ReverseActionExecutor.undo_action_safe(
                        state, action.action_type, action.params
                    )

                    if new_state and new_state != state:
                        state_hash = hash(new_state)
                        if state_hash not in seen_state_hashes:
                            results.append((action, new_state))
                            seen_state_hashes.add(state_hash)

        # Undo unload-truck actions
        for pkg in state.packages:
            if pkg in state.at and pkg not in state.in_vehicle:
                pkg_loc = state.at[pkg]
                for truck in state.trucks:
                    if truck in state.at and state.at[truck] == pkg_loc:

                        action = Action(ActionType.UNLOAD_TRUCK, [pkg, truck, pkg_loc])
                        new_state = ReverseActionExecutor.undo_action_safe(
                            state, action.action_type, action.params
                        )

                        if new_state and new_state != state:
                            state_hash = hash(new_state)
                            if state_hash not in seen_state_hashes:
                                results.append((action, new_state))
                                seen_state_hashes.add(state_hash)

        # Undo load-airplane actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                vehicle = state.in_vehicle[pkg]
                if vehicle in state.airplanes and vehicle in state.at:
                    loc = state.at[vehicle]
                    if loc in state.airports:

                        action = Action(ActionType.LOAD_AIRPLANE, [pkg, vehicle, loc])
                        new_state = ReverseActionExecutor.undo_action_safe(
                            state, action.action_type, action.params
                        )

                        if new_state and new_state != state:
                            state_hash = hash(new_state)
                            if state_hash not in seen_state_hashes:
                                results.append((action, new_state))
                                seen_state_hashes.add(state_hash)

        # Undo unload-airplane actions
        for pkg in state.packages:
            if pkg in state.at and pkg not in state.in_vehicle:
                pkg_loc = state.at[pkg]
                if pkg_loc in state.airports:
                    for airplane in state.airplanes:
                        if airplane in state.at and state.at[airplane] == pkg_loc:

                            action = Action(ActionType.UNLOAD_AIRPLANE, [pkg, airplane, pkg_loc])
                            new_state = ReverseActionExecutor.undo_action_safe(
                                state, action.action_type, action.params
                            )

                            if new_state and new_state != state:
                                state_hash = hash(new_state)
                                if state_hash not in seen_state_hashes:
                                    results.append((action, new_state))
                                    seen_state_hashes.add(state_hash)

        # Undo drive-truck actions
        for truck in state.trucks:
            if truck in state.at:
                current_loc = state.at[truck]
                current_city = state.in_city.get(current_loc)
                if current_city:
                    for other_loc in state.locations:
                        if (state.in_city.get(other_loc) == current_city and
                                other_loc != current_loc):

                            action = Action(ActionType.DRIVE_TRUCK,
                                            [truck, other_loc, current_loc, current_city])
                            new_state = ReverseActionExecutor.undo_action_safe(
                                state, action.action_type, action.params
                            )

                            if new_state and new_state != state:
                                state_hash = hash(new_state)
                                if state_hash not in seen_state_hashes:
                                    results.append((action, new_state))
                                    seen_state_hashes.add(state_hash)

        # Undo fly-airplane actions
        for airplane in state.airplanes:
            if airplane in state.at:
                current_loc = state.at[airplane]
                if current_loc in state.airports:
                    for other_airport in state.airports:
                        if other_airport != current_loc:

                            action = Action(ActionType.FLY_AIRPLANE,
                                            [airplane, other_airport, current_loc])
                            new_state = ReverseActionExecutor.undo_action_safe(
                                state, action.action_type, action.params
                            )

                            if new_state and new_state != state:
                                state_hash = hash(new_state)
                                if state_hash not in seen_state_hashes:
                                    results.append((action, new_state))
                                    seen_state_hashes.add(state_hash)

        return results


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

    # MODIFY BackwardProblemGenerator.generate_problem METHOD

    def generate_problem(
            self,
            difficulty: str,
            generation_params: Optional[LogisticsGenerationParams] = None,
            target_plan_length: Optional[int] = None,
            archetype: Optional[GoalArchetype] = None,
            tolerance: int = 2,  # INCREASED from 1
            max_retries: int = 100  # INCREASED from 50
    ) -> Tuple[LogisticsState, LogisticsState, List[Action], GoalArchetype]:
        """
        Generate a problem with 100% validity guarantee.

        FIX #3+: Enhanced cycle detection, adaptive target lengths, and fallback strategies.
        """
        from config import DIFFICULTY_TIERS, DEFAULT_LOGISTICS_PARAMS
        from problem_validator import ProblemValidator

        if generation_params is None:
            generation_params = DEFAULT_LOGISTICS_PARAMS.get(difficulty)
        if target_plan_length is None:
            tier = DIFFICULTY_TIERS.get(difficulty)
            target_plan_length = tier.target_length if tier else 10

        min_length = max(1, target_plan_length - tolerance)
        max_length = target_plan_length + tolerance * 3  # INCREASED range

        logger.info(f"[GEN] Target length: {target_plan_length}±{tolerance}, "
                    f"accepting [{min_length}, {max_length}]")

        for retry in range(max_retries):
            try:
                # Step 1: Build valid world
                world, packages, trucks, airplanes = LogisticsProblemBuilder.build_world(
                    generation_params,
                    random_seed=self.random_seed + retry if self.random_seed else None
                )

                is_valid, error = world.is_valid()
                if not is_valid:
                    logger.debug(f"[GEN {retry}] Invalid world: {error}")
                    continue

                # Step 2: Generate goal dict with archetype
                goal_dict, used_archetype = self.generate_goal_dict_robust_with_archetype(
                    world,
                    packages,
                    len(packages),
                    max_attempts=50
                )

                if not goal_dict:
                    logger.debug(f"[GEN {retry}] Failed to generate goal dict")
                    continue

                # Step 3: Create goal state
                goal_state = world.copy()
                for pkg, dest_loc in goal_dict.items():
                    if pkg in goal_state.in_vehicle:
                        del goal_state.in_vehicle[pkg]
                    goal_state.at[pkg] = dest_loc

                is_valid, error = goal_state.is_valid()
                if not is_valid:
                    logger.debug(f"[GEN {retry}] Goal state invalid: {error}")
                    continue

                if goal_state == world:
                    logger.debug(f"[GEN {retry}] Trivial problem (initial == goal)")
                    continue

                # FIX #3: Step 4 - Backward search with ADAPTIVE termination
                deduplicator = StateDeduplicator(max_states=5000)
                current_state = goal_state.copy()
                plan = []
                iteration = 0
                max_iterations = max(target_plan_length * 20, 2000)  # INCREASED

                no_progress_count = 0
                max_no_progress = 50

                while (len(plan) < max_length and
                       iteration < max_iterations and
                       not deduplicator.cycle_detected and
                       no_progress_count < max_no_progress):

                    iteration += 1

                    if deduplicator.is_visited(current_state):
                        logger.debug(f"[GEN {retry}] Cycle detected at iteration {iteration}")
                        break

                    deduplicator.mark_visited(current_state)

                    # Get applicable reverse actions
                    reverse_actions = ReverseActionExecutor.get_applicable_reverse_actions(current_state)
                    if not reverse_actions:
                        logger.debug(f"[GEN {retry}] No reverse actions at iteration {iteration}")
                        break

                    # Shuffle for variety
                    random.shuffle(reverse_actions)
                    action_found = False

                    for action, new_state in reverse_actions:
                        # Skip if state already visited
                        if deduplicator.is_visited(new_state):
                            continue

                        is_valid, _ = new_state.is_valid()
                        if not is_valid:
                            continue

                        if new_state == current_state:
                            continue

                        # Accept this action
                        plan.insert(0, action)
                        current_state = new_state
                        action_found = True
                        no_progress_count = 0
                        break

                    if not action_found:
                        no_progress_count += 1

                initial_state = current_state

                # Step 5: Check plan length with adaptive acceptance
                if len(plan) < min_length:
                    logger.debug(f"[GEN {retry}] Plan too short: {len(plan)} < {min_length}")
                    continue

                if len(plan) > max_length:
                    logger.debug(f"[GEN {retry}] Plan too long: {len(plan)} > {max_length}")
                    continue

                # Step 6: Comprehensive validation
                is_valid, reason = ProblemValidator.validate_complete_problem(
                    initial_state,
                    goal_state,
                    plan
                )

                if is_valid:
                    logger.info(f"[GEN {retry}] SUCCESS: {used_archetype.value}, length={len(plan)}")
                    return initial_state, goal_state, plan, used_archetype
                else:
                    logger.debug(f"[GEN {retry}] Validation failed: {reason}")

            except Exception as e:
                logger.debug(f"[GEN {retry}] Exception: {str(e)[:100]}")
                continue

        raise ValueError(
            f"Failed to generate valid {difficulty} problem after {max_retries} retries. "
            f"Target plan length: {target_plan_length}±{tolerance}. "
            f"Try: (1) reducing difficulty tier, (2) increasing world complexity in config.py, "
            f"or (3) reducing target_plan_length."
        )

    def _validate_reverse_action_soundness(
            self,
            state_before: LogisticsState,
            state_after: LogisticsState,
            action: Action
    ) -> bool:
        """
        FIX #3: Validate that the reverse action is truly sound.

        Check that if we applied the FORWARD action to state_after,
        we get back to state_before.
        """
        from actions import ActionExecutor

        forward_state = ActionExecutor.execute_forward(state_after, action)
        if forward_state is None:
            return False

        # Check if states match
        return forward_state == state_before

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