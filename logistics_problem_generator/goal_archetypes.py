"""
Goal archetypes for Logistics domain.

Defines different types of goal structures to ensure structural diversity (Requirement #2).
Active archetype selection ensures variety in problem structures.
"""

import random
from typing import List, Tuple, Dict
from enum import Enum
from state import LogisticsState


class GoalArchetype(Enum):
    """Different goal archetype types for Logistics."""
    INTRA_CITY = "intra_city"
    INTER_CITY_SIMPLE = "inter_city_simple"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class GoalArchetypeGenerator:
    """
    Generates goal states according to different archetypes.

    Requirement #4: Active sampling of goal archetypes to guarantee variety.
    """

    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    # In goal_archetypes.py - Replace the generate methods

    def generate_intra_city(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """
        Archetype: All packages stay within the same city.
        """
        if not packages:
            return {}

        # Try to find a city with at least 2 locations
        valid_cities = [
            city for city in state.cities
            if len([loc for loc in state.locations if state.in_city.get(loc) == city]) >= 2
        ]

        if not valid_cities:
            return {}

        city = random.choice(valid_cities)
        city_locs = [loc for loc in state.locations if state.in_city.get(loc) == city]

        goal = {}
        # Select packages that are IN this city
        pkgs_in_city = [
            pkg for pkg in packages
            if state.in_city.get(state.at.get(pkg)) == city
        ]

        if not pkgs_in_city:
            # If no packages are in city, just pick some and move them within city
            selected_pkgs = random.sample(packages, min(num_packages, len(packages)))
            for pkg in selected_pkgs:
                goal[pkg] = random.choice(city_locs)
            return goal

        selected_pkgs = random.sample(pkgs_in_city, min(num_packages, len(pkgs_in_city)))

        for pkg in selected_pkgs:
            current_loc = state.at.get(pkg)
            other_locs = [l for l in city_locs if l != current_loc]
            if other_locs:
                goal[pkg] = random.choice(other_locs)

        return goal

    def generate_inter_city_simple(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """
        Archetype: Packages move between two specific cities.
        """
        if not packages or len(state.cities) < 2:
            return {}

        cities_list = list(state.cities)
        random.shuffle(cities_list)
        source_city = cities_list[0]
        dest_city = cities_list[1]

        source_locs = [loc for loc in state.locations if state.in_city.get(loc) == source_city]
        dest_locs = [loc for loc in state.locations if state.in_city.get(loc) == dest_city]

        if not source_locs or not dest_locs:
            return {}

        goal = {}
        # Get packages in source city
        pkgs_in_source = [
            pkg for pkg in packages
            if state.in_city.get(state.at.get(pkg)) == source_city
        ]

        # If no packages in source, just pick some
        if not pkgs_in_source:
            selected_pkgs = random.sample(packages, min(num_packages, len(packages)))
        else:
            selected_pkgs = random.sample(pkgs_in_source, min(num_packages, len(pkgs_in_source)))

        for pkg in selected_pkgs:
            goal[pkg] = random.choice(dest_locs)

        return goal

    def generate_one_to_many(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """
        Archetype: One-to-Many distribution from a hub.
        """
        if not packages or not state.locations:
            return {}

        hub = random.choice(list(state.locations))
        other_locs = [loc for loc in state.locations if loc != hub]

        if not other_locs:
            return {}

        goal = {}
        for pkg in random.sample(packages, min(num_packages, len(packages))):
            goal[pkg] = random.choice(other_locs)

        return goal if goal else {}

    def generate_many_to_one(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """
        Archetype: Many-to-One collection at hub.
        """
        if not packages:
            return {}

        hub = random.choice(list(state.locations))

        goal = {}
        for pkg in random.sample(packages, min(num_packages, len(packages))):
            goal[pkg] = hub

        return goal if goal else {}

    def generate_many_to_many(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """
        Archetype: Many-to-Many complex exchange.
        """
        if not packages or not state.locations:
            return {}

        goal = {}
        for pkg in random.sample(packages, min(num_packages, len(packages))):
            current = state.at.get(pkg)
            other_locs = [l for l in state.locations if l != current]
            if other_locs:
                goal[pkg] = random.choice(other_locs)

        return goal if goal else {}

    def generate_archetype(
            self,
            archetype: GoalArchetype,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Dict[str, str]:
        """Generate a goal for the given archetype with validation."""

        for attempt in range(10):  # Retry loop
            if archetype == GoalArchetype.INTRA_CITY:
                goal = self.generate_intra_city(state, packages, num_packages)
            elif archetype == GoalArchetype.INTER_CITY_SIMPLE:
                goal = self.generate_inter_city_simple(state, packages, num_packages)
            elif archetype == GoalArchetype.ONE_TO_MANY:
                goal = self.generate_one_to_many(state, packages, num_packages)
            elif archetype == GoalArchetype.MANY_TO_ONE:
                goal = self.generate_many_to_one(state, packages, num_packages)
            elif archetype == GoalArchetype.MANY_TO_MANY:
                goal = self.generate_many_to_many(state, packages, num_packages)
            else:
                return {}

            # Validate goal
            if goal:
                is_achievable, reason = is_goal_achievable(state, goal)
                if is_achievable:
                    return goal

        return {}  # Return empty if all attempts fail

    def generate_random_archetype(
            self,
            state: LogisticsState,
            packages: List[str],
            num_packages: int
    ) -> Tuple[Dict[str, str], GoalArchetype]:
        """Generate a random archetype (Requirement #4)."""
        archetype = random.choice(list(GoalArchetype))
        goal = self.generate_archetype(archetype, state, packages, num_packages)
        return goal, archetype


def create_goal_state_from_dict(
        initial_state: LogisticsState,
        goal_dict: Dict[str, str]
) -> LogisticsState:
    """
    Create a goal state from a goal specification dictionary.

    The goal state is the initial state with packages moved to their goal locations
    (but not in any vehicle).

    Args:
        initial_state: The initial state
        goal_dict: Mapping of package -> goal location

    Returns:
        Goal state with packages at specified locations
    """
    goal_state = initial_state.copy()

    # Move all specified packages to their goal locations
    for pkg, dest_loc in goal_dict.items():
        if pkg in goal_state.in_vehicle:
            del goal_state.in_vehicle[pkg]
        goal_state.at[pkg] = dest_loc

    is_valid, error = goal_state.is_valid()
    if not is_valid:
        raise ValueError(f"Invalid goal state: {error}")

    return goal_state


# goal_archetypes.py - REPLACE is_goal_achievable FUNCTION

# In goal_archetypes.py, REPLACE is_goal_achievable function with this:

def is_goal_achievable(
        initial_state: LogisticsState,
        goal_dict: Dict[str, str]
) -> Tuple[bool, str]:
    """
    FIX #4: Comprehensive goal achievability check.

    Returns: (is_achievable, reason)
    """
    # Check 1: Empty goal
    if not goal_dict:
        return False, "Empty goal"

    # Check 2: All goal packages exist in initial state
    for pkg, goal_loc in goal_dict.items():
        if pkg not in initial_state.packages:
            return False, f"Package {pkg} does not exist"
        if goal_loc not in initial_state.locations:
            return False, f"Goal location {goal_loc} does not exist"

    # Check 3: Package not already at goal (non-trivial)
    for pkg, goal_loc in goal_dict.items():
        current_loc = initial_state.at.get(pkg)
        if current_loc == goal_loc:
            return False, f"Package {pkg} already at goal (trivial)"

    # Check 4: All packages have valid initial positions
    for pkg in goal_dict.keys():
        current_loc = initial_state.at.get(pkg)
        if current_loc is None:
            if pkg in initial_state.in_vehicle:
                vehicle = initial_state.in_vehicle[pkg]
                if vehicle not in initial_state.at:
                    return False, f"Package {pkg} in vehicle {vehicle} with no location"
            else:
                return False, f"Package {pkg} has no location or vehicle"

    # Check 5: At least one vehicle exists
    if not initial_state.trucks and not initial_state.airplanes:
        return False, "No vehicles to perform transport"

    # Check 6: Validate city connectivity
    for pkg, goal_loc in goal_dict.items():
        initial_loc = initial_state.at.get(pkg)
        if not initial_loc:
            if pkg in initial_state.in_vehicle:
                vehicle = initial_state.in_vehicle[pkg]
                initial_loc = initial_state.at.get(vehicle)
            if not initial_loc:
                return False, f"Cannot determine initial location of {pkg}"

        goal_city = initial_state.in_city.get(goal_loc)
        initial_city = initial_state.in_city.get(initial_loc)

        if not goal_city or not initial_city:
            return False, f"Invalid city mapping"

        # Intra-city case
        if goal_city == initial_city:
            trucks_in_city = [
                t for t in initial_state.trucks
                if initial_state.in_city.get(initial_state.at.get(t)) == goal_city
            ]
            if not trucks_in_city:
                return False, f"No trucks in {goal_city}"
        else:
            # Inter-city case
            if not initial_state.airplanes:
                return False, "Inter-city goal requires airplanes"
            if not initial_state.airports:
                return False, "Inter-city goal requires airports"

            # FIX #4: Stricter airport checks
            source_airports = [
                a for a in initial_state.airports
                if initial_state.in_city.get(a) == initial_city
            ]
            dest_airports = [
                a for a in initial_state.airports
                if initial_state.in_city.get(a) == goal_city
            ]

            if not source_airports:
                return False, f"Source city {initial_city} has no airport"
            if not dest_airports:
                return False, f"Dest city {goal_city} has no airport"

            # Check trucks in both cities
            source_trucks = [
                t for t in initial_state.trucks
                if initial_state.in_city.get(initial_state.at.get(t)) == initial_city
            ]
            dest_trucks = [
                t for t in initial_state.trucks
                if initial_state.in_city.get(initial_state.at.get(t)) == goal_city
            ]

            if not source_trucks:
                return False, f"Source city {initial_city} has no trucks"
            if not dest_trucks:
                return False, f"Dest city {goal_city} has no trucks"

    return True, "Goal is achievable"