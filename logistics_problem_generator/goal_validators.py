"""
Goal validation utilities for Logistics domain.

Ensures generated goals are achievable before attempting backward search.
"""

from typing import Tuple, List
from state import LogisticsState


class GoalValidator:
    """Validates that goals are achievable with current resource configuration."""

    @staticmethod
    def can_achieve_intra_city_delivery(
            state: LogisticsState,
            goal_dict: dict
    ) -> Tuple[bool, str]:
        """Check if all intra-city deliveries are feasible."""

        for pkg, goal_loc in goal_dict.items():
            initial_loc = state.at.get(pkg)
            if not initial_loc:
                return False, f"Package {pkg} has no initial location"

            initial_city = state.in_city.get(initial_loc)
            goal_city = state.in_city.get(goal_loc)

            if initial_city != goal_city:
                return False, f"Goal specifies inter-city but checking intra-city"

            # Need a truck in this city
            trucks_in_city = [
                t for t in state.trucks
                if state.in_city.get(state.at.get(t)) == initial_city
            ]
            if not trucks_in_city:
                return False, f"No trucks in city {initial_city}"

        return True, "Intra-city delivery feasible"

    @staticmethod
    def can_achieve_inter_city_delivery(
            state: LogisticsState,
            goal_dict: dict
    ) -> Tuple[bool, str]:
        """Check if inter-city deliveries are feasible."""

        # Need airplanes
        if not state.airplanes:
            return False, "No airplanes for inter-city delivery"

        # Need airports
        if not state.airports:
            return False, "No airports"

        # Need at least 2 cities
        if len(state.cities) < 2:
            return False, "Need at least 2 cities for inter-city"

        # Need trucks in source cities
        for pkg, goal_loc in goal_dict.items():
            initial_loc = state.at.get(pkg)
            if not initial_loc:
                return False, f"Package {pkg} has no initial location"

            initial_city = state.in_city.get(initial_loc)
            goal_city = state.in_city.get(goal_loc)

            if initial_city == goal_city:
                continue  # Intra-city is fine too

            # Need truck in source city
            trucks = [t for t in state.trucks if state.in_city.get(state.at.get(t)) == initial_city]
            if not trucks:
                return False, f"No trucks in source city {initial_city}"

            # Need airport in source city
            source_airports = [a for a in state.airports if state.in_city.get(a) == initial_city]
            if not source_airports:
                return False, f"No airport in source city {initial_city}"

            # Need airport in dest city
            dest_airports = [a for a in state.airports if state.in_city.get(a) == goal_city]
            if not dest_airports:
                return False, f"No airport in destination city {goal_city}"

            # Need truck in dest city
            trucks_dest = [t for t in state.trucks if state.in_city.get(state.at.get(t)) == goal_city]
            if not trucks_dest:
                return False, f"No trucks in destination city {goal_city}"

        return True, "Inter-city delivery feasible"

    @staticmethod
    def validate_goal_achievability(
            state: LogisticsState,
            goal_dict: dict
    ) -> Tuple[bool, str]:
        """
        Comprehensive check: Can this goal be achieved?

        Returns (is_achievable, reason)
        """

        if not goal_dict:
            return False, "Empty goal"

        if not state.packages:
            return False, "No packages in initial state"

        # Determine if intra-city or inter-city
        has_inter_city = False
        for pkg, goal_loc in goal_dict.items():
            initial_loc = state.at.get(pkg)
            if not initial_loc:
                if pkg in state.in_vehicle:
                    vehicle = state.in_vehicle[pkg]
                    initial_loc = state.at.get(vehicle)
                if not initial_loc:
                    return False, f"Package {pkg} has no location"

            initial_city = state.in_city.get(initial_loc)
            goal_city = state.in_city.get(goal_loc)

            if initial_city and goal_city and initial_city != goal_city:
                has_inter_city = True
                break

        if has_inter_city:
            return GoalValidator.can_achieve_inter_city_delivery(state, goal_dict)
        else:
            return GoalValidator.can_achieve_intra_city_delivery(state, goal_dict)