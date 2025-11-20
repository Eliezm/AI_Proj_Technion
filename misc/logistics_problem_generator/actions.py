"""
Action definitions and execution for Logistics domain.

Implements forward and reverse execution of the six actions:
1. load-truck ?obj ?truck ?loc
2. unload-truck ?obj ?truck ?loc
3. load-airplane ?obj ?airplane ?loc
4. unload-airplane ?obj ?airplane ?loc
5. drive-truck ?truck ?loc-from ?loc-to ?city
6. fly-airplane ?airplane ?loc-from ?loc-to
"""

from typing import List, Optional
from enum import Enum
from state import LogisticsState


class ActionType(Enum):
    LOAD_TRUCK = "load-truck"
    UNLOAD_TRUCK = "unload-truck"
    LOAD_AIRPLANE = "load-airplane"
    UNLOAD_AIRPLANE = "unload-airplane"
    DRIVE_TRUCK = "drive-truck"
    FLY_AIRPLANE = "fly-airplane"


class Action:
    """Represents an action in Logistics domain."""

    def __init__(self, action_type: ActionType, params: List[str]):
        self.action_type = action_type
        self.params = params

    def __repr__(self):
        return f"{self.action_type.value}({', '.join(self.params)})"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.action_type == other.action_type and self.params == other.params

    def __hash__(self):
        return hash((self.action_type, tuple(self.params)))


class ActionExecutor:
    """Executes actions on Logistics states."""

    @staticmethod
    def can_execute_load_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> bool:
        """
        Check preconditions for load-truck:
        - truck is at location
        - object is at location
        - object is not in any vehicle
        """
        return (
                truck in state.trucks and
                obj in state.packages and
                loc in state.locations and
                state.at.get(truck) == loc and
                state.at.get(obj) == loc and
                obj not in state.in_vehicle
        )

    @staticmethod
    def execute_load_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """Execute load-truck action."""
        if not ActionExecutor.can_execute_load_truck(state, obj, truck, loc):
            return None

        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = truck

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_unload_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> bool:
        """
        Check preconditions for unload-truck:
        - truck is at location
        - object is in truck
        """
        return (
                truck in state.trucks and
                obj in state.packages and
                loc in state.locations and
                state.at.get(truck) == loc and
                state.in_vehicle.get(obj) == truck
        )

    @staticmethod
    def execute_unload_truck(state: LogisticsState, obj: str, truck: str, loc: str) -> Optional[LogisticsState]:
        """Execute unload-truck action."""
        if not ActionExecutor.can_execute_unload_truck(state, obj, truck, loc):
            return None

        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_load_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> bool:
        """
        Check preconditions for load-airplane:
        - airplane is at airport location
        - object is at location
        - object is not in any vehicle
        """
        return (
                airplane in state.airplanes and
                obj in state.packages and
                loc in state.locations and
                loc in state.airports and  # Must be at airport
                state.at.get(airplane) == loc and
                state.at.get(obj) == loc and
                obj not in state.in_vehicle
        )

    @staticmethod
    def execute_load_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """Execute load-airplane action."""
        if not ActionExecutor.can_execute_load_airplane(state, obj, airplane, loc):
            return None

        new_state = state.copy()
        del new_state.at[obj]
        new_state.in_vehicle[obj] = airplane

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_unload_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> bool:
        """
        Check preconditions for unload-airplane:
        - airplane is at airport location
        - object is in airplane
        """
        return (
                airplane in state.airplanes and
                obj in state.packages and
                loc in state.locations and
                loc in state.airports and  # Must be at airport
                state.at.get(airplane) == loc and
                state.in_vehicle.get(obj) == airplane
        )

    @staticmethod
    def execute_unload_airplane(state: LogisticsState, obj: str, airplane: str, loc: str) -> Optional[LogisticsState]:
        """Execute unload-airplane action."""
        if not ActionExecutor.can_execute_unload_airplane(state, obj, airplane, loc):
            return None

        new_state = state.copy()
        del new_state.in_vehicle[obj]
        new_state.at[obj] = loc

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_drive_truck(state: LogisticsState, truck: str, loc_from: str, loc_to: str, city: str) -> bool:
        """
        Check preconditions for drive-truck:
        - truck is at loc_from
        - both locations are in the same city
        """
        return (
                truck in state.trucks and
                loc_from in state.locations and
                loc_to in state.locations and
                city in state.cities and
                state.at.get(truck) == loc_from and
                state.in_city.get(loc_from) == city and
                state.in_city.get(loc_to) == city and
                loc_from != loc_to
        )

    @staticmethod
    def execute_drive_truck(state: LogisticsState, truck: str, loc_from: str, loc_to: str, city: str) -> Optional[
        LogisticsState]:
        """Execute drive-truck action."""
        if not ActionExecutor.can_execute_drive_truck(state, truck, loc_from, loc_to, city):
            return None

        new_state = state.copy()
        new_state.at[truck] = loc_to

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_fly_airplane(state: LogisticsState, airplane: str, loc_from: str, loc_to: str) -> bool:
        """
        Check preconditions for fly-airplane:
        - airplane is at loc_from
        - both locations are airports
        """
        return (
                airplane in state.airplanes and
                loc_from in state.locations and
                loc_to in state.locations and
                loc_from in state.airports and
                loc_to in state.airports and
                state.at.get(airplane) == loc_from and
                loc_from != loc_to
        )

    @staticmethod
    def execute_fly_airplane(state: LogisticsState, airplane: str, loc_from: str, loc_to: str) -> Optional[
        LogisticsState]:
        """Execute fly-airplane action."""
        if not ActionExecutor.can_execute_fly_airplane(state, airplane, loc_from, loc_to):
            return None

        new_state = state.copy()
        new_state.at[airplane] = loc_to

        is_valid, _ = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def execute_forward(state: LogisticsState, action: Action) -> Optional[LogisticsState]:
        """Execute an action in the forward direction."""
        if action.action_type == ActionType.LOAD_TRUCK:
            return ActionExecutor.execute_load_truck(state, action.params[0], action.params[1], action.params[2])
        elif action.action_type == ActionType.UNLOAD_TRUCK:
            return ActionExecutor.execute_unload_truck(state, action.params[0], action.params[1], action.params[2])
        elif action.action_type == ActionType.LOAD_AIRPLANE:
            return ActionExecutor.execute_load_airplane(state, action.params[0], action.params[1], action.params[2])
        elif action.action_type == ActionType.UNLOAD_AIRPLANE:
            return ActionExecutor.execute_unload_airplane(state, action.params[0], action.params[1], action.params[2])
        elif action.action_type == ActionType.DRIVE_TRUCK:
            return ActionExecutor.execute_drive_truck(state, action.params[0], action.params[1], action.params[2],
                                                      action.params[3])
        elif action.action_type == ActionType.FLY_AIRPLANE:
            return ActionExecutor.execute_fly_airplane(state, action.params[0], action.params[1], action.params[2])
        else:
            return None

    # actions.py - REPLACE get_applicable_actions METHOD

    @staticmethod
    def get_applicable_actions(state: LogisticsState) -> List[Action]:
        """Get all applicable actions in the current state (forward direction).

        FIX: Deduplicate actions to avoid generating the same action multiple times.
        """
        applicable = []
        seen = set()  # FIX: Track seen actions

        # Load-truck actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                continue  # Skip if already in vehicle
            pkg_loc = state.at.get(pkg)
            for truck in state.trucks:
                truck_loc = state.at.get(truck)
                if pkg_loc == truck_loc and ActionExecutor.can_execute_load_truck(state, pkg, truck, pkg_loc):
                    action = Action(ActionType.LOAD_TRUCK, [pkg, truck, pkg_loc])
                    if action not in seen:
                        applicable.append(action)
                        seen.add(action)

        # Unload-truck actions
        for pkg in state.packages:
            if pkg not in state.in_vehicle:
                continue  # Skip if not in vehicle
            vehicle = state.in_vehicle[pkg]
            if vehicle not in state.trucks:
                continue
            truck_loc = state.at.get(vehicle)
            if truck_loc and ActionExecutor.can_execute_unload_truck(state, pkg, vehicle, truck_loc):
                action = Action(ActionType.UNLOAD_TRUCK, [pkg, vehicle, truck_loc])
                if action not in seen:
                    applicable.append(action)
                    seen.add(action)

        # Load-airplane actions
        for pkg in state.packages:
            if pkg in state.in_vehicle:
                continue
            pkg_loc = state.at.get(pkg)
            if pkg_loc not in state.airports:
                continue
            for airplane in state.airplanes:
                airplane_loc = state.at.get(airplane)
                if pkg_loc == airplane_loc and ActionExecutor.can_execute_load_airplane(state, pkg, airplane, pkg_loc):
                    action = Action(ActionType.LOAD_AIRPLANE, [pkg, airplane, pkg_loc])
                    if action not in seen:
                        applicable.append(action)
                        seen.add(action)

        # Unload-airplane actions
        for pkg in state.packages:
            if pkg not in state.in_vehicle:
                continue
            vehicle = state.in_vehicle[pkg]
            if vehicle not in state.airplanes:
                continue
            airplane_loc = state.at.get(vehicle)
            if airplane_loc and airplane_loc in state.airports and ActionExecutor.can_execute_unload_airplane(state,
                                                                                                              pkg,
                                                                                                              vehicle,
                                                                                                              airplane_loc):
                action = Action(ActionType.UNLOAD_AIRPLANE, [pkg, vehicle, airplane_loc])
                if action not in seen:
                    applicable.append(action)
                    seen.add(action)

        # Drive-truck actions
        for truck in state.trucks:
            truck_loc = state.at.get(truck)
            if not truck_loc:
                continue
            truck_city = state.in_city.get(truck_loc)
            if not truck_city:
                continue
            for dest_loc in state.locations:
                if dest_loc != truck_loc and state.in_city.get(dest_loc) == truck_city:
                    if ActionExecutor.can_execute_drive_truck(state, truck, truck_loc, dest_loc, truck_city):
                        action = Action(ActionType.DRIVE_TRUCK, [truck, truck_loc, dest_loc, truck_city])
                        if action not in seen:
                            applicable.append(action)
                            seen.add(action)

        # Fly-airplane actions
        for airplane in state.airplanes:
            airplane_loc = state.at.get(airplane)
            if not airplane_loc or airplane_loc not in state.airports:
                continue
            for dest_airport in state.airports:
                if dest_airport != airplane_loc:
                    if ActionExecutor.can_execute_fly_airplane(state, airplane, airplane_loc, dest_airport):
                        action = Action(ActionType.FLY_AIRPLANE, [airplane, airplane_loc, dest_airport])
                        if action not in seen:
                            applicable.append(action)
                            seen.add(action)

        return applicable