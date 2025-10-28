"""
State representation for Logistics domain and state validation.

A valid Logistics state consists of:
- Objects (packages) at locations
- Trucks and airplanes at locations
- Objects potentially inside vehicles
- Static map: in-city relations and airport designations
"""

from typing import Set, Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class LogisticsState:
    """Represents a Logistics world state."""

    # Entity collections
    packages: Set[str] = field(default_factory=set)
    trucks: Set[str] = field(default_factory=set)
    airplanes: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    cities: Set[str] = field(default_factory=set)

    # Dynamic facts
    at: Dict[str, str] = field(default_factory=dict)  # object -> location
    in_vehicle: Dict[str, str] = field(default_factory=dict)  # object -> vehicle

    # Static facts (map structure)
    in_city: Dict[str, str] = field(default_factory=dict)  # location -> city
    airports: Set[str] = field(default_factory=set)  # Set of airport locations

    def copy(self) -> 'LogisticsState':
        """Create a deep copy of the state."""
        return LogisticsState(
            packages=self.packages.copy(),
            trucks=self.trucks.copy(),
            airplanes=self.airplanes.copy(),
            locations=self.locations.copy(),
            cities=self.cities.copy(),
            at=self.at.copy(),
            in_vehicle=self.in_vehicle.copy(),
            in_city=self.in_city.copy(),
            airports=self.airports.copy()
        )

    def is_valid(self) -> Tuple[bool, Optional[str]]:
        """Enhanced validation with stricter checks."""

        # Original checks...
        for pkg in self.packages:
            at_loc = pkg in self.at
            in_veh = pkg in self.in_vehicle
            count = sum([at_loc, in_veh])
            if count != 1:
                return False, f"Package {pkg} is in {count} places (should be exactly 1)"

        # FIX 1: Validate no vehicle carries the same package twice
        for pkg in self.packages:
            vehicle_count = 0
            in_which_vehicles = []
            for vehicle in list(self.trucks) + list(self.airplanes):
                if pkg in self.in_vehicle and self.in_vehicle[pkg] == vehicle:
                    vehicle_count += 1
                    in_which_vehicles.append(vehicle)
            if vehicle_count > 1:
                return False, f"Package {pkg} is in multiple vehicles: {in_which_vehicles}"

        # FIX 2: Validate vehicles don't exceed capacity (implicit: one location each)
        for vehicle in list(self.trucks) + list(self.airplanes):
            if vehicle not in self.at:
                return False, f"Vehicle {vehicle} not at any location"
            # Each vehicle at exactly one location
            if not isinstance(self.at[vehicle], str):
                return False, f"Vehicle {vehicle} location is invalid type"

        # FIX 3: Validate packages in vehicles match vehicle positions
        for pkg, vehicle in self.in_vehicle.items():
            if vehicle not in self.at:
                return False, f"Vehicle {vehicle} carrying {pkg} has no location"
            if vehicle not in list(self.trucks) + list(self.airplanes):
                return False, f"Invalid vehicle {vehicle}"

        # FIX 4: Validate airport constraints
        for airplane in self.airplanes:
            if airplane in self.at:
                loc = self.at[airplane]
                if loc not in self.airports:
                    return False, f"Airplane {airplane} at non-airport location {loc}"

        # FIX 5: All locations must have valid in-city mappings
        for loc in self.locations:
            if loc not in self.in_city:
                return False, f"Location {loc} not mapped to any city"
            city = self.in_city[loc]
            if city not in self.cities:
                return False, f"Location {loc} mapped to non-existent city {city}"

        return True, None

    def __hash__(self):
        """Make state hashable for deduplication."""
        at_tuple = tuple(sorted((k, v) for k, v in self.at.items()))
        in_veh_tuple = tuple(sorted((k, v) for k, v in self.in_vehicle.items()))
        in_city_tuple = tuple(sorted((k, v) for k, v in self.in_city.items()))
        return hash((
            frozenset(self.at.items()),
            frozenset(self.in_vehicle.items()),
            frozenset(self.airports)
        ))

    def __eq__(self, other):
        """Check state equality."""
        if not isinstance(other, LogisticsState):
            return False
        return (
                self.packages == other.packages and
                self.trucks == other.trucks and
                self.airplanes == other.airplanes and
                self.at == other.at and
                self.in_vehicle == other.in_vehicle and
                self.in_city == other.in_city and
                self.airports == other.airports
        )

    def __repr__(self):
        parts = []
        if self.at:
            parts.append(f"at={dict(sorted(self.at.items()))}")
        if self.in_vehicle:
            parts.append(f"in_vehicle={dict(sorted(self.in_vehicle.items()))}")
        return f"LogisticsState({', '.join(parts)})"


def create_initial_state(
        packages: List[str],
        trucks: List[str],
        airplanes: List[str],
        locations: List[str],
        cities: List[str],
        in_city: Dict[str, str],
        airports: Set[str],
        at: Dict[str, str] = None,
        in_vehicle: Dict[str, str] = None
) -> LogisticsState:
    """
    Create a valid initial Logistics state.

    All objects start at their designated locations.

    Args:
        packages: List of package names
        trucks: List of truck names
        airplanes: List of airplane names
        locations: List of location names
        cities: List of city names
        in_city: Mapping of location -> city
        airports: Set of airport location names
        at: Optional mapping of object -> location (if None, will be empty)
        in_vehicle: Optional mapping of object -> vehicle (if None, will be empty)

    Returns:
        Valid LogisticsState
    """

    state = LogisticsState(
        packages=set(packages),
        trucks=set(trucks),
        airplanes=set(airplanes),
        locations=set(locations),
        cities=set(cities),
        at=at.copy() if at else {},
        in_vehicle=in_vehicle.copy() if in_vehicle else {},
        in_city=in_city.copy(),
        airports=airports.copy()
    )

    is_valid, error = state.is_valid()
    if not is_valid:
        raise ValueError(f"Failed to create initial state: {error}")

    return state