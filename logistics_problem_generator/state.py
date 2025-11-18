"""
State representation for Logistics domain and state validation.
"""

from typing import Set, Dict, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class LogisticsState:
    """Represents a Logistics world state."""

    packages: Set[str] = field(default_factory=set)
    trucks: Set[str] = field(default_factory=set)
    airplanes: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    cities: Set[str] = field(default_factory=set)

    at: Dict[str, str] = field(default_factory=dict)
    in_vehicle: Dict[str, str] = field(default_factory=dict)

    in_city: Dict[str, str] = field(default_factory=dict)
    airports: Set[str] = field(default_factory=set)

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
        """Enhanced validation with comprehensive checks."""

        if not self.packages:
            return False, "No packages"
        if not self.trucks:
            return False, "No trucks"
        if not self.locations:
            return False, "No locations"
        if not self.cities:
            return False, "No cities"

        # FIX 1: Package location uniqueness
        package_locations = {}
        for pkg in self.packages:
            at_count = 0
            in_count = 0

            if pkg in self.at:
                at_count = 1
                if self.at[pkg] not in self.locations:
                    return False, f"Package {pkg} at invalid location {self.at[pkg]}"
                if pkg in package_locations:
                    return False, f"Package {pkg} location defined twice"
                package_locations[pkg] = ('at', self.at[pkg])

            if pkg in self.in_vehicle:
                in_count = 1
                vehicle = self.in_vehicle[pkg]
                if vehicle not in list(self.trucks) + list(self.airplanes):
                    return False, f"Package {pkg} in invalid vehicle {vehicle}"
                if pkg in package_locations:
                    return False, f"Package {pkg} in-vehicle and at defined simultaneously"
                package_locations[pkg] = ('in', vehicle)

            total = at_count + in_count
            if total != 1:
                return False, f"Package {pkg} in {total} places (should be exactly 1)"

        # FIX 2: No duplicate packages in vehicles
        vehicle_contents = {}
        for pkg, vehicle in self.in_vehicle.items():
            if pkg not in self.packages:
                return False, f"Unknown package in vehicle: {pkg}"
            if vehicle not in vehicle_contents:
                vehicle_contents[vehicle] = []
            vehicle_contents[vehicle].append(pkg)

        # FIX 3: Truck validation
        for truck in self.trucks:
            if truck not in self.at:
                return False, f"Truck {truck} has no location"
            loc = self.at[truck]
            if loc not in self.locations:
                return False, f"Truck {truck} at invalid location {loc}"
            if truck in self.in_vehicle:
                return False, f"Truck {truck} cannot be in another vehicle"

        # FIX 4: Airplane validation
        for airplane in self.airplanes:
            if airplane not in self.at:
                return False, f"Airplane {airplane} has no location"
            loc = self.at[airplane]
            if loc not in self.airports:
                return False, f"Airplane {airplane} at non-airport location {loc}"
            if airplane in self.in_vehicle:
                return False, f"Airplane {airplane} cannot be in another vehicle"

        # FIX 5: Location validation
        for loc in self.locations:
            if loc not in self.in_city:
                return False, f"Location {loc} not mapped to any city"
            city = self.in_city[loc]
            if city not in self.cities:
                return False, f"Location {loc} mapped to non-existent city {city}"

        # FIX 6: Airport validation
        for airport in self.airports:
            if airport not in self.locations:
                return False, f"Airport {airport} not in locations set"
            if airport not in self.in_city:
                return False, f"Airport {airport} not mapped to any city"

        # FIX 7: At dict validation
        for obj, loc in self.at.items():
            if obj not in list(self.packages) + list(self.trucks) + list(self.airplanes):
                return False, f"Unknown object in 'at': {obj}"
            if loc not in self.locations:
                return False, f"Invalid location in 'at': {loc}"

        # FIX 8: In-vehicle dict validation
        for obj, vehicle in self.in_vehicle.items():
            if obj not in self.packages:
                return False, f"Non-package object in vehicle: {obj}"
            if vehicle not in list(self.trucks) + list(self.airplanes):
                return False, f"Invalid vehicle: {vehicle}"
            if vehicle not in self.at:
                return False, f"Vehicle {vehicle} with no location"

        # FIX 9: Consistency check
        for obj in list(self.packages) + list(self.trucks) + list(self.airplanes):
            locations_count = 0
            if obj in self.at:
                locations_count += 1
            if obj in self.in_vehicle:
                locations_count += 1

            if obj in self.packages and locations_count != 1:
                return False, f"Package {obj} in {locations_count} places (not exactly 1)"

        # FIX 10: Airport distribution for inter-city
        if len(self.cities) > 1:
            cities_with_airports = set()
            for airport in self.airports:
                city = self.in_city.get(airport)
                if city:
                    cities_with_airports.add(city)

            if len(cities_with_airports) < len(self.cities):
                return False, f"Not all cities have airports (required for inter-city)"

        return True, None

    # FIX #1: REMOVE DUPLICATE - KEEP ONLY THIS ONE __hash__ METHOD
    def __hash__(self):
        """Make state hashable for deduplication."""
        at_tuple = tuple(sorted((k, v) for k, v in self.at.items()))
        in_veh_tuple = tuple(sorted((k, v) for k, v in self.in_vehicle.items()))
        return hash((
            frozenset(self.packages),
            frozenset(self.trucks),
            frozenset(self.airplanes),
            at_tuple,
            in_veh_tuple,
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
    """Create a valid initial Logistics state."""

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