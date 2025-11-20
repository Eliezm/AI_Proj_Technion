"""
Utility for building Logistics problems with structured world generation.
"""

import random
from typing import List, Tuple
from state import LogisticsState, create_initial_state
from config import LogisticsGenerationParams


class LogisticsProblemBuilder:
    """Builds a complete Logistics problem with world structure."""

    @staticmethod
    def build_world(
            params: LogisticsGenerationParams,
            random_seed: int = None
    ) -> Tuple[LogisticsState, List[str], List[str], List[str]]:
        """
        FIX #7: Build valid Logistics world with strict airport guarantees.

        Ensures:
        - Every city has at least one airport (if multi-city)
        - Trucks distributed across cities
        - Airplanes at airports
        """

        if random_seed is not None:
            random.seed(random_seed)

        if params.num_cities < 1:
            raise ValueError("Must have at least 1 city")
        if params.locations_per_city < 1:
            raise ValueError("Must have at least 1 location per city")
        if params.num_packages < 1:
            raise ValueError("Must have at least 1 package")
        if params.num_trucks < 1:
            raise ValueError("Must have at least 1 truck")
        if params.num_airplanes < 1:
            raise ValueError("Must have at least 1 airplane")

        cities = [f"city-{i}" for i in range(params.num_cities)]
        locations = []
        in_city = {}

        # Create locations
        for city in cities:
            for j in range(params.locations_per_city):
                loc = f"loc-{city}-{j}"
                locations.append(loc)
                in_city[loc] = city

        # FIX #7: GUARANTEE airports in every city for multi-city
        airports = set()

        if params.num_cities > 1 and params.num_airplanes > 0:
            # Ensure ONE airport per city minimum
            for city in cities:
                city_locs = [loc for loc in locations if in_city[loc] == city]
                if city_locs:
                    airport = random.choice(city_locs)
                    airports.add(airport)
        else:
            # Single city: at least one airport
            if locations:
                airports.add(random.choice(locations))

        # Add extra airports randomly
        remaining_locs = [loc for loc in locations if loc not in airports]
        for loc in remaining_locs:
            if random.random() < params.prob_airport:
                airports.add(loc)

        # FIX #7: Validate airports
        if params.num_cities > 1 and len(airports) < params.num_cities:
            raise ValueError(f"Insufficient airports: {len(airports)} < {params.num_cities} cities")

        # Create vehicles
        trucks = [f"truck-{i}" for i in range(params.num_trucks)]
        airplanes = [f"airplane-{i}" for i in range(params.num_airplanes)]
        packages = [f"pkg-{i}" for i in range(params.num_packages)]

        # Position vehicles
        at_dict = {}

        # FIX #7: Distribute trucks across cities
        for i, truck in enumerate(trucks):
            city = cities[i % len(cities)]
            city_locs = [loc for loc in locations if in_city[loc] == city]
            at_dict[truck] = random.choice(city_locs) if city_locs else locations[0]

        # FIX #7: Position airplanes at airports only
        if not airports:
            raise ValueError("No airports: multi-city transport impossible")

        for airplane in airplanes:
            at_dict[airplane] = random.choice(list(airports))

        # Position packages randomly
        for pkg in packages:
            at_dict[pkg] = random.choice(locations)

        # Create and validate state
        initial_state = create_initial_state(
            packages=packages,
            trucks=trucks,
            airplanes=airplanes,
            locations=locations,
            cities=cities,
            in_city=in_city,
            airports=airports,
            at=at_dict,
            in_vehicle={}
        )

        is_valid, error = initial_state.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid world: {error}")

        return initial_state, packages, trucks, airplanes