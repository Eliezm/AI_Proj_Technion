"""
Utility for building Logistics problems with structured world generation.

Generates a complete Logistics world: cities, locations, vehicles, and packages.
"""

import random
from typing import List, Tuple
from state import LogisticsState, create_initial_state
from config import LogisticsGenerationParams


class LogisticsProblemBuilder:
    """Builds a complete Logistics problem with world structure."""

    # logistics_problem_builder.py - MODIFY build_world METHOD

    @staticmethod
    def build_world(
            params: LogisticsGenerationParams,
            random_seed: int = None
    ) -> Tuple[LogisticsState, List[str], List[str], List[str]]:
        """Build a valid Logistics world with guarantees.

        FIX: Ensure airports are properly distributed for inter-city transport.
        """

        if random_seed is not None:
            random.seed(random_seed)

        # Validate parameters
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

        # FIX: Guarantee at least one airport per city for inter-city problems
        airports = set()

        if params.num_cities > 1 and params.num_airplanes > 0:
            # For multi-city: ensure EVERY city has at least one airport
            for city in cities:
                city_locs = [loc for loc in locations if in_city[loc] == city]
                if city_locs:
                    airports.add(random.choice(city_locs))
        else:
            # For single city: just one airport is enough
            if locations:
                airports.add(locations[0])

        # Add extra airports randomly if requested
        remaining_locs = [loc for loc in locations if loc not in airports]
        for loc in remaining_locs:
            if random.random() < params.prob_airport:
                airports.add(loc)

        # FIX: Validate airports are in locations
        invalid_airports = [a for a in airports if a not in locations]
        if invalid_airports:
            raise ValueError(f"Invalid airports: {invalid_airports}")

        # FIX: Validate all cities have airports if multiple cities
        if params.num_cities > 1:
            cities_with_airports = set()
            for airport in airports:
                city = in_city.get(airport)
                if city:
                    cities_with_airports.add(city)

            missing_cities = [c for c in cities if c not in cities_with_airports]
            if missing_cities:
                # Add airport to missing cities
                for city in missing_cities:
                    city_locs = [loc for loc in locations if in_city[loc] == city]
                    if city_locs:
                        airports.add(random.choice(city_locs))

        # Create vehicles
        trucks = [f"truck-{i}" for i in range(params.num_trucks)]
        airplanes = [f"airplane-{i}" for i in range(params.num_airplanes)]
        packages = [f"pkg-{i}" for i in range(params.num_packages)]

        # Position vehicles and packages
        at_dict = {}

        # Trucks at various locations (spread across cities)
        for i, truck in enumerate(trucks):
            city = cities[i % len(cities)]
            city_locs = [loc for loc in locations if in_city[loc] == city]
            at_dict[truck] = random.choice(city_locs) if city_locs else locations[0]

        # Airplanes at airports
        if not airports:
            raise ValueError("No airports generated; inter-city transport impossible")

        for airplane in airplanes:
            at_dict[airplane] = random.choice(list(airports))

        # Packages at random locations
        for pkg in packages:
            at_dict[pkg] = random.choice(locations)

        # Create state
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

        # Validate world
        is_valid, error = initial_state.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid world: {error}")

        return initial_state, packages, trucks, airplanes