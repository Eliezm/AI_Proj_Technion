# """
# PDDL file generation for Logistics domain.
#
# Outputs valid PDDL domain and problem files according to Requirement #10.
# """
#
# from state import LogisticsState
#
#
# class PDDLWriter:
#     """Writes PDDL domain and problem files for Logistics."""
#
#     DOMAIN_TEMPLATE = """(define (domain logistics-strips)
#   (:requirements :strips)
#   (:predicates
#     (OBJ ?obj)
#     (TRUCK ?truck)
#     (AIRPLANE ?airplane)
#     (LOCATION ?loc)
#     (CITY ?city)
#     (AIRPORT ?airport)
#     (at ?obj ?loc)
#     (in ?obj1 ?obj2)
#     (in-city ?loc ?city)
#   )
#
#   (:action LOAD-TRUCK
#     :parameters (?obj ?truck ?loc)
#     :precondition
#       (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
#            (at ?truck ?loc) (at ?obj ?loc))
#     :effect
#       (and (not (at ?obj ?loc)) (in ?obj ?truck))
#   )
#
#   (:action UNLOAD-TRUCK
#     :parameters (?obj ?truck ?loc)
#     :precondition
#       (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
#            (at ?truck ?loc) (in ?obj ?truck))
#     :effect
#       (and (not (in ?obj ?truck)) (at ?obj ?loc))
#   )
#
#   (:action LOAD-AIRPLANE
#     :parameters (?obj ?airplane ?loc)
#     :precondition
#       (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
#            (at ?airplane ?loc) (at ?obj ?loc))
#     :effect
#       (and (not (at ?obj ?loc)) (in ?obj ?airplane))
#   )
#
#   (:action UNLOAD-AIRPLANE
#     :parameters (?obj ?airplane ?loc)
#     :precondition
#       (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
#            (at ?airplane ?loc) (in ?obj ?airplane))
#     :effect
#       (and (not (in ?obj ?airplane)) (at ?obj ?loc))
#   )
#
#   (:action DRIVE-TRUCK
#     :parameters (?truck ?loc-from ?loc-to ?city)
#     :precondition
#       (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
#            (at ?truck ?loc-from)
#            (in-city ?loc-from ?city)
#            (in-city ?loc-to ?city))
#     :effect
#       (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to))
#   )
#
#   (:action FLY-AIRPLANE
#     :parameters (?airplane ?loc-from ?loc-to)
#     :precondition
#       (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
#            (at ?airplane ?loc-from))
#     :effect
#       (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to))
#   )
# )
# """
#
#     @staticmethod
#     def write_domain(filepath: str) -> None:
#         """Write the Logistics domain file."""
#         with open(filepath, 'w') as f:
#             f.write(PDDLWriter.DOMAIN_TEMPLATE)
#
#     @staticmethod
#     def state_to_objects_pddl(state: LogisticsState) -> str:
#         """Convert a state to PDDL :objects format with proper typing."""
#         objects = []
#
#         # Group objects by type
#         if state.packages:
#             objects.append(f"{' '.join(sorted(state.packages))} - OBJ")
#         if state.trucks:
#             objects.append(f"{' '.join(sorted(state.trucks))} - TRUCK")
#         if state.airplanes:
#             objects.append(f"{' '.join(sorted(state.airplanes))} - AIRPLANE")
#         if state.locations:
#             objects.append(f"{' '.join(sorted(state.locations))} - LOCATION")
#         if state.cities:
#             objects.append(f"{' '.join(sorted(state.cities))} - CITY")
#
#         return "\n    ".join(objects)
#
#     @staticmethod
#     def state_to_init_pddl(state: LogisticsState) -> str:
#         """Convert a state to PDDL :init format WITH type predicates."""
#         facts = []
#
#         # Type predicates (REQUIRED for proper PDDL)
#         for obj in sorted(state.packages):
#             facts.append(f"(OBJ {obj})")
#         for truck in sorted(state.trucks):
#             facts.append(f"(TRUCK {truck})")
#         for airplane in sorted(state.airplanes):
#             facts.append(f"(AIRPLANE {airplane})")
#         for loc in sorted(state.locations):
#             facts.append(f"(LOCATION {loc})")
#         for city in sorted(state.cities):
#             facts.append(f"(CITY {city})")
#
#         # Airport type predicates
#         for airport in sorted(state.airports):
#             facts.append(f"(AIRPORT {airport})")
#
#         # at facts
#         for obj, loc in sorted(state.at.items()):
#             facts.append(f"(at {obj} {loc})")
#
#         # in facts
#         for obj, vehicle in sorted(state.in_vehicle.items()):
#             facts.append(f"(in {obj} {vehicle})")
#
#         # in-city facts (static)
#         for loc, city in sorted(state.in_city.items()):
#             facts.append(f"(in-city {loc} {city})")
#
#         return " ".join(facts)
#
#     # pddl_writer.py - REPLACE state_to_goal_pddl METHOD
#
#     @staticmethod
#     def state_to_goal_pddl(state: LogisticsState) -> str:
#         """
#         Convert a state to PDDL :goal format.
#
#         FIX: Handle packages that are in vehicles at goal time.
#         """
#         facts = []
#
#         # at facts for packages only
#         for pkg in sorted(state.packages):
#             if pkg in state.at:
#                 facts.append(f"(at {pkg} {state.at[pkg]})")
#             elif pkg in state.in_vehicle:
#                 # FIX: Package is in a vehicle - extract vehicle location
#                 vehicle = state.in_vehicle[pkg]
#                 if vehicle in state.at:
#                     # Goal location is same as vehicle location
#                     facts.append(f"(at {pkg} {state.at[vehicle]})")
#                 else:
#                     raise ValueError(f"Vehicle {vehicle} carrying {pkg} has no location in goal state")
#             else:
#                 raise ValueError(f"Package {pkg} has no location or vehicle in goal state")
#
#         if not facts:
#             raise ValueError("Goal state has no package locations")
#
#         if len(facts) == 1:
#             return facts[0]
#
#         return "(and " + " ".join(facts) + ")"
#
#     @staticmethod
#     def write_problem(
#             filepath: str,
#             problem_name: str,
#             initial_state: LogisticsState,
#             goal_state: LogisticsState
#     ) -> None:
#         """
#         Write a PDDL problem file.
#
#         Requirement #10: Standard .pddl file format.
#         """
#         objects_str = PDDLWriter.state_to_objects_pddl(initial_state)
#         init_str = PDDLWriter.state_to_init_pddl(initial_state)
#         goal_str = PDDLWriter.state_to_goal_pddl(goal_state)
#
#         problem_pddl = f"""(define (problem {problem_name})
#   (:domain logistics-strips)
#   (:objects
#     {objects_str}
#   )
#   (:init
#     {init_str}
#   )
#   (:goal
#     {goal_str}
#   )
# )
# """
#         with open(filepath, 'w') as f:
#             f.write(problem_pddl)

# logistics_problem_generator/pddl_writer.py

"""
PDDL file generation for Logistics domain.

Outputs valid PDDL domain and problem files according to Requirement #10.
"""

from state import LogisticsState


class PDDLWriter:
    """Writes PDDL domain and problem files for Logistics."""

    # FIXED: Simpler domain without type predicates
    DOMAIN_TEMPLATE = """(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates
    (at ?obj ?loc)
    (in ?obj ?vehicle)
    (in-city ?loc ?city)
    (is-location ?loc)
    (is-city ?city)
    (is-truck ?truck)
    (is-airplane ?airplane)
    (is-airport ?loc)
    (is-package ?pkg)
  )

  (:action load-truck
    :parameters (?pkg ?truck ?loc)
    :precondition
      (and (is-package ?pkg) (is-truck ?truck) (is-location ?loc)
           (at ?truck ?loc) (at ?pkg ?loc))
    :effect
      (and (not (at ?pkg ?loc)) (in ?pkg ?truck))
  )

  (:action unload-truck
    :parameters (?pkg ?truck ?loc)
    :precondition
      (and (is-package ?pkg) (is-truck ?truck) (is-location ?loc)
           (at ?truck ?loc) (in ?pkg ?truck))
    :effect
      (and (not (in ?pkg ?truck)) (at ?pkg ?loc))
  )

  (:action load-airplane
    :parameters (?pkg ?airplane ?loc)
    :precondition
      (and (is-package ?pkg) (is-airplane ?airplane) (is-airport ?loc)
           (at ?airplane ?loc) (at ?pkg ?loc))
    :effect
      (and (not (at ?pkg ?loc)) (in ?pkg ?airplane))
  )

  (:action unload-airplane
    :parameters (?pkg ?airplane ?loc)
    :precondition
      (and (is-package ?pkg) (is-airplane ?airplane) (is-airport ?loc)
           (at ?airplane ?loc) (in ?pkg ?airplane))
    :effect
      (and (not (in ?pkg ?airplane)) (at ?pkg ?loc))
  )

  (:action drive-truck
    :parameters (?truck ?loc-from ?loc-to ?city)
    :precondition
      (and (is-truck ?truck) (is-location ?loc-from) (is-location ?loc-to) (is-city ?city)
           (at ?truck ?loc-from)
           (in-city ?loc-from ?city)
           (in-city ?loc-to ?city))
    :effect
      (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to))
  )

  (:action fly-airplane
    :parameters (?airplane ?loc-from ?loc-to)
    :precondition
      (and (is-airplane ?airplane) (is-airport ?loc-from) (is-airport ?loc-to)
           (at ?airplane ?loc-from))
    :effect
      (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to))
  )
)
"""

    @staticmethod
    def write_domain(filepath: str) -> None:
        """Write the Logistics domain file."""
        with open(filepath, 'w') as f:
            f.write(PDDLWriter.DOMAIN_TEMPLATE)

    @staticmethod
    def state_to_objects_pddl(state: LogisticsState) -> str:
        """Convert state to PDDL objects (no typing to keep it simple)."""
        objects = []

        if state.packages:
            objects.append(f"{' '.join(sorted(state.packages))}")
        if state.trucks:
            objects.append(f"{' '.join(sorted(state.trucks))}")
        if state.airplanes:
            objects.append(f"{' '.join(sorted(state.airplanes))}")
        if state.locations:
            objects.append(f"{' '.join(sorted(state.locations))}")
        if state.cities:
            objects.append(f"{' '.join(sorted(state.cities))}")

        return " ".join(objects)

    @staticmethod
    def state_to_init_pddl(state: LogisticsState) -> str:
        """Convert state to PDDL :init format."""
        facts = []

        # Type facts
        for pkg in sorted(state.packages):
            facts.append(f"(is-package {pkg})")
        for truck in sorted(state.trucks):
            facts.append(f"(is-truck {truck})")
        for airplane in sorted(state.airplanes):
            facts.append(f"(is-airplane {airplane})")
        for loc in sorted(state.locations):
            facts.append(f"(is-location {loc})")
        for city in sorted(state.cities):
            facts.append(f"(is-city {city})")
        for airport in sorted(state.airports):
            facts.append(f"(is-airport {airport})")

        # Positional facts
        for obj, loc in sorted(state.at.items()):
            facts.append(f"(at {obj} {loc})")

        # Vehicle containment
        for pkg, vehicle in sorted(state.in_vehicle.items()):
            facts.append(f"(in {pkg} {vehicle})")

        # City mappings (static)
        for loc, city in sorted(state.in_city.items()):
            facts.append(f"(in-city {loc} {city})")

        return " ".join(facts)

    @staticmethod
    def state_to_goal_pddl(state: LogisticsState) -> str:
        """Convert state to PDDL :goal format."""
        facts = []

        # Goal: all packages at their goal locations (not in vehicles)
        for pkg in sorted(state.packages):
            if pkg in state.at:
                facts.append(f"(at {pkg} {state.at[pkg]})")
            elif pkg in state.in_vehicle:
                # Shouldn't happen if goal generation is correct
                vehicle = state.in_vehicle[pkg]
                if vehicle in state.at:
                    facts.append(f"(at {pkg} {state.at[vehicle]})")
                else:
                    raise ValueError(f"Cannot determine location for {pkg} in goal state")
            else:
                raise ValueError(f"Package {pkg} has no location in goal state")

        if not facts:
            raise ValueError("Goal state has no package locations")

        if len(facts) == 1:
            return facts[0]

        return "(and " + " ".join(facts) + ")"

    @staticmethod
    def write_problem(
            filepath: str,
            problem_name: str,
            initial_state: LogisticsState,
            goal_state: LogisticsState
    ) -> None:
        """Write a PDDL problem file."""
        objects_str = PDDLWriter.state_to_objects_pddl(initial_state)
        init_str = PDDLWriter.state_to_init_pddl(initial_state)
        goal_str = PDDLWriter.state_to_goal_pddl(goal_state)

        problem_pddl = f"""(define (problem {problem_name})
  (:domain logistics-strips)
  (:objects
    {objects_str}
  )
  (:init
    {init_str}
  )
  (:goal
    {goal_str}
  )
)
"""
        with open(filepath, 'w') as f:
            f.write(problem_pddl)