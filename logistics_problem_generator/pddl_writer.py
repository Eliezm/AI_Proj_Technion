"""
PDDL file generation for Logistics domain.

Outputs valid PDDL domain and problem files according to Requirement #10.
"""

from state import LogisticsState


class PDDLWriter:
    """Writes PDDL domain and problem files for Logistics."""

    DOMAIN_TEMPLATE = """(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates
    (OBJ ?obj)
    (TRUCK ?truck)
    (AIRPLANE ?airplane)
    (LOCATION ?loc)
    (CITY ?city)
    (AIRPORT ?airport)
    (at ?obj ?loc)
    (in ?obj1 ?obj2)
    (in-city ?loc ?city)
  )

  (:action LOAD-TRUCK
    :parameters (?obj ?truck ?loc)
    :precondition
      (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
           (at ?truck ?loc) (at ?obj ?loc))
    :effect
      (and (not (at ?obj ?loc)) (in ?obj ?truck))
  )

  (:action UNLOAD-TRUCK
    :parameters (?obj ?truck ?loc)
    :precondition
      (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
           (at ?truck ?loc) (in ?obj ?truck))
    :effect
      (and (not (in ?obj ?truck)) (at ?obj ?loc))
  )

  (:action LOAD-AIRPLANE
    :parameters (?obj ?airplane ?loc)
    :precondition
      (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
           (at ?airplane ?loc) (at ?obj ?loc))
    :effect
      (and (not (at ?obj ?loc)) (in ?obj ?airplane))
  )

  (:action UNLOAD-AIRPLANE
    :parameters (?obj ?airplane ?loc)
    :precondition
      (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
           (at ?airplane ?loc) (in ?obj ?airplane))
    :effect
      (and (not (in ?obj ?airplane)) (at ?obj ?loc))
  )

  (:action DRIVE-TRUCK
    :parameters (?truck ?loc-from ?loc-to ?city)
    :precondition
      (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
           (at ?truck ?loc-from)
           (in-city ?loc-from ?city)
           (in-city ?loc-to ?city))
    :effect
      (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to))
  )

  (:action FLY-AIRPLANE
    :parameters (?airplane ?loc-from ?loc-to)
    :precondition
      (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
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
        """Convert a state to PDDL :objects format with proper typing."""
        objects = []

        # Group objects by type
        if state.packages:
            objects.append(f"{' '.join(sorted(state.packages))} - OBJ")
        if state.trucks:
            objects.append(f"{' '.join(sorted(state.trucks))} - TRUCK")
        if state.airplanes:
            objects.append(f"{' '.join(sorted(state.airplanes))} - AIRPLANE")
        if state.locations:
            objects.append(f"{' '.join(sorted(state.locations))} - LOCATION")
        if state.cities:
            objects.append(f"{' '.join(sorted(state.cities))} - CITY")

        return "\n    ".join(objects)

    @staticmethod
    def state_to_init_pddl(state: LogisticsState) -> str:
        """Convert a state to PDDL :init format (without type predicates)."""
        facts = []

        # at facts
        for obj, loc in sorted(state.at.items()):
            facts.append(f"(at {obj} {loc})")

        # in facts
        for obj, vehicle in sorted(state.in_vehicle.items()):
            facts.append(f"(in {obj} {vehicle})")

        # in-city facts (static)
        for loc, city in sorted(state.in_city.items()):
            facts.append(f"(in-city {loc} {city})")

        # AIRPORT predicates
        for airport in sorted(state.airports):
            facts.append(f"(AIRPORT {airport})")

        return " ".join(facts)

    @staticmethod
    def state_to_goal_pddl(state: LogisticsState) -> str:
        """Convert a state to PDDL :goal format (packages only)."""
        facts = []

        # at facts for packages (packages should be at specific locations)
        for pkg in sorted(state.packages):
            if pkg in state.at:
                facts.append(f"(at {pkg} {state.at[pkg]})")

        if not facts:
            facts.append("(at pkg-0 loc-0)")  # Fallback

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
        """
        Write a PDDL problem file.

        Requirement #10: Standard .pddl file format.
        """
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