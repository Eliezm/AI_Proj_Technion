"""
PDDL file generation.

Outputs valid PDDL domain and problem files according to Requirement #10.
"""

from typing import List, Optional
from state import BlocksWorldState
from actions import Action, ActionType


class PDDLWriter:
    """Writes PDDL domain and problem files."""

    DOMAIN_TEMPLATE = """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (clear ?x)
               (on-table ?x)
               (arm-empty)
               (holding ?x)
               (on ?x ?y))

  (:action pickup
    :parameters (?ob)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) 
                 (not (arm-empty))))

  (:action putdown
    :parameters  (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob) 
                 (not (holding ?ob))))

  (:action stack
    :parameters  (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
                 (not (clear ?underob)) (not (holding ?ob))))

  (:action unstack
    :parameters  (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) (clear ?underob)
                 (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
"""

    @staticmethod
    def write_domain(filepath: str) -> None:
        """Write the Blocksworld domain file."""
        with open(filepath, 'w') as f:
            f.write(PDDLWriter.DOMAIN_TEMPLATE)

    @staticmethod
    def state_to_init_pddl(state: BlocksWorldState) -> str:
        """Convert a state to PDDL :init format."""
        facts = []

        # on-table facts
        for block in sorted(state.on_table):
            facts.append(f"(on-table {block})")

        # on facts
        for block, underblock in sorted(state.on.items()):
            facts.append(f"(on {block} {underblock})")

        # clear facts
        for block in sorted(state.clear):
            facts.append(f"(clear {block})")

        # arm state
        if state.arm_empty:
            facts.append("(arm-empty)")
        if state.holding:
            facts.append(f"(holding {state.holding})")

        return " ".join(facts)

    @staticmethod
    def state_to_goal_pddl(state: BlocksWorldState) -> str:
        """Convert a state to PDDL :goal format."""
        facts = []

        # on-table facts
        for block in sorted(state.on_table):
            facts.append(f"(on-table {block})")

        # on facts
        for block, underblock in sorted(state.on.items()):
            facts.append(f"(on {block} {underblock})")

        # clear facts
        for block in sorted(state.clear):
            facts.append(f"(clear {block})")

        # arm state (usually arm-empty in goals)
        if state.arm_empty:
            facts.append("(arm-empty)")
        if state.holding:
            facts.append(f"(holding {state.holding})")

        if len(facts) == 1:
            return facts[0]
        return "(and " + " ".join(facts) + ")"

    @staticmethod
    def write_problem(
            filepath: str,
            problem_name: str,
            initial_state: BlocksWorldState,
            goal_state: BlocksWorldState
    ) -> None:
        """
        Write a PDDL problem file.

        Requirement #10: Standard .pddl file format.
        """
        blocks = sorted(initial_state.blocks)
        objects_str = " ".join(blocks)

        init_str = PDDLWriter.state_to_init_pddl(initial_state)
        goal_str = PDDLWriter.state_to_goal_pddl(goal_state)

        problem_pddl = f"""(define (problem {problem_name})
  (:domain blocksworld)
  (:objects {objects_str})
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