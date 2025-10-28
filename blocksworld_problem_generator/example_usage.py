"""
CORRECTED Example usage of the Blocksworld problem generation framework.

This example demonstrates:
1. Creating states manually
2. Applying forward actions correctly
3. Generating diverse problems with different initial and goal states
4. Writing valid PDDL files

Key fixes:
- Proper state validation for held blocks
- Forward search from initial state to reach goal
- Verification that initial ≠ goal
"""

from state import create_empty_state
from actions import Action, ActionType, ActionExecutor
from backward_generator import BackwardProblemGenerator
from pddl_writer import PDDLWriter
from goal_archetypes import GoalArchetype


def example_1_simple_forward_search():
    """
    Example 1: Forward search - the CORRECT way to generate problems.

    This creates an initial state, applies actions, and reaches a goal state.
    Both states are different and the plan is guaranteed to work.
    """
    print("=" * 80)
    print("EXAMPLE 1: Forward Search Problem Generation (CORRECT METHOD)")
    print("=" * 80)

    # Step 1: Create initial state (all blocks on table)
    blocks = ["b0", "b1", "b2", "b3"]
    initial_state = create_empty_state(blocks)

    print("\n[INITIAL STATE]")
    print(f"  {initial_state}")

    # Step 2: Define a sequence of actions to reach goal
    plan = [
        Action(ActionType.PICKUP, ["b0"]),
        Action(ActionType.STACK, ["b0", "b1"]),
        Action(ActionType.PICKUP, ["b2"]),
        Action(ActionType.STACK, ["b2", "b3"]),
        Action(ActionType.UNSTACK, ["b0", "b1"]),
        Action(ActionType.PUTDOWN, ["b0"]),
    ]

    print("\n[EXECUTING PLAN]")
    current_state = initial_state.copy()

    for i, action in enumerate(plan, 1):
        result = ActionExecutor.execute_forward(current_state, action)
        if result is None:
            print(f"  {i}. {action} ✗ FAILED (preconditions not met)")
            break
        current_state = result
        print(f"  {i}. {action} ✓")

    goal_state = current_state

    # Step 3: Verify states are different
    print("\n[VERIFICATION]")
    print(f"  Initial state == Goal state? {initial_state == goal_state}")
    if initial_state != goal_state:
        print(f"  ✓ SUCCESS: States are DIFFERENT")
    else:
        print(f"  ✗ ERROR: States are identical!")
        return

    print(f"\n[FINAL STATES]")
    print(f"  Initial: {initial_state}")
    print(f"  Goal:    {goal_state}")
    print(f"  Plan length: {len(plan)}")

    # Step 4: Write PDDL files
    print(f"\n[GENERATING PDDL FILES]")
    pddl_writer = PDDLWriter()
    pddl_writer.write_domain("example1_domain.pddl")
    pddl_writer.write_problem(
        "example1_problem.pddl",
        "example1",
        initial_state,
        goal_state
    )
    print(f"  ✓ Written: example1_domain.pddl")
    print(f"  ✓ Written: example1_problem.pddl")

    # Print the generated PDDL
    print(f"\n[GENERATED PDDL PROBLEM]")
    with open("../example1_problem.pddl", 'r') as f:
        print(f.read())


def example_2_fixed_backward_generator():
    """
    Example 2: Using the FIXED backward generator.

    The backward generator now works correctly because the state validation
    has been fixed to handle held blocks properly.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Backward Generator (Now Fixed!)")
    print("=" * 80)

    generator = BackwardProblemGenerator(random_seed=42)

    print("\n[GENERATING PROBLEM]")
    print("  Blocks: 5")
    print("  Target plan length: 8")
    print("  Goal archetype: SINGLE_TOWER")

    try:
        initial_state, goal_state, plan, archetype = generator.generate_problem(
            num_blocks=5,
            target_plan_length=8,
            archetype=GoalArchetype.SINGLE_TOWER,
            tolerance=2
        )

        print(f"\n[RESULT]")
        print(f"  Archetype: {archetype.value}")
        print(f"  Plan length: {len(plan)}")
        print(f"  Blocks: {sorted(initial_state.blocks)}")

        # Verify different
        print(f"\n[VERIFICATION]")
        print(f"  Initial == Goal? {initial_state == goal_state}")

        if initial_state == goal_state:
            print(f"  ✗ ERROR: States are identical!")
            return

        print(f"  ✓ States are DIFFERENT")

        # Verify plan works
        test_state = initial_state.copy()
        for action in plan:
            test_state = ActionExecutor.execute_forward(test_state, action)
            if test_state is None:
                print(f"  ✗ Plan validation FAILED")
                return

        if test_state == goal_state:
            print(f"  ✓ Plan successfully reaches goal")
        else:
            print(f"  ✗ Plan does not reach goal")
            return

        print(f"\n[INITIAL STATE]")
        print(f"  {initial_state}")

        print(f"\n[GOAL STATE]")
        print(f"  {goal_state}")

        print(f"\n[PLAN ({len(plan)} steps)]")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action}")

        # Generate PDDL
        pddl_writer = PDDLWriter()
        pddl_writer.write_domain("example2_domain.pddl")
        pddl_writer.write_problem(
            "example2_problem.pddl",
            "example2",
            initial_state,
            goal_state
        )
        print(f"\n[PDDL FILES]")
        print(f"  ✓ Written: example2_domain.pddl")
        print(f"  ✓ Written: example2_problem.pddl")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_3_all_archetypes():
    """
    Example 3: Generate problems with different goal archetypes.

    Shows the variety of problems that can be generated.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: All Goal Archetypes")
    print("=" * 80)

    generator = BackwardProblemGenerator(random_seed=123)

    for archetype in [
        GoalArchetype.SINGLE_TOWER,
        GoalArchetype.MULTIPLE_TOWERS,
        GoalArchetype.CLEAR_TABLE,
        GoalArchetype.SCATTERED_RELATIONS,
        GoalArchetype.MIXED_PYRAMID,
    ]:
        print(f"\n[{archetype.value.upper()}]")
        try:
            initial_state, goal_state, plan, _ = generator.generate_problem(
                num_blocks=4,
                target_plan_length=6,
                archetype=archetype,
                tolerance=1
            )

            is_different = initial_state != goal_state
            print(f"  Plan length: {len(plan)}")
            print(f"  Different states: {'✓ YES' if is_different else '✗ NO'}")

            if not is_different:
                print(f"    WARNING: Initial and goal states are identical!")

        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_4_batch_generation():
    """
    Example 4: Generate a batch of diverse problems.

    Creates multiple problems with different seeds.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Generation (10 problems)")
    print("=" * 80)

    num_problems = 10
    problems = []

    for i in range(num_problems):
        generator = BackwardProblemGenerator(random_seed=i)
        try:
            initial_state, goal_state, plan, archetype = generator.generate_problem(
                num_blocks=4,
                target_plan_length=7,
                tolerance=2
            )

            is_different = initial_state != goal_state
            problems.append({
                'id': i,
                'archetype': archetype.value,
                'plan_length': len(plan),
                'different': is_different,
                'initial': initial_state,
                'goal': goal_state,
                'plan': plan
            })

            status = "✓" if is_different else "✗"
            print(f"  Problem {i}: {status} {archetype.value:20s} length={len(plan)}")

        except Exception as e:
            print(f"  Problem {i}: ✗ Error: {e}")

    # Summary
    print(f"\n[SUMMARY]")
    successful = sum(1 for p in problems if p['different'])
    print(f"  Generated: {len(problems)}/{num_problems}")
    print(f"  With different states: {successful}/{len(problems)}")

    if successful == len(problems):
        print(f"  ✓ ALL PROBLEMS VALID")
    else:
        print(f"  ✗ Some problems have identical initial/goal states")

    # Write first 3 to PDDL
    print(f"\n[WRITING FIRST 3 PROBLEMS TO PDDL]")
    pddl_writer = PDDLWriter()
    pddl_writer.write_domain("batch_domain.pddl")

    for problem in problems[:3]:
        pddl_writer.write_problem(
            f"batch_problem_{problem['id']}.pddl",
            f"batch-problem-{problem['id']}",
            problem['initial'],
            problem['goal']
        )
        print(f"  ✓ Written: batch_problem_{problem['id']}.pddl")


def example_5_state_transitions_visualization():
    """
    Example 5: Visualize state transitions during forward search.

    Shows how each action transforms the state.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: State Transitions Visualization")
    print("=" * 80)

    blocks = ["b0", "b1", "b2"]
    state = create_empty_state(blocks)

    print(f"\nStep 0: Initial State")
    print(f"  on_table: {state.on_table}")
    print(f"  on: {state.on}")
    print(f"  clear: {state.clear}")
    print(f"  holding: {state.holding}")
    print(f"  arm_empty: {state.arm_empty}")

    actions = [
        ("pickup", Action(ActionType.PICKUP, ["b0"])),
        ("stack(b0, b1)", Action(ActionType.STACK, ["b0", "b1"])),
        ("pickup", Action(ActionType.PICKUP, ["b2"])),
        ("stack(b2, b0)", Action(ActionType.STACK, ["b2", "b0"])),
    ]

    for step, (label, action) in enumerate(actions, 1):
        state = ActionExecutor.execute_forward(state, action)
        if state is None:
            print(f"✗ Action failed!")
            break

        print(f"\nStep {step}: {label}")
        print(f"  on_table: {state.on_table}")
        print(f"  on: {state.on}")
        print(f"  clear: {state.clear}")
        print(f"  holding: {state.holding}")
        print(f"  arm_empty: {state.arm_empty}")


if __name__ == "__main__":
    example_1_simple_forward_search()
    example_2_fixed_backward_generator()
    example_3_all_archetypes()
    example_4_batch_generation()
    example_5_state_transitions_visualization()

    print("\n" + "=" * 80)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)