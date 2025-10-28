"""
Comprehensive Example Usage of the Logistics Problem Generation Framework

This demonstrates:
1. Manual state creation and validation
2. Forward action execution
3. Backward problem generation with thorough validation
4. PDDL file writing
5. Batch problem generation with quality checks
6. Plan verification and debugging
"""

from state import LogisticsState, create_initial_state
from actions import Action, ActionType, ActionExecutor
from backward_generator import BackwardProblemGenerator, ReverseActionExecutor
from pddl_writer import PDDLWriter
from goal_archetypes import GoalArchetype, create_goal_state_from_dict
from logistics_problem_builder import LogisticsProblemBuilder
from config import DEFAULT_LOGISTICS_PARAMS, DIFFICULTY_TIERS
import random


def example_1_manual_state_creation():
    """
    Example 1: Create and validate states manually.

    Shows how to construct a valid Logistics state from scratch.
    """
    print("=" * 80)
    print("EXAMPLE 1: Manual State Creation and Validation")
    print("=" * 80)

    # Define world structure
    packages = ["pkg-0", "pkg-1"]
    trucks = ["truck-0"]
    airplanes = ["airplane-0"]
    cities = ["city-0", "city-1"]
    locations = ["loc-city-0-0", "loc-city-0-1", "loc-city-1-0", "loc-city-1-1"]

    # Build mappings
    in_city = {
        "loc-city-0-0": "city-0",
        "loc-city-0-1": "city-0",
        "loc-city-1-0": "city-1",
        "loc-city-1-1": "city-1",
    }

    airports = {"loc-city-0-1", "loc-city-1-1"}

    # Position objects
    at_dict = {
        "pkg-0": "loc-city-0-0",
        "pkg-1": "loc-city-1-0",
        "truck-0": "loc-city-0-0",
        "airplane-0": "loc-city-0-1",
    }

    print("\n[WORLD STRUCTURE]")
    print(f"  Cities: {cities}")
    print(f"  Locations: {locations}")
    print(f"  Airports: {airports}")
    print(f"  Packages: {packages}")
    print(f"  Trucks: {trucks}")
    print(f"  Airplanes: {airplanes}")

    print("\n[CREATING INITIAL STATE]")
    try:
        initial_state = create_initial_state(
            packages=packages,
            trucks=trucks,
            airplanes=airplanes,
            locations=locations,
            cities=cities,
            in_city=in_city,
            airports=airports,
            at=at_dict
        )
        print("  ✓ State created successfully")
        print(f"\n{initial_state}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    # Validate state
    print("\n[VALIDATION]")
    is_valid, error = initial_state.is_valid()
    if is_valid:
        print("  ✓ State is valid")
    else:
        print(f"  ✗ State is invalid: {error}")


def example_2_forward_actions():
    """
    Example 2: Execute actions in forward direction.

    Demonstrates how actions modify state and how to chain them.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Forward Action Execution")
    print("=" * 80)

    # Create simple world
    packages = ["pkg-0"]
    trucks = ["truck-0"]
    airplanes = []
    cities = ["city-0"]
    locations = ["loc-0", "loc-1"]

    in_city = {"loc-0": "city-0", "loc-1": "city-0"}
    airports = set()

    at_dict = {
        "pkg-0": "loc-0",
        "truck-0": "loc-0",
    }

    initial_state = create_initial_state(
        packages=packages,
        trucks=trucks,
        airplanes=airplanes,
        locations=locations,
        cities=cities,
        in_city=in_city,
        airports=airports,
        at=at_dict
    )

    print("\n[INITIAL STATE]")
    print(f"  {initial_state}")

    # Define action sequence
    actions = [
        Action(ActionType.LOAD_TRUCK, ["pkg-0", "truck-0", "loc-0"]),
        Action(ActionType.DRIVE_TRUCK, ["truck-0", "loc-0", "loc-1", "city-0"]),
        Action(ActionType.UNLOAD_TRUCK, ["pkg-0", "truck-0", "loc-1"]),
    ]

    print("\n[EXECUTING PLAN]")
    current_state = initial_state.copy()

    for i, action in enumerate(actions, 1):
        result = ActionExecutor.execute_forward(current_state, action)
        if result is None:
            print(f"  {i}. {action} ✗ FAILED")
            break
        current_state = result
        print(f"  {i}. {action} ✓")
        print(f"      State: {current_state}")

    print(f"\n[FINAL STATE]")
    print(f"  {current_state}")

    # Verify pkg-0 is at loc-1
    if current_state.at.get("pkg-0") == "loc-1":
        print(f"\n  ✓ SUCCESS: Package delivered to destination!")
    else:
        print(f"\n  ✗ FAILED: Package not at destination")


def example_3_build_world():
    """
    Example 3: Use LogisticsProblemBuilder to create a world.

    Automatic world generation with proper validation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Automated World Building")
    print("=" * 80)

    params = DEFAULT_LOGISTICS_PARAMS['small']

    print(f"\n[WORLD PARAMETERS]")
    print(f"  Cities: {params.num_cities}")
    print(f"  Locations per city: {params.locations_per_city}")
    print(f"  Packages: {params.num_packages}")
    print(f"  Trucks: {params.num_trucks}")
    print(f"  Airplanes: {params.num_airplanes}")
    print(f"  Airport probability: {params.prob_airport}")

    print(f"\n[BUILDING WORLD]")
    try:
        world, packages, trucks, airplanes = LogisticsProblemBuilder.build_world(
            params,
            random_seed=42
        )

        print(f"  ✓ World built successfully")
        print(f"\n  Cities: {sorted(world.cities)}")
        print(f"  Locations: {sorted(world.locations)}")
        print(f"  Airports: {sorted(world.airports)}")
        print(f"  Packages: {packages}")
        print(f"  Trucks: {trucks}")
        print(f"  Airplanes: {airplanes}")

        print(f"\n[INITIAL STATE]")
        print(f"  {world}")

        is_valid, error = world.is_valid()
        if is_valid:
            print(f"\n  ✓ World state is valid")
        else:
            print(f"\n  ✗ World state is INVALID: {error}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_4_backward_generation_single():
    """
    Example 4: Generate a single problem using backward search.

    Shows detailed problem generation with validation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Single Problem Generation (with Validation)")
    print("=" * 80)

    generator = BackwardProblemGenerator(random_seed=42)

    print(f"\n[GENERATING PROBLEM]")
    print(f"  Difficulty: small")
    print(f"  Target plan length: 7")

    try:
        initial_state, goal_state, plan, archetype = generator.generate_problem(
            difficulty='small',
            target_plan_length=7,
            tolerance=2
        )

        print(f"  ✓ Problem generated successfully")

        print(f"\n[WORLD INFO]")
        print(f"  Cities: {len(initial_state.cities)}")
        print(f"  Locations: {len(initial_state.locations)}")
        print(f"  Packages: {len(initial_state.packages)}")
        print(f"  Trucks: {len(initial_state.trucks)}")
        print(f"  Airplanes: {len(initial_state.airplanes)}")

        print(f"\n[RESULT]")
        print(f"  Archetype: {archetype.value}")
        print(f"  Plan length: {len(plan)} (target: 7)")
        print(f"  Initial ≠ Goal: {initial_state != goal_state}")

        print(f"\n[PACKAGE DISTRIBUTION]")
        print(f"  Initial state:")
        for pkg in sorted(initial_state.packages):
            loc_or_vehicle = initial_state.at.get(pkg)
            if loc_or_vehicle is None:
                # FIX: Convert sets to lists before concatenation
                for vehicle in list(initial_state.trucks) + list(initial_state.airplanes):
                    if initial_state.in_vehicle.get(pkg) == vehicle:
                        loc_or_vehicle = f"in {vehicle}"
                        break
            print(f"    {pkg}: {loc_or_vehicle}")

        print(f"  Goal state:")
        for pkg in sorted(goal_state.packages):
            loc = goal_state.at.get(pkg, "UNKNOWN")
            print(f"    {pkg}: {loc}")

        if plan:
            print(f"\n[PLAN ({len(plan)} actions)]")
            for i, action in enumerate(plan, 1):
                print(f"  {i:2d}. {action}")

            # Verify plan
            print(f"\n[VERIFICATION]")
            test_state = initial_state.copy()
            all_valid = True
            for i, action in enumerate(plan, 1):
                test_state = ActionExecutor.execute_forward(test_state, action)
                if test_state is None:
                    print(f"  ✗ Action {i} failed: {action}")
                    all_valid = False
                    break

            if all_valid:
                # Check goal
                all_packages_at_goal = True
                for pkg in goal_state.packages:
                    goal_loc = goal_state.at.get(pkg)
                    current_loc = test_state.at.get(pkg)
                    if goal_loc != current_loc:
                        print(f"  ✗ {pkg}: at {current_loc}, goal is {goal_loc}")
                        all_packages_at_goal = False

                if all_packages_at_goal:
                    print(f"  ✓ All packages at goal locations!")
                    print(f"  ✓ PROBLEM IS VALID")
                else:
                    print(f"  ✗ Not all packages at goal")
        else:
            print(f"\n[NOTE] Empty plan generated")
            if initial_state == goal_state:
                print(f"  Initial state equals goal state (trivial problem)")
            else:
                print(f"  WARNING: States differ but no plan was generated!")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_5_pddl_generation():
    """
    Example 5: Generate PDDL files from a problem.

    Shows PDDL output and validation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: PDDL File Generation")
    print("=" * 80)

    generator = BackwardProblemGenerator(random_seed=42)

    try:
        initial_state, goal_state, plan, archetype = generator.generate_problem(
            difficulty='small',
            target_plan_length=7,
            tolerance=2
        )

        pddl_writer = PDDLWriter()

        print(f"\n[WRITING PDDL FILES]")
        pddl_writer.write_domain("logistics_domain.pddl")
        print(f"  ✓ Written: logistics_domain.pddl")

        pddl_writer.write_problem(
            "logistics_problem.pddl",
            "logistics-example",
            initial_state,
            goal_state
        )
        print(f"  ✓ Written: logistics_problem.pddl")

        print(f"\n[GENERATED DOMAIN FILE (excerpt)]")
        with open("logistics_domain.pddl", 'r') as f:
            content = f.read()
            lines = content.split('\n')[:20]
            for line in lines:
                print(f"  {line}")
            print(f"  ...")

        print(f"\n[GENERATED PROBLEM FILE]")
        with open("logistics_problem.pddl", 'r') as f:
            content = f.read()
            print(content[:800])
            if len(content) > 800:
                print("  ...")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_6_batch_generation_validated():
    """
    Example 6: Generate a batch of diverse, validated problems.

    THIS IS THE KEY EXAMPLE - Shows how to robustly generate valid problems.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Batch Problem Generation (with Validation)")
    print("=" * 80)

    num_problems = 10
    problems = []
    failed = []

    print(f"\n[GENERATING {num_problems} SMALL PROBLEMS]")
    print(f"Target plan length: 7 (±2)")
    print()

    for i in range(num_problems):
        try:
            generator = BackwardProblemGenerator(random_seed=i)
            initial_state, goal_state, plan, archetype = generator.generate_problem(
                difficulty='small',
                target_plan_length=7,
                tolerance=2
            )

            # Validate the problem
            if initial_state == goal_state:
                status = "⚠ TRIVIAL"
                failed.append((i, "Trivial problem (initial == goal)"))
            elif len(plan) == 0:
                status = "⚠ EMPTY"
                failed.append((i, "Empty plan generated"))
            else:
                # Verify plan execution
                test_state = initial_state.copy()
                plan_valid = True
                for action in plan:
                    test_state = ActionExecutor.execute_forward(test_state, action)
                    if test_state is None:
                        plan_valid = False
                        break

                if not plan_valid:
                    status = "✗ INVALID"
                    failed.append((i, "Plan execution failed"))
                else:
                    # Check goal
                    goal_reached = True
                    for pkg in goal_state.packages:
                        if goal_state.at.get(pkg) != test_state.at.get(pkg):
                            goal_reached = False
                            break

                    if goal_reached:
                        status = "✓ VALID"
                        problems.append({
                            'id': i,
                            'archetype': archetype.value,
                            'plan_length': len(plan),
                            'initial': initial_state,
                            'goal': goal_state,
                            'plan': plan
                        })
                    else:
                        status = "✗ GOAL"
                        failed.append((i, "Plan does not reach goal"))

            print(f"  Problem {i:2d}: {status:15s} | {archetype.value:20s} | length={len(plan):2d}")

        except Exception as e:
            error_msg = str(e)[:50]
            print(f"  Problem {i:2d}: ✗ ERROR          | {error_msg}")
            failed.append((i, str(e)))

    print(f"\n[SUMMARY]")
    print(f"  Generated: {len(problems)}/{num_problems} valid problems")
    print(f"  Failed: {len(failed)}/{num_problems}")

    if failed:
        print(f"\n[FAILURES]")
        for prob_id, reason in failed[:5]:
            print(f"  Problem {prob_id}: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if problems:
        print(f"\n[STATISTICS]")
        plan_lengths = [p['plan_length'] for p in problems]
        print(f"  Min plan length: {min(plan_lengths)}")
        print(f"  Max plan length: {max(plan_lengths)}")
        print(f"  Avg plan length: {sum(plan_lengths) / len(plan_lengths):.1f}")

        archetypes = {}
        for p in problems:
            arch = p['archetype']
            archetypes[arch] = archetypes.get(arch, 0) + 1
        print(f"  Archetype distribution:")
        for arch, count in sorted(archetypes.items()):
            print(f"    {arch}: {count}")

    if problems:
        print(f"\n[WRITING FIRST 3 VALID PROBLEMS TO PDDL]")
        pddl_writer = PDDLWriter()
        pddl_writer.write_domain("batch_domain.pddl")
        print(f"  ✓ Written: batch_domain.pddl")

        for problem in problems[:3]:
            filename = f"batch_problem_{problem['id']}.pddl"
            pddl_writer.write_problem(
                filename,
                f"batch-problem-{problem['id']}",
                problem['initial'],
                problem['goal']
            )
            print(f"  ✓ Written: {filename} (plan length: {problem['plan_length']})")


def example_7_inter_city_problem():
    """
    Example 7: Generate a problem that requires inter-city transport.

    Shows a more complex scenario with multiple cities and airplanes.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Inter-City Problem Generation")
    print("=" * 80)

    print(f"\n[GENERATING INTER-CITY PROBLEM]")
    print(f"  Difficulty: medium")
    print(f"  Target plan length: 12 (±2)")

    generator = BackwardProblemGenerator(random_seed=100)

    try:
        initial_state, goal_state, plan, archetype = generator.generate_problem(
            difficulty='medium',
            target_plan_length=12,
            tolerance=2
        )

        print(f"\n  ✓ Problem generated")
        print(f"  Archetype: {archetype.value}")
        print(f"  Plan length: {len(plan)}")

        print(f"\n[WORLD STRUCTURE]")
        print(f"  Cities: {sorted(initial_state.cities)}")
        print(f"  Total locations: {len(initial_state.locations)}")
        print(f"  Airports: {sorted(initial_state.airports)}")

        # Analyze if this requires inter-city transport
        requires_inter_city = False
        for pkg in goal_state.packages:
            initial_loc = initial_state.at.get(pkg)
            goal_loc = goal_state.at.get(pkg)
            if initial_loc and goal_loc:
                initial_city = initial_state.in_city.get(initial_loc)
                goal_city = initial_state.in_city.get(goal_loc)
                if initial_city != goal_city:
                    requires_inter_city = True
                    print(f"  {pkg}: {initial_city} → {goal_city} (INTER-CITY)")
                else:
                    print(f"  {pkg}: {initial_city} → {goal_city} (intra-city)")

        print(f"\n[PLAN ANALYSIS]")
        fly_count = sum(1 for a in plan if a.action_type == ActionType.FLY_AIRPLANE)
        drive_count = sum(1 for a in plan if a.action_type == ActionType.DRIVE_TRUCK)
        load_count = sum(1 for a in plan if 'LOAD' in a.action_type.value.upper())
        unload_count = sum(1 for a in plan if 'UNLOAD' in a.action_type.value.upper())

        print(f"  FLY-AIRPLANE: {fly_count}")
        print(f"  DRIVE-TRUCK: {drive_count}")
        print(f"  LOAD actions: {load_count}")
        print(f"  UNLOAD actions: {unload_count}")

        if requires_inter_city and fly_count > 0:
            print(f"\n  ✓ This is a genuine multi-modal problem!")
        elif fly_count == 0 and not requires_inter_city:
            print(f"\n  ✓ This is a valid intra-city delivery problem")
        else:
            print(f"\n  Note: Check if this problem structure matches expectations")

        # Execute and verify
        print(f"\n[VERIFICATION]")
        test_state = initial_state.copy()
        valid = True
        for i, action in enumerate(plan):
            test_state = ActionExecutor.execute_forward(test_state, action)
            if test_state is None:
                print(f"  ✗ Action {i} failed")
                valid = False
                break

        if valid:
            goal_reached = all(
                goal_state.at.get(pkg) == test_state.at.get(pkg)
                for pkg in goal_state.packages
            )
            if goal_reached:
                print(f"  ✓ Plan is valid and reaches goal!")
            else:
                print(f"  ✗ Plan executed but goal not reached")
        else:
            print(f"  ✗ Plan execution failed")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# At the end of example_usage.py, add:

def example_8_robust_batch_generation():
    """
    Example 8: Robust batch generation with fixes.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Robust Batch Generation (Fixed)")
    print("=" * 80)

    num_problems = 15
    valid_problems = []
    failed = []

    print(f"\n[GENERATING {num_problems} ROBUST SMALL PROBLEMS]")
    print()

    for i in range(num_problems):
        try:
            generator = BackwardProblemGenerator(random_seed=i * 42)
            initial_state, goal_state, plan, archetype = generator.generate_problem(
                difficulty='small',
                target_plan_length=7,
                tolerance=2
            )

            # Validate
            if initial_state == goal_state:
                print(f"  Problem {i:2d}: ✗ TRIVIAL")
                failed.append((i, "Trivial"))
                continue

            if not plan:
                print(f"  Problem {i:2d}: ✗ EMPTY_PLAN")
                failed.append((i, "Empty plan"))
                continue

            # Verify plan execution
            test_state = initial_state.copy()
            plan_valid = True
            for action in plan:
                test_state = ActionExecutor.execute_forward(test_state, action)
                if test_state is None:
                    plan_valid = False
                    break

            if not plan_valid:
                print(f"  Problem {i:2d}: ✗ INVALID_PLAN")
                failed.append((i, "Invalid plan"))
                continue

            # Check goal
            goal_reached = all(
                goal_state.at.get(pkg) == test_state.at.get(pkg)
                for pkg in goal_state.packages
            )

            if not goal_reached:
                print(f"  Problem {i:2d}: ✗ GOAL_NOT_REACHED")
                failed.append((i, "Goal not reached"))
                continue

            print(f"  Problem {i:2d}: ✓ VALID | {archetype.value:20s} | length={len(plan):2d}")
            valid_problems.append(i)

        except Exception as e:
            print(f"  Problem {i:2d}: ✗ ERROR | {str(e)[:40]}")
            failed.append((i, str(e)[:40]))

    print(f"\n[SUMMARY]")
    print(f"  Generated: {len(valid_problems)}/{num_problems} valid problems")
    print(f"  Success rate: {len(valid_problems) / num_problems * 100:.1f}%")

    if failed:
        print(f"\n[FAILURE ANALYSIS]")
        failure_types = {}
        for _, reason in failed:
            failure_types[reason] = failure_types.get(reason, 0) + 1
        for reason, count in failure_types.items():
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "LOGISTICS PROBLEM GENERATION FRAMEWORK - COMPREHENSIVE EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    example_1_manual_state_creation()
    example_2_forward_actions()
    example_3_build_world()
    example_4_backward_generation_single()
    example_5_pddl_generation()
    example_6_batch_generation_validated()
    example_7_inter_city_problem()
    example_8_robust_batch_generation()

    print("\n" + "=" * 80)
    print("✓ ALL EXAMPLES COMPLETED!")
    print("=" * 80 + "\n")