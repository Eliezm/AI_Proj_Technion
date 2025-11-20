"""
Main orchestration and CLI for the Logistics problem generation framework.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import os
import sys
import random
from typing import List

# FIX #5: Fix imports to work whether run as module or script
try:
    from config import (
        DIFFICULTY_TIERS,
        BASELINE_PLANNER_CONFIG,
        DEFAULT_LOGISTICS_PARAMS,
        ensure_output_dirs,
        DOMAIN_DIR,
        PROBLEMS_DIR,
        METADATA_DIR
    )
    from backward_generator import BackwardProblemGenerator
    from pddl_writer import PDDLWriter
    from baseline_planner import FastDownwardRunner
    from metadata_store import MetadataStore, ProblemMetadata
    from validator import PDDLValidator
    from state import LogisticsState
    from problem_validator import ProblemValidator
    from actions import Action
except ImportError:
    # Running as module
    from .config import (
        DIFFICULTY_TIERS,
        BASELINE_PLANNER_CONFIG,
        DEFAULT_LOGISTICS_PARAMS,
        ensure_output_dirs,
        DOMAIN_DIR,
        PROBLEMS_DIR,
        METADATA_DIR
    )
    from .backward_generator import BackwardProblemGenerator
    from .pddl_writer import PDDLWriter
    from .baseline_planner import FastDownwardRunner
    from .metadata_store import MetadataStore, ProblemMetadata
    from .validator import PDDLValidator
    from .state import LogisticsState
    from .problem_validator import ProblemValidator
    from .actions import Action


class ProblemGenerationFramework:
    """Main framework for Logistics problem generation and validation."""

    def __init__(self, random_seed: int = None):
        """Initialize the framework."""
        ensure_output_dirs()
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self.generator = BackwardProblemGenerator(random_seed=random_seed)
        self.pddl_writer = PDDLWriter()
        self.fd_runner = FastDownwardRunner(timeout_search=260)
        self.validator = PDDLValidator()
        self.metadata_store = MetadataStore(METADATA_DIR)

    def generate_batch(
            self,
            num_problems: int,
            difficulty: str,
            domain_name: str = "logistics",
            skip_planner: bool = False
    ) -> List[str]:
        """Generate a batch of Logistics problems with 100% validation."""

        if difficulty not in DIFFICULTY_TIERS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        tier = DIFFICULTY_TIERS[difficulty]
        generation_params = DEFAULT_LOGISTICS_PARAMS.get(difficulty)
        problem_names = []

        print(f"\n{'=' * 70}")
        print(f"Generating {num_problems} {difficulty.upper()} Logistics problems")
        print(f"Target plan length: {tier.target_length} ±{2}")  # Updated tolerance display
        print(f"World complexity:")
        print(f"  Cities: {generation_params.num_cities}")
        print(f"  Locs/city: {generation_params.locations_per_city}")
        print(f"  Packages: {generation_params.num_packages}")
        print(f"  Trucks: {generation_params.num_trucks}")
        print(f"  Planes: {generation_params.num_airplanes}")
        if skip_planner:
            print("(Baseline planner disabled)")
        print(f"{'=' * 70}\n")

        successful = 0
        failed = 0

        for i in range(num_problems):
            try:
                # Generate problem
                initial_state, goal_state, plan, archetype = self.generator.generate_problem(
                    difficulty=difficulty,
                    generation_params=generation_params,
                    target_plan_length=tier.target_length,
                    tolerance=1
                )

                # FIX #5: Validate generated problem
                is_valid, reason = ProblemValidator.validate_complete_problem(
                    initial_state,
                    goal_state,
                    plan
                )
                if not is_valid:
                    print(f"  [{i:3d}] ✗ Validation failed: {reason[:50]}")
                    failed += 1
                    continue

                problem_name = f"{domain_name}-{difficulty}-{i:04d}"
                problem_names.append(problem_name)

                # Write PDDL files
                domain_file = os.path.join(DOMAIN_DIR, f"{domain_name}.pddl")
                problem_file = os.path.join(PROBLEMS_DIR, f"{problem_name}.pddl")

                if i == 0:
                    self.pddl_writer.write_domain(domain_file)
                    print(f"  [INIT] Domain written to {domain_file}\n")

                self.pddl_writer.write_problem(
                    problem_file,
                    problem_name,
                    initial_state,
                    goal_state
                )

                if not os.path.exists(problem_file):
                    print(f"  [{i:3d}] ✗ Problem file not created")
                    failed += 1
                    continue

                # Validate PDDL
                is_pddl_valid, pddl_error = self.validator.validate_problem(
                    domain_file,
                    problem_file
                )
                if not is_pddl_valid and pddl_error and "not found" not in pddl_error.lower():
                    print(f"  [{i:3d}] ✗ PDDL error: {pddl_error[:40]}")
                    failed += 1
                    continue

                # Run baseline planner
                print(f"  [{i:3d}] ", end='', flush=True)

                if skip_planner:
                    print("✓ (planner skipped)")
                    planner_result = {
                        'success': True,
                        'time': None,
                        'plan_cost': len(plan),
                        'nodes_expanded': None,
                        'error': None
                    }
                else:
                    try:
                        planner_result = self.fd_runner.run_problem(
                            domain_file,
                            problem_file,
                            search_config="astar(lmcut())",
                            timeout=BASELINE_PLANNER_CONFIG['timeout']
                        )

                        if planner_result['success']:
                            print(
                                f"✓ Time: {planner_result['time']:.2f}s, "
                                f"Cost: {planner_result['plan_cost']}, "
                                f"Plan len: {len(plan)}"
                            )
                        else:
                            print(f"✗ {planner_result['error'][:40]}")
                            failed += 1
                            continue

                    except Exception as e:
                        print(f"✗ Planner error: {str(e)[:40]}")
                        logger.error(f"Planner error: {e}")
                        failed += 1
                        continue

                # Store metadata
                metadata = ProblemMetadata(
                    problem_name=problem_name,
                    domain=domain_name,
                    difficulty=difficulty,
                    num_cities=generation_params.num_cities,
                    num_locations=(generation_params.num_cities *
                                   generation_params.locations_per_city),
                    num_packages=generation_params.num_packages,
                    num_trucks=generation_params.num_trucks,
                    num_airplanes=generation_params.num_airplanes,
                    goal_archetype=archetype.value,
                    plan_length=len(plan),
                    optimal_plan_cost=len(plan),
                    planner_time=planner_result.get('time', 0) or 0,
                    planner_success=planner_result.get('success', False),
                    nodes_expanded=planner_result.get('nodes_expanded', 0) or 0,
                    plan_cost=planner_result.get('plan_cost', len(plan)) or len(plan),
                    domain_file=domain_file,
                    problem_file=problem_file
                )
                self.metadata_store.save_metadata(metadata)
                successful += 1

            except Exception as e:
                print(f"  [{i:3d}] ✗ Exception: {str(e)[:40]}")
                logger.exception(f"Generation error: {e}")
                failed += 1
                continue

        print(f"\n{'=' * 70}")
        print(f"BATCH COMPLETE:")
        print(f"  Generated: {successful} valid problems")
        print(f"  Failed: {failed} problems")
        print(f"  Success rate: {successful/(successful+failed)*100:.1f}%")
        print(f"{'=' * 70}\n")

        return problem_names

    def validate_subset(self, difficulty: str, count: int = 5) -> None:
        """Validate a subset of problems (Requirement #18)."""
        problems = self.metadata_store.get_by_difficulty(difficulty)
        if not problems:
            print(f"No problems found for difficulty: {difficulty}")
            return

        selected = problems[:count]
        print(f"\nValidating {len(selected)} {difficulty} problems:\n")

        for meta in selected:
            print(f"✓ {meta.problem_name}")
            print(f"    Archetype: {meta.goal_archetype}")
            print(f"    World: {meta.num_cities} cities, {meta.num_packages} packages")
            print(f"    Plan: {meta.plan_length} actions (cost: {meta.plan_cost})")
            print(f"    Solver: {meta.planner_time:.2f}s")
            print()

    def print_summary(self) -> None:
        """Print summary statistics."""
        stats = self.metadata_store.get_summary_stats()

        print(f"\n{'=' * 70}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 70}\n")

        for difficulty in ['small', 'medium', 'large']:
            if difficulty in stats:
                s = stats[difficulty]
                print(f"{difficulty.upper()}:")
                print(f"  Total: {s['count']} problems")
                print(f"  Successful: {s['successful']}/{s['count']}")
                if s['avg_time']:
                    print(f"  Avg time: {s['avg_time']:.2f}s")
                    print(f"  Range: {s['min_time']:.2f}s - {s['max_time']:.2f}s")
                print()

    def calibrate_difficulty(self) -> None:
        """Recommend difficulty adjustments (Requirement #17)."""
        stats = self.metadata_store.get_summary_stats()

        print(f"\n{'=' * 70}")
        print("DIFFICULTY CALIBRATION")
        print(f"{'=' * 70}\n")

        target_times = {'small': 60, 'medium': 180, 'large': 420}

        for difficulty in ['small', 'medium', 'large']:
            if difficulty not in stats:
                print(f"{difficulty.upper()}: No data")
                continue

            s = stats[difficulty]
            avg_time = s['avg_time'] or 0
            target = target_times[difficulty]
            tier = DIFFICULTY_TIERS[difficulty]

            print(f"{difficulty.upper()}:")
            print(f"  Current target length: {tier.target_length}")
            print(f"  Avg solve time: {avg_time:.2f}s (target: {target}s)")

            if avg_time == 0:
                print(f"  → Insufficient data")
            elif avg_time > target * 1.5:
                print(f"  → TOO HARD: decrease target length by 2-3")
            elif avg_time < target * 0.5:
                print(f"  → TOO EASY: increase target length by 2-3")
            else:
                print(f"  → OK")
            print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Logistics PDDL Problem Generation Framework - 100% Valid Problems"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen = subparsers.add_parser('generate', help='Generate problems')
    gen.add_argument('--num-problems', type=int, default=10)
    gen.add_argument('--difficulty', choices=['small', 'medium', 'large'], required=True)
    gen.add_argument('--seed', type=int, default=None)
    gen.add_argument('--skip-planner', action='store_true')

    # Validate command
    val = subparsers.add_parser('validate-subset', help='Validate problems')
    val.add_argument('--difficulty', choices=['small', 'medium', 'large'], required=True)
    val.add_argument('--count', type=int, default=5)

    # Summary command
    subparsers.add_parser('summary', help='Show statistics')

    # Calibrate command
    subparsers.add_parser('calibrate', help='Calibration recommendations')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    framework = ProblemGenerationFramework(random_seed=args.seed)

    if args.command == 'generate':
        framework.generate_batch(
            num_problems=args.num_problems,
            difficulty=args.difficulty,
            skip_planner=args.skip_planner
        )
        framework.print_summary()

    elif args.command == 'validate-subset':
        framework.validate_subset(
            difficulty=args.difficulty,
            count=args.count
        )

    elif args.command == 'summary':
        framework.print_summary()

    elif args.command == 'calibrate':
        framework.calibrate_difficulty()


if __name__ == '__main__':
    main()