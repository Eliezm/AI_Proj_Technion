"""
Main orchestration and CLI for the problem generation framework.

Requirement #9: Scalable generation of arbitrary numbers of problems.
Requirement #13: Simple, modular architecture.
Requirement #18: Selective manual validation.
Requirement #19: Executable Python codebase.
"""

import logging
logger = logging.getLogger(__name__)

import argparse
import os
import sys
import random
from typing import List, Dict, Optional
from pathlib import Path
import json

from config import (
    DIFFICULTY_TIERS,
    BASELINE_PLANNER_CONFIG,
    ensure_output_dirs,
    OUTPUT_DIR,
    DOMAIN_DIR,
    PROBLEMS_DIR,
    METADATA_DIR
)
from state import create_empty_state
from backward_generator import BackwardProblemGenerator
from pddl_writer import PDDLWriter
from baseline_planner import FastDownwardRunner
from metadata_store import MetadataStore, ProblemMetadata
from validator import PDDLValidator
from goal_archetypes import GoalArchetype


class ProblemGenerationFramework:
    """Main framework for problem generation and validation."""

    def __init__(self, random_seed: int = None):
        """Initialize the framework."""
        ensure_output_dirs()
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self.generator = BackwardProblemGenerator(random_seed=random_seed)
        self.pddl_writer = PDDLWriter()
        self.fd_runner = FastDownwardRunner()
        self.validator = PDDLValidator()
        self.metadata_store = MetadataStore(METADATA_DIR)

    def generate_batch(
            self,
            num_problems: int,
            difficulty: str,
            num_blocks: int = 4,
            domain_name: str = "blocksworld",
            skip_planner: bool = False,
            timeout: Optional[int] = None  # <-- ADD THIS
    ) -> List[str]:
        """Generate a batch of problems (Requirement #9)."""

        if difficulty not in DIFFICULTY_TIERS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        tier = DIFFICULTY_TIERS[difficulty]

        # FIXED: Ensure timeout is set
        if timeout is None:
            timeout = BASELINE_PLANNER_CONFIG['timeout']


        problem_names = []

        print(f"\n{'=' * 70}")
        print(f"Generating {num_problems} {difficulty} problems")
        print(f"Target plan length: {tier.target_length}")
        print(f"Number of blocks: {num_blocks}")
        print(f"Planner timeout: {timeout} seconds per problem")  # ADDED
        if skip_planner:
            print("(Baseline planner disabled)")
        print(f"{'=' * 70}\n")

        for i in range(num_problems):
            try:
                # Generate problem
                initial_state, goal_state, plan, archetype = self.generator.generate_problem(
                    num_blocks=num_blocks,
                    target_plan_length=tier.target_length,
                    tolerance=1
                )

                problem_name = f"{domain_name}-{difficulty}-{i}"
                problem_names.append(problem_name)

                # Write PDDL files
                domain_file = os.path.join(DOMAIN_DIR, f"{domain_name}.pddl")
                problem_file = os.path.join(PROBLEMS_DIR, f"{problem_name}.pddl")

                if i == 0:
                    self.pddl_writer.write_domain(domain_file)

                self.pddl_writer.write_problem(
                    problem_file,
                    problem_name,
                    initial_state,
                    goal_state
                )

                # Validate PDDL syntax (Requirement #8)
                is_valid, error = self.validator.validate_problem(domain_file, problem_file)
                if not is_valid:
                    print(f"  Problem {i}: PDDL validation failed: {error}")
                    continue

                print(f"  Problem {i}: ", end='', flush=True)

                # Run baseline planner if available (Requirement #12)
                if skip_planner:
                    print("✓ Generated (planner skipped)")
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
                            timeout=timeout  # <-- CHANGE THIS LINE
                        )

                        if planner_result['success']:
                            print(
                                f"✓ Time: {planner_result['time']:.2f}s, "
                                f"Cost: {planner_result['plan_cost']}, "
                                f"Nodes: {planner_result['nodes_expanded']}"
                            )
                        else:
                            print(f"✗ Failed: {planner_result['error']}")
                    except Exception as e:
                        logger.error(f"Planner execution error: {e}")
                        planner_result = {
                            'success': False,
                            'time': 0,
                            'plan_cost': None,
                            'nodes_expanded': None,
                            'error': str(e)[:200]
                        }
                        print(f"✗ Planner error: {str(e)[:50]}")

                # Store metadata (Requirement #6, #12)
                metadata = ProblemMetadata(
                    problem_name=problem_name,
                    domain=domain_name,
                    difficulty=difficulty,
                    num_blocks=num_blocks,
                    goal_archetype=archetype.value,
                    plan_length=len(plan),
                    optimal_plan_cost=len(plan),
                    planner_time=planner_result.get('time', 0),
                    planner_success=planner_result.get('success', False),
                    nodes_expanded=planner_result.get('nodes_expanded', 0),
                    plan_cost=planner_result.get('plan_cost', len(plan)),
                    domain_file=domain_file,
                    problem_file=problem_file
                )
                self.metadata_store.save_metadata(metadata)

            except Exception as e:
                print(f"  Problem {i}: Error: {e}")
                logger.error(f"Problem generation error: {e}", exc_info=True)
                continue

        print(f"\nGenerated {len(problem_names)}/{num_problems} problems successfully\n")
        return problem_names

    def validate_subset(self, difficulty: str, count: int = 5) -> None:
        """
        Validate a subset of problems (Requirement #18).

        Manually run and inspect specific problems.
        """
        problems = self.metadata_store.get_by_difficulty(difficulty)
        if not problems:
            print(f"No problems found for difficulty: {difficulty}")
            return

        selected = problems[:count]
        print(f"\nValidating {len(selected)} {difficulty} problems:\n")

        for meta in selected:
            print(f"Problem: {meta.problem_name}")
            print(f"  Generated plan length: {meta.plan_length}")
            print(f"  Baseline planner time: {meta.planner_time:.2f}s")
            print(f"  Plan cost: {meta.plan_cost}")
            print(f"  Nodes expanded: {meta.nodes_expanded}")
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
                print(f"  Count: {s['count']}")
                print(f"  Successful: {s['successful']}")
                if s['avg_time']:
                    print(f"  Avg time: {s['avg_time']:.2f}s")
                    print(f"  Min time: {s['min_time']:.2f}s")
                    print(f"  Max time: {s['max_time']:.2f}s")
                print()

    def calibrate_difficulty(self) -> None:
        """
        Recommend difficulty tier adjustments (Requirement #17).

        Analyze baseline planner results and suggest parameter changes.
        """
        stats = self.metadata_store.get_summary_stats()

        print(f"\n{'=' * 70}")
        print("DIFFICULTY CALIBRATION RECOMMENDATIONS")
        print(f"{'=' * 70}\n")

        target_times = {'small': 1, 'medium': 180, 'large': 420}  # seconds

        for difficulty in ['small', 'medium', 'large']:
            if difficulty not in stats:
                print(f"{difficulty.upper()}: No data")
                continue

            s = stats[difficulty]
            avg_time = s['avg_time'] or 0
            target_time = target_times[difficulty]

            tier = DIFFICULTY_TIERS[difficulty]
            print(f"{difficulty.upper()}:")
            print(f"  Current target plan length: {tier.target_length}")
            print(f"  Average solve time: {avg_time:.2f}s (target: {target_time}s)")

            if avg_time > target_time * 1.5:
                suggested = max(tier.target_length - 2, 3)
                print(f"  → Problems too hard; reduce plan length to ~{suggested}")
            elif avg_time < target_time * 0.5:
                suggested = tier.target_length + 3
                print(f"  → Problems too easy; increase plan length to ~{suggested}")
            else:
                print(f"  → OK, keep as is")
            print()





def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Blocksworld PDDL Problem Generation Framework"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate problems')
    gen_parser.add_argument(
        '--num-problems',
        type=int,
        default=10,
        help='Number of problems to generate'
    )
    gen_parser.add_argument(
        '--difficulty',
        choices=['small', 'medium', 'large'],
        required=True,
        help='Difficulty tier'
    )
    gen_parser.add_argument(
        '--num-blocks',
        type=int,
        default=4,
        help='Number of blocks per problem'
    )
    gen_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    gen_parser.add_argument(
        '--skip-planner',
        action='store_true',
        help='Skip baseline planner (useful if Fast Downward not installed)'
    )

    gen_parser.add_argument(
        '--timeout',
        type=int,
        default=BASELINE_PLANNER_CONFIG['timeout'],
        help='Max planner time in seconds per problem (default: 600)'
    )

    # Validate subset command
    val_parser = subparsers.add_parser('validate-subset', help='Validate a subset of problems')
    val_parser.add_argument(
        '--difficulty',
        choices=['small', 'medium', 'large'],
        required=True,
        help='Difficulty tier'
    )
    val_parser.add_argument(
        '--count',
        type=int,
        default=5,
        help='Number of problems to validate'
    )

    # NEW: Generate by time command
    time_parser = subparsers.add_parser(
        'generate-by-time',
        help='Generate problems targeting specific solving time'
    )
    time_parser.add_argument(
        '--difficulty',
        choices=['small', 'medium', 'large'],
        required=True,
        help='Difficulty tier'
    )
    time_parser.add_argument(
        '--count',
        type=int,
        default=20,
        help='Number of problems to generate (default: 20)'
    )
    time_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    # Summary command
    subparsers.add_parser('summary', help='Print summary statistics')

    # Calibrate command
    subparsers.add_parser('calibrate', help='Calibrate difficulty tiers')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize framework
    framework = ProblemGenerationFramework(random_seed=args.seed)

    # Execute command
    if args.command == 'generate':
        print(f"Timeout per problem: {args.timeout}s")  # ✓ Show what we're using

        framework.generate_batch(
            num_problems=args.num_problems,
            difficulty=args.difficulty,
            num_blocks=args.num_blocks,
            skip_planner=args.skip_planner,  # NEW
            timeout=args.timeout  # <-- ADD THIS LINE
        )
        framework.print_summary()

    # NEW: Time-based generation command
    elif args.command == 'generate-by-time':
        from time_based_generator import TimeBasedProblemGenerator

        generator = TimeBasedProblemGenerator(
            difficulty=args.difficulty,
            domain_dir=DOMAIN_DIR,
            problems_dir=PROBLEMS_DIR,
            metadata_dir=METADATA_DIR,
            random_seed=args.seed
        )

        # Calibration phase
        if generator.calibrate():
            # Generation phase
            generator.generate_batch(target_count=args.count)

            # Summary
            framework.print_summary()
        else:
            print("✗ Calibration failed. Adjust TIME_DIFFICULTY_TIERS in config.py")
            sys.exit(1)

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