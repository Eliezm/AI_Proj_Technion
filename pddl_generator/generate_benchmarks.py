# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# COMPLETE BENCHMARK GENERATION PIPELINE
# ======================================
#
# Usage:
#     python generate_benchmarks.py --domain blocksworld --output benchmarks
#     python generate_benchmarks.py --all --output benchmarks
#     python generate_benchmarks.py --validate --benchmark-dir benchmarks
# """
#
# import argparse
# import logging
# from pathlib import Path
#
# from problem_generator import ComprehensiveProblemGenerator
# from problem_validator import validate_all_benchmarks
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
#
# # FILE: generate_benchmarks.py
# # REPLACE the main() function with this:
#
# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Generate and validate PDDL benchmarks"
# #     )
# #
# #     parser.add_argument(
# #         "--domain",
# #         choices=["blocksworld", "logistics", "parking"],
# #         help="Generate for specific domain"
# #     )
# #
# #     parser.add_argument(
# #         "--all",
# #         action="store_true",
# #         help="Generate for all domains (default if no domain specified)"
# #     )
# #
# #     parser.add_argument(
# #         "--validate",
# #         action="store_true",
# #         help="Validate generated benchmarks"
# #     )
# #
# #     parser.add_argument(
# #         "--output",
# #         default="benchmarks",
# #         help="Output directory"
# #     )
# #
# #     parser.add_argument(
# #         "--benchmark-dir",
# #         default="benchmarks",
# #         help="Benchmark directory to validate"
# #     )
# #
# #     parser.add_argument(
# #         "--problems-per-size",
# #         type=int,
# #         default=15,
# #         help="Number of problems per size category"
# #     )
# #
# #     parser.add_argument(
# #         "--seed",
# #         type=int,
# #         default=42,
# #         help="Random seed for reproducibility"
# #     )
# #
# #     # FILE: generate_benchmarks.py
# #     # INSIDE main()
# #     # --- ADD THESE TWO ARGUMENTS ---
# #     parser.add_argument(
# #         "--validate-domain",
# #         type=str,
# #         default=None,
# #         help="Only validate this specific domain (e.g., blocksworld)"
# #     )
# #     parser.add_argument(
# #         "--validate-size",
# #         type=str,
# #         default=None,
# #         help="Only validate this specific size (e.g., small)"
# #     )
# #     # --- END OF NEW ARGUMENTS ---
# #
# #
# #     args = parser.parse_args()
# #
# #     if args.validate:
# #         logger.info("\n" + "=" * 80)
# #         logger.info("VALIDATING BENCHMARKS")
# #         logger.info(f"  Domain filter: {args.validate_domain or 'All'}")
# #         logger.info(f"  Size filter:   {args.validate_size or 'All'}")
# #         logger.info("=" * 80 + "\n")
# #
# #         # Pass the new filter arguments to the validation function
# #         validate_all_benchmarks(
# #             args.benchmark_dir,
# #             domain_filter=args.validate_domain,
# #             size_filter=args.validate_size
# #         )
# #
# #     else:
# #         logger.info("\n" + "=" * 80)
# #         logger.info("GENERATING BENCHMARKS")
# #         logger.info("=" * 80 + "\n")
# #
# #         gen = ComprehensiveProblemGenerator(
# #             output_dir=args.output,
# #             seed=args.seed
# #         )
# #
# #         domains_to_generate = []
# #         if args.all:
# #             domains_to_generate = ["blocksworld", "logistics", "parking"]
# #         elif args.domain:
# #             domains_to_generate = [args.domain]
# #         else:
# #             # Default behavior: run all if no specific domain is given
# #             logger.info("No domain specified and --all not set. Defaulting to all domains.")
# #             domains_to_generate = ["blocksworld", "logistics", "parking"]
# #
# #         logger.info(f"Target domains: {domains_to_generate}")
# #
# #         # ✅ FIX: Loop through the selected domains
# #         gen.generate_all(
# #             problems_per_size=args.problems_per_size,
# #             domains_to_generate=domains_to_generate  # Pass the list to the generator
# #         )
# #
# #         logger.info("\n✅ Generation complete!")
# #         logger.info(f"Benchmarks saved to: {args.output}")
# #         logger.info("\nNext step: Validate benchmarks")
# #         logger.info(f"  python generate_benchmarks.py --validate --benchmark-dir {args.output}")
#
#
# # FILE: generate_benchmarks.py
# # REPLACE your entire main() function with this
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate and validate PDDL benchmarks"
#     )
#
#     parser.add_argument(
#         "--domain",
#         choices=["blocksworld", "logistics", "parking"],
#         help="Generate for specific domain"
#     )
#
#     parser.add_argument(
#         "--all",
#         action="store_true",
#         help="Generate for all domains (default if no domain specified)"
#     )
#
#     parser.add_argument(
#         "--validate",
#         action="store_true",
#         help="Validate generated benchmarks"
#     )
#
#     parser.add_argument(
#         "--output",
#         default="benchmarks",
#         help="Output directory"
#     )
#
#     parser.add_argument(
#         "--benchmark-dir",
#         default="benchmarks",
#         help="Benchmark directory to validate"
#     )
#
#     parser.add_argument(
#         "--problems-per-size",
#         type=int,
#         default=15,
#         help="Number of problems per size category"
#     )
#
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for reproducibility"
#     )
#
#     # --- ADD THESE TWO ARGUMENTS ---
#     parser.add_argument(
#         "--validate-domain",
#         type=str,
#         default=None,
#         help="Only validate this specific domain (e.g., blocksworld)"
#     )
#     parser.add_argument(
#         "--validate-size",
#         type=str,
#         default=None,
#         help="Only validate this specific size (e.g., small)"
#     )
#     # --- END OF NEW ARGUMENTS ---
#
#     args = parser.parse_args()
#
#     if args.validate:
#         logger.info("\n" + "=" * 80)
#         logger.info("VALIDATING BENCHMARKS")
#         logger.info(f"  Domain filter: {args.validate_domain or 'All'}")
#         logger.info(f"  Size filter:   {args.validate_size or 'All'}")
#         logger.info("=" * 80 + "\n")
#
#         # Pass the new filter arguments to the validation function
#         validate_all_benchmarks(
#             args.benchmark_dir,
#             domain_filter=args.validate_domain,
#             size_filter=args.validate_size
#         )
#
#     else:
#         logger.info("\n" + "=" * 80)
#         logger.info("GENERATING BENCHMARKS")
#         logger.info("=" * 80 + "\n")
#
#         gen = ComprehensiveProblemGenerator(
#             output_dir=args.output,
#             seed=args.seed
#         )
#
#         domains_to_generate = []
#         if args.all:
#             domains_to_generate = ["blocksworld", "logistics", "parking"]
#         elif args.domain:
#             domains_to_generate = [args.domain]
#         else:
#             # Default behavior: run all if no specific domain is given
#             logger.info("No domain specified and --all not set. Defaulting to all domains.")
#             domains_to_generate = ["blocksworld", "logistics", "parking"]
#
#         logger.info(f"Target domains: {domains_to_generate}")
#
#         gen.generate_all(
#             problems_per_size=args.problems_per_size,
#             domains_to_generate=domains_to_generate
#         )
#
#         logger.info("\n✅ Generation complete!")
#         logger.info(f"Benchmarks saved to: {args.output}")
#         logger.info("\nNext step: Validate benchmarks")
#         logger.info(f"  python generate_benchmarks.py --validate --benchmark-dir {args.output}")
#
# if __name__ == "__main__":
#     main()

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED BENCHMARK GENERATION PIPELINE
======================================

Usage:
    python generate_benchmarks.py --domain blocksworld --output benchmarks
    python generate_benchmarks.py --all --output benchmarks
    python generate_benchmarks.py --validate --benchmark-dir benchmarks
    python generate_benchmarks.py --validate-size blocksworld small
"""

import argparse
import logging
from pathlib import Path

# Import enhanced generators
from problem_generator import EnhancedProblemGenerator
from problem_validator import EnhancedProblemValidator, validate_all_benchmarks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate PDDL benchmarks with difficulty targeting"
    )

    parser.add_argument(
        "--domain",
        choices=["blocksworld", "logistics", "parking"],
        help="Generate for specific domain"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all domains (default if no domain specified)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all generated benchmarks"
    )

    parser.add_argument(
        "--validate-size",
        nargs=2,
        metavar=("DOMAIN", "SIZE"),
        help="Validate specific domain/size combination (e.g., blocksworld small)"
    )

    parser.add_argument(
        "--output",
        default="benchmarks",
        help="Output directory for generation"
    )

    parser.add_argument(
        "--benchmark-dir",
        default="benchmarks",
        help="Benchmark directory to validate"
    )

    parser.add_argument(
        "--problems-per-size",
        type=int,
        default=15,
        help="Number of problems per size category (default: 15)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # ========================================================================
    # GENERATION MODE
    # ========================================================================

    if args.validate_size:
        # Validate specific domain/size
        domain, size = args.validate_size
        logger.info(f"\n{'=' * 80}")
        logger.info(f"VALIDATING: {domain.upper()} / {size.upper()}")
        logger.info(f"{'=' * 80}\n")

        validator = EnhancedProblemValidator()
        benchmark_path = Path(args.benchmark_dir) / domain / size

        if not benchmark_path.exists():
            logger.error(f"Directory not found: {benchmark_path}")
            return 1

        domain_file = benchmark_path / "domain.pddl"
        if not domain_file.exists():
            logger.error(f"Domain file not found: {domain_file}")
            return 1

        problem_files = sorted(benchmark_path.glob("problem_*.pddl"))

        valid_count = 0
        for problem_file in problem_files:
            metadata_file = problem_file.with_suffix('.json')
            is_valid, difficulty, error = validator.validate_and_classify(
                str(domain_file),
                str(problem_file),
                str(metadata_file)
            )

            if is_valid:
                valid_count += 1

        logger.info(f"\n✅ Validation complete: {valid_count}/{len(problem_files)} problems valid")
        return 0

    elif args.validate:
        # Validate all benchmarks
        logger.info(f"\n{'=' * 80}")
        logger.info("VALIDATING ALL BENCHMARKS")
        logger.info(f"{'=' * 80}\n")
        validate_all_benchmarks(args.benchmark_dir)
        return 0

    else:
        # Generation mode
        logger.info(f"\n{'=' * 80}")
        logger.info("GENERATING BENCHMARKS WITH DIFFICULTY TARGETING")
        logger.info(f"{'=' * 80}\n")

        gen = EnhancedProblemGenerator(
            output_dir=args.output,
            seed=args.seed
        )

        domains_to_generate = []
        if args.all:
            domains_to_generate = ["blocksworld", "logistics", "parking"]
        elif args.domain:
            domains_to_generate = [args.domain]
        else:
            # Default: all domains
            logger.info("No domain specified - generating for all domains")
            domains_to_generate = ["blocksworld", "logistics", "parking"]

        logger.info(f"Domains: {domains_to_generate}")
        logger.info(f"Problems per size: {args.problems_per_size}")
        logger.info(f"Seed: {args.seed}\n")

        gen.generate_all(
            problems_per_size=args.problems_per_size,
            domains_to_generate=domains_to_generate,
            max_generation_attempts=50
        )

        logger.info(f"\n{'=' * 80}")
        logger.info("GENERATION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"\nNext step: Validate benchmarks")
        logger.info(f"  python generate_benchmarks.py --validate --benchmark-dir {args.output}")
        logger.info(f"\nOr validate a specific size:")
        logger.info(f"  python generate_benchmarks.py --validate-size blocksworld small")

        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())