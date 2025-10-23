# -*- coding: utf-8 -*-
"""
Core PDDL problem generator.
Creates valid, solvable problems of varying sizes.
"""

import os
import random
from typing import List, Dict, Tuple
from pathlib import Path

from domain_templates import DOMAINS
from size_config import SIZE_CONFIGS


class PDDLProblemGenerator:
    """Generates valid PDDL problems for training."""

    def __init__(self, output_dir: str = "benchmarks", seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        random.seed(seed)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # BLOCKS WORLD PROBLEM GENERATION
    # ========================================================================

    def generate_blocks_world_problem(
            self,
            num_blocks: int,
            num_stacks: int,
            problem_id: int
    ) -> str:
        """
        Generate a Blocks World problem.

        Creates a solvable problem: random initial state → all blocks on table →
        then builds stacks as goal.

        Args:
            num_blocks: Total number of blocks
            num_stacks: Number of stacks to create in goal
            problem_id: Unique problem identifier

        Returns:
            PDDL problem string
        """
        blocks = [f"b{i}" for i in range(num_blocks)]

        # Initial state: all blocks on table
        init_facts = [f"(ontable {b})" for b in blocks]
        init_facts.append("(arm-empty)")
        for b in blocks:
            init_facts.append(f"(clear {b})")

        # Goal: create stacks
        # Stack 0: b0, b1, b2, ... (blocks_per_stack elements)
        # Stack 1: b_k, b_k+1, ...
        # etc.
        blocks_per_stack = num_blocks // num_stacks
        goal_facts = []

        block_idx = 0
        for stack_id in range(num_stacks):
            stack_blocks = blocks[block_idx:block_idx + blocks_per_stack]
            if stack_blocks:
                # Stack blocks: b0 on b1 on b2 on table
                for i in range(len(stack_blocks) - 1):
                    goal_facts.append(f"(on {stack_blocks[i]} {stack_blocks[i + 1]})")
                goal_facts.append(f"(ontable {stack_blocks[-1]})")
            block_idx += blocks_per_stack

        # Add remaining blocks on table
        for b in blocks[block_idx:]:
            goal_facts.append(f"(ontable {b})")

        problem = f"""(define (problem blocks-world-{problem_id})
  (:domain blocks-world)
  (:objects {' '.join(blocks)} - block)
  (:init
    {' '.join(init_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""
        return problem

    # ========================================================================
    # LOGISTICS PROBLEM GENERATION
    # ========================================================================

    def generate_logistics_problem(
            self,
            num_cities: int,
            locations_per_city: int,
            trucks_per_city: int,
            num_objects: int,
            problem_id: int
    ) -> str:
        """
        Generate a Logistics problem.

        Creates a solvable problem: objects scattered across cities →
        deliver all to a central location.
        """
        cities = [f"city{i}" for i in range(num_cities)]
        locations = []
        trucks = []
        objects = [f"obj{i}" for i in range(num_objects)]

        # Create locations and trucks
        loc_idx = 0
        for city in cities:
            for loc_in_city in range(locations_per_city):
                loc_name = f"loc-{city}-{loc_in_city}"
                locations.append((loc_name, city))
                loc_idx += 1

            for truck_in_city in range(trucks_per_city):
                truck_name = f"truck-{city}-{truck_in_city}"
                trucks.append((truck_name, city))

        # Initial state: objects at random locations, trucks at first location of their city
        init_facts = []
        for obj in objects:
            loc_choice = random.choice(locations)
            init_facts.append(f"(at-obj {obj} {loc_choice[0]})")
            init_facts.append(f"(obj-at-city {obj} {loc_choice[1]})")

        for truck, city in trucks:
            init_facts.append(f"(at {truck} loc-{city}-0)")
            init_facts.append(f"(truck-at-city {truck} {city})")

        # Connected locations within each city (line graph)
        for city in cities:
            city_locs = [loc for loc, c in locations if c == city]
            for i in range(len(city_locs) - 1):
                init_facts.append(f"(connected {city_locs[i]} {city_locs[i + 1]})")
                init_facts.append(f"(connected {city_locs[i + 1]} {city_locs[i]})")

        # Connect cities: connect first location of each city
        for i in range(len(cities) - 1):
            loc1 = f"loc-{cities[i]}-0"
            loc2 = f"loc-{cities[i + 1]}-0"
            init_facts.append(f"(connected {loc1} {loc2})")
            init_facts.append(f"(connected {loc2} {loc1})")

        # Goal: all objects at first location of first city
        goal_loc = f"loc-{cities[0]}-0"
        goal_facts = [f"(at-obj {obj} {goal_loc})" for obj in objects]

        # Object type definitions
        obj_str = " ".join(objects)
        loc_str = " ".join([loc for loc, _ in locations])
        truck_str = " ".join([truck for truck, _ in trucks])
        city_str = " ".join(cities)

        problem = f"""(define (problem logistics-{problem_id})
  (:domain logistics)
  (:objects
    {truck_str} - truck
    {loc_str} - location
    {obj_str} - object
    {city_str} - city
  )
  (:init
    {' '.join(init_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""
        return problem

    # ========================================================================
    # GRIPPER PROBLEM GENERATION
    # ========================================================================

    def generate_gripper_problem(
            self,
            num_rooms: int,
            num_objects: int,
            num_grippers: int,
            problem_id: int
    ) -> str:
        """
        Generate a Gripper problem.

        Classic gripper problem: move objects from room A to room B.
        """
        rooms = [f"room{i}" for i in range(num_rooms)]
        objects = [f"obj{i}" for i in range(num_objects)]
        grippers = [f"gripper{i}" for i in range(num_grippers)]

        # Initial state: all objects in first room, robot in first room
        init_facts = [f"(at-robot {rooms[0]})"]
        for obj in objects:
            init_facts.append(f"(at {obj} {rooms[0]})")
        for gripper in grippers:
            init_facts.append(f"(free {gripper})")

        # Connect rooms in a line graph
        for i in range(len(rooms) - 1):
            init_facts.append(f"(connect {rooms[i]} {rooms[i + 1]})")
            init_facts.append(f"(connect {rooms[i + 1]} {rooms[i]})")

        # Goal: all objects in last room
        goal_loc = rooms[-1]
        goal_facts = [f"(at {obj} {goal_loc})" for obj in objects]

        room_str = " ".join(rooms)
        obj_str = " ".join(objects)
        gripper_str = " ".join(grippers)

        problem = f"""(define (problem gripper-{problem_id})
  (:domain gripper)
  (:objects
    {room_str} - room
    {obj_str} - object
    {gripper_str} - gripper
  )
  (:init
    {' '.join(init_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""
        return problem

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def generate_problem_set(
            self,
            domain: str,
            size: str,  # "small", "medium", "large"
            num_problems: int = 5
    ) -> List[str]:
        """
        Generate a set of problems for a domain at a specific size.

        Args:
            domain: "blocks_world", "logistics", or "gripper"
            size: "small", "medium", or "large"
            num_problems: Number of problems to generate

        Returns:
            List of PDDL problem strings
        """
        if domain not in SIZE_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}")
        if size not in SIZE_CONFIGS[domain]:
            raise ValueError(f"Unknown size: {size}")

        config = SIZE_CONFIGS[domain][size]
        problems = []

        for problem_id in range(num_problems):
            if domain == "blocks_world":
                problem = self.generate_blocks_world_problem(
                    num_blocks=config["num_blocks"],
                    num_stacks=config["num_stacks"],
                    problem_id=problem_id
                )
            elif domain == "logistics":
                problem = self.generate_logistics_problem(
                    num_cities=config["num_cities"],
                    locations_per_city=config["locations_per_city"],
                    trucks_per_city=config["trucks_per_city"],
                    num_objects=config["objects"],
                    problem_id=problem_id
                )
            elif domain == "gripper":
                problem = self.generate_gripper_problem(
                    num_rooms=config["num_rooms"],
                    num_objects=config["num_objects"],
                    num_grippers=config["num_grippers"],
                    problem_id=problem_id
                )

            problems.append(problem)

        return problems

    def save_domain_and_problems(
            self,
            domain: str,
            size: str,
            num_problems: int = 5
    ) -> Tuple[str, List[str]]:
        """
        Generate and save domain and problems to disk.

        Returns:
            (domain_path, problem_paths)
        """
        # Create directory
        domain_dir = Path(self.output_dir) / domain / size
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Save domain
        domain_path = domain_dir / "domain_new.pddl"
        with open(domain_path, "w") as f:
            f.write(DOMAINS[domain])
        print(f"✓ Domain saved: {domain_path}")

        # Generate and save problems
        problems = self.generate_problem_set(domain, size, num_problems)
        problem_paths = []

        for i, problem_str in enumerate(problems):
            problem_path = domain_dir / f"problem_{size}_{i:02d}.pddl"
            with open(problem_path, "w") as f:
                f.write(problem_str)
            problem_paths.append(str(problem_path))

        print(f"✓ Generated {len(problems)} problems at {domain_dir}")
        return str(domain_path), problem_paths


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_all_benchmarks(output_dir: str = "benchmarks"):
    """Generate all domain/size combinations."""
    gen = PDDLProblemGenerator(output_dir)

    domains = ["blocks_world", "logistics", "gripper"]
    sizes = ["small", "medium", "large"]

    for domain in domains:
        for size in sizes:
            print(f"\nGenerating {domain} - {size}...")
            gen.save_domain_and_problems(domain, size, num_problems=15)

    print("\n✅ All benchmarks generated!")


if __name__ == "__main__":
    generate_all_benchmarks()