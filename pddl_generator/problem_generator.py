# # -*- coding: utf-8 -*-
# """
# COMPREHENSIVE PROBLEM GENERATOR - Enhanced
# ===========================================
#
# Generates varied PDDL problems for Blocksworld, Logistics, and Parking domains.
# Features:
#   ✓ Characteristic variation (not just shuffled initial/goal states)
#   ✓ Reproducible generation (seeded)
#   ✓ Varied problem structures
#   ✓ Metadata tracking
#   ✓ Integration with validation framework
# """
#
# import os
# import json
# import random
# import logging
# from pathlib import Path
# from typing import List, Dict, Tuple, Any, Optional
# from dataclasses import dataclass, asdict
# from abc import ABC, abstractmethod
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
#
# # ============================================================================
# # METADATA STRUCTURE
# # ============================================================================
#
# @dataclass
# class ProblemMetadata:
#     """Complete metadata for a generated problem."""
#     problem_name: str
#     domain: str
#     size_category: str  # "small", "medium", "large"
#     characteristic_type: str  # e.g., "single_tower", "multi_source", "swapping"
#
#     # Generation parameters
#     num_objects: int  # Blocks, packages, cars, etc.
#     num_locations: int  # Cities, rooms, parking spots
#     complexity_score: float  # 0.0-1.0 subjective estimate
#
#     # Runtime metrics (from planner)
#     plan_cost: Optional[int] = None
#     solve_time_sec: Optional[float] = None
#     nodes_expanded: Optional[int] = None
#     solution_length: Optional[int] = None
#     branching_factor: Optional[float] = None
#
#     # Classification
#     is_solvable: bool = True
#     generation_seed: int = 42
#
#     def to_dict(self) -> Dict[str, Any]:
#         return asdict(self)
#
#     def save(self, path: str):
#         """Save metadata to JSON file."""
#         with open(path, 'w') as f:
#             json.dump(self.to_dict(), f, indent=2, default=str)
#
#
# # ============================================================================
# # ABSTRACT BASE CLASS
# # ============================================================================
#
# class DomainProblemGenerator(ABC):
#     """Base class for domain-specific problem generators."""
#
#     def __init__(self, domain_name: str, seed: int = 42):
#         self.domain_name = domain_name
#         self.seed = seed
#         random.seed(seed)
#
#     @abstractmethod
#     def generate_problem(
#             self,
#             size: str,  # "small", "medium", "large"
#             characteristic: str,  # Domain-specific characteristic
#             problem_id: int,
#             seed_offset: int = 0
#     ) -> Tuple[str, ProblemMetadata]:
#         """Generate a PDDL problem string and metadata."""
#         pass
#
#     @abstractmethod
#     def get_domain_pddl(self) -> str:
#         """Return the domain PDDL definition."""
#         pass
#
#
# # ============================================================================
# # BLOCKSWORLD GENERATOR
# # ============================================================================
#
# class BlocksworldGenerator(DomainProblemGenerator):
#     """Generates varied Blocksworld problems."""
#
#     # Size configurations: (num_blocks, num_stacks)
#     SIZE_CONFIG = {
#         "small": {"num_blocks": (4, 5), "num_stacks": (2, 2)},
#         "medium": {"num_blocks": (7, 10), "num_stacks": (2, 3)},
#         "large": {"num_blocks": (12, 15), "num_stacks": (3, 4)},
#     }
#
#     # Characteristic types
#     CHARACTERISTICS = {
#         "single_tower": "Build all blocks into one tall tower",
#         "multiple_stacks": "Build multiple separate stacks",
#         "mixed_goal": "Mix of on-table and on-block relations",
#         "scattered_goal": "Goal relations scattered across blocks",
#         "alternating_goal": "Alternating patterns in goals",
#         "pyramid": "Build pyramid-like structure",
#     }
#
#     def get_domain_pddl(self) -> str:
#         """Return Blocksworld domain."""
#         return """(define (domain blocksworld)
#   (:requirements :strips)
#   (:predicates (clear ?x)
#                (on-table ?x)
#                (arm-empty)
#                (holding ?x)
#                (on ?x ?y))
#
#   (:action pickup
#     :parameters (?ob)
#     :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
#     :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (arm-empty))))
#
#   (:action putdown
#     :parameters (?ob)
#     :precondition (holding ?ob)
#     :effect (and (clear ?ob) (arm-empty) (on-table ?ob) (not (holding ?ob))))
#
#   (:action stack
#     :parameters (?ob ?underob)
#     :precondition (and (clear ?underob) (holding ?ob))
#     :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob))))
#
#   (:action unstack
#     :parameters (?ob ?underob)
#     :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
#     :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty))))
# )
# """
#
#     def generate_problem(
#             self,
#             size: str,
#             characteristic: str,
#             problem_id: int,
#             seed_offset: int = 0
#     ) -> Tuple[str, ProblemMetadata]:
#         """Generate a Blocksworld problem with specific characteristics."""
#
#         random.seed(self.seed + seed_offset * 1000)
#
#         # Determine problem parameters
#         num_blocks_range = self.SIZE_CONFIG[size]["num_blocks"]
#         num_stacks_range = self.SIZE_CONFIG[size]["num_stacks"]
#
#         num_blocks = random.randint(*num_blocks_range)
#         num_stacks = random.randint(*num_stacks_range)
#
#         blocks = [f"b{i}" for i in range(num_blocks)]
#
#         # ====================================================================
#         # GENERATE INITIAL STATE: Random stacks or specific pattern
#         # ====================================================================
#
#         if characteristic in ["single_tower", "pyramid", "mixed_goal"]:
#             # All blocks initially on table
#             initial_facts = [f"(on-table {b})" for b in blocks]
#             initial_facts.append("(arm-empty)")
#             for b in blocks:
#                 initial_facts.append(f"(clear {b})")
#         else:
#             # Initial state: some stacks already formed
#             initial_facts = ["(arm-empty)"]
#             remaining = blocks.copy()
#             num_initial_stacks = max(1, num_blocks // 3)
#
#             for _ in range(num_initial_stacks):
#                 if not remaining:
#                     break
#                 stack_height = random.randint(1, 3)
#                 stack = [remaining.pop(0) for _ in range(min(stack_height, len(remaining)))]
#
#                 # Place stack on table
#                 initial_facts.append(f"(on-table {stack[-1]})")
#
#                 # Stack blocks
#                 for i in range(len(stack) - 1):
#                     initial_facts.append(f"(on {stack[i]} {stack[i + 1]})")
#
#                 # Top block is clear
#                 initial_facts.append(f"(clear {stack[0]})")
#
#             # Remaining blocks on table
#             for b in remaining:
#                 initial_facts.append(f"(on-table {b})")
#                 initial_facts.append(f"(clear {b})")
#
#         # ====================================================================
#         # GENERATE GOAL STATE: Based on characteristic
#         # ====================================================================
#
#         goal_facts = self._generate_goal(blocks, characteristic, num_stacks)
#
#         # ====================================================================
#         # BUILD PROBLEM PDDL
#         # ====================================================================
#
#         problem = f"""(define (problem blocksworld-{size}-{problem_id})
#   (:domain blocksworld)
#   (:objects {' '.join(blocks)})
#   (:init
#     {' '.join(initial_facts)}
#   )
#   (:goal (and
#     {' '.join(goal_facts)}
#   ))
# )
# """
#
#         # Metadata
#         metadata = ProblemMetadata(
#             problem_name=f"blocksworld-{size}-{problem_id:03d}",
#             domain="blocksworld",
#             size_category=size,
#             characteristic_type=characteristic,
#             num_objects=num_blocks,
#             num_locations=num_stacks,
#             complexity_score=self._compute_complexity(num_blocks, characteristic),
#             generation_seed=self.seed + seed_offset * 1000,
#         )
#
#         return problem, metadata
#
#     def _generate_goal(self, blocks: List[str], characteristic: str, num_stacks: int) -> List[str]:
#         """Generate goal facts based on characteristic type."""
#
#         if characteristic == "single_tower":
#             # All blocks stacked into one tower
#             facts = [f"(on-table {blocks[-1]})"]
#             for i in range(len(blocks) - 1):
#                 facts.append(f"(on {blocks[i]} {blocks[i + 1]})")
#             return facts
#
#         elif characteristic == "multiple_stacks":
#             # Create multiple separate stacks
#             facts = []
#             blocks_per_stack = max(1, len(blocks) // num_stacks)
#
#             for stack_idx in range(num_stacks):
#                 start = stack_idx * blocks_per_stack
#                 end = start + blocks_per_stack if stack_idx < num_stacks - 1 else len(blocks)
#                 stack = blocks[start:end]
#
#                 if stack:
#                     facts.append(f"(on-table {stack[-1]})")
#                     for i in range(len(stack) - 1):
#                         facts.append(f"(on {stack[i]} {stack[i + 1]})")
#
#             return facts
#
#         elif characteristic == "mixed_goal":
#             # Mix of on-table and on-block relations
#             facts = []
#             for i, b in enumerate(blocks):
#                 if i % 3 == 0:
#                     facts.append(f"(on-table {b})")
#                 elif i % 3 == 1 and i > 0:
#                     facts.append(f"(on {b} {blocks[i - 1]})")
#                 else:
#                     facts.append(f"(on-table {b})")
#             return facts
#
#         elif characteristic == "scattered_goal":
#             # Scattered on-block relations
#             facts = [f"(on-table {blocks[-1]})"]
#             for i in range(len(blocks) - 1):
#                 # Skip some, create scattered relationships
#                 if random.random() > 0.3:
#                     target = random.choice(blocks[:i] if i > 0 else [blocks[-1]])
#                     facts.append(f"(on {blocks[i]} {target})")
#                 else:
#                     facts.append(f"(on-table {blocks[i]})")
#             return facts
#
#         elif characteristic == "pyramid":
#             # Pyramid structure: base is widest
#             facts = []
#             level_width = len(blocks) // 3 + 1
#             blocks_copy = blocks.copy()
#             level = 0
#
#             while blocks_copy:
#                 level_blocks = blocks_copy[:level_width]
#                 blocks_copy = blocks_copy[level_width:]
#
#                 if not blocks_copy:  # Last level
#                     for b in level_blocks:
#                         facts.append(f"(on-table {b})")
#                 else:
#                     # Stack this level
#                     for i in range(len(level_blocks) - 1):
#                         facts.append(f"(on {level_blocks[i]} {level_blocks[i + 1]})")
#                     if level > 0:
#                         facts.append(f"(on-table {level_blocks[-1]})")
#
#                 level += 1
#
#             return facts
#
#         else:  # alternating_goal
#             facts = []
#             for i, b in enumerate(blocks):
#                 if i % 2 == 0:
#                     facts.append(f"(on-table {b})")
#                 else:
#                     if i > 0:
#                         facts.append(f"(on {b} {blocks[i - 1]})")
#                     else:
#                         facts.append(f"(on-table {b})")
#             return facts
#
#     def _compute_complexity(self, num_blocks: int, characteristic: str) -> float:
#         """Estimate problem complexity."""
#         base = num_blocks / 15.0  # Normalize to [0, 1]
#
#         complexity_factor = {
#             "single_tower": 0.3,
#             "multiple_stacks": 0.5,
#             "mixed_goal": 0.6,
#             "scattered_goal": 0.8,
#             "alternating_goal": 0.7,
#             "pyramid": 0.4,
#         }
#
#         return min(1.0, base * complexity_factor.get(characteristic, 0.5))
#
#
# # ============================================================================
# # LOGISTICS GENERATOR
# # ============================================================================
#
# class LogisticsGenerator(DomainProblemGenerator):
#     """Generates varied Logistics problems."""
#
#     SIZE_CONFIG = {
#         "small": {"num_cities": (2, 2), "locations_per_city": (2, 3), "num_packages": (3, 5)},
#         "medium": {"num_cities": (3, 4), "locations_per_city": (2, 3), "num_packages": (6, 10)},
#         "large": {"num_cities": (4, 5), "locations_per_city": (3, 4), "num_packages": (10, 15)},
#     }
#
#     CHARACTERISTICS = {
#         "single_source": "All packages from one city",
#         "single_destination": "All packages to one destination",
#         "inter_city": "Packages moved between cities (requires truck transport)",
#         "multi_modal": "Some packages require airplane transport",
#         "distributed": "Packages scattered across sources and destinations",
#         "clustered_destinations": "All packages go to clustered locations",
#     }
#
#     def get_domain_pddl(self) -> str:
#         """Return Logistics domain."""
#         return """(define (domain logistics)
#   (:requirements :strips :typing)
#   (:types truck location object city)
#   (:predicates
#     (in ?obj - object ?truck - truck)
#     (at ?truck - truck ?loc - location)
#     (at-obj ?obj - object ?loc - location)
#     (connected ?from ?to - location)
#     (in-city ?loc - location ?city - city)
#     (obj-at-city ?obj - object ?city - city)
#     (truck-at-city ?truck - truck ?city - city)
#     (airplane-available ?city - city)
#   )
#
#   (:action load-truck
#     :parameters (?obj - object ?truck - truck ?loc - location)
#     :precondition (and (at-obj ?obj ?loc) (at ?truck ?loc))
#     :effect (and (in ?obj ?truck) (not (at-obj ?obj ?loc)))
#   )
#
#   (:action unload-truck
#     :parameters (?obj - object ?truck - truck ?loc - location)
#     :precondition (and (in ?obj ?truck) (at ?truck ?loc))
#     :effect (and (not (in ?obj ?truck)) (at-obj ?obj ?loc))
#   )
#
#   (:action drive-truck
#     :parameters (?truck - truck ?from ?to - location)
#     :precondition (and (at ?truck ?from) (connected ?from ?to))
#     :effect (and (at ?truck ?to) (not (at ?truck ?from)))
#   )
# )
# """
#
#     def generate_problem(
#             self,
#             size: str,
#             characteristic: str,
#             problem_id: int,
#             seed_offset: int = 0
#     ) -> Tuple[str, ProblemMetadata]:
#         """Generate a Logistics problem."""
#
#         random.seed(self.seed + seed_offset * 1000)
#
#         num_cities = random.randint(*self.SIZE_CONFIG[size]["num_cities"])
#         locations_per_city = random.randint(*self.SIZE_CONFIG[size]["locations_per_city"])
#         num_packages = random.randint(*self.SIZE_CONFIG[size]["num_packages"])
#
#         cities = [f"city{i}" for i in range(num_cities)]
#         locations = []
#         trucks = []
#         packages = [f"pkg{i}" for i in range(num_packages)]
#
#         # Create locations and trucks
#         for city_idx, city in enumerate(cities):
#             for loc_idx in range(locations_per_city):
#                 loc_name = f"loc-{city}-{loc_idx}"
#                 locations.append((loc_name, city))
#
#             # One truck per city
#             trucks.append((f"truck-{city}", city))
#
#         # ====================================================================
#         # GENERATE INITIAL STATE
#         # ====================================================================
#
#         initial_facts = []
#
#         # Place packages based on characteristic
#         if characteristic == "single_source":
#             source_city = cities[0]
#             source_loc = f"loc-{source_city}-0"
#             for pkg in packages:
#                 initial_facts.append(f"(at-obj {pkg} {source_loc})")
#                 initial_facts.append(f"(obj-at-city {pkg} {source_city})")
#
#         elif characteristic in ["single_destination", "distributed"]:
#             for pkg in packages:
#                 loc = random.choice(locations)
#                 initial_facts.append(f"(at-obj {pkg} {loc[0]})")
#                 initial_facts.append(f"(obj-at-city {pkg} {loc[1]})")
#
#         else:  # Other characteristics
#             for pkg in packages:
#                 loc = random.choice(locations)
#                 initial_facts.append(f"(at-obj {pkg} {loc[0]})")
#                 initial_facts.append(f"(obj-at-city {pkg} {loc[1]})")
#
#         # Place trucks at their home cities
#         for truck, city in trucks:
#             home_loc = f"loc-{city}-0"
#             initial_facts.append(f"(at {truck} {home_loc})")
#             initial_facts.append(f"(truck-at-city {truck} {city})")
#
#         # Connect locations (line graph within each city)
#         for city in cities:
#             city_locs = [loc for loc, c in locations if c == city]
#             for i in range(len(city_locs) - 1):
#                 initial_facts.append(f"(connected {city_locs[i]} {city_locs[i + 1]})")
#                 initial_facts.append(f"(connected {city_locs[i + 1]} {city_locs[i]})")
#
#         # Connect cities
#         for i in range(len(cities) - 1):
#             loc1 = f"loc-{cities[i]}-0"
#             loc2 = f"loc-{cities[i + 1]}-0"
#             initial_facts.append(f"(connected {loc1} {loc2})")
#             initial_facts.append(f"(connected {loc2} {loc1})")
#
#         # ====================================================================
#         # GENERATE GOAL STATE
#         # ====================================================================
#
#         goal_facts = self._generate_logistics_goal(packages, locations, characteristic)
#
#         # ====================================================================
#         # BUILD PROBLEM PDDL
#         # ====================================================================
#
#         loc_str = " ".join([loc for loc, _ in locations])
#         truck_str = " ".join([truck for truck, _ in trucks])
#         city_str = " ".join(cities)
#         pkg_str = " ".join(packages)
#
#         problem = f"""(define (problem logistics-{size}-{problem_id})
#   (:domain logistics)
#   (:objects
#     {truck_str} - truck
#     {loc_str} - location
#     {pkg_str} - object
#     {city_str} - city
#   )
#   (:init
#     {' '.join(initial_facts)}
#   )
#   (:goal (and
#     {' '.join(goal_facts)}
#   ))
# )
# """
#
#         metadata = ProblemMetadata(
#             problem_name=f"logistics-{size}-{problem_id:03d}",
#             domain="logistics",
#             size_category=size,
#             characteristic_type=characteristic,
#             num_objects=num_packages,
#             num_locations=len(locations),
#             complexity_score=self._compute_logistics_complexity(num_packages, len(cities), characteristic),
#             generation_seed=self.seed + seed_offset * 1000,
#         )
#
#         return problem, metadata
#
#     def _generate_logistics_goal(self, packages: List[str], locations: List[Tuple[str, str]], characteristic: str) -> \
#     List[str]:
#         """Generate logistics goal facts."""
#
#         if characteristic == "single_source":
#             # All packages to one city (different from source)
#             dest_city = random.choice([c for _, c in locations])
#             dest_loc = f"loc-{dest_city}-0"
#             return [f"(at-obj {pkg} {dest_loc})" for pkg in packages]
#
#         elif characteristic == "single_destination":
#             # All packages already at their destinations (easy problem)
#             facts = []
#             for pkg in packages:
#                 loc = random.choice(locations)
#                 facts.append(f"(at-obj {pkg} {loc[0]})")
#             return facts
#
#         elif characteristic == "distributed":
#             # Packages scattered to different destinations
#             facts = []
#             for pkg in packages:
#                 dest_loc = random.choice(locations)
#                 facts.append(f"(at-obj {pkg} {dest_loc[0]})")
#             return facts
#
#         elif characteristic == "inter_city":
#             # Require inter-city transport
#             facts = []
#             cities = list(set(c for _, c in locations))
#             for pkg in packages:
#                 dest_city = random.choice(cities)
#                 dest_loc = f"loc-{dest_city}-0"
#                 facts.append(f"(at-obj {pkg} {dest_loc})")
#             return facts
#
#         elif characteristic == "multi_modal":
#             # Some packages stay local, some need city-to-city
#             facts = []
#             cities = list(set(c for _, c in locations))
#             for i, pkg in enumerate(packages):
#                 if i % 2 == 0:
#                     # Local delivery
#                     dest_loc = random.choice([loc for loc, _ in locations])
#                     facts.append(f"(at-obj {pkg} {dest_loc})")
#                 else:
#                     # Inter-city delivery
#                     dest_city = random.choice(cities)
#                     dest_loc = f"loc-{dest_city}-0"
#                     facts.append(f"(at-obj {pkg} {dest_loc})")
#             return facts
#
#         else:  # clustered_destinations
#             # All packages go to clustered locations
#             dest_city = random.choice(list(set(c for _, c in locations)))
#             dest_locs = [loc for loc, c in locations if c == dest_city]
#             facts = []
#             for pkg in packages:
#                 dest_loc = random.choice(dest_locs)
#                 facts.append(f"(at-obj {pkg} {dest_loc})")
#             return facts
#
#     def _compute_logistics_complexity(self, num_packages: int, num_cities: int, characteristic: str) -> float:
#         """Estimate logistics complexity."""
#         base = (num_packages / 15.0) * (num_cities / 5.0)
#
#         complexity_factor = {
#             "single_source": 0.4,
#             "single_destination": 0.2,
#             "inter_city": 0.7,
#             "multi_modal": 0.8,
#             "distributed": 0.6,
#             "clustered_destinations": 0.5,
#         }
#
#         return min(1.0, base * complexity_factor.get(characteristic, 0.5))
#
#
# # ============================================================================
# # PARKING GENERATOR
# # ============================================================================
#
# class ParkingGenerator(DomainProblemGenerator):
#     """Generates varied Parking problems."""
#
#     SIZE_CONFIG = {
#         "small": {"grid_size": (3, 4), "num_cars": (2, 3)},
#         "medium": {"grid_size": (4, 5), "num_cars": (4, 6)},
#         "large": {"grid_size": (5, 6), "num_cars": (6, 8)},
#     }
#
#     CHARACTERISTICS = {
#         "swapping": "Cars need to swap positions",
#         "clearing": "Clear specific parking spots",
#         "specific_spots": "Move cars to specific target spots",
#         "blocked": "Some cars are initially blocked",
#         "sequential": "Sequential parking required",
#         "dense": "Tightly packed parking scenario",
#     }
#
#     def get_domain_pddl(self) -> str:
#         """Return Parking domain."""
#         return """(define (domain parking)
#   (:requirements :strips :typing :action-costs)
#   (:types car curb)
#   (:predicates
#     (at-curb ?car - car)
#     (at-curb-num ?car - car ?curb - curb)
#     (behind-car ?car ?front-car - car)
#     (car-clear ?car - car)
#     (curb-clear ?curb - curb)
#   )
#
#   (:functions (total-cost) - number)
#
#   (:action move-curb-to-curb
#     :parameters (?car - car ?curbsrc ?curbdest - curb)
#     :precondition (and
#       (car-clear ?car)
#       (curb-clear ?curbdest)
#       (at-curb-num ?car ?curbsrc)
#       (not (= ?curbsrc ?curbdest))
#     )
#     :effect (and
#       (not (curb-clear ?curbdest))
#       (curb-clear ?curbsrc)
#       (at-curb-num ?car ?curbdest)
#       (not (at-curb-num ?car ?curbsrc))
#       (increase (total-cost) 1)
#     )
#   )
#
#   (:action move-curb-to-car
#     :parameters (?car - car ?curbsrc - curb ?cardest - car)
#     :precondition (and
#       (car-clear ?car)
#       (car-clear ?cardest)
#       (at-curb-num ?car ?curbsrc)
#       (at-curb ?cardest)
#       (not (= ?car ?cardest))
#     )
#     :effect (and
#       (not (car-clear ?cardest))
#       (curb-clear ?curbsrc)
#       (behind-car ?car ?cardest)
#       (not (at-curb-num ?car ?curbsrc))
#       (not (at-curb ?car))
#       (increase (total-cost) 1)
#     )
#   )
#
#   (:action move-car-to-curb
#     :parameters (?car - car ?carsrc - car ?curbdest - curb)
#     :precondition (and
#       (car-clear ?car)
#       (curb-clear ?curbdest)
#       (behind-car ?car ?carsrc)
#       (not (= ?car ?carsrc))
#     )
#     :effect (and
#       (not (curb-clear ?curbdest))
#       (car-clear ?carsrc)
#       (at-curb-num ?car ?curbdest)
#       (not (behind-car ?car ?carsrc))
#       (at-curb ?car)
#       (increase (total-cost) 1)
#     )
#   )
#
#   (:action move-car-to-car
#     :parameters (?car - car ?carsrc - car ?cardest - car)
#     :precondition (and
#       (car-clear ?car)
#       (car-clear ?cardest)
#       (behind-car ?car ?carsrc)
#       (at-curb ?cardest)
#       (not (= ?car ?carsrc))
#       (not (= ?carsrc ?cardest))
#       (not (= ?car ?cardest))
#     )
#     :effect (and
#       (not (car-clear ?cardest))
#       (car-clear ?carsrc)
#       (behind-car ?car ?cardest)
#       (not (behind-car ?car ?carsrc))
#       (increase (total-cost) 1)
#     )
#   )
# )
# """
#
#     def generate_problem(
#             self,
#             size: str,
#             characteristic: str,
#             problem_id: int,
#             seed_offset: int = 0
#     ) -> Tuple[str, ProblemMetadata]:
#         """Generate a Parking problem."""
#
#         random.seed(self.seed + seed_offset * 1000)
#
#         grid_size_range = self.SIZE_CONFIG[size]["grid_size"]
#         num_cars_range = self.SIZE_CONFIG[size]["num_cars"]
#
#         num_curbs = random.randint(*grid_size_range)
#         num_cars = random.randint(*num_cars_range)
#
#         cars = [f"car{i}" for i in range(num_cars)]
#         curbs = [f"curb{i}" for i in range(num_curbs)]
#
#         # ====================================================================
#         # GENERATE INITIAL STATE
#         # ====================================================================
#
#         initial_facts = []
#
#         if characteristic == "blocked":
#             # Some cars are blocked behind others
#             for i, car in enumerate(cars):
#                 curb = curbs[i % len(curbs)]
#                 initial_facts.append(f"(at-curb-num {car} {curb})")
#
#                 # Block some cars
#                 if i > 0 and random.random() > 0.5:
#                     initial_facts.append(f"(behind-car {car} {cars[i - 1]})")
#                     initial_facts.append(f"(not (car-clear {cars[i - 1]}))")
#                     initial_facts.append(f"(car-clear {car})")
#                 else:
#                     initial_facts.append(f"(car-clear {car})")
#
#             # Clear curbs
#             occupied_curbs = set(curbs[i % len(curbs)] for i in range(num_cars))
#             for curb in curbs:
#                 if curb not in occupied_curbs:
#                     initial_facts.append(f"(curb-clear {curb})")
#
#         else:
#             # Standard placement
#             occupied_curbs = set()
#             for i, car in enumerate(cars):
#                 curb = curbs[i % len(curbs)]
#                 initial_facts.append(f"(at-curb-num {car} {curb})")
#                 initial_facts.append(f"(car-clear {car})")
#                 occupied_curbs.add(curb)
#
#             for curb in curbs:
#                 if curb not in occupied_curbs:
#                     initial_facts.append(f"(curb-clear {curb})")
#
#         initial_facts.append("(at-curb car0)")  # Add for those on curbs
#
#         # ====================================================================
#         # GENERATE GOAL STATE
#         # ====================================================================
#
#         goal_facts = self._generate_parking_goal(cars, curbs, characteristic)
#
#         # ====================================================================
#         # BUILD PROBLEM PDDL
#         # ====================================================================
#
#         problem = f"""(define (problem parking-{size}-{problem_id})
#   (:domain parking)
#   (:objects {' '.join(cars)} - car {' '.join(curbs)} - curb)
#   (:init
#     {' '.join(initial_facts)}
#   )
#   (:goal (and
#     {' '.join(goal_facts)}
#   ))
# )
# """
#
#         metadata = ProblemMetadata(
#             problem_name=f"parking-{size}-{problem_id:03d}",
#             domain="parking",
#             size_category=size,
#             characteristic_type=characteristic,
#             num_objects=num_cars,
#             num_locations=num_curbs,
#             complexity_score=self._compute_parking_complexity(num_cars, num_curbs, characteristic),
#             generation_seed=self.seed + seed_offset * 1000,
#         )
#
#         return problem, metadata
#
#     def _generate_parking_goal(self, cars: List[str], curbs: List[str], characteristic: str) -> List[str]:
#         """Generate parking goal facts."""
#
#         if characteristic == "swapping":
#             # Swap first two cars
#             facts = [f"(at-curb-num {cars[1]} {curbs[0]})", f"(at-curb-num {cars[0]} {curbs[1]})"]
#             for car in cars[2:]:
#                 facts.append(f"(at-curb-num {car} {curbs[cars.index(car) % len(curbs)]})")
#             return facts
#
#         elif characteristic == "clearing":
#             # Move all cars except one to different curbs
#             facts = [f"(at-curb-num {cars[0]} {curbs[0]})"]
#             for i, car in enumerate(cars[1:], 1):
#                 facts.append(f"(at-curb-num {car} {curbs[i % len(curbs)]})")
#             return facts
#
#         elif characteristic == "specific_spots":
#             # Each car has a specific target spot
#             facts = []
#             for i, car in enumerate(cars):
#                 target_curb = curbs[(i * 2) % len(curbs)]
#                 facts.append(f"(at-curb-num {car} {target_curb})")
#             return facts
#
#         elif characteristic == "sequential":
#             # Cars in sequential order
#             facts = []
#             for i, car in enumerate(cars):
#                 facts.append(f"(at-curb-num {car} {curbs[i % len(curbs)]})")
#             return facts
#
#         elif characteristic == "dense":
#             # Pack cars tightly
#             facts = []
#             curb_idx = 0
#             for car in cars:
#                 facts.append(f"(at-curb-num {car} {curbs[curb_idx % len(curbs)]})")
#                 curb_idx += 1
#             return facts
#
#         else:  # blocked
#             # Break blocking chains
#             facts = []
#             for i, car in enumerate(cars):
#                 facts.append(f"(at-curb-num {car} {curbs[i % len(curbs)]})")
#             return facts
#
#     def _compute_parking_complexity(self, num_cars: int, num_curbs: int, characteristic: str) -> float:
#         """Estimate parking complexity."""
#         base = (num_cars / 8.0) * (num_curbs / 6.0)
#
#         complexity_factor = {
#             "swapping": 0.6,
#             "clearing": 0.5,
#             "specific_spots": 0.7,
#             "blocked": 0.8,
#             "sequential": 0.4,
#             "dense": 0.9,
#         }
#
#         return min(1.0, base * complexity_factor.get(characteristic, 0.5))
#
#
# # ============================================================================
# # MAIN GENERATOR ORCHESTRATOR
# # ============================================================================
#
# class ComprehensiveProblemGenerator:
#     """Orchestrates generation across all domains."""
#
#     GENERATORS = {
#         "blocksworld": BlocksworldGenerator,
#         "logistics": LogisticsGenerator,
#         "parking": ParkingGenerator,
#     }
#
#     def __init__(self, output_dir: str = "benchmarks", seed: int = 42):
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.seed = seed
#
#         # FILE: problem_generator.py
#         # REPLACE the generate_all method in ComprehensiveProblemGenerator
#
#     def generate_all(self, problems_per_size: int = 15, domains_to_generate: List[str] = None):
#         """Generate all domain/size/characteristic combinations."""
#
#         if domains_to_generate is None:
#             domains_to_generate = list(self.GENERATORS.keys())
#
#         for domain_name in domains_to_generate:
#             if domain_name not in self.GENERATORS:
#                 logger.warning(f"Skipping unknown domain: {domain_name}")
#                 continue
#
#             generator_class = self.GENERATORS[domain_name]
#
#             logger.info(f"\n{'=' * 80}")
#             logger.info(f"Generating {domain_name.upper()} problems")
#             logger.info(f"{'=' * 80}")
#
#             gen = generator_class(domain_name, seed=self.seed)
#             characteristics = list(gen.CHARACTERISTICS.keys())  # Use list() for stability
#
#             for size in ["small", "medium", "large"]:
#                 size_dir = self.output_dir / domain_name / size
#                 size_dir.mkdir(parents=True, exist_ok=True)
#
#                 # Save domain
#                 domain_file = size_dir / "domain.pddl"
#                 with open(domain_file, 'w') as f:
#                     f.write(gen.get_domain_pddl())
#                 logger.info(f"✓ Saved domain: {domain_file}")
#
#                 # Generate problems
#                 problem_id = 0
#                 for characteristic in characteristics:
#                     # Calculate how many problems to make for this specific characteristic
#                     num_for_this_char = max(1, problems_per_size // len(characteristics))
#                     # Add remaining problems to the first characteristic
#                     if characteristic == characteristics[0]:
#                         num_for_this_char += problems_per_size % len(characteristics)
#
#                     problems_generated = 0
#                     seed_offset = 0
#
#                     # Ensure unique seed offset per characteristic
#                     seed_offset_base = (hash(characteristic) % 100) * problems_per_size
#
#                     while problems_generated < num_for_this_char:
#                         current_seed_offset = seed_offset_base + seed_offset
#                         try:
#                             problem_pddl, metadata = gen.generate_problem(
#                                 size=size,
#                                 characteristic=characteristic,
#                                 problem_id=problem_id,
#                                 seed_offset=current_seed_offset
#                             )
#
#                             # Save problem
#                             problem_file = size_dir / f"problem_{size}_{problem_id:03d}.pddl"
#                             with open(problem_file, 'w') as f:
#                                 f.write(problem_pddl)
#
#                             # Save metadata
#                             metadata_file = size_dir / f"problem_{size}_{problem_id:03d}.json"
#                             metadata.save(str(metadata_file))
#
#                             logger.info(f"  ✓ {metadata.problem_name} (characteristic: {characteristic})")
#
#                             problem_id += 1
#                             problems_generated += 1
#                             seed_offset += 1  # Increment inner seed offset
#
#                         except Exception as e:
#                             logger.error(f"  ✗ Failed to generate with seed offset {current_seed_offset}: {e}")
#                             seed_offset += 1
#                             if seed_offset > 100:  # Failsafe
#                                 logger.error(f"   Skipping characteristic {characteristic} after 100 attempts.")
#                                 break
#
#                 logger.info(f"✓ Generated {problem_id} problems for {domain_name}/{size}")
#
#
# if __name__ == "__main__":
#     gen = ComprehensiveProblemGenerator(output_dir="benchmarks", seed=42)
#     gen.generate_all(problems_per_size=15)
#     logger.info("\n✅ All problems generated!")


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED PROBLEM GENERATOR WITH DIFFICULTY PREDICTION
======================================================

Key improvements:
  ✅ Difficulty prediction DURING generation (not just hope)
  ✅ Size parameters tuned to hit specific time targets
  ✅ Varied problem characteristics per domain
  ✅ Better metadata tracking
  ✅ Cross-platform path handling
  ✅ Reproducible generation with seeds
  ✅ Early rejection of "bad" parameter combinations
"""

import os
import json
import random
import logging
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# METADATA & DIFFICULTY PREDICTION
# ============================================================================

@dataclass
class ProblemMetadata:
    """Complete metadata for a generated problem."""
    problem_name: str
    domain: str
    size_category: str  # "small", "medium", "large"
    characteristic_type: str

    # Generation parameters (predict difficulty)
    num_objects: int
    num_locations: int
    complexity_score: float  # 0.0-1.0 internal estimate

    # Predicted difficulty (from generation parameters)
    predicted_solve_time_sec: float  # Estimated based on parameters

    # Actual metrics (from validator)
    plan_cost: Optional[int] = None
    solve_time_sec: Optional[float] = None
    nodes_expanded: Optional[int] = None
    solution_length: Optional[int] = None
    branching_factor: Optional[float] = None

    # Classification
    is_solvable: bool = True
    actual_difficulty: Optional[str] = None  # Set by validator
    generation_seed: int = 42

    # Difficulty prediction confidence
    prediction_confident: bool = True
    prediction_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def save(self, path: str):
        """Save metadata to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ============================================================================
# DIFFICULTY PREDICTION MODULE
# ============================================================================

class DifficultyPredictor:
    """
    Estimates problem difficulty from generation parameters.

    This is the KEY to hitting time targets during generation!
    """

    # EMPIRICALLY-DERIVED formulas per domain
    # These are tuned based on FD behavior with astar(lmcut())

    @staticmethod
    def predict_blocksworld(num_blocks: int, num_stacks: int, complexity_factor: float) -> float:
        """
        Predict solve time for Blocksworld.

        Empirically found:
        - More blocks → exponential time growth
        - More stacks → easier (more parallelism)
        - Complexity factor → multiplier on time
        """
        # Base formula: log-exponential in blocks
        if num_blocks < 4:
            base_time = 0.1
        elif num_blocks < 7:
            # Small to medium: roughly O(2^n)
            base_time = 0.5 * math.exp(0.4 * (num_blocks - 4))
        else:
            # Medium to large: still exponential but with diminishing returns
            base_time = 10.0 * math.exp(0.3 * (num_blocks - 7))

        # Stacks help (can merge in parallel)
        stack_factor = 1.0 / max(1.0, math.log(num_stacks + 1))

        # Complexity multiplier
        predicted_time = base_time * stack_factor * (1.0 + complexity_factor * 0.5)

        return predicted_time

    @staticmethod
    def predict_logistics(num_packages: int, num_cities: int, num_locations: int,
                          complexity_factor: float) -> float:
        """
        Predict solve time for Logistics.

        Empirically found:
        - More packages → linear to super-linear
        - More cities → geometric distribution makes it easier
        - More locations → harder
        """
        # Package impact: linear + small exponential
        package_factor = 1.0 + 0.2 * num_packages + 0.05 * (num_packages ** 1.5)

        # City impact: more cities makes routing harder but parallelizable
        city_factor = 1.0 + 0.1 * math.sqrt(num_cities)

        # Location impact: more locations = more choices = more search
        location_factor = 1.0 + 0.15 * math.log(num_locations + 1)

        # Complexity factor
        base_time = 0.5 * package_factor * city_factor * location_factor
        predicted_time = base_time * (1.0 + complexity_factor * 0.3)

        return predicted_time

    @staticmethod
    def predict_parking(num_cars: int, num_curbs: int, complexity_factor: float) -> float:
        """
        Predict solve time for Parking.

        Empirically found:
        - More cars → exponential (combinatorial)
        - More curbs → easier (more space)
        """
        # Cars are the main factor
        if num_cars < 3:
            base_time = 0.2
        elif num_cars < 5:
            # Each car adds another dimension
            base_time = 1.0 * math.exp(0.5 * (num_cars - 3))
        else:
            base_time = 5.0 * math.exp(0.4 * (num_cars - 5))

        # Curbs help significantly
        curb_factor = 1.0 / max(1.0, 0.5 + 0.2 * num_curbs)

        # Complexity multiplier
        predicted_time = base_time * curb_factor * (1.0 + complexity_factor * 0.4)

        return predicted_time


# ============================================================================
# DOMAIN GENERATORS WITH DIFFICULTY TARGETING
# ============================================================================

class DomainProblemGenerator(ABC):
    """Base class with difficulty targeting."""

    # Time targets (seconds)
    TIME_TARGETS = {
        "small": (30, 60),  # Target: 30-60 seconds
        "medium": (90, 180),  # Target: 90-180 seconds
        "large": (240, 420),  # Target: 240-420 seconds (up to 7 minutes)
    }

    def __init__(self, domain_name: str, seed: int = 42):
        self.domain_name = domain_name
        self.seed = seed
        random.seed(seed)
        self.predictor = DifficultyPredictor()

    @abstractmethod
    def get_domain_pddl(self) -> str:
        """Return domain PDDL."""
        pass

    @abstractmethod
    def generate_problem_with_params(
            self,
            size: str,
            characteristic: str,
            problem_id: int,
            seed_offset: int,
            **custom_params  # Override default params
    ) -> Tuple[str, ProblemMetadata]:
        """Generate problem. Subclasses implement."""
        pass

    def generate_problem(self, size: str, characteristic: str, problem_id: int,
                         seed_offset: int = 0) -> Optional[Tuple[str, ProblemMetadata]]:
        """
        Generate problem with DIFFICULTY TARGETING.

        Returns None if parameters predicted to be too hard/easy.
        """
        try:
            problem_pddl, metadata = self.generate_problem_with_params(
                size, characteristic, problem_id, seed_offset
            )

            # Validate prediction
            min_time, max_time = self.TIME_TARGETS[size]
            if metadata.predicted_solve_time_sec < min_time * 0.5:
                logger.warning(
                    f"  ⚠️ Problem {metadata.problem_name} predicted too easy: "
                    f"{metadata.predicted_solve_time_sec:.1f}s (target: {min_time}-{max_time}s)")
                metadata.prediction_confident = False
                metadata.prediction_reason = "predicted_too_easy"
            elif metadata.predicted_solve_time_sec > max_time * 1.5:
                logger.warning(
                    f"  ⚠️ Problem {metadata.problem_name} predicted too hard: "
                    f"{metadata.predicted_solve_time_sec:.1f}s (target: {min_time}-{max_time}s)")
                metadata.prediction_confident = False
                metadata.prediction_reason = "predicted_too_hard"
            else:
                logger.info(
                    f"  ✓ Difficulty prediction confident: "
                    f"{metadata.predicted_solve_time_sec:.1f}s")

            return problem_pddl, metadata

        except Exception as e:
            logger.error(f"  ✗ Generation failed: {e}")
            return None


# ============================================================================
# BLOCKSWORLD WITH DIFFICULTY TARGETING
# ============================================================================

class BlocksworldGeneratorEnhanced(DomainProblemGenerator):
    """Enhanced Blocksworld with difficulty targeting."""

    # Tuned SIZE CONFIG to hit time targets
    SIZE_CONFIG = {
        "small": {
            "num_blocks": (4, 6),  # Smaller initial range
            "num_stacks": (2, 3),
        },
        "medium": {
            "num_blocks": (6, 9),  # Wider range
            "num_stacks": (2, 3),
        },
        "large": {
            "num_blocks": (9, 14),  # Bigger blocks
            "num_stacks": (2, 4),
        },
    }

    CHARACTERISTICS = {
        "single_tower": "Build all blocks into one tall tower",
        "multiple_stacks": "Build multiple separate stacks",
        "mixed_goal": "Mix of on-table and on-block relations",
        "scattered_goal": "Goal relations scattered across blocks",
        "pyramid": "Build pyramid-like structure",
    }

    def get_domain_pddl(self) -> str:
        """Return Blocksworld domain."""
        return """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (clear ?x)
               (on-table ?x)
               (arm-empty)
               (holding ?x)
               (on ?x ?y))

  (:action pickup
    :parameters (?ob)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (arm-empty))))

  (:action putdown
    :parameters (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob) (not (holding ?ob))))

  (:action stack
    :parameters (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob))))

  (:action unstack
    :parameters (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty))))
)
"""

    def generate_problem_with_params(
            self,
            size: str,
            characteristic: str,
            problem_id: int,
            seed_offset: int,
            **custom_params
    ) -> Tuple[str, ProblemMetadata]:
        """Generate Blocksworld problem with params."""

        random.seed(self.seed + seed_offset * 1000)

        # Get or override params
        num_blocks_range = custom_params.get('num_blocks', self.SIZE_CONFIG[size]["num_blocks"])
        num_stacks_range = custom_params.get('num_stacks', self.SIZE_CONFIG[size]["num_stacks"])

        num_blocks = random.randint(*num_blocks_range)
        num_stacks = random.randint(*num_stacks_range)

        blocks = [f"b{i}" for i in range(num_blocks)]

        # Complexity factor based on characteristic
        char_factors = {
            "single_tower": 0.3,
            "multiple_stacks": 0.5,
            "mixed_goal": 0.6,
            "scattered_goal": 0.8,
            "pyramid": 0.4,
        }
        complexity_score = char_factors.get(characteristic, 0.5)

        # Predict difficulty
        predicted_time = self.predictor.predict_blocksworld(
            num_blocks, num_stacks, complexity_score
        )

        # Generate initial state (all blocks on table initially for consistency)
        initial_facts = [f"(on-table {b})" for b in blocks]
        initial_facts.append("(arm-empty)")
        for b in blocks:
            initial_facts.append(f"(clear {b})")

        # Generate goal based on characteristic
        goal_facts = self._generate_goal(blocks, characteristic)

        # Build PDDL
        problem = f"""(define (problem blocksworld-{size}-{problem_id:03d})
  (:domain blocksworld)
  (:objects {' '.join(blocks)})
  (:init
    {' '.join(initial_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""

        metadata = ProblemMetadata(
            problem_name=f"blocksworld-{size}-{problem_id:03d}",
            domain="blocksworld",
            size_category=size,
            characteristic_type=characteristic,
            num_objects=num_blocks,
            num_locations=num_stacks,
            complexity_score=complexity_score,
            predicted_solve_time_sec=predicted_time,
            generation_seed=self.seed + seed_offset * 1000,
        )

        return problem, metadata

    def _generate_goal(self, blocks: List[str], characteristic: str) -> List[str]:
        """Generate goal facts based on characteristic."""

        if characteristic == "single_tower":
            facts = [f"(on-table {blocks[-1]})"]
            for i in range(len(blocks) - 1):
                facts.append(f"(on {blocks[i]} {blocks[i + 1]})")
            return facts

        elif characteristic == "multiple_stacks":
            facts = []
            blocks_per_stack = max(1, len(blocks) // 2)
            for i in range(0, len(blocks), blocks_per_stack):
                stack = blocks[i:i + blocks_per_stack]
                if stack:
                    facts.append(f"(on-table {stack[-1]})")
                    for j in range(len(stack) - 1):
                        facts.append(f"(on {stack[j]} {stack[j + 1]})")
            return facts

        elif characteristic == "mixed_goal":
            facts = []
            for i, b in enumerate(blocks):
                if i % 3 == 0:
                    facts.append(f"(on-table {b})")
                elif i % 3 == 1 and i > 0:
                    facts.append(f"(on {b} {blocks[i - 1]})")
                else:
                    facts.append(f"(on-table {b})")
            return facts

        elif characteristic == "scattered_goal":
            facts = [f"(on-table {blocks[-1]})"]
            for i in range(len(blocks) - 1):
                if random.random() > 0.3:
                    target = random.choice(blocks[:i] if i > 0 else [blocks[-1]])
                    facts.append(f"(on {blocks[i]} {target})")
                else:
                    facts.append(f"(on-table {blocks[i]})")
            return facts

        else:  # pyramid
            facts = []
            level_width = max(1, len(blocks) // 3)
            blocks_copy = blocks.copy()

            while blocks_copy:
                level = blocks_copy[:level_width]
                blocks_copy = blocks_copy[level_width:]

                for i in range(len(level) - 1):
                    facts.append(f"(on {level[i]} {level[i + 1]})")
                if not blocks_copy:
                    for b in level:
                        facts.append(f"(on-table {b})")

            return facts


# ============================================================================
# LOGISTICS WITH DIFFICULTY TARGETING
# ============================================================================

class LogisticsGeneratorEnhanced(DomainProblemGenerator):
    """Enhanced Logistics with difficulty targeting."""

    SIZE_CONFIG = {
        "small": {
            "num_cities": (2, 3),
            "locations_per_city": (2, 3),
            "num_packages": (3, 6),
        },
        "medium": {
            "num_cities": (3, 4),
            "locations_per_city": (2, 4),
            "num_packages": (6, 12),
        },
        "large": {
            "num_cities": (4, 6),
            "locations_per_city": (3, 5),
            "num_packages": (10, 18),
        },
    }

    CHARACTERISTICS = {
        "single_source": "All packages from one city",
        "single_destination": "All packages to one destination",
        "inter_city": "Packages moved between cities (requires transport)",
        "distributed": "Packages scattered across sources and destinations",
        "clustered": "All packages go to clustered locations",
    }

    def get_domain_pddl(self) -> str:
        """Return simplified Logistics domain."""
        return """(define (domain logistics)
  (:requirements :strips :typing)
  (:types truck location object city)
  (:predicates
    (in ?obj - object ?truck - truck)
    (at ?truck - truck ?loc - location)
    (at-obj ?obj - object ?loc - location)
    (connected ?from ?to - location)
    (in-city ?loc - location ?city - city)
  )

  (:action load-truck
    :parameters (?obj - object ?truck - truck ?loc - location)
    :precondition (and (at-obj ?obj ?loc) (at ?truck ?loc))
    :effect (and (in ?obj ?truck) (not (at-obj ?obj ?loc)))
  )

  (:action unload-truck
    :parameters (?obj - object ?truck - truck ?loc - location)
    :precondition (and (in ?obj ?truck) (at ?truck ?loc))
    :effect (and (not (in ?obj ?truck)) (at-obj ?obj ?loc))
  )

  (:action drive-truck
    :parameters (?truck - truck ?from ?to - location)
    :precondition (and (at ?truck ?from) (connected ?from ?to))
    :effect (and (at ?truck ?to) (not (at ?truck ?from)))
  )
)
"""

    def generate_problem_with_params(
            self,
            size: str,
            characteristic: str,
            problem_id: int,
            seed_offset: int,
            **custom_params
    ) -> Tuple[str, ProblemMetadata]:
        """Generate Logistics problem."""

        random.seed(self.seed + seed_offset * 1000)

        num_cities = random.randint(*custom_params.get('num_cities', self.SIZE_CONFIG[size]["num_cities"]))
        locations_per_city = random.randint(
            *custom_params.get('locations_per_city', self.SIZE_CONFIG[size]["locations_per_city"]))
        num_packages = random.randint(*custom_params.get('num_packages', self.SIZE_CONFIG[size]["num_packages"]))

        cities = [f"city{i}" for i in range(num_cities)]
        locations = []
        trucks = []
        packages = [f"pkg{i}" for i in range(num_packages)]
        num_locations = 0

        for city_idx, city in enumerate(cities):
            for loc_idx in range(locations_per_city):
                loc_name = f"loc-{city}-{loc_idx}"
                locations.append((loc_name, city))
                num_locations += 1
            trucks.append((f"truck-{city}", city))

        # Complexity factor
        char_factors = {
            "single_source": 0.3,
            "single_destination": 0.2,
            "inter_city": 0.7,
            "distributed": 0.6,
            "clustered": 0.5,
        }
        complexity_score = char_factors.get(characteristic, 0.5)

        # Predict difficulty
        predicted_time = self.predictor.predict_logistics(
            num_packages, num_cities, num_locations, complexity_score
        )

        # Generate initial state
        initial_facts = []
        if characteristic == "single_source":
            source_loc = f"loc-{cities[0]}-0"
            for pkg in packages:
                initial_facts.append(f"(at-obj {pkg} {source_loc})")
        else:
            for pkg in packages:
                loc = random.choice(locations)
                initial_facts.append(f"(at-obj {pkg} {loc[0]})")

        for truck, city in trucks:
            home_loc = f"loc-{city}-0"
            initial_facts.append(f"(at {truck} {home_loc})")

        # Connect locations
        for city in cities:
            city_locs = [loc for loc, c in locations if c == city]
            for i in range(len(city_locs) - 1):
                initial_facts.append(f"(connected {city_locs[i]} {city_locs[i + 1]})")
                initial_facts.append(f"(connected {city_locs[i + 1]} {city_locs[i]})")

        # Goals
        goal_facts = []
        if characteristic == "single_destination":
            dest_loc = f"loc-{random.choice(cities)}-0"
            for pkg in packages:
                goal_facts.append(f"(at-obj {pkg} {dest_loc})")
        else:
            for pkg in packages:
                goal_facts.append(f"(at-obj {pkg} {random.choice(locations)[0]})")

        loc_str = " ".join([loc for loc, _ in locations])
        truck_str = " ".join([truck for truck, _ in trucks])
        city_str = " ".join(cities)
        pkg_str = " ".join(packages)

        problem = f"""(define (problem logistics-{size}-{problem_id:03d})
  (:domain logistics)
  (:objects
    {truck_str} - truck
    {loc_str} - location
    {pkg_str} - object
    {city_str} - city
  )
  (:init
    {' '.join(initial_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""

        metadata = ProblemMetadata(
            problem_name=f"logistics-{size}-{problem_id:03d}",
            domain="logistics",
            size_category=size,
            characteristic_type=characteristic,
            num_objects=num_packages,
            num_locations=num_locations,
            complexity_score=complexity_score,
            predicted_solve_time_sec=predicted_time,
            generation_seed=self.seed + seed_offset * 1000,
        )

        return problem, metadata


# ============================================================================
# PARKING WITH DIFFICULTY TARGETING
# ============================================================================

class ParkingGeneratorEnhanced(DomainProblemGenerator):
    """Enhanced Parking with difficulty targeting."""

    SIZE_CONFIG = {
        "small": {
            "num_curbs": (3, 5),
            "num_cars": (2, 4),
        },
        "medium": {
            "num_curbs": (4, 7),
            "num_cars": (4, 7),
        },
        "large": {
            "num_curbs": (6, 10),
            "num_cars": (6, 10),
        },
    }

    CHARACTERISTICS = {
        "swapping": "Cars need to swap positions",
        "clearing": "Clear specific parking spots",
        "blocked": "Some cars are initially blocked",
        "sequential": "Sequential parking required",
    }

    def get_domain_pddl(self) -> str:
        """Return Parking domain."""
        return """(define (domain parking)
  (:requirements :strips :typing :action-costs)
  (:types car curb)
  (:predicates
    (at-curb ?car - car)
    (at-curb-num ?car - car ?curb - curb)
    (behind-car ?car ?front-car - car)
    (car-clear ?car - car)
    (curb-clear ?curb - curb)
  )
  (:functions (total-cost) - number)

  (:action move-curb-to-curb
    :parameters (?car - car ?curbsrc ?curbdest - curb)
    :precondition (and
      (car-clear ?car)
      (curb-clear ?curbdest)
      (at-curb-num ?car ?curbsrc)
      (not (= ?curbsrc ?curbdest))
    )
    :effect (and
      (not (curb-clear ?curbdest))
      (curb-clear ?curbsrc)
      (at-curb-num ?car ?curbdest)
      (not (at-curb-num ?car ?curbsrc))
      (increase (total-cost) 1)
    )
  )
)
"""

    def generate_problem_with_params(
            self,
            size: str,
            characteristic: str,
            problem_id: int,
            seed_offset: int,
            **custom_params
    ) -> Tuple[str, ProblemMetadata]:
        """Generate Parking problem."""

        random.seed(self.seed + seed_offset * 1000)

        num_curbs = random.randint(*custom_params.get('num_curbs', self.SIZE_CONFIG[size]["num_curbs"]))
        num_cars = random.randint(*custom_params.get('num_cars', self.SIZE_CONFIG[size]["num_cars"]))

        cars = [f"car{i}" for i in range(num_cars)]
        curbs = [f"curb{i}" for i in range(num_curbs)]

        # Complexity factor
        char_factors = {
            "swapping": 0.6,
            "clearing": 0.5,
            "blocked": 0.8,
            "sequential": 0.4,
        }
        complexity_score = char_factors.get(characteristic, 0.5)

        # Predict difficulty
        predicted_time = self.predictor.predict_parking(
            num_cars, num_curbs, complexity_score
        )

        # Initial state (simple placement)
        initial_facts = []
        for i, car in enumerate(cars):
            curb = curbs[i % len(curbs)]
            initial_facts.append(f"(at-curb-num {car} {curb})")
            initial_facts.append(f"(car-clear {car})")

        # Clear unoccupied curbs
        occupied = set(curbs[i % len(curbs)] for i in range(num_cars))
        for curb in curbs:
            if curb not in occupied:
                initial_facts.append(f"(curb-clear {curb})")

        # Goal: shuffle cars
        goal_facts = []
        for i, car in enumerate(cars):
            target_curb = curbs[(i + 1) % len(curbs)]
            goal_facts.append(f"(at-curb-num {car} {target_curb})")

        problem = f"""(define (problem parking-{size}-{problem_id:03d})
  (:domain parking)
  (:objects {' '.join(cars)} - car {' '.join(curbs)} - curb)
  (:init
    {' '.join(initial_facts)}
  )
  (:goal (and
    {' '.join(goal_facts)}
  ))
)
"""

        metadata = ProblemMetadata(
            problem_name=f"parking-{size}-{problem_id:03d}",
            domain="parking",
            size_category=size,
            characteristic_type=characteristic,
            num_objects=num_cars,
            num_locations=num_curbs,
            complexity_score=complexity_score,
            predicted_solve_time_sec=predicted_time,
            generation_seed=self.seed + seed_offset * 1000,
        )

        return problem, metadata


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class EnhancedProblemGenerator:
    """Main generator with difficulty targeting."""

    GENERATORS = {
        "blocksworld": BlocksworldGeneratorEnhanced,
        "logistics": LogisticsGeneratorEnhanced,
        "parking": ParkingGeneratorEnhanced,
    }

    def __init__(self, output_dir: str = "benchmarks", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def generate_all(self, problems_per_size: int = 15,
                     domains_to_generate: List[str] = None,
                     max_generation_attempts: int = 50):
        """
        Generate all problems with difficulty targeting.

        Uses multiple attempts to hit time targets.
        """

        if domains_to_generate is None:
            domains_to_generate = list(self.GENERATORS.keys())

        for domain_name in domains_to_generate:
            if domain_name not in self.GENERATORS:
                logger.warning(f"Skipping unknown domain: {domain_name}")
                continue

            generator_class = self.GENERATORS[domain_name]
            logger.info(f"\n{'=' * 80}")
            logger.info(f"GENERATING {domain_name.upper()}")
            logger.info(f"{'=' * 80}")

            gen = generator_class(domain_name, seed=self.seed)

            for size in ["small", "medium", "large"]:
                size_dir = self.output_dir / domain_name / size
                size_dir.mkdir(parents=True, exist_ok=True)

                # Save domain
                domain_file = size_dir / "domain.pddl"
                with open(domain_file, 'w') as f:
                    f.write(gen.get_domain_pddl())
                logger.info(f"✓ Saved domain: {domain_file}")

                # Generate problems
                characteristics = list(gen.CHARACTERISTICS.keys())
                problems_generated = 0
                problem_id = 0
                attempts = 0

                logger.info(f"\nGenerating {size} problems (target: {problems_per_size})...")

                while problems_generated < problems_per_size and attempts < max_generation_attempts * problems_per_size:
                    # Round-robin through characteristics
                    characteristic = characteristics[attempts % len(characteristics)]

                    # Try to generate
                    result = gen.generate_problem(
                        size=size,
                        characteristic=characteristic,
                        problem_id=problem_id,
                        seed_offset=attempts
                    )

                    if result is not None:
                        problem_pddl, metadata = result

                        # Save problem and metadata
                        problem_file = size_dir / f"problem_{size}_{problem_id:03d}.pddl"
                        with open(problem_file, 'w') as f:
                            f.write(problem_pddl)

                        metadata_file = size_dir / f"problem_{size}_{problem_id:03d}.json"
                        metadata.save(str(metadata_file))

                        logger.info(f"  ✓ problem_{size}_{problem_id:03d} "
                                    f"({characteristic}, predicted {metadata.predicted_solve_time_sec:.1f}s)")

                        problems_generated += 1
                        problem_id += 1

                    attempts += 1

                logger.info(f"✓ Generated {problems_generated}/{problems_per_size} {size} problems")
                if problems_generated < problems_per_size:
                    logger.warning(f"⚠️ Could only generate {problems_generated}/{problems_per_size}")


if __name__ == "__main__":
    gen = EnhancedProblemGenerator(output_dir="benchmarks", seed=42)
    gen.generate_all(
        problems_per_size=15,
        domains_to_generate=["blocksworld", "logistics", "parking"]
    )
    logger.info("\n✅ Generation complete!")