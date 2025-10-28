"""
Goal archetypes for Blocksworld.

Defines different types of goals to ensure structural diversity (Requirement #2).
Active archetype selection ensures variety in problem structures.
"""

import random
from typing import List, Tuple, Callable
from enum import Enum
from state import BlocksWorldState


class GoalArchetype(Enum):
    """Different goal archetype types."""
    SINGLE_TOWER = "single_tower"
    MULTIPLE_TOWERS = "multiple_towers"
    CLEAR_TABLE = "clear_table"
    SCATTERED_RELATIONS = "scattered_relations"
    MIXED_PYRAMID = "mixed_pyramid"


class GoalArchetypeGenerator:
    """
    Generates goal states according to different archetypes.

    Requirement #4: Active sampling of goal archetypes to guarantee variety.
    """

    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    def generate_single_tower(self, blocks: List[str]) -> BlocksWorldState:
        """
        Archetype: Single tall tower.
        All blocks stacked into one tall tower.
        """
        state = BlocksWorldState(
            blocks=set(blocks),
            on_table={blocks[0]},
            on={},
            clear=set(),
            arm_empty=True,
            holding=None
        )

        # Stack remaining blocks on top
        for i in range(1, len(blocks)):
            state.on[blocks[i]] = blocks[i - 1]

        state.clear.add(blocks[-1])

        is_valid, error = state.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid single tower state: {error}")
        return state

    def generate_multiple_towers(self, blocks: List[str], num_towers: int = 2) -> BlocksWorldState:
        """
        Archetype: Multiple towers.
        Distribute blocks among num_towers separate stacks.
        """
        num_towers = min(num_towers, len(blocks))

        state = BlocksWorldState(
            blocks=set(blocks),
            on_table=set(),
            on={},
            clear=set(),
            arm_empty=True,
            holding=None
        )

        # Distribute blocks among towers
        tower_assignments = [[] for _ in range(num_towers)]
        for i, block in enumerate(blocks):
            tower_assignments[i % num_towers].append(block)

        # Build each tower
        for tower_blocks in tower_assignments:
            if tower_blocks:
                state.on_table.add(tower_blocks[0])
                for i in range(1, len(tower_blocks)):
                    state.on[tower_blocks[i]] = tower_blocks[i - 1]
                state.clear.add(tower_blocks[-1])

        is_valid, error = state.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid multiple towers state: {error}")
        return state

    def generate_clear_table(self, blocks: List[str]) -> BlocksWorldState:
        """
        Archetype: All blocks on table.
        No stacking; all blocks sitting on the table, all clear.
        """
        state = BlocksWorldState(
            blocks=set(blocks),
            on_table=set(blocks),
            on={},
            clear=set(blocks),
            arm_empty=True,
            holding=None
        )

        is_valid, error = state.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid clear table state: {error}")
        return state

    def generate_scattered_relations(self, blocks: List[str]) -> BlocksWorldState:
        """
        Archetype: Scattered on/on-table relations.
        Create a few random on/on-table relations without forming complete towers.
        """
        state = BlocksWorldState(
            blocks=set(blocks),
            on_table=set(),
            on={},
            clear=set(),
            arm_empty=True,
            holding=None
        )

        # Randomly decide which blocks are on the table
        shuffled = blocks.copy()
        random.shuffle(shuffled)
        base_blocks = set(shuffled[: max(1, len(blocks) // 2)])

        state.on_table = base_blocks.copy()
        state.clear = base_blocks.copy()

        # Place remaining blocks on random base blocks
        non_base = [b for b in blocks if b not in base_blocks]
        for block in non_base:
            if state.clear:  # Only place if there's a clear block
                under = random.choice(list(state.clear))
                state.on[block] = under
                state.clear.discard(under)

        state.clear.update([b for b in blocks if b not in state.on.values()])

        is_valid, error = state.is_valid()
        if not is_valid:
            # Fallback to simple valid state
            return self.generate_multiple_towers(blocks, 2)
        return state

    def generate_mixed_pyramid(self, blocks: List[str]) -> BlocksWorldState:
        """
        Archetype: Pyramid-like structure.
        Create a structure with decreasing blocks per level.
        """
        if len(blocks) < 3:
            return self.generate_multiple_towers(blocks, 2)

        state = BlocksWorldState(
            blocks=set(blocks),
            on_table=set(),
            on={},
            clear=set(),
            arm_empty=True,
            holding=None
        )

        shuffled = blocks.copy()
        random.shuffle(shuffled)

        # First block(s) on table
        state.on_table.add(shuffled[0])
        idx = 1

        # Stack blocks in a pyramid
        for i in range(1, len(shuffled)):
            if i % 2 == 0:  # Decide structure
                state.on[shuffled[i]] = shuffled[i - 1]
            else:
                if len(state.on_table) < max(1, len(blocks) // 3):
                    state.on_table.add(shuffled[i])
                else:
                    state.on[shuffled[i]] = shuffled[i - 1]

        # Update clear
        state.clear = {b for b in blocks if b not in state.on.values()}

        is_valid, error = state.is_valid()
        if not is_valid:
            return self.generate_multiple_towers(blocks, 2)
        return state

    def generate_archetype(self, archetype: GoalArchetype, blocks: List[str]) -> BlocksWorldState:
        """Generate a goal state for the given archetype."""
        if archetype == GoalArchetype.SINGLE_TOWER:
            return self.generate_single_tower(blocks)
        elif archetype == GoalArchetype.MULTIPLE_TOWERS:
            num_towers = random.randint(2, max(2, len(blocks) // 2))
            return self.generate_multiple_towers(blocks, num_towers)
        elif archetype == GoalArchetype.CLEAR_TABLE:
            return self.generate_clear_table(blocks)
        elif archetype == GoalArchetype.SCATTERED_RELATIONS:
            return self.generate_scattered_relations(blocks)
        elif archetype == GoalArchetype.MIXED_PYRAMID:
            return self.generate_mixed_pyramid(blocks)
        else:
            raise ValueError(f"Unknown archetype: {archetype}")

    def generate_random_archetype(self, blocks: List[str]) -> Tuple[BlocksWorldState, GoalArchetype]:
        """Generate a random archetype (Requirement #4)."""
        archetype = random.choice(list(GoalArchetype))
        state = self.generate_archetype(archetype, blocks)
        return state, archetype