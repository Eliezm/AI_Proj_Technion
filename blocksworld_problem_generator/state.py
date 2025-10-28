"""
State representation for Blocksworld and state validation.

A valid Blocksworld state consists of:
- on_table: set of blocks on the table
- on: dict mapping block -> block (stacking relations)
- clear: set of blocks with nothing on top (and not held)
- arm_empty: boolean
- holding: block or None
"""

from typing import Set, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BlocksWorldState:
    """Represents a Blocksworld state."""
    blocks: Set[str]
    on_table: Set[str] = field(default_factory=set)
    on: Dict[str, str] = field(default_factory=dict)  # block -> block it's on
    clear: Set[str] = field(default_factory=set)
    arm_empty: bool = True
    holding: Optional[str] = None

    def copy(self) -> 'BlocksWorldState':
        """Create a deep copy of the state."""
        return BlocksWorldState(
            blocks=self.blocks.copy(),
            on_table=self.on_table.copy(),
            on=self.on.copy(),
            clear=self.clear.copy(),
            arm_empty=self.arm_empty,
            holding=self.holding
        )

    def is_valid(self) -> Tuple[bool, Optional[str]]:
        """
        Validate state against Blocksworld constraints.

        Returns:
            (is_valid, error_message)
        """
        # Check: Each block must be exactly one of: on_table, on another block, or held
        for block in self.blocks:
            on_table = block in self.on_table
            on_another = block in self.on
            held = block == self.holding

            count = sum([on_table, on_another, held])
            if count != 1:
                return False, f"Block {block} is in {count} positions (should be exactly 1)"

        # Check: Arm state consistency
        if self.arm_empty and self.holding is not None:
            return False, "Arm cannot be both empty and holding a block"
        if not self.arm_empty and self.holding is None:
            return False, "Arm is not empty but not holding any block"
        if self.holding is not None and self.holding not in self.blocks:
            return False, f"Arm is holding non-existent block {self.holding}"

        # Check: on relations point to existing blocks
        for on_block, under_block in self.on.items():
            if under_block not in self.blocks:
                return False, f"Block {on_block} claims to be on non-existent block {under_block}"
            if under_block == on_block:
                return False, f"Block {on_block} cannot be on itself"

        # Check: Clear predicate validity
        # FIX: A block is clear if it's NOT held AND nothing is on top of it
        for block in self.blocks:
            is_clear = (self.holding != block and block not in self.on.values())
            should_be_clear = block in self.clear
            if is_clear != should_be_clear:
                return False, f"Clear predicate mismatch for block {block} (is_clear={is_clear}, in_clear_set={should_be_clear})"

        return True, None

    def __hash__(self):
        """Make state hashable for deduplication."""
        on_tuple = tuple(sorted((k, v) for k, v in self.on.items()))
        return hash((
            frozenset(self.on_table),
            on_tuple,
            frozenset(self.clear),
            self.arm_empty,
            self.holding
        ))

    def __eq__(self, other):
        """Check state equality."""
        if not isinstance(other, BlocksWorldState):
            return False
        return (
                self.on_table == other.on_table and
                self.on == other.on and
                self.clear == other.clear and
                self.arm_empty == other.arm_empty and
                self.holding == other.holding
        )

    def __repr__(self):
        parts = []
        if self.on_table:
            parts.append(f"on_table={self.on_table}")
        if self.on:
            parts.append(f"on={self.on}")
        if self.clear:
            parts.append(f"clear={self.clear}")
        parts.append(f"arm_empty={self.arm_empty}")
        if self.holding:
            parts.append(f"holding={self.holding}")
        return f"State({', '.join(parts)})"


def create_empty_state(block_names: list) -> BlocksWorldState:
    """
    Create an empty initial state: all blocks on table, arm empty, all clear.

    Requirement #14: Start with a valid state.
    """
    state = BlocksWorldState(
        blocks=set(block_names),
        on_table=set(block_names),
        on={},
        clear=set(block_names),
        arm_empty=True,
        holding=None
    )
    is_valid, error = state.is_valid()
    if not is_valid:
        raise ValueError(f"Failed to create empty state: {error}")
    return state