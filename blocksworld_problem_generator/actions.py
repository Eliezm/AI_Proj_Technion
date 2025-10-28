"""
Action definitions and execution for Blocksworld.

Implements forward and reverse execution of the four actions:
1. pickup ?ob (from table)
2. putdown ?ob (onto table)
3. stack ?ob ?underob (onto another block)
4. unstack ?ob ?underob (from another block)
"""

from typing import List, Optional, Tuple
from enum import Enum
from state import BlocksWorldState


class ActionType(Enum):
    PICKUP = "pickup"
    PUTDOWN = "putdown"
    STACK = "stack"
    UNSTACK = "unstack"


class Action:
    """Represents an action in Blocksworld."""

    def __init__(self, action_type: ActionType, params: List[str]):
        self.action_type = action_type
        self.params = params

    def __repr__(self):
        return f"{self.action_type.value}({', '.join(self.params)})"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.action_type == other.action_type and self.params == other.params

    def __hash__(self):
        return hash((self.action_type, tuple(self.params)))


class ActionExecutor:
    """Executes actions on Blocksworld states."""

    @staticmethod
    def can_execute_pickup(state: BlocksWorldState, ob: str) -> bool:
        """
        Check preconditions for pickup:
        - arm-empty
        - on-table ?ob
        - clear ?ob
        """
        return (
                state.arm_empty and
                ob in state.on_table and
                ob in state.clear
        )

    @staticmethod
    def execute_pickup(state: BlocksWorldState, ob: str) -> Optional[BlocksWorldState]:
        """Execute pickup action."""
        if not ActionExecutor.can_execute_pickup(state, ob):
            return None

        new_state = state.copy()
        new_state.arm_empty = False
        new_state.holding = ob
        new_state.on_table.discard(ob)
        new_state.clear.discard(ob)

        is_valid, error = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_putdown(state: BlocksWorldState, ob: str) -> bool:
        """
        Check preconditions for putdown:
        - holding ?ob
        """
        return state.holding == ob

    @staticmethod
    def execute_putdown(state: BlocksWorldState, ob: str) -> Optional[BlocksWorldState]:
        """Execute putdown action."""
        if not ActionExecutor.can_execute_putdown(state, ob):
            return None

        new_state = state.copy()
        new_state.arm_empty = True
        new_state.holding = None
        new_state.on_table.add(ob)
        new_state.clear.add(ob)

        is_valid, error = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_stack(state: BlocksWorldState, ob: str, underob: str) -> bool:
        """
        Check preconditions for stack:
        - holding ?ob
        - clear ?underob
        - ob != underob
        """
        return (
                state.holding == ob and
                underob in state.clear and
                ob != underob
        )

    @staticmethod
    def execute_stack(state: BlocksWorldState, ob: str, underob: str) -> Optional[BlocksWorldState]:
        """Execute stack action."""
        if not ActionExecutor.can_execute_stack(state, ob, underob):
            return None

        new_state = state.copy()
        new_state.arm_empty = True
        new_state.holding = None
        new_state.on[ob] = underob
        new_state.clear.add(ob)
        new_state.clear.discard(underob)

        is_valid, error = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def can_execute_unstack(state: BlocksWorldState, ob: str, underob: str) -> bool:
        """
        Check preconditions for unstack:
        - on ?ob ?underob
        - clear ?ob
        - arm-empty
        """
        return (
                state.on.get(ob) == underob and
                ob in state.clear and
                state.arm_empty
        )

    @staticmethod
    def execute_unstack(state: BlocksWorldState, ob: str, underob: str) -> Optional[BlocksWorldState]:
        """Execute unstack action."""
        if not ActionExecutor.can_execute_unstack(state, ob, underob):
            return None

        new_state = state.copy()
        new_state.arm_empty = False
        new_state.holding = ob
        del new_state.on[ob]
        new_state.clear.discard(ob)
        new_state.clear.add(underob)

        is_valid, error = new_state.is_valid()
        return new_state if is_valid else None

    @staticmethod
    def execute_forward(state: BlocksWorldState, action: Action) -> Optional[BlocksWorldState]:
        """
        Execute an action in the forward direction.

        Returns the resulting state, or None if action is not applicable.
        """
        if action.action_type == ActionType.PICKUP:
            return ActionExecutor.execute_pickup(state, action.params[0])
        elif action.action_type == ActionType.PUTDOWN:
            return ActionExecutor.execute_putdown(state, action.params[0])
        elif action.action_type == ActionType.STACK:
            return ActionExecutor.execute_stack(state, action.params[0], action.params[1])
        elif action.action_type == ActionType.UNSTACK:
            return ActionExecutor.execute_unstack(state, action.params[0], action.params[1])
        else:
            return None

    @staticmethod
    def get_applicable_actions(state: BlocksWorldState) -> List[Action]:
        """Get all applicable actions in the current state (forward direction)."""
        applicable = []

        # Pickup actions
        for block in state.blocks:
            if ActionExecutor.can_execute_pickup(state, block):
                applicable.append(Action(ActionType.PICKUP, [block]))

        # Putdown actions
        if state.holding:
            if ActionExecutor.can_execute_putdown(state, state.holding):
                applicable.append(Action(ActionType.PUTDOWN, [state.holding]))

        # Stack actions
        if state.holding:
            for underob in state.blocks:
                if underob != state.holding:
                    if ActionExecutor.can_execute_stack(state, state.holding, underob):
                        applicable.append(Action(ActionType.STACK, [state.holding, underob]))

        # Unstack actions
        if state.arm_empty:
            for ob, underob in state.on.items():
                if ActionExecutor.can_execute_unstack(state, ob, underob):
                    applicable.append(Action(ActionType.UNSTACK, [ob, underob]))

        return applicable