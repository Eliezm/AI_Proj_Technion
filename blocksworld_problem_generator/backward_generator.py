"""
Core backward search generator for problem generation.

Requirement #14: Generate problems using backward state-space search.
- Start with a structurally diverse goal state
- Apply valid actions in reverse
- Walk back to a derived initial state
"""

import random
from typing import List, Tuple, Optional
from state import BlocksWorldState, create_empty_state
from actions import Action, ActionType, ActionExecutor
from goal_archetypes import GoalArchetypeGenerator, GoalArchetype


class ReverseActionExecutor:
    """
    Executes actions in reverse to generate problems.

    For each forward action, we implement its reverse:
    - pickup (forward) <-> putdown (reverse)
    - stack (forward) <-> unstack (reverse)
    - unstack (forward) <-> pickup (reverse)
    - putdown (forward) <-> stack (reverse)
    """

    @staticmethod
    def reverse_pickup(state: BlocksWorldState, ob: str) -> Optional[BlocksWorldState]:
        """
        Reverse of pickup: arm was holding block, put it back on table.
        This is equivalent to forward putdown.
        """
        return ActionExecutor.execute_putdown(state, ob)

    @staticmethod
    def reverse_putdown(state: BlocksWorldState, ob: str) -> Optional[BlocksWorldState]:
        """
        Reverse of putdown: arm was empty, pick up the block from table.
        This is equivalent to forward pickup.
        """
        return ActionExecutor.execute_pickup(state, ob)

    @staticmethod
    def reverse_stack(state: BlocksWorldState, ob: str, underob: str) -> Optional[BlocksWorldState]:
        """
        Reverse of stack: unstack the block.
        This is equivalent to forward unstack.
        """
        return ActionExecutor.execute_unstack(state, ob, underob)

    @staticmethod
    def reverse_unstack(state: BlocksWorldState, ob: str, underob: str) -> Optional[BlocksWorldState]:
        """
        Reverse of unstack: arm was empty, stack the block.
        This is equivalent to forward stack.
        """
        return ActionExecutor.execute_stack(state, ob, underob)

    @staticmethod
    def execute_reverse(state: BlocksWorldState, action: Action) -> Optional[BlocksWorldState]:
        """Execute an action in reverse."""
        if action.action_type == ActionType.PICKUP:
            return ReverseActionExecutor.reverse_pickup(state, action.params[0])
        elif action.action_type == ActionType.PUTDOWN:
            return ReverseActionExecutor.reverse_putdown(state, action.params[0])
        elif action.action_type == ActionType.STACK:
            return ReverseActionExecutor.reverse_stack(state, action.params[0], action.params[1])
        elif action.action_type == ActionType.UNSTACK:
            return ReverseActionExecutor.reverse_unstack(state, action.params[0], action.params[1])
        else:
            return None

    @staticmethod
    def get_applicable_reverse_actions(state: BlocksWorldState) -> List[Action]:
        """
        Get all actions that can be applied in reverse from this state.

        An action can be reversed if its reverse application produces a
        valid new state that is different from the current state.

        Requirement #14: Ensure backward search actually progresses.
        """
        applicable = []

        # Try all possible actions
        for block in state.blocks:
            # Try reverse_pickup (applies putdown)
            result = ReverseActionExecutor.reverse_pickup(state, block)
            if result is not None and result != state:
                # Avoid cycles
                applicable.append(Action(ActionType.PICKUP, [block]))

            # Try reverse_putdown (applies pickup)
            result = ReverseActionExecutor.reverse_putdown(state, block)
            if result is not None and result != state:
                applicable.append(Action(ActionType.PUTDOWN, [block]))

            # Try reverse stack/unstack with all other blocks
            for other_block in state.blocks:
                if block != other_block:
                    # Try reverse_stack (applies unstack)
                    result = ReverseActionExecutor.reverse_stack(state, block, other_block)
                    if result is not None and result != state:
                        applicable.append(Action(ActionType.STACK, [block, other_block]))

                    # Try reverse_unstack (applies stack)
                    result = ReverseActionExecutor.reverse_unstack(state, block, other_block)
                    if result is not None and result != state:
                        applicable.append(Action(ActionType.UNSTACK, [block, other_block]))

        # Remove duplicates while preserving order
        seen = set()
        unique_applicable = []
        for action in applicable:
            action_tuple = (action.action_type, tuple(action.params))
            if action_tuple not in seen:
                seen.add(action_tuple)
                unique_applicable.append(action)

        return unique_applicable


class BackwardProblemGenerator:
    """
    Generate Blocksworld problems using backward state-space search.

    Algorithm:
    1. Select a goal archetype
    2. Generate a valid goal state
    3. Perform N random valid reverse actions from the goal
    4. Result: derived initial state + path = known plan

    Requirement #14: Guaranteed solvability and known plan cost.
    """

    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
        self.archetype_gen = GoalArchetypeGenerator(random_seed)
        if random_seed is not None:
            random.seed(random_seed)

    def generate_problem(
            self,
            num_blocks: int,
            target_plan_length: int,
            archetype: Optional[GoalArchetype] = None,
            tolerance: int = 1
    ) -> Tuple[BlocksWorldState, BlocksWorldState, List[Action], GoalArchetype]:
        """
        Generate a problem using backward search.

        Args:
            num_blocks: Number of blocks to use
            target_plan_length: Desired length of the plan (path)
            archetype: Goal archetype (random if None)
            tolerance: Allowed deviation from target length

        Returns:
            (initial_state, goal_state, plan, archetype_used)

        Requirement #15: Inherent guarantees (solvability and known plan).
        """
        block_names = [f"b{i}" for i in range(num_blocks)]

        # Step 1: Generate or select goal state (Req #3, #4)
        if archetype is None:
            goal_state, archetype = self.archetype_gen.generate_random_archetype(block_names)
        else:
            goal_state = self.archetype_gen.generate_archetype(archetype, block_names)

        # Verify goal is valid (Req #14)
        is_valid, error = goal_state.is_valid()
        if not is_valid:
            raise ValueError(f"Generated invalid goal state: {error}")

        # Step 2: Backward search from goal state
        current_state = goal_state.copy()
        plan = []
        max_attempts = target_plan_length + tolerance + 10  # Safety margin

        while len(plan) < target_plan_length and max_attempts > 0:
            # Get all reverse actions applicable in current state
            reverse_actions = ReverseActionExecutor.get_applicable_reverse_actions(current_state)

            if not reverse_actions:
                # No more actions available; stop here
                break

            # Randomly select a reverse action
            action = random.choice(reverse_actions)
            new_state = ReverseActionExecutor.execute_reverse(current_state, action)

            if new_state is None:
                max_attempts -= 1
                continue

            is_valid, error = new_state.is_valid()
            if not is_valid:
                max_attempts -= 1
                continue

            # Record the action in reverse order (for forward execution)
            plan.insert(0, action)
            current_state = new_state
            max_attempts -= 1

        # Step 3: Validate plan length
        if not (target_plan_length - tolerance <= len(plan) <= target_plan_length + tolerance):
            # Plan length out of target range; return anyway but note the deviation
            pass

        initial_state = current_state

        # Requirement #15: Verify the plan is valid
        if not self._verify_plan(initial_state, goal_state, plan):
            raise RuntimeError("Generated plan does not reach goal from initial state")

        return initial_state, goal_state, plan, archetype

    def _verify_plan(
            self,
            initial_state: BlocksWorldState,
            goal_state: BlocksWorldState,
            plan: List[Action]
    ) -> bool:
        """
        Verify that the plan reaches goal from initial state.
        """
        current = initial_state.copy()
        for action in plan:
            current = ActionExecutor.execute_forward(current, action)
            if current is None:
                return False
        return current == goal_state