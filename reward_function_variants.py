# -*- coding: utf-8 -*-
"""
REWARD FUNCTION VARIANTS WITH BAD MERGE DETECTION
==================================================

This file contains multiple reward function implementations for merge strategy learning.
Each variant emphasizes different aspects of merge quality, with enhanced bad merge detection.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from reward_info_extractor import MergeInfo

logger = logging.getLogger(__name__)


class RewardFunctionBase:
    """Base class for all reward functions with bad merge detection."""

    def __init__(self, name: str):
        self.name = name
        self.component_values = {}
        self.bad_merge_reasons: List[str] = []

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """
        Computes reward for a merge.

        Args:
            merge_info: Extracted merge information
            search_expansions: Number of states expanded so far
            plan_cost: Current plan cost (if terminal)
            is_terminal: Whether this is the final merge

        Returns:
            Scalar reward value (may be heavily penalized if bad merge detected)
        """
        raise NotImplementedError

    def get_components_dict(self) -> Dict[str, float]:
        """Returns the constituent components of the reward for logging."""
        return self.component_values.copy()

    def _log_bad_merge_detected(self, reason: str) -> None:
        """Log a bad merge detection event."""
        self.bad_merge_reasons.append(reason)
        logger.warning(f"  ⚠️  BAD MERGE DETECTED: {reason}")


class SimpleStabilityReward(RewardFunctionBase):
    """VARIANT 1: Simple reward - penalizes F-value changes and state explosion."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, lambda_shrink: float = 0.02,
                 f_threshold: float = 5.0):
        super().__init__("SimpleStabilityReward")
        self.alpha = alpha
        self.beta = beta
        self.lambda_shrink = lambda_shrink
        self.f_threshold = f_threshold

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Compute reward with bad merge detection."""

        logger.info(f"\n[REWARD] Computing {self.name}")

        # Component 1: F-value stability
        f_stability_term = self.alpha * merge_info.f_value_stability
        logger.info(f"  [1] F-stability: {f_stability_term:.4f}")

        # Component 2: State explosion
        state_explosion = merge_info.state_explosion_penalty
        state_term = -self.lambda_shrink * state_explosion
        logger.info(f"  [2] State explosion penalty: {state_term:.4f}")

        # Component 3: Search effort
        if search_expansions > 0:
            exp_norm = min(search_expansions / 200_000.0, 1.0)
            exp_term = -self.beta * exp_norm
        else:
            exp_term = 0.0
        logger.info(f"  [3] Search effort: {exp_term:.4f}")

        # Combine
        reward = f_stability_term + state_term + exp_term
        logger.info(f"  Base reward: {reward:.4f}")

        # ✅ BAD MERGE DETECTION
        bad_merge_penalty = self._detect_bad_merges(merge_info)
        reward += bad_merge_penalty

        if bad_merge_penalty < 0:
            logger.warning(f"  ⚠️  Applied bad merge penalty: {bad_merge_penalty:.4f}")
            logger.warning(f"  Final reward: {reward:.4f}")

        self.component_values = {
            'f_stability': f_stability_term,
            'state_explosion': state_term,
            'search_effort': exp_term,
            'bad_merge_penalty': bad_merge_penalty,
            'total': reward
        }

        return reward

    def _detect_bad_merges(self, merge_info: MergeInfo) -> float:
        """✅ NEW: Detect bad merges and apply penalties."""
        penalty = 0.0

        # CHECK 1: State explosion (TS1_size × TS2_size explosion not controlled)
        expected_merged_size = merge_info.ts1_size * merge_info.ts2_size
        if merge_info.states_after > expected_merged_size * 1.2:
            # Shrinking didn't work as expected
            explosion_ratio = (merge_info.states_after - expected_merged_size) / max(expected_merged_size, 1)
            explosion_penalty = -0.5 * min(explosion_ratio, 2.0)
            penalty += explosion_penalty
            self._log_bad_merge_detected(f"State explosion not controlled: {merge_info.states_after} vs {expected_merged_size}")

        # CHECK 2: F-stability degradation (low preservation)
        if merge_info.f_value_stability < 0.3:
            penalty -= 0.8
            self._log_bad_merge_detected(f"Critical F-value degradation: {merge_info.f_value_stability:.4f}")

        # CHECK 3: Many unreachable states
        reachable_count = sum(1 for f in merge_info.f_after if f != float('inf') and f < 1_000_000_000)
        unreachable_ratio = 1.0 - (reachable_count / max(merge_info.states_after, 1))
        if unreachable_ratio > 0.7:
            penalty -= 1.0
            self._log_bad_merge_detected(f"High unreachability: {unreachable_ratio*100:.1f}% unreachable")

        # CHECK 4: Significant F-value changes (unstable)
        if merge_info.num_significant_f_changes > merge_info.states_after * 0.5:
            penalty -= 0.3
            self._log_bad_merge_detected(f"Significant F-changes: {merge_info.num_significant_f_changes} changes")

        return penalty


class InformationPreservationReward(RewardFunctionBase):
    """VARIANT 2: Information preservation - minimizes heuristic quality loss."""

    def __init__(self, alpha: float = 2.0, beta: float = 0.05, lambda_density: float = 0.1):
        super().__init__("InformationPreservationReward")
        self.alpha = alpha
        self.beta = beta
        self.lambda_density = lambda_density

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Reward information preservation with bad merge detection."""

        info_preserve = self.alpha * merge_info.f_value_stability
        state_penalty = -self.beta * (merge_info.delta_states / max(merge_info.states_before, 10))
        transition_penalty = -self.lambda_density * abs(merge_info.transition_density_change)

        reward = info_preserve + state_penalty + transition_penalty

        # ✅ BAD MERGE DETECTION
        bad_merge_penalty = self._detect_bad_merges(merge_info)
        reward += bad_merge_penalty

        self.component_values = {
            'info_preservation': info_preserve,
            'state_penalty': state_penalty,
            'transition_penalty': transition_penalty,
            'bad_merge_penalty': bad_merge_penalty,
            'total': reward
        }

        return reward

    def _detect_bad_merges(self, merge_info: MergeInfo) -> float:
        """Detect bad merges in information preservation context."""
        penalty = 0.0

        # Very low F-stability is catastrophic for information preservation
        if merge_info.f_value_stability < 0.2:
            penalty = -1.5
            self._log_bad_merge_detected(f"Critical info loss: f_stability={merge_info.f_value_stability:.4f}")

        # Explosive state growth
        if merge_info.delta_states > merge_info.states_before * 2.0:
            penalty -= 0.7
            self._log_bad_merge_detected(f"State count tripled: {merge_info.states_before} → {merge_info.states_after}")

        # Transition density explosion
        if merge_info.transition_density_change > 0.5:
            penalty -= 0.4
            self._log_bad_merge_detected(f"Transition density explosion: +{merge_info.transition_density_change:.2f}")

        return penalty


class HybridMergeQualityReward(RewardFunctionBase):
    """VARIANT 3: Hybrid - balances multiple quality metrics."""

    def __init__(self, w_f_stability: float = 0.4, w_state_control: float = 0.3,
                 w_transition: float = 0.1, w_search: float = 0.2):
        super().__init__("HybridMergeQualityReward")

        total_w = w_f_stability + w_state_control + w_transition + w_search
        self.w_f_stability = w_f_stability / total_w
        self.w_state_control = w_state_control / total_w
        self.w_transition = w_transition / total_w
        self.w_search = w_search / total_w

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Weighted combination with bad merge detection."""

        f_score = merge_info.f_value_stability

        if merge_info.states_before > 0:
            explosion_ratio = merge_info.delta_states / merge_info.states_before
            state_score = max(0, 1.0 - abs(explosion_ratio))
        else:
            state_score = 0.5

        trans_change = merge_info.transition_density_change
        transition_score = 0.5 - min(0.5, trans_change / 2.0) if trans_change >= 0 else 0.5
        transition_score = float(np.clip(transition_score, 0.0, 1.0))

        total_states_after = merge_info.states_after
        if total_states_after > 0:
            valid_f_count = sum(1 for f in merge_info.f_after if f != float('inf'))
            reachability_score = valid_f_count / total_states_after
        else:
            reachability_score = 0.5

        composite = (
            self.w_f_stability * f_score +
            self.w_state_control * state_score +
            self.w_transition * transition_score +
            self.w_search * (1.0 - min(search_expansions / 100_000.0, 1.0))
        )

        reward = 2.0 * composite - 1.0

        # ✅ BAD MERGE DETECTION
        bad_merge_penalty = self._detect_bad_merges(merge_info)
        reward += bad_merge_penalty

        self.component_values = {
            'f_stability': f_score,
            'state_control': state_score,
            'transition': transition_score,
            'composite': composite,
            'bad_merge_penalty': bad_merge_penalty,
            'total': float(reward)
        }

        return float(reward)

    def _detect_bad_merges(self, merge_info: MergeInfo) -> float:
        """Detect bad merges in hybrid context."""
        penalty = 0.0

        # Multiple bad indicators simultaneously
        issues_count = 0

        if merge_info.f_value_stability < 0.35:
            issues_count += 1
        if merge_info.delta_states > merge_info.states_before * 1.5:
            issues_count += 1
        if merge_info.transition_density_change > 0.3:
            issues_count += 1

        # Cumulative penalty for multiple issues
        if issues_count >= 2:
            penalty = -0.5 * issues_count
            self._log_bad_merge_detected(f"Multiple quality issues detected: {issues_count} indicators")

        return penalty


class ConservativeReward(RewardFunctionBase):
    """VARIANT 4: Conservative - heavily penalizes risky merges."""

    def __init__(self, stability_threshold: float = 0.7):
        super().__init__("ConservativeReward")
        self.stability_threshold = stability_threshold

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Conservative approach with strict bad merge detection."""

        if merge_info.f_value_stability < self.stability_threshold:
            base_reward = -2.0
            self._log_bad_merge_detected(f"Below stability threshold: {merge_info.f_value_stability:.4f}")
        elif merge_info.delta_states > 0:
            explosion_ratio = merge_info.delta_states / max(merge_info.states_before, 1)
            base_reward = -0.5 * explosion_ratio
            if explosion_ratio > 0.5:
                self._log_bad_merge_detected(f"State explosion: {explosion_ratio*100:.1f}%")
        else:
            base_reward = merge_info.f_value_stability * 0.5

        reward = base_reward

        self.component_values = {
            'stability_check': base_reward,
            'total': reward
        }

        return reward


class ProgressiveReward(RewardFunctionBase):
    """VARIANT 5: Progressive - adapts based on episode progress."""

    def __init__(self):
        super().__init__("ProgressiveReward")
        self.merge_count = 0
        self._episode_initialized = False

    def reset_episode(self):
        """Call at episode start."""
        self.merge_count = 0
        self._episode_initialized = True

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Adapts reward based on merge count."""
        self.merge_count += 1
        progress = min(self.merge_count / 50.0, 1.0)

        if progress < 0.3:
            conservatism = 1.0
        elif progress < 0.7:
            conservatism = 0.5
        else:
            conservatism = 0.1

        stability_reward = merge_info.f_value_stability * 1.0
        state_penalty = -(1.0 - conservatism) * (
            merge_info.delta_states / max(merge_info.states_before, 1)
        )
        reward = stability_reward + state_penalty

        self.component_values = {
            'progress': progress,
            'conservatism': conservatism,
            'stability_reward': stability_reward,
            'state_penalty': state_penalty,
            'total': reward
        }

        return reward


class RichMergeQualityReward(RewardFunctionBase):
    """VARIANT 6: Rich - combines multiple quality signals."""

    def __init__(self,
                 w_f_stability: float = 0.35,
                 w_state_efficiency: float = 0.30,
                 w_transition_quality: float = 0.20,
                 w_reachability: float = 0.15):
        super().__init__("RichMergeQualityReward")

        total_w = w_f_stability + w_state_efficiency + w_transition_quality + w_reachability
        self.w_f_stability = w_f_stability / total_w
        self.w_state_efficiency = w_state_efficiency / total_w
        self.w_transition_quality = w_transition_quality / total_w
        self.w_reachability = w_reachability / total_w

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """Rich reward with comprehensive bad merge detection."""

        f_stability_score = merge_info.f_value_stability

        if merge_info.states_before > 0:
            explosion_ratio = merge_info.delta_states / merge_info.states_before
            state_efficiency = max(-1.0, 1.0 - explosion_ratio)
        else:
            state_efficiency = 0.5

        trans_change = merge_info.transition_density_change
        if trans_change < 0:
            transition_quality = 0.5
        else:
            transition_quality = 0.5 - min(0.5, trans_change / 2.0)
        transition_quality = float(np.clip(transition_quality, 0.0, 1.0))

        total_states_after = merge_info.states_after
        if total_states_after > 0:
            valid_f_count = sum(1 for f in merge_info.f_after if f != float('inf'))
            reachability_score = valid_f_count / total_states_after
        else:
            reachability_score = 0.5

        composite = (
            self.w_f_stability * f_stability_score +
            self.w_state_efficiency * state_efficiency +
            self.w_transition_quality * transition_quality +
            self.w_reachability * reachability_score
        )

        reward = 2.0 * composite - 1.0

        # ✅ BAD MERGE DETECTION - Enhanced for rich variant
        bad_merge_penalty = self._detect_bad_merges(merge_info, composite)
        reward += bad_merge_penalty

        if is_terminal and merge_info.f_value_stability > 0.8:
            reward += 0.5

        self.component_values = {
            'f_stability': f_stability_score,
            'state_efficiency': state_efficiency,
            'transition_quality': transition_quality,
            'reachability': reachability_score,
            'composite': composite,
            'bad_merge_penalty': bad_merge_penalty,
            'total': float(reward)
        }

        return float(reward)

    def _detect_bad_merges(self, merge_info: MergeInfo, composite: float) -> float:
        """✅ COMPREHENSIVE: Detect bad merges across all dimensions."""
        penalty = 0.0

        # CHECK 1: F-stability catastrophic failure
        if merge_info.f_value_stability < 0.25:
            penalty -= 1.2
            self._log_bad_merge_detected(
                f"F-stability catastrophic: {merge_info.f_value_stability:.4f} (threshold: 0.25)")

        # CHECK 2: State explosion uncontrolled
        expected_product = merge_info.ts1_size * merge_info.ts2_size
        if merge_info.states_after > expected_product and merge_info.states_after > merge_info.states_before * 3:
            penalty -= 1.0
            self._log_bad_merge_detected(
                f"Uncontrolled state explosion: {merge_info.states_before} → {merge_info.states_after}")

        # CHECK 3: Most states became unreachable
        unreachable_count = sum(1 for f in merge_info.f_after if f == float('inf') or f >= 1_000_000_000)
        unreachable_ratio = unreachable_count / max(merge_info.states_after, 1)
        if unreachable_ratio > 0.8:
            penalty -= 1.5
            self._log_bad_merge_detected(
                f"Critical unreachability: {unreachable_ratio*100:.1f}% of states unreachable")

        # CHECK 4: Goal unreachability
        goal_reachable = any(
            f != float('inf') and f < 1_000_000_000
            for f in merge_info.f_after
        )
        if not goal_reachable:
            penalty -= 2.0
            self._log_bad_merge_detected("CRITICAL: Goal unreachable after merge!")

        # CHECK 5: Transition density explosion
        if merge_info.transition_density_change > 0.5:
            penalty -= 0.6
            self._log_bad_merge_detected(
                f"Transition density explosion: +{merge_info.transition_density_change:.2f}")

        # CHECK 6: Significant F-value instability
        if merge_info.num_significant_f_changes > merge_info.states_after * 0.6:
            penalty -= 0.8
            self._log_bad_merge_detected(
                f"High F-value instability: {merge_info.num_significant_f_changes} changes")

        # CHECK 7: Composite score tells us overall quality
        if composite < 0.2:
            penalty -= 0.4
            self._log_bad_merge_detected(f"Poor composite score: {composite:.4f}")

        return penalty

class AStarSearchReward(RewardFunctionBase):
    """
    ✅ ENHANCED: A*-informed reward with robust signal for long-term learning.

    Key improvements:
    - Robust F-value stability measurement
    - Softer bad merge penalties that preserve learning signal
    - Better A* signal normalization
    - Reward shaping for smoother learning curves
    - Success bonuses for terminal states
    """

    def __init__(self,
                 w_search_efficiency: float = 0.25,
                 w_solution_quality: float = 0.20,
                 w_f_stability: float = 0.40,
                 w_state_control: float = 0.15):
        super().__init__("EnhancedAStarSearchReward")

        # Normalize weights
        total_w = w_search_efficiency + w_solution_quality + w_f_stability + w_state_control
        self.w_search_efficiency = w_search_efficiency / total_w
        self.w_solution_quality = w_solution_quality / total_w
        self.w_f_stability = w_f_stability / total_w
        self.w_state_control = w_state_control / total_w

        # ✅ NEW: Calibration parameters for robust normalization
        self.bf_comfort_zone = 3.0  # BF values 1-3 are good
        self.depth_comfort_zone = 50  # Typical depth for many problems
        self.stability_threshold = 0.3  # Threshold for acceptable stability

    def compute(self, merge_info: MergeInfo, search_expansions: int = 0,
                plan_cost: int = 0, is_terminal: bool = False) -> float:
        """✅ ENHANCED: Robust A*-informed reward with better learning signal."""

        logger.info(f"\n[REWARD] Computing {self.name}")

        # ====================================================================
        # COMPONENT 1: SEARCH EFFICIENCY (Branching Factor)
        # ====================================================================

        # ✅ IMPROVED: More nuanced branching factor scoring
        bf = merge_info.branching_factor

        if bf < 1.0 or np.isnan(bf) or np.isinf(bf):
            bf_score = 0.5
            bf_reason = "invalid BF, using neutral"
        elif bf <= 1.1:
            bf_score = 1.0  # Optimal: nearly linear branching
            bf_reason = "optimal"
        elif bf <= self.bf_comfort_zone:
            # Smooth interpolation in comfort zone
            bf_score = 1.0 - (bf - 1.0) / (self.bf_comfort_zone - 1.0) * 0.3
            bf_reason = f"good (in comfort zone)"
        elif bf <= 6.0:
            # Tolerable range with increasing penalty
            bf_score = 0.7 - (bf - self.bf_comfort_zone) / (6.0 - self.bf_comfort_zone) * 0.4
            bf_reason = f"tolerable"
        else:
            # Beyond 6: still provide some credit for trying
            bf_score = max(0.1, 0.3 - (bf - 6.0) / 10.0)
            bf_reason = "high BF, minimal credit"

        bf_score = float(np.clip(bf_score, 0.0, 1.0))
        logger.info(f"  [1] BRANCHING FACTOR: {bf:.3f} → score {bf_score:.3f} ({bf_reason})")

        # ====================================================================
        # COMPONENT 2: SOLUTION QUALITY (Search Depth)
        # ====================================================================

        # ✅ IMPROVED: Depth scoring with solution bonus
        depth_score = 0.5  # Default
        solution_bonus = 0.0

        if merge_info.solution_found:
            depth = merge_info.search_depth

            if depth < 1:
                depth_score = 0.5
            elif depth <= self.depth_comfort_zone:
                # Good depths: log scale to avoid over-penalizing
                depth_score = 1.0 - np.log(depth + 1) / np.log(self.depth_comfort_zone + 1) * 0.5
                depth_score = float(np.clip(depth_score, 0.5, 1.0))
            else:
                # Beyond comfort zone: still positive credit
                depth_score = max(0.2, 0.5 - (np.log(depth + 1) - np.log(self.depth_comfort_zone + 1)) / 2.0)
                depth_score = float(np.clip(depth_score, 0.0, 0.5))

            # ✅ NEW: Solution found is genuinely good
            solution_bonus = 0.3
            logger.info(f"  [2] SOLUTION QUALITY: depth={depth} → score {depth_score:.3f} + bonus {solution_bonus:.3f}")
        else:
            depth_score = 0.2  # Penalty for no solution
            logger.info(f"  [2] SOLUTION QUALITY: NO SOLUTION → score {depth_score:.3f}")

        # ====================================================================
        # COMPONENT 3: F-VALUE STABILITY (CRITICAL FOR LEARNING)
        # ====================================================================

        # ✅ ENHANCED: Better F-stability measurement
        f_stability = self._compute_robust_f_stability(merge_info)


        logger.info(f"  [3] F-VALUE STABILITY: {f_stability:.3f}")

        # ✅ NEW: Stability-based learning signal
        if f_stability > 0.7:
            stability_bonus = 0.2
            stability_reason = "excellent stability"
        elif f_stability > 0.5:
            stability_bonus = 0.1
            stability_reason = "good stability"
        elif f_stability > 0.3:
            stability_bonus = 0.0
            stability_reason = "acceptable stability"
        else:
            stability_bonus = -0.1
            stability_reason = "poor stability (but not penalized heavily)"

        logger.info(f"      → {stability_reason}, bonus: {stability_bonus:.3f}")

        # ====================================================================
        # COMPONENT 4: STATE CONTROL
        # ====================================================================

        # ✅ IMPROVED: More sophisticated state explosion handling
        if merge_info.states_before > 0:
            explosion_ratio = merge_info.delta_states / merge_info.states_before

            if explosion_ratio < -0.5:
                # State reduction is excellent
                state_score = 1.0
                state_reason = "excellent reduction"
            elif explosion_ratio < 0:
                # Some reduction
                state_score = 0.8
                state_reason = "reduction"
            elif explosion_ratio < 0.3:
                # Moderate increase is acceptable
                state_score = 0.8 - explosion_ratio * 0.3
                state_reason = "minor increase (acceptable)"
            elif explosion_ratio < 1.0:
                # Noticeable increase
                state_score = 0.65 - (explosion_ratio - 0.3) * 0.2
                state_reason = "significant increase"
            else:
                # Major explosion: still provide small credit
                state_score = max(0.1, 0.4 - np.log(explosion_ratio + 1) * 0.2)
                state_reason = "major explosion (penalized)"
        else:
            state_score = 0.5
            state_reason = "no baseline"

        state_score = float(np.clip(state_score, 0.0, 1.0))
        logger.info(
            f"  [4] STATE CONTROL: ratio={merge_info.delta_states / max(merge_info.states_before, 1):.3f} → score {state_score:.3f} ({state_reason})")

        # ====================================================================
        # WEIGHTED COMBINATION
        # ====================================================================

        composite = (
                self.w_search_efficiency * bf_score +
                self.w_solution_quality * depth_score +
                self.w_f_stability * f_stability +
                self.w_state_control * state_score
        )

        logger.info(f"\n  [COMPOSITE]: {composite:.3f}")
        logger.info(f"    = {self.w_search_efficiency:.3f}*{bf_score:.3f} (bf)")
        logger.info(f"    + {self.w_solution_quality:.3f}*{depth_score:.3f} (depth)")
        logger.info(f"    + {self.w_f_stability:.3f}*{f_stability:.3f} (stability)")
        logger.info(f"    + {self.w_state_control:.3f}*{state_score:.3f} (state)")

        # Scale to [-1, 1]
        reward = 2.0 * composite - 1.0

        # ====================================================================
        # BONUSES & PENALTIES (MILD - Preserve Learning Signal)
        # ====================================================================

        # ✅ Solution bonus
        reward += solution_bonus
        logger.info(f"\n  [BONUSES]:")
        logger.info(f"    + Solution bonus: {solution_bonus:.3f}")

        # ✅ Stability bonus
        reward += stability_bonus
        logger.info(f"    + Stability bonus: {stability_bonus:.3f}")

        # ✅ BAD MERGE DETECTION (SOFT PENALTIES)
        bad_merge_penalty = self._detect_bad_merges_soft(merge_info)
        reward += bad_merge_penalty

        if bad_merge_penalty != 0.0:
            logger.info(f"    + Bad merge penalty: {bad_merge_penalty:.3f}")

        # ✅ TERMINAL BONUS
        if is_terminal and merge_info.solution_found:
            terminal_bonus = 0.3
            reward += terminal_bonus
            logger.info(f"    + Terminal bonus: {terminal_bonus:.3f}")

        # ====================================================================
        # FINAL CLIPPING & LOGGING
        # ====================================================================

        reward = float(np.clip(reward, -1.0, 1.0))

        logger.info(f"\n  [FINAL REWARD]: {reward:.4f}")
        logger.info(f"  [RANGE]: [-1.0, +1.0] (clipped)\n")

        # Store components
        self.component_values = {
            'bf_score': float(bf_score),
            'depth_score': float(depth_score),
            'solution_bonus': float(solution_bonus),
            'f_stability': float(f_stability),
            'stability_bonus': float(stability_bonus),
            'state_score': float(state_score),
            'bad_merge_penalty': float(bad_merge_penalty),
            'composite': float(composite),
            'total': float(reward)
        }

        return reward

    def _compute_robust_f_stability(self, merge_info: MergeInfo) -> float:
        """
        ✅ ENHANCED: Compute F-stability with robust handling of edge cases.

        This is the MOST IMPORTANT metric for learning quality.
        """

        # ✅ NEW: Better handling of valid vs invalid values
        f_before_valid = [
            f for f in (merge_info.f_before if merge_info.f_before else [])
            if f != float('inf') and f >= 0 and f < 1_000_000_000
        ]

        f_after_valid = [
            f for f in (merge_info.f_after if merge_info.f_after else [])
            if f != float('inf') and f >= 0 and f < 1_000_000_000
        ]

        # Edge case 1: No valid data
        if not f_before_valid or not f_after_valid:
            logger.debug(
                f"      [F-STABILITY] Insufficient valid data: {len(f_before_valid)} before, {len(f_after_valid)} after")
            return 0.5  # Neutral score

        # Edge case 2: Too few samples
        if len(f_after_valid) < 2:
            return 0.5

        # ✅ IMPROVED: Use median-based stability (robust to outliers)
        before_median = float(np.median(f_before_valid))
        after_median = float(np.median(f_after_valid))

        before_std = float(np.std(f_before_valid)) if len(f_before_valid) > 1 else 0.0
        after_std = float(np.std(f_after_valid)) if len(f_after_valid) > 1 else 0.0

        # Normalized change in median
        if before_median > 0:
            median_change = abs(after_median - before_median) / before_median
        else:
            median_change = 0.0

        # Normalized change in variance
        if before_std > 0:
            std_change = abs(after_std - before_std) / before_std
        else:
            std_change = 0.0 if after_std == 0 else 1.0

        # ✅ IMPROVED: Weighted combination (median is more important than std)
        change_metric = 0.7 * np.clip(median_change, 0, 1) + 0.3 * np.clip(std_change, 0, 1)

        # Convert to stability score: lower change = higher stability
        f_stability = max(0.0, 1.0 - change_metric)

        logger.debug(
            f"      [F-STABILITY] median: {before_median:.1f}→{after_median:.1f}, std: {before_std:.1f}→{after_std:.1f}")
        logger.debug(f"      [F-STABILITY] change_metric: {change_metric:.3f}, stability: {f_stability:.3f}")

        return float(np.clip(f_stability, 0.0, 1.0))

    def _detect_bad_merges_soft(self, merge_info: MergeInfo) -> float:
        """
        ✅ ENHANCED: Detect bad merges with SOFT penalties that preserve learning signal.

        Key: Don't over-penalize—let the main reward components handle it.
        Only penalize CRITICAL failures.
        """
        penalty = 0.0
        self.bad_merge_reasons = []

        # CRITICAL CHECK 1: Goal becomes unreachable
        goal_reachable = any(
            f != float('inf') and f < 1_000_000_000
            for f in (merge_info.f_after if merge_info.f_after else [])
        )

        if not goal_reachable:
            penalty -= 0.5  # ✅ MILD penalty for critical failure
            self._log_bad_merge_detected("Goal unreachable (CRITICAL)")
            return penalty  # Return early—this is catastrophic

        # CHECK 2: Very poor F-stability (only if REALLY bad)
        if merge_info.f_value_stability < 0.15:
            penalty -= 0.15  # ✅ VERY MILD
            self._log_bad_merge_detected(f"F-stability extremely poor: {merge_info.f_value_stability:.3f}")

        # CHECK 3: Unreachable states > 90% (indicates broken abstraction)
        unreachable_count = sum(
            1 for f in (merge_info.f_after if merge_info.f_after else [])
            if f == float('inf') or f >= 1_000_000_000
        )

        total_states = len(merge_info.f_after) if merge_info.f_after else 1
        unreachability_ratio = unreachable_count / max(total_states, 1)

        if unreachability_ratio > 0.9:
            penalty -= 0.1  # ✅ MILD: most states unreachable
            self._log_bad_merge_detected(f"High unreachability: {unreachability_ratio * 100:.1f}%")

        # ✅ REMOVED: Other penalties (let composite score handle them)

        return penalty


def create_reward_function(variant: str, **kwargs) -> RewardFunctionBase:
    """
    Creates a reward function instance by name.

    Supported variants:
    - 'simple_stability': Basic stability-focused reward
    - 'information_preservation': Preserve heuristic information
    - 'hybrid': Balance multiple metrics
    - 'conservative': Risk-averse approach
    - 'progressive': Adaptive based on progress
    - 'rich': Comprehensive multi-signal reward
    - 'astar_search': A*-informed with bad merge detection (RECOMMENDED)
    """
    variants = {
        'simple_stability': SimpleStabilityReward,
        'information_preservation': InformationPreservationReward,
        'hybrid': HybridMergeQualityReward,
        'conservative': ConservativeReward,
        'progressive': ProgressiveReward,
        'rich': RichMergeQualityReward,
        'astar_search': AStarSearchReward,
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Supported: {list(variants.keys())}")

    return variants[variant](**kwargs)