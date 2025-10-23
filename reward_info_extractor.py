# FILE: reward_info_extractor.py (COMPLETE REPLACEMENT - CRITICAL FIXES)
import os
import json
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS (MOVED TO TOP)
# ============================================================================

# Fast Downward's INF constant (typical value)
# This is the maximum possible distance in FD
LARGE_VALUE_THRESHOLD = 1_000_000_000  # 1 billion (clearly unreachable)
FD_INF_CANDIDATES = [
    2 ** 31 - 1,  # Typical on 32-bit systems
    2 ** 31,  # Alternative
    2 ** 32 - 1,  # Alternative
    999999999,  # Some versions use this
    1_000_000_000,  # Common threshold
    float('inf'),  # JSON Infinity (if properly encoded)
]


def is_unreachable_value(value: Any) -> bool:
    """
    ✅ FIXED: Properly detect unreachable states.

    A value is unreachable if:
    - It's NaN
    - It's infinity (float or large int)
    - It's negative (error code)
    """
    try:
        # Handle float('inf')
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return True

        # Handle integers that represent INF
        if isinstance(value, int):
            if value < 0:  # Negative = error/unreachable
                return True
            if value in FD_INF_CANDIDATES:  # Known INF encodings
                return True
            if value > LARGE_VALUE_THRESHOLD:  # Heuristic: suspiciously large
                return True

        return False
    except (TypeError, ValueError):
        return True


def is_valid_fvalue(value: Any) -> bool:
    """
    ✅ NEW: Validate that an F-value is actually valid.

    Valid F-values are:
    - Non-negative numbers
    - Less than a reasonable threshold (reachability assumption)
    - Not NaN or Inf
    """
    if is_unreachable_value(value):
        return False

    try:
        num_value = float(value)
        # Sanity check: F-values should be reasonable distances
        # (0 to ~1000 for typical planning problems)
        if num_value < 0 or num_value > LARGE_VALUE_THRESHOLD:
            return False
        return True
    except (TypeError, ValueError):
        return False


def safe_compute_average(values: List[Any], context: str = "") -> Tuple[float, int]:
    """
    ✅ NEW: Safely compute average with detailed validation.

    Returns:
        (average, valid_count)
    """
    if not values:
        logger.warning(f"[{context}] Empty value list for averaging")
        return 0.0, 0

    # Filter to only valid values
    valid_values = [v for v in values if is_valid_fvalue(v)]

    removed_count = len(values) - len(valid_values)
    if removed_count > 0:
        logger.info(f"[{context}] Removed {removed_count}/{len(values)} invalid F-values")

    if not valid_values:
        logger.error(f"[{context}] ALL values were invalid! Using 0.5 as default")
        return 0.5, 0

    # Compute with numpy (handles float conversion)
    avg = float(np.mean(np.array(valid_values, dtype=np.float32)))

    # Final sanity check
    if np.isnan(avg) or np.isinf(avg):
        logger.error(f"[{context}] Computed average is NaN/Inf! Values: {valid_values[:10]}")
        return 0.5, len(valid_values)

    return avg, len(valid_values)


@dataclass
class MergeInfo:
    """✅ COMPLETE: Container with ALL necessary fields."""
    iteration: int
    states_before: int
    states_after: int
    delta_states: int
    f_before: List[int]
    f_after: List[int]
    f_value_stability: float
    num_significant_f_changes: int
    avg_f_change: float
    max_f_change: float
    ts1_id: int
    ts2_id: int
    ts1_size: int
    ts2_size: int
    ts1_transitions: int
    ts2_transitions: int
    merged_transitions: int
    state_explosion_penalty: float
    f_preservation_score: float
    transition_density_change: float
    nodes_expanded: int = 0
    search_depth: int = 0
    solution_cost: int = 0
    branching_factor: float = 1.0
    solution_found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'states_before': self.states_before,
            'states_after': self.states_after,
            'delta_states': self.delta_states,
            'f_value_stability': float(self.f_value_stability),
            'num_significant_f_changes': self.num_significant_f_changes,
            'avg_f_change': float(self.avg_f_change),
            'max_f_change': float(self.max_f_change),
            'state_explosion_penalty': float(self.state_explosion_penalty),
            'f_preservation_score': float(self.f_preservation_score),
            'transition_density_change': float(self.transition_density_change),
            'nodes_expanded': self.nodes_expanded,
            'search_depth': self.search_depth,
            'solution_cost': self.solution_cost,
            'branching_factor': float(self.branching_factor),
            'solution_found': self.solution_found,
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """✅ ENHANCED: Comprehensive validation."""
        issues = []

        # Check numeric ranges
        if not (0 <= self.f_value_stability <= 1):
            issues.append(f"f_value_stability out of [0,1]: {self.f_value_stability}")

        if self.states_before <= 0 or self.states_after <= 0:
            issues.append(f"Invalid state counts: {self.states_before} → {self.states_after}")

        if self.branching_factor < 1.0:
            self.branching_factor = 1.0
            issues.append("Corrected branching_factor to 1.0")

        # Check for NaN/Inf
        for field in ['f_value_stability', 'avg_f_change', 'max_f_change', 'state_explosion_penalty',
                      'transition_density_change']:
            val = getattr(self, field)
            if isinstance(val, float):
                if np.isnan(val):
                    setattr(self, field, 0.0)
                    issues.append(f"Replaced NaN in {field}")
                elif np.isinf(val):
                    setattr(self, field, 1.0 if 'stability' in field or 'preservation' in field else 0.0)
                    issues.append(f"Replaced Inf in {field}")

        return len(issues) == 0, issues


class RewardInfoExtractor:
    """✅ COMPLETELY REFACTORED: Robust F-value extraction."""

    def __init__(self, fd_output_dir: str = "downward/fd_output"):
        self.fd_output_dir = fd_output_dir
        self.f_change_threshold = 5.0
        os.makedirs(fd_output_dir, exist_ok=True)
        logger.info(f"✓ RewardInfoExtractor initialized (output_dir: {fd_output_dir})")

    def extract_merge_info(self, iteration: int, timeout: float = 30.0) -> Optional[MergeInfo]:
        """✅ ENHANCED: Complete extraction with F-value validation."""

        logger.info(f"\n{'=' * 80}")
        logger.info(f"[EXTRACT] Iteration {iteration}: Extracting merge info")
        logger.info(f"{'=' * 80}")

        try:
            before_path = os.path.join(self.fd_output_dir, f"merge_before_{iteration}.json")
            after_path = os.path.join(self.fd_output_dir, f"merge_after_{iteration}.json")

            # ====================================================================
            # LOAD FILES
            # ====================================================================
            logger.info(f"\n[LOAD] Loading files...")
            logger.info(f"  - before: {before_path}")
            logger.info(f"  - after:  {after_path}")

            before_data = self._load_and_validate_json(before_path, 'before', timeout)
            after_data = self._load_and_validate_json(after_path, 'after', timeout)

            if before_data is None or after_data is None:
                logger.error(f"[LOAD] ✗ Failed to load before or after data")
                return None

            logger.info(f"[LOAD] ✓ Both files loaded successfully")

            # ====================================================================
            # EXTRACT F-VALUES (WITH COMPREHENSIVE VALIDATION)
            # ====================================================================
            logger.info(f"\n[F-VALUES] Extracting F-values...")

            ts1_f = before_data.get("ts1_f_values", [])
            ts2_f = before_data.get("ts2_f_values", [])
            f_after_raw = after_data.get("f_values", [])

            logger.info(f"  - Raw ts1_f length: {len(ts1_f)}")
            logger.info(f"  - Raw ts2_f length: {len(ts2_f)}")
            logger.info(f"  - Raw f_after length: {len(f_after_raw)}")

            # ✅ AGGRESSIVE FILTERING: Remove all unreachable values
            logger.info(f"\n[VALIDATE] Filtering invalid F-values...")

            ts1_valid = [f for f in ts1_f if is_valid_fvalue(f)]
            logger.info(f"  - ts1_valid: {len(ts1_valid)} / {len(ts1_f)} "
                        f"(removed {len(ts1_f) - len(ts1_valid)})")

            ts2_valid = [f for f in ts2_f if is_valid_fvalue(f)]
            logger.info(f"  - ts2_valid: {len(ts2_valid)} / {len(ts2_f)} "
                        f"(removed {len(ts2_f) - len(ts2_f) - len(ts2_valid)})")

            f_after_valid = [f for f in f_after_raw if is_valid_fvalue(f)]
            logger.info(f"  - after_valid: {len(f_after_valid)} / {len(f_after_raw)} "
                        f"(removed {len(f_after_raw) - len(f_after_valid)})")

            # ====================================================================
            # TS IDs
            # ====================================================================
            logger.info(f"\n[IDS] Extracting TS IDs...")
            ts1_id = before_data.get("ts1_id", -1)
            ts2_id = before_data.get("ts2_id", -1)

            logger.info(f"  - ts1_id: {ts1_id}")
            logger.info(f"  - ts2_id: {ts2_id}")

            if ts1_id == ts2_id:
                logger.error(f"[IDS] ✗ TS1 and TS2 have same ID")
                return None

            logger.info(f"[IDS] ✓ IDs are different")

            # ====================================================================
            # F-STABILITY (WITH PROPER VALIDATION)
            # ====================================================================
            logger.info(f"\n[STABILITY] Computing F-stability...")

            product_mapping = before_data.get("product_mapping", {})
            ts1_size = before_data.get("ts1_size", 1)
            ts2_size = before_data.get("ts2_size", 1)

            logger.info(f"  - ts1_size: {ts1_size}, ts2_size: {ts2_size}")
            logger.info(f"  - Product space size: {ts1_size * ts2_size}")

            # ✅ PRODUCT MAPPING METHOD (if available)
            if product_mapping:
                logger.info(f"  - Using product state mapping")

                f_before_product = []
                for s in range(ts1_size * ts2_size):
                    s_key = str(s)
                    if s_key in product_mapping:
                        s1 = product_mapping[s_key].get("s1", -1)
                        s2 = product_mapping[s_key].get("s2", -1)

                        if (s1 >= 0 and s1 < len(ts1_f) and
                                s2 >= 0 and s2 < len(ts2_f)):
                            f1_val = ts1_f[s1]
                            f2_val = ts2_f[s2]

                            # ✅ ONLY add if BOTH are valid
                            if is_valid_fvalue(f1_val) and is_valid_fvalue(f2_val):
                                combined_f = max(float(f1_val), float(f2_val))
                                f_before_product.append(combined_f)

                logger.info(f"  - f_before_product length: {len(f_before_product)}")

                # ✅ SAFE AVERAGING
                if f_before_product and f_after_valid:
                    avg_f_before, cnt_before = safe_compute_average(f_before_product, "f_before_product")
                    avg_f_after, cnt_after = safe_compute_average(f_after_valid, "f_after")

                    logger.info(f"  - avg_f_before: {avg_f_before:.4f} ({cnt_before} valid states)")
                    logger.info(f"  - avg_f_after:  {avg_f_after:.4f} ({cnt_after} valid states)")

                    delta = avg_f_after - avg_f_before
                    logger.info(f"  - delta: {delta:.4f}")

                    if avg_f_before > 0:
                        f_stability = 1.0 - abs(delta) / avg_f_before
                        f_stability = float(np.clip(f_stability, 0.0, 1.0))
                        logger.info(f"  - f_stability: 1.0 - |{delta:.4f}| / {avg_f_before:.4f} = {f_stability:.4f}")
                    else:
                        f_stability = 0.5
                        logger.warning(f"  - avg_f_before is 0, using default 0.5")
                else:
                    avg_f_before = 1.0
                    avg_f_after = 1.0
                    f_stability = 0.5
                    logger.warning(f"  - Insufficient data for product mapping method")

            else:
                # ✅ FALLBACK: Weighted average
                logger.info(f"  - Using weighted average method")

                if ts1_valid and ts2_valid:
                    w1 = len(ts1_valid) / (len(ts1_valid) + len(ts2_valid))
                    w2 = len(ts2_valid) / (len(ts1_valid) + len(ts2_valid))
                    avg_f_before, _ = safe_compute_average(ts1_valid + ts2_valid, "weighted_before")

                    logger.info(f"  - w1: {w1:.4f}, w2: {w2:.4f}")
                    logger.info(f"  - weighted avg_f_before: {avg_f_before:.4f}")
                else:
                    avg_f_before = 1.0
                    logger.warning(f"  - One of ts1_valid or ts2_valid is empty")

                avg_f_after, _ = safe_compute_average(f_after_valid, "after")
                logger.info(f"  - avg_f_after: {avg_f_after:.4f}")

                if avg_f_before > 0:
                    f_stability = 1.0 - abs(avg_f_after - avg_f_before) / avg_f_before
                    f_stability = float(np.clip(f_stability, 0.0, 1.0))
                    logger.info(f"  - f_stability: {f_stability:.4f}")
                else:
                    f_stability = 0.5
                    logger.warning(f"  - avg_f_before is 0, using default 0.5")

            # ====================================================================
            # SIGNIFICANT CHANGES
            # ====================================================================
            logger.info(f"\n[CHANGES] Counting significant F-changes (threshold: {self.f_change_threshold})...")

            num_sig_changes = 0
            if ts1_valid and f_after_valid and len(ts1_valid) > 0 and len(f_after_valid) > 0:
                combined_before = ts1_valid + ts2_valid
                min_len = min(len(combined_before), len(f_after_valid))

                try:
                    abs_changes = np.abs(
                        np.array(combined_before[:min_len], dtype=np.float32) -
                        np.array(f_after_valid[:min_len], dtype=np.float32)
                    )
                    num_sig_changes = int(np.sum(abs_changes > self.f_change_threshold))
                    logger.info(f"  - Compared {min_len} values")
                    logger.info(f"  - Significant changes (> {self.f_change_threshold}): {num_sig_changes}")
                except Exception as e:
                    logger.warning(f"  - Error computing changes: {e}")
                    num_sig_changes = 0

            # ====================================================================
            # SIZES & TRANSITIONS
            # ====================================================================
            logger.info(f"\n[SIZES] Extracting sizes and transitions...")

            states_before = max(1, ts1_size * ts2_size)
            states_after = max(1, len(f_after_raw))
            delta_states = states_after - states_before

            logger.info(f"  - states_before: {states_before}")
            logger.info(f"  - states_after: {states_after}")
            logger.info(f"  - delta_states: {delta_states}")

            state_explosion = self._compute_state_explosion_penalty(delta_states, states_before)
            logger.info(f"  - state_explosion_penalty: {state_explosion:.4f}")

            ts1_trans = before_data.get("ts1_num_transitions", 0)
            ts2_trans = before_data.get("ts2_num_transitions", 0)
            merged_trans = after_data.get("num_transitions", 0)

            logger.info(f"  - ts1_transitions: {ts1_trans}")
            logger.info(f"  - ts2_transitions: {ts2_trans}")
            logger.info(f"  - merged_transitions: {merged_trans}")

            trans_density = self._compute_transition_density_change(
                ts1_trans, ts2_trans, merged_trans,
                ts1_size, ts2_size, states_after
            )
            logger.info(f"  - transition_density_change: {trans_density:.4f}")

            # ====================================================================
            # A* SIGNALS
            # ====================================================================
            logger.info(f"\n[A*] Extracting A* search signals...")

            search_signals = after_data.get("search_signals", {})
            logger.info(f"  - search_signals present: {bool(search_signals)}")

            nodes_expanded = int(search_signals.get("nodes_expanded", 0))
            search_depth = int(search_signals.get("search_depth", 0))
            solution_cost = int(search_signals.get("solution_cost", 0))
            branching_factor = float(search_signals.get("branching_factor", 1.0))
            solution_found = bool(search_signals.get("solution_found", False))

            logger.info(f"  - nodes_expanded: {nodes_expanded}")
            logger.info(f"  - search_depth: {search_depth}")
            logger.info(f"  - solution_cost: {solution_cost}")
            logger.info(f"  - branching_factor: {branching_factor:.4f}")
            logger.info(f"  - solution_found: {solution_found}")

            # Safety checks
            if nodes_expanded < 0:
                nodes_expanded = 0
            if search_depth < 0:
                search_depth = 0
            if solution_cost < 0:
                solution_cost = 0
            if branching_factor < 1.0 or np.isnan(branching_factor) or np.isinf(branching_factor):
                branching_factor = 1.0

            # ====================================================================
            # CREATE MERGE INFO
            # ====================================================================
            logger.info(f"\n[CREATE] Creating MergeInfo object...")

            # ✅ COMPUTE max_f_change safely
            if ts1_valid and ts2_valid:
                combined_before = np.array(ts1_valid + ts2_valid, dtype=np.float32)
                max_f_change = float(np.max(np.abs(combined_before - np.mean(combined_before))))
            else:
                max_f_change = 0.0

            # ✅ COMPUTE avg_f_change safely
            if len(ts1_valid) > 0 and len(ts2_valid) > 0:
                avg_f_before_tmp, _ = safe_compute_average(ts1_valid + ts2_valid, "for_delta")
                avg_f_after_tmp, _ = safe_compute_average(f_after_valid, "for_delta")
                avg_f_change = float(avg_f_before_tmp - avg_f_after_tmp)
            else:
                avg_f_change = 0.0

            merge_info = MergeInfo(
                iteration=iteration,
                states_before=states_before,
                states_after=states_after,
                delta_states=delta_states,
                f_before=ts1_f + ts2_f,
                f_after=f_after_raw,
                f_value_stability=f_stability,
                num_significant_f_changes=num_sig_changes,
                avg_f_change=avg_f_change,
                max_f_change=max_f_change,
                ts1_id=ts1_id,
                ts2_id=ts2_id,
                ts1_size=ts1_size,
                ts2_size=ts2_size,
                ts1_transitions=ts1_trans,
                ts2_transitions=ts2_trans,
                merged_transitions=merged_trans,
                state_explosion_penalty=state_explosion,
                f_preservation_score=f_stability,
                transition_density_change=trans_density,
                nodes_expanded=nodes_expanded,
                search_depth=search_depth,
                solution_cost=solution_cost,
                branching_factor=branching_factor,
                solution_found=solution_found,
            )

            logger.info(f"[CREATE] ✓ MergeInfo created")

            # Validate
            is_valid, issues = merge_info.validate()
            if issues:
                for issue in issues:
                    logger.info(f"  - {issue}")

            logger.info(f"\n[SUMMARY] Iteration {iteration}:")
            logger.info(f"  - TS{ts1_id}({ts1_size}) + TS{ts2_id}({ts2_size}) → {states_after} states")
            logger.info(f"  - f_stability={f_stability:.3f}")
            logger.info(f"  - A* nodes_expanded={nodes_expanded}")
            logger.info(f"  - solution_found={solution_found}")

            return merge_info

        except Exception as e:
            logger.error(f"\n[ERROR] Failed to extract merge info for iteration {iteration}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _load_and_validate_json(self, path: str, phase: str, timeout: float = 30.0) -> Optional[Dict]:
        """✅ Load JSON with timeout and basic validation."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not os.path.exists(path):
                time.sleep(0.1)
                continue

            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    logger.warning(f"JSON at {path} is not a dict")
                    return None

                return data

            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse error at {path}: {e}, retrying...")
                time.sleep(0.1)

        logger.error(f"Timeout waiting for valid JSON at {path}")
        return None

    def _compute_state_explosion_penalty(self, delta_states: int, states_before: int) -> float:
        """Compute state explosion penalty safely."""
        if states_before <= 0:
            return 0.0
        pct_increase = delta_states / float(max(states_before, 1))
        penalty = min(1.0, max(0.0, pct_increase / 0.5))
        return float(penalty)

    def _compute_transition_density_change(self, ts1_trans: int, ts2_trans: int,
                                           merged_trans: int, ts1_size: int,
                                           ts2_size: int, merged_size: int) -> float:
        """Change in transition density."""
        if ts1_size <= 0 or ts2_size <= 0 or merged_size <= 0:
            return 0.0
        try:
            density_before_ts1 = ts1_trans / float(ts1_size)
            density_before_ts2 = ts2_trans / float(ts2_size)
            density_after = merged_trans / float(merged_size)
            expected_density = (ts1_trans + ts2_trans) / float(max(merged_size, 1))
            density_change = density_after - expected_density
            return float(density_change)
        except Exception as e:
            logger.warning(f"Error computing transition density: {e}")
            return 0.0


def validate_extracted_info(merge_info: MergeInfo) -> Tuple[bool, List[str]]:
    """✅ COMPLETE: Full validation of extracted info."""
    if merge_info is None:
        return False, ["merge_info is None"]

    return merge_info.validate()