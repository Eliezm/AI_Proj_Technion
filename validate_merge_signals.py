# FILE: validate_merge_signals.py
"""
✅ NEW: Validate that merge signals are meaningful before reward computation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_merge_signals(merge_info: 'MergeInfo') -> tuple:
    """
    Validate that extracted signals are physically meaningful.

    Returns:
        (is_valid, issues_found)
    """
    issues = []

    # CHECK 1: F-values are reasonable
    if merge_info.states_before <= 0 or merge_info.states_after <= 0:
        issues.append(f"Invalid state counts: {merge_info.states_before} → {merge_info.states_after}")
        return False, issues

    # CHECK 2: F-stability in valid range
    if not (0.0 <= merge_info.f_value_stability <= 1.0):
        issues.append(f"F-stability out of range: {merge_info.f_value_stability}")
        merge_info.f_value_stability = np.clip(merge_info.f_value_stability, 0.0, 1.0)

    # CHECK 3: Branching factor >= 1
    if merge_info.branching_factor < 1.0:
        issues.append(f"Branching factor < 1.0: {merge_info.branching_factor}")
        merge_info.branching_factor = 1.0

    # CHECK 4: Explosion penalty is reasonable
    if merge_info.state_explosion_penalty < 0.0 or merge_info.state_explosion_penalty > 1.0:
        issues.append(f"Explosion penalty out of range: {merge_info.state_explosion_penalty}")
        merge_info.state_explosion_penalty = np.clip(merge_info.state_explosion_penalty, 0.0, 1.0)

    # CHECK 5: F-value lists have data
    if len(merge_info.f_after) == 0:
        issues.append("No f_after values")
        return False, issues

    # CHECK 6: No NaN/Inf in critical fields
    critical_fields = [
        'f_value_stability', 'branching_factor', 'state_explosion_penalty',
        'transition_density_change'
    ]
    for field in critical_fields:
        val = getattr(merge_info, field)
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                issues.append(f"{field} is NaN/Inf")
                setattr(merge_info, field, 0.0)

    if issues:
        logger.warning(f"[VALIDATE] Found {len(issues)} signal issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return len(issues) <= 2, issues  # Allow up to 2 minor issues

    logger.info("[VALIDATE] ✅ All signals valid")
    return True, []