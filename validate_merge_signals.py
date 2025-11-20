# FILE: validate_merge_signals.py (COMPLETE)
# -*- coding: utf-8 -*-
"""
✅ FIXED: Validate merge signals with comprehensive checks
"""

import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


def validate_merge_signals(merge_info) -> Tuple[bool, List[str]]:
    """
    Validate that extracted merge signals are physically meaningful.

    Returns:
        (is_valid, issues_found)
    """
    if merge_info is None:
        return False, ["merge_info is None"]

    issues = []

    # CHECK 1: State counts are positive
    if merge_info.states_before <= 0:
        issues.append(f"states_before <= 0: {merge_info.states_before}")
        return False, issues

    if merge_info.states_after <= 0:
        issues.append(f"states_after <= 0: {merge_info.states_after}")
        return False, issues

    # CHECK 2: F-stability in valid range
    if not (0.0 <= merge_info.f_value_stability <= 1.0):
        issues.append(f"f_value_stability out of range: {merge_info.f_value_stability}")
        merge_info.f_value_stability = np.clip(merge_info.f_value_stability, 0.0, 1.0)

    # CHECK 3: Branching factor >= 1
    if merge_info.branching_factor < 1.0:
        issues.append(f"branching_factor < 1.0: {merge_info.branching_factor}")
        merge_info.branching_factor = 1.0

    if np.isnan(merge_info.branching_factor) or np.isinf(merge_info.branching_factor):
        issues.append(f"branching_factor is NaN/Inf: {merge_info.branching_factor}")
        merge_info.branching_factor = 1.0

    # CHECK 4: Explosion penalty is reasonable
    if merge_info.state_explosion_penalty < 0.0 or merge_info.state_explosion_penalty > 1.0:
        issues.append(f"explosion_penalty out of range: {merge_info.state_explosion_penalty}")
        merge_info.state_explosion_penalty = np.clip(merge_info.state_explosion_penalty, 0.0, 1.0)

    # CHECK 5: F-value lists have data
    if len(merge_info.f_after) == 0:
        issues.append("No f_after values")
        return False, issues

    # CHECK 6: No NaN/Inf in critical fields
    critical_fields = [
        'f_value_stability', 'branching_factor', 'state_explosion_penalty',
        'transition_density_change', 'avg_f_change', 'max_f_change'
    ]

    for field in critical_fields:
        if not hasattr(merge_info, field):
            continue

        val = getattr(merge_info, field)
        if isinstance(val, float):
            if np.isnan(val):
                issues.append(f"{field} is NaN")
                setattr(merge_info, field, 0.0)
            elif np.isinf(val):
                issues.append(f"{field} is Inf")
                setattr(merge_info, field, 0.0)

    # CHECK 7: TS sizes reasonable
    if merge_info.ts1_size <= 0 or merge_info.ts2_size <= 0:
        issues.append(f"TS sizes invalid: {merge_info.ts1_size} x {merge_info.ts2_size}")
        return False, issues

    # Allow up to 3 minor issues
    has_critical_issues = any(issue in issues for issue in [
        "merge_info is None",
        "states_before <= 0",
        "states_after <= 0",
        "No f_after values",
        "TS sizes invalid",
    ])

    if has_critical_issues:
        logger.error(f"[VALIDATE] CRITICAL ISSUES: {issues}")
        return False, issues

    if len(issues) > 3:
        logger.warning(f"[VALIDATE] Multiple issues ({len(issues)}): {issues[:3]}")

    if issues:
        logger.info(f"[VALIDATE] Minor issues detected and corrected: {issues}")
        return True, issues  # Valid with warnings

    logger.debug("[VALIDATE] ✅ All signals valid")
    return True, []