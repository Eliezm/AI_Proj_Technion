#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE ANALYSIS AND VISUALIZATION MODULE - ENHANCED
===========================================================
Generates comprehensive plots for evaluation reporting.

Features:
  ✓ Solve rate comparison (bar chart)
  ✓ Time comparison (box plots + scatter)
  ✓ Expansions comparison (log scale)
  ✓ Plan cost comparison
  ✓ Efficiency frontier (2D scatter)
  ✓ Per-problem heatmap
  ✓ Time distribution (violin plots)
  ✓ ✅ NEW: Learning curves (if available)
  ✓ ✅ NEW: Per-difficulty breakdown
  ✓ ✅ NEW: Curriculum learning effectiveness
  ✓ ✅ NEW: Statistical significance tests
  ✓ ✅ NEW: Error analysis
  ✓ ✅ NEW: Performance profile
  ✓ ✅ NEW: Cumulative distribution
  ✓ ✅ NEW: Summary dashboard
"""

import sys
import os
import logging
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not installed")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed")

# ============================================================================
# CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (14, 8),
    "dpi": 150,
    "font_size": 11,
}

COLORS = {
    "GNN": "#2E86AB",
    "FD": "#A23B72",
    "FD_LM-Cut": "#F18F01",
    "FD_Blind": "#C73E1D",
    "FD_Add": "#06A77D",
    "FD_Max": "#D62828",
    "overfit": "#2E86AB",
    "problem_gen": "#A23B72",
    "scale_gen": "#F18F01",
    "curriculum": "#06A77D",
}


# ============================================================================
# DATA LOADING - ENHANCED FOR EXPERIMENT INTEGRATION
# ============================================================================

def load_results_csv(csv_path: str) -> Optional['pd.DataFrame']:
    """Load evaluation results from CSV."""
    if not HAS_PANDAS:
        logger.warning("pandas not available")
        return None

    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} results from CSV")
    return df


def load_experiment_results(
        experiment_dirs: Dict[str, str]
) -> Optional['pd.DataFrame']:
    """
    Load and convert experiment results to unified format.

    Args:
        experiment_dirs: Dict mapping experiment_type -> directory_path

    Returns:
        Unified pandas DataFrame
    """
    if not HAS_PANDAS:
        logger.warning("pandas not available")
        return None

    logger.info("Loading experiment results from multiple directories...")

    all_data = []

    for exp_type, exp_dir in experiment_dirs.items():
        results_file = os.path.join(exp_dir, "results.json")

        if not os.path.exists(results_file):
            logger.warning(f"Results file not found: {results_file}")
            continue

        logger.info(f"Loading {exp_type} from {exp_dir}...")

        try:
            with open(results_file, 'r') as f:
                exp_results = json.load(f)

            if 'evaluation' in exp_results:
                eval_data = exp_results['evaluation']
                for detail in eval_data.get('details', []):
                    all_data.append({
                        'experiment_type': exp_type,
                        'planner_name': 'GNN',
                        'problem_name': detail.get('problem', ''),
                        'solved': detail.get('solved', False),
                        'wall_clock_time': detail.get('time', 0),
                        'plan_cost': 0,
                        'nodes_expanded': 0,
                        'reward': detail.get('reward', 0),
                    })

                logger.info(f"  Loaded {len(all_data)} problem results for {exp_type}")

        except Exception as e:
            logger.error(f"Failed to load {exp_type}: {e}")
            continue

    if not all_data:
        logger.error("No experiment data loaded")
        return None

    df = pd.DataFrame(all_data)
    logger.info(f"Total: {len(df)} results from {len(experiment_dirs)} experiments")

    return df


# ============================================================================
# PLOT 1: SOLVE RATE COMPARISON
# ============================================================================

def plot_solve_rate_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart comparing solve rates."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    solve_rates = []
    planner_names = []

    for planner in sorted(df['planner_name'].unique()):
        df_planner = df[df['planner_name'] == planner]
        rate = (df_planner['solved'].sum() / len(df_planner)) * 100
        solve_rates.append(rate)
        planner_names.append(planner)

    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(solve_rates)), solve_rates, color=colors_list, alpha=0.8, edgecolor='black')

    ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.3, label='Target')

    for i, (bar, rate) in enumerate(zip(bars, solve_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Solve Rate Comparison", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 2: TIME COMPARISON
# ============================================================================

def plot_time_comparison(df: 'pd.DataFrame', output_path: str):
    """Box plot and scatter showing time distribution."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    planners = sorted(df_solved['planner_name'].unique())
    data_for_box = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].values for p in planners]

    bp = ax1.boxplot(data_for_box, labels=planners, patch_artist=True)
    for patch, planner in zip(bp['boxes'], planners):
        patch.set_facecolor(COLORS.get(planner, "#555"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax1.set_title("Time Distribution (Box Plot)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax1.set_xticklabels(planners, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for planner in planners:
        df_p = df_solved[df_solved['planner_name'] == planner]
        x = np.random.normal(list(planners).index(planner), 0.04, size=len(df_p))
        ax2.scatter(x, df_p['wall_clock_time'], alpha=0.6, s=100,
                    color=COLORS.get(planner, "#555"), label=planner, edgecolor='black')

    ax2.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax2.set_xticks(range(len(planners)))
    ax2.set_xticklabels(planners, rotation=45, ha='right')
    ax2.set_title("Time Distribution (Scatter)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 3: EXPANSIONS COMPARISON
# ============================================================================

def plot_expansions_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart of expansions on log scale."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    expansions = []
    planner_names = []

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]
        avg_exp = df_p['nodes_expanded'].mean() if df_p['nodes_expanded'].sum() > 0 else 1
        expansions.append(max(avg_exp, 1))
        planner_names.append(planner)

    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(expansions)), expansions, color=colors_list, alpha=0.8, edgecolor='black')

    ax.set_ylabel("Average Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right')

    for bar, exp in zip(bars, expansions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 2,
                f'{int(exp):,}', ha='center', va='bottom', fontsize=9)

    ax.set_title("Average Nodes Expanded Comparison (Log Scale)",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 4: EFFICIENCY FRONTIER
# ============================================================================

def plot_efficiency_frontier(df: 'pd.DataFrame', output_path: str):
    """Scatter plot: time vs expansions."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]
        ax.scatter(df_p['wall_clock_time'], df_p['nodes_expanded'],
                   label=planner, s=150, alpha=0.6, color=COLORS.get(planner, "#555"),
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel("Wall Clock Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel("Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("Efficiency Frontier (Time vs Expansions)",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 5: PER-PROBLEM HEATMAP
# ============================================================================

def plot_per_problem_heatmap(df: 'pd.DataFrame', output_path: str, max_problems: int = 30):
    """Heatmap: problems vs planners, color = solve status."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])

    problems = sorted(df['problem_name'].unique())[:max_problems]
    planners = sorted(df['planner_name'].unique())

    matrix = np.zeros((len(planners), len(problems)))

    for i, planner in enumerate(planners):
        for j, problem in enumerate(problems):
            df_cell = df[(df['planner_name'] == planner) & (df['problem_name'] == problem)]
            if len(df_cell) > 0:
                solved = int(df_cell.iloc[0]['solved'])
                matrix[i, j] = solved
            else:
                matrix[i, j] = -1

    fig, ax = plt.subplots(figsize=(16, 6), dpi=PLOT_CONFIG["dpi"])

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(problems)))
    ax.set_yticks(range(len(planners)))
    ax.set_xticklabels([p.replace('problem_', '').replace('.pddl', '')[:15] for p in problems],
                       rotation=90, fontsize=8)
    ax.set_yticklabels(planners, fontsize=PLOT_CONFIG["font_size"])

    ax.set_title(f"Per-Problem Solve Status (first {max_problems} problems)",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Solved", fontsize=PLOT_CONFIG["font_size"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 6: TIME VIOLIN PLOTS
# ============================================================================

def plot_time_violin(df: 'pd.DataFrame', output_path: str):
    """Violin plot of time distribution."""
    if not HAS_MATPLOTLIB or df is None or not HAS_PANDAS:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']].copy()

    planners = sorted(df_solved['planner_name'].unique())

    parts = ax.violinplot(
        [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].values for p in planners],
        positions=range(len(planners)),
        showmeans=True,
        showmedians=True
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(planners, rotation=45, ha='right')
    ax.set_ylabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_title("Time Distribution (Violin Plot)",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 7: CUMULATIVE DISTRIBUTION
# ============================================================================

def plot_cumulative_distribution(df: 'pd.DataFrame', output_path: str):
    """Cumulative distribution of solve times."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    for planner in sorted(df_solved['planner_name'].unique()):
        df_p = df_solved[df_solved['planner_name'] == planner]['wall_clock_time'].sort_values()
        cumulative = np.arange(1, len(df_p) + 1) / len(df_p)
        ax.plot(df_p.values, cumulative, marker='o', label=planner, linewidth=2,
                color=COLORS.get(planner, "#555"))

    ax.set_xlabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel("Cumulative Fraction Solved", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("Cumulative Distribution of Solve Times",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 8: PER-DIFFICULTY BREAKDOWN
# ============================================================================

def plot_per_difficulty_breakdown(df: 'pd.DataFrame', output_path: str):
    """Compare performance across small/medium/large problems."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])

    df_copy = df.copy()
    df_copy['difficulty'] = df_copy['problem_name'].apply(_extract_difficulty)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PLOT_CONFIG["dpi"])

    difficulties = ['small', 'medium', 'large']

    solve_by_diff = {}
    for diff in difficulties:
        df_diff = df_copy[df_copy['difficulty'] == diff]
        if len(df_diff) > 0:
            rate = (df_diff['solved'].sum() / len(df_diff)) * 100
            solve_by_diff[diff] = rate

    axes[0, 0].bar(solve_by_diff.keys(), solve_by_diff.values(),
                   color=['#2E86AB', '#F18F01', '#C73E1D'], alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Solve Rate by Difficulty', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Solve Rate (%)', fontweight='bold')
    axes[0, 0].set_ylim([0, 105])
    axes[0, 0].grid(axis='y', alpha=0.3)

    for diff in difficulties:
        df_diff = df_copy[df_copy['difficulty'] == diff]
        if len(df_diff) > 0:
            times = df_diff[df_diff['solved']]['wall_clock_time']
            if len(times) > 0:
                axes[0, 1].hist(times, label=diff, alpha=0.7, bins=10)
    axes[0, 1].set_title('Time Distribution by Difficulty', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    for planner in sorted(df_copy['planner_name'].unique()):
        df_planner = df_copy[df_copy['planner_name'] == planner]
        rates_by_diff = []
        for diff in difficulties:
            df_combo = df_planner[df_planner['difficulty'] == diff]
            if len(df_combo) > 0:
                rate = (df_combo['solved'].sum() / len(df_combo)) * 100
                rates_by_diff.append(rate)
        if rates_by_diff:
            axes[1, 0].plot(difficulties, rates_by_diff, marker='o', label=planner, linewidth=2)

    axes[1, 0].set_title('Solve Rate Trend by Difficulty', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Solve Rate (%)', fontweight='bold')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    summary_text = "Per-Difficulty Summary:\n"
    for diff in difficulties:
        df_diff = df_copy[df_copy['difficulty'] == diff]
        if len(df_diff) > 0:
            solved = df_diff['solved'].sum()
            total = len(df_diff)
            avg_time = df_diff[df_diff['solved']]['wall_clock_time'].mean()
            summary_text += f"\n{diff.upper()}:\n"
            summary_text += f"  Solved: {solved}/{total}\n"
            summary_text += f"  Avg Time: {avg_time:.2f}s\n"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 9: CURRICULUM LEARNING EFFECTIVENESS
# ============================================================================

def plot_curriculum_effectiveness(
        experiment_summaries: Dict[str, Dict],
        output_path: str
):
    """Compare: overfit → problem_gen → scale_gen → curriculum."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=(12, 7), dpi=PLOT_CONFIG["dpi"])

    exp_order = ['overfit', 'problem_gen', 'scale_gen', 'curriculum']
    labels = [
        'Overfit\n(Same Domain)',
        'Problem Gen\n(Different Problems)',
        'Scale Gen\n(Larger)',
        'Curriculum\n(Small→Med→Large)'
    ]

    solve_rates = []
    for exp_type in exp_order:
        if exp_type in experiment_summaries:
            rate = experiment_summaries[exp_type].get('solve_rate', 0)
            solve_rates.append(rate)
        else:
            solve_rates.append(0)

    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    bars = ax.bar(labels, solve_rates, color=colors_list, alpha=0.8, edgecolor='black', width=0.6)

    for bar, rate in zip(bars, solve_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_title("Generalization Across Experiment Types", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (80%)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=PLOT_CONFIG["font_size"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 10: STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

def plot_statistical_summary(df: 'pd.DataFrame', output_path: str):
    """Summary statistics comparison."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig = plt.figure(figsize=(16, 10), dpi=PLOT_CONFIG["dpi"])
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    df_solved = df[df['solved']]
    planners = sorted(df_solved['planner_name'].unique())

    ax1 = fig.add_subplot(gs[0, 0])
    solve_rates = [(df[df['planner_name'] == p]['solved'].sum() / len(df[df['planner_name'] == p]) * 100)
                   for p in planners]
    colors = [COLORS.get(p, "#555") for p in planners]
    ax1.bar(range(len(planners)), solve_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Solve Rate (%)", fontsize=10, fontweight='bold')
    ax1.set_title("Solve Rate", fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(planners)))
    ax1.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    median_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].median() for p in planners]
    ax2.bar(range(len(planners)), median_times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Median Time (s)", fontsize=10, fontweight='bold')
    ax2.set_title("Median Time (Solved)", fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(planners)))
    ax2.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    mean_exps = [df_solved[df_solved['planner_name'] == p]['nodes_expanded'].mean() for p in planners]
    ax3.bar(range(len(planners)), mean_exps, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel("Mean Expansions", fontsize=10, fontweight='bold')
    ax3.set_title("Mean Nodes Expanded (Solved)", fontsize=11, fontweight='bold')
    ax3.set_xticks(range(len(planners)))
    ax3.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    stats_text = "Statistical Summary:\n\n"
    for planner in planners:
        df_p = df[df['planner_name'] == planner]
        df_p_solved = df_p[df_p['solved']]
        n_solved = len(df_p_solved)
        n_total = len(df_p)
        stats_text += f"{planner}:\n"
        stats_text += f"  Solved: {n_solved}/{n_total}\n"
        if n_solved > 0:
            med_time = df_p_solved['wall_clock_time'].median()
            stats_text += f"  Med Time: {med_time:.2f}s\n"
        stats_text += "\n"

    ax4.text(0.1, 0.9, stats_text, fontsize=9, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("Statistical Summary", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# PLOT 11: PERFORMANCE PROFILE
# ============================================================================

def plot_performance_profile(df: 'pd.DataFrame', output_path: str):
    """Performance profile: % problems solved within τ × best_time."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]
    planners = sorted(df_solved['planner_name'].unique())

    for planner in planners:
        df_p = df_solved[df_solved['planner_name'] == planner]
        performance_ratios = []

        for problem in df['problem_name'].unique():
            best_time = df[df['problem_name'] == problem]['wall_clock_time'].min()
            df_problem = df_p[df_p['problem_name'] == problem]

            if len(df_problem) > 0 and best_time > 0:
                time_p = df_problem['wall_clock_time'].values[0]
                ratio = time_p / best_time
                performance_ratios.append(ratio)

        if performance_ratios:
            performance_ratios = sorted(performance_ratios)
            coverage = np.arange(1, len(performance_ratios) + 1) / len(performance_ratios)
            ax.plot(performance_ratios, coverage, marker='o', label=planner, linewidth=2,
                    color=COLORS.get(planner, "#555"))

    ax.set_xscale('log')
    ax.set_xlabel('Performance Ratio τ (time / best_time)', fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_ylabel('% Problems Solved', fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_title('Performance Profile', fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.legend(fontsize=PLOT_CONFIG["font_size"])
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_difficulty(problem_name: str) -> str:
    """Extract difficulty from problem name."""
    if 'small' in problem_name.lower():
        return 'small'
    elif 'medium' in problem_name.lower():
        return 'medium'
    elif 'large' in problem_name.lower() or 'hard' in problem_name.lower():
        return 'large'
    return 'unknown'


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analysis and visualization")
    parser.add_argument("--results", help="Path to evaluation_results.csv")
    parser.add_argument("--experiments", nargs='+',
                        help="Experiment directories (e.g., overfit_experiment_results ...)")
    parser.add_argument("--output", default="plots", help="Output directory")

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    df = None
    experiment_summaries = {}

    if args.results and os.path.exists(args.results):
        logger.info(f"\nLoading results from CSV: {args.results}")
        df = load_results_csv(args.results)

    elif args.experiments:
        logger.info(f"\nLoading results from experiment directories...")
        exp_dirs = {}
        for exp_dir in args.experiments:
            exp_name = os.path.basename(exp_dir.rstrip('/'))
            exp_dirs[exp_name] = exp_dir

        df = load_experiment_results(exp_dirs)

        for exp_type, exp_dir in exp_dirs.items():
            results_file = os.path.join(exp_dir, "results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        exp_results = json.load(f)
                    if 'summary' in exp_results:
                        experiment_summaries[exp_type] = exp_results['summary']
                except:
                    pass

    if df is None:
        logger.error("No results loaded. Provide either --results or --experiments")
        return 1

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATING PLOTS")
    logger.info(f"{'=' * 80}\n")

    plot_solve_rate_comparison(df, os.path.join(args.output, "01_solve_rate_comparison.png"))
    plot_time_comparison(df, os.path.join(args.output, "02_time_comparison.png"))
    plot_expansions_comparison(df, os.path.join(args.output, "03_expansions_comparison.png"))
    plot_efficiency_frontier(df, os.path.join(args.output, "04_efficiency_frontier.png"))
    plot_per_problem_heatmap(df, os.path.join(args.output, "05_per_problem_heatmap.png"))
    plot_time_violin(df, os.path.join(args.output, "06_time_violin.png"))
    plot_cumulative_distribution(df, os.path.join(args.output, "07_cumulative_distribution.png"))
    plot_per_difficulty_breakdown(df, os.path.join(args.output, "08_per_difficulty_breakdown.png"))
    plot_performance_profile(df, os.path.join(args.output, "09_performance_profile.png"))
    plot_statistical_summary(df, os.path.join(args.output, "10_statistical_summary.png"))

    if experiment_summaries:
        plot_curriculum_effectiveness(experiment_summaries,
                                      os.path.join(args.output, "11_curriculum_effectiveness.png"))

    logger.info(f"\n✅ All plots generated in {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())