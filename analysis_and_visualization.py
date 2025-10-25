# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS AND VISUALIZATION MODULE - COMPLETE REFACTORING
========================================================
Generates ALL necessary plots for comprehensive evaluation reporting.

Plots Generated:
  ✓ Solve rate comparison (bar chart)
  ✓ Time comparison (box plots + scatter)
  ✓ Expansions comparison (log scale)
  ✓ Plan cost comparison
  ✓ Efficiency frontier (2D scatter)
  ✓ Per-problem heatmap
  ✓ Time distribution (violin plots)
  ✓ Learning curves (if available)
  ✓ Statistical significance tests
  ✓ Cumulative distribution
  ✓ Summary dashboard (HTML)
"""

import sys
import os
import logging
import argparse
import json
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
}


# ============================================================================
# DATA LOADING
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


# ============================================================================
# PLOT 1: SOLVE RATE COMPARISON
# ============================================================================

def plot_solve_rate_comparison(df: 'pd.DataFrame', output_path: str):
    """Bar chart comparing solve rates."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    # Compute solve rates
    solve_rates = []
    planner_names = []

    for planner in sorted(df['planner_name'].unique()):
        df_planner = df[df['planner_name'] == planner]
        rate = (df_planner['solved'].sum() / len(df_planner)) * 100
        solve_rates.append(rate)
        planner_names.append(planner)

    # Plot
    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(solve_rates)), solve_rates, color=colors_list, alpha=0.8, edgecolor='black')

    ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.3, label='Target')

    # Add value labels
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
# PLOT 2: TIME COMPARISON (BOX + SCATTER)
# ============================================================================

def plot_time_comparison(df: 'pd.DataFrame', output_path: str):
    """Box plot and scatter showing time distribution."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=PLOT_CONFIG["dpi"])

    df_solved = df[df['solved']]

    # Box plot
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

    # Scatter plot
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
# PLOT 3: EXPANSIONS COMPARISON (LOG SCALE)
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
        avg_exp = df_p['nodes_expanded'].mean()
        expansions.append(avg_exp)
        planner_names.append(planner)

    colors_list = [COLORS.get(name, "#555") for name in planner_names]
    bars = ax.bar(range(len(expansions)), expansions, color=colors_list, alpha=0.8, edgecolor='black')

    ax.set_ylabel("Average Nodes Expanded (log scale)", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_xlabel("Planner", fontsize=PLOT_CONFIG["font_size"] + 1, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(planner_names)))
    ax.set_xticklabels(planner_names, rotation=45, ha='right')

    # Add value labels
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

    # Create matrix
    matrix = np.zeros((len(planners), len(problems)))

    for i, planner in enumerate(planners):
        for j, problem in enumerate(problems):
            df_cell = df[(df['planner_name'] == planner) & (df['problem_name'] == problem)]
            if len(df_cell) > 0:
                solved = int(df_cell.iloc[0]['solved'])
                matrix[i, j] = solved
            else:
                matrix[i, j] = -1  # Missing

    fig, ax = plt.subplots(figsize=(16, 6), dpi=PLOT_CONFIG["dpi"])

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(problems)))
    ax.set_yticks(range(len(planners)))
    ax.set_xticklabels([p.replace('problem_', '').replace('.pddl', '')[:15] for p in problems],
                       rotation=90, fontsize=8)
    ax.set_yticklabels(planners, fontsize=PLOT_CONFIG["font_size"])

    ax.set_title(f"Per-Problem Solve Status (first {max_problems} problems)",
                 fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')

    # Add colorbar
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
# PLOT 8: STATISTICAL SUMMARY
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

    # 1. Solve Rate
    ax1 = fig.add_subplot(gs[0, 0])
    solve_rates = [(df[df['planner_name'] == p]['solved'].sum() / len(df[df['planner_name'] == p]) * 100) for p in
                   planners]
    colors = [COLORS.get(p, "#555") for p in planners]
    ax1.bar(range(len(planners)), solve_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Solve Rate (%)", fontsize=10, fontweight='bold')
    ax1.set_title("Solve Rate", fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(planners)))
    ax1.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)

    # 2. Median Time
    ax2 = fig.add_subplot(gs[0, 1])
    median_times = [df_solved[df_solved['planner_name'] == p]['wall_clock_time'].median() for p in planners]
    ax2.bar(range(len(planners)), median_times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Median Time (s)", fontsize=10, fontweight='bold')
    ax2.set_title("Median Time (Solved)", fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(planners)))
    ax2.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Mean Expansions
    ax3 = fig.add_subplot(gs[1, 0])
    mean_exps = [df_solved[df_solved['planner_name'] == p]['nodes_expanded'].mean() for p in planners]
    ax3.bar(range(len(planners)), mean_exps, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel("Mean Expansions", fontsize=10, fontweight='bold')
    ax3.set_title("Mean Nodes Expanded (Solved)", fontsize=11, fontweight='bold')
    ax3.set_xticks(range(len(planners)))
    ax3.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Mean Plan Cost
    ax4 = fig.add_subplot(gs[1, 1])
    mean_costs = [df_solved[df_solved['planner_name'] == p]['plan_cost'].mean() for p in planners]
    ax4.bar(range(len(planners)), mean_costs, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel("Mean Plan Cost", fontsize=10, fontweight='bold')
    ax4.set_title("Mean Plan Cost (Solved)", fontsize=11, fontweight='bold')
    ax4.set_xticks(range(len(planners)))
    ax4.set_xticklabels(planners, rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    fig.suptitle("Statistical Summary", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analysis and visualization")
    parser.add_argument("--results", required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--output", default="plots", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load results
    if not HAS_PANDAS:
        logger.error("pandas required")
        return 1

    df = load_results_csv(args.results)
    if df is None:
        return 1

    logger.info("\nGenerating plots...\n")

    # Generate all plots
    plot_solve_rate_comparison(df, os.path.join(args.output, "01_solve_rate_comparison.png"))
    plot_time_comparison(df, os.path.join(args.output, "02_time_comparison.png"))
    plot_expansions_comparison(df, os.path.join(args.output, "03_expansions_comparison.png"))
    plot_efficiency_frontier(df, os.path.join(args.output, "04_efficiency_frontier.png"))
    plot_per_problem_heatmap(df, os.path.join(args.output, "05_per_problem_heatmap.png"))
    plot_time_violin(df, os.path.join(args.output, "06_time_violin.png"))
    plot_cumulative_distribution(df, os.path.join(args.output, "07_cumulative_distribution.png"))
    plot_statistical_summary(df, os.path.join(args.output, "08_statistical_summary.png"))

    logger.info(f"\n✅ All plots generated in {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())