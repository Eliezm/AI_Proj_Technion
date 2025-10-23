#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSIS AND VISUALIZATION MODULE
==================================
Generates plots, learning curves, and statistical analysis from evaluation results.

Creates:
  - Comparison plots (solve rate, time, expansions)
  - Learning curves from TensorBoard logs
  - Efficiency frontier analysis
  - Statistical significance testing
  - HTML interactive dashboard

Usage:
    python analysis_and_visualization.py \
        --results evaluation_results/evaluation_results.csv \
        --tb-logs tb_logs/ \
        --output plots/
"""

import sys
import os
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

# Data science imports
try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed - visualization skipped")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "figsize": (12, 7),
    "dpi": 150,
    "font_size": 11,
}

COLORS = {
    "GNN": "#2E86AB",
    "FD LM-Cut": "#A23B72",
    "FD Blind": "#F18F01",
    "FD Add": "#C73E1D",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results_csv(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV."""
    if not HAS_PANDAS:
        logger.warning("pandas not available - skipping CSV loading")
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} result rows")
    return df


def load_tensorboard_logs(log_dir: str) -> Dict:
    """Extract learning curves from TensorBoard logs."""
    import os
    from pathlib import Path

    # TensorBoard data is in events files
    # For now, return placeholder
    logger.warning("TensorBoard log extraction not yet implemented")
    return {}


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_solve_rate_comparison(df: pd.DataFrame, output_path: str):
    """Plot solve rate comparison across planners."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    # Compute solve rates
    solve_rates = df.groupby("planner_name")["solved"].apply(lambda x: (x.sum() / len(x)) * 100)

    # Plot
    bars = ax.bar(range(len(solve_rates)), solve_rates.values,
                  color=[COLORS.get(name, "#555") for name in solve_rates.index])
    ax.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_xticks(range(len(solve_rates)))
    ax.set_xticklabels(solve_rates.index, rotation=45, ha='right')
    ax.set_ylim([0, 105])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.0f}%', ha='center', va='bottom')

    ax.set_title("Solve Rate Comparison", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_time_comparison(df: pd.DataFrame, output_path: str):
    """Plot average time comparison (solved problems only)."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    # Filter to solved only
    df_solved = df[df["solved"]]

    # Compute average times
    avg_times = df_solved.groupby("planner_name")["time_sec"].mean()

    bars = ax.bar(range(len(avg_times)), avg_times.values, color=[COLORS.get(name, "#555") for name in avg_times.index])
    ax.set_ylabel("Average Time (seconds)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_xticks(range(len(avg_times)))
    ax.set_xticklabels(avg_times.index, rotation=45, ha='right')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}s', ha='center', va='bottom')

    ax.set_title("Average Solution Time (Solved Problems)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_expansions_comparison(df: pd.DataFrame, output_path: str):
    """Plot expansions comparison (log scale)."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    # Filter to solved only
    df_solved = df[df["solved"]]

    # Compute average expansions
    avg_exp = df_solved.groupby("planner_name")["expansions"].mean()

    bars = ax.bar(range(len(avg_exp)), avg_exp.values, color=[COLORS.get(name, "#555") for name in avg_exp.index])
    ax.set_ylabel("Average Expansions (log scale)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_yscale('log')
    ax.set_xticks(range(len(avg_exp)))
    ax.set_xticklabels(avg_exp.index, rotation=45, ha='right')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.2,
                f'{height:.0f}', ha='center', va='bottom')

    ax.set_title("Average Nodes Expanded (Solved Problems, Log Scale)", fontsize=PLOT_CONFIG["font_size"] + 2,
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_efficiency_frontier(df: pd.DataFrame, output_path: str):
    """Plot efficiency frontier (time vs expansions)."""
    if not HAS_MATPLOTLIB or df is None:
        return

    plt.style.use(PLOT_CONFIG["style"])
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"], dpi=PLOT_CONFIG["dpi"])

    # Filter to solved only
    df_solved = df[df["solved"]]

    # Plot each planner
    for planner in df_solved["planner_name"].unique():
        df_planner = df_solved[df_solved["planner_name"] == planner]
        ax.scatter(df_planner["time_sec"], df_planner["expansions"],
                   label=planner, s=100, alpha=0.6, color=COLORS.get(planner, "#555"))

    ax.set_xlabel("Time (seconds)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Nodes Expanded", fontsize=PLOT_CONFIG["font_size"])
    ax.set_yscale('log')
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1)
    ax.set_title("Efficiency Frontier (Time vs Expansions)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_problem_comparison(df: pd.DataFrame, output_path: str, max_problems: int = 20):
    """Plot per-problem time and solve rate."""
    if not HAS_MATPLOTLIB or df is None or not HAS_PANDAS:
        return

    # Get unique problems (limit to first N)
    problems = sorted(df["problem_name"].unique())[:max_problems]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=PLOT_CONFIG["dpi"])

    # Plot 1: Solve rate per problem
    for planner in df["planner_name"].unique():
        df_planner = df[df["planner_name"] == planner]
        solve_rates = []
        for problem in problems:
            df_problem = df_planner[df_planner["problem_name"] == problem]
            rate = (df_problem["solved"].sum() / len(df_problem) * 100) if len(df_problem) > 0 else 0
            solve_rates.append(rate)
        ax1.plot(problems, solve_rates, marker='o', label=planner, color=COLORS.get(planner, "#555"))

    ax1.set_ylabel("Solve Rate (%)", fontsize=PLOT_CONFIG["font_size"])
    ax1.set_title("Per-Problem Solve Rate", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(problems, rotation=45, ha='right')

    # Plot 2: Average time per problem
    for planner in df["planner_name"].unique():
        df_planner = df[df["planner_name"] == planner]
        avg_times = []
        for problem in problems:
            df_problem = df_planner[df_planner["problem_name"] == problem]
            df_solved = df_problem[df_problem["solved"]]
            avg_time = df_solved["time_sec"].mean() if len(df_solved) > 0 else 0
            avg_times.append(avg_time)
        ax2.plot(problems, avg_times, marker='s', label=planner, color=COLORS.get(planner, "#555"))

    ax2.set_ylabel("Average Time (seconds)", fontsize=PLOT_CONFIG["font_size"])
    ax2.set_xlabel("Problem", fontsize=PLOT_CONFIG["font_size"])
    ax2.set_title("Per-Problem Average Time (Solved)", fontsize=PLOT_CONFIG["font_size"] + 2, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels(problems, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analysis and visualization")
    parser.add_argument("--results", required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--tb-logs", help="Path to TensorBoard logs")
    parser.add_argument("--output", default="plots", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if not HAS_PANDAS:
        logger.error("pandas required for analysis")
        return 1

    df = load_results_csv(args.results)
    if df is None:
        return 1

    logger.info("Generating plots...")

    # Generate all plots
    plot_solve_rate_comparison(df, str(output_dir / "solve_rate_comparison.png"))
    plot_time_comparison(df, str(output_dir / "time_comparison.png"))
    plot_expansions_comparison(df, str(output_dir / "expansions_comparison.png"))
    plot_efficiency_frontier(df, str(output_dir / "efficiency_frontier.png"))
    plot_per_problem_comparison(df, str(output_dir / "per_problem_comparison.png"))

    logger.info(f"✅ All plots generated in {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())