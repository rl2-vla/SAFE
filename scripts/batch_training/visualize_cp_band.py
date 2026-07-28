#!/usr/bin/env python3
"""
Visualize CP (Conformal Prediction) bands from saved numpy files.

This script loads the cp_band_by_alpha.npy file and creates visualizations
similar to those in train.py's eval_save_videos_functional_cp function.

Usage:
    python visualize_cp_band.py <path_to_cp_band_by_alpha.npy>

Example:
    python visualize_cp_band.py logs/openvla-simpler_0-3-lstm-lstm/SAVED/openvla_simpler_tasks0to3_175854/cp_band_by_alpha.npy
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_cp_bands(npy_path: str, alpha: float = 0.2) -> dict:
    """
    Load the CP bands from a numpy file.

    Args:
        npy_path: Path to the cp_band_by_alpha.npy or cp_bands_by_task.npy file
        alpha: Specific alpha value to load (default: 0.2)

    Returns:
        Dictionary mapping task_name to CP band (or single CP band if pooled format)
        If pooled: {'pooled': cp_band}
        If per-task: {'task_name_1': cp_band, 'task_name_2': cp_band, ...}
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"CP band not found at {npy_path}")

    cp_band_data = np.load(npy_path, allow_pickle=True).item()

    # Detect format: check if first key is a string (per-task with task names) or float (pooled)
    first_key = list(cp_band_data.keys())[0]
    is_per_task_legacy = isinstance(first_key, (int, np.integer))
    is_per_task_new = isinstance(first_key, str)

    print(f"{'='*80}")
    print(f"Loaded CP bands from: {npy_path}")

    cp_bands_by_task = {}

    if is_per_task_new or is_per_task_legacy:
        # Per-task format: {task_name: {alpha: cp_band}} or legacy {task_id: {alpha: cp_band}}
        format_type = "PER-TASK CP bands (by task name)" if is_per_task_new else "PER-TASK CP bands (legacy by task ID)"
        print(f"Format: {format_type}")
        for task_key in sorted(cp_band_data.keys()):
            if alpha not in cp_band_data[task_key]:
                raise ValueError(f"Task '{task_key}' missing alpha={alpha}")
            cp_band = cp_band_data[task_key][alpha]
            if len(cp_band.shape) == 2 and cp_band.shape[0] == 1:
                cp_band = cp_band[0]
            cp_bands_by_task[task_key] = cp_band
            print(f"  Task '{task_key}': shape={cp_band.shape}, range=[{cp_band.min():.4f}, {cp_band.max():.4f}]")
    else:
        # Pooled format: {alpha: cp_band}
        print(f"Format: POOLED CP band")
        if alpha not in cp_band_data:
            available_alphas = list(cp_band_data.keys())
            raise ValueError(f"CP band for alpha={alpha} not found. Available: {available_alphas}")
        cp_band = cp_band_data[alpha]
        if len(cp_band.shape) == 2 and cp_band.shape[0] == 1:
            cp_band = cp_band[0]
        cp_bands_by_task['pooled'] = cp_band
        print(f"  Pooled: shape={cp_band.shape}, range=[{cp_band.min():.4f}, {cp_band.max():.4f}]")

    print(f"{'='*80}\n")

    return cp_bands_by_task


def plot_cp_bands(cp_bands_by_task: dict, alpha: float = 0.2, save_dir: str = None, exec_horizon: int = 5):
    """
    Create visualization of the CP band for a specific alpha value.

    Args:
        cp_bands_by_task: Dictionary mapping task_name to CP bands
        alpha: Alpha value to plot (default: 0.2)
        save_dir: Directory to save plots (optional)
        exec_horizon: Execution horizon for x-axis scaling (default: 5)
    """
    task_keys = sorted(cp_bands_by_task.keys())

    # If pooled, create single plot
    if task_keys == ['pooled']:
        cp_band = cp_bands_by_task['pooled']
        x = np.arange(len(cp_band)) * exec_horizon

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.plot(x, cp_band, label=f'α = {alpha}', color='green', linewidth=2)
        ax.fill_between(x, 0, cp_band, color='green', alpha=0.2)

        ax.set_xlabel('Time step $t$', fontsize=12)
        ax.set_ylabel('Score Threshold $s_t$', fontsize=12)
        ax.set_title(f'CP Band (α = {alpha})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, len(cp_band) * exec_horizon)

        plt.tight_layout()
    else:
        # Per-task: create subplots (2x2 for 4 tasks)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        axes = axes.flatten()

        for idx, task_key in enumerate(task_keys):
            cp_band = cp_bands_by_task[task_key]
            x = np.arange(len(cp_band)) * exec_horizon

            ax = axes[idx]
            ax.plot(x, cp_band, label=f'α = {alpha}', color='green', linewidth=2)
            ax.fill_between(x, 0, cp_band, color='green', alpha=0.2)

            ax.set_xlabel('Time step $t$', fontsize=12)
            ax.set_ylabel('Score Threshold $s_t$', fontsize=12)
            ax.set_title(f'{task_key}: CP Band (α = {alpha})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, len(cp_band) * exec_horizon)

        plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'cp_band_alpha_{alpha}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved α={alpha} plot to: {save_path}")

        # Also save as PDF
        save_path_pdf = os.path.join(save_dir, f'cp_band_alpha_{alpha}.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"Saved α={alpha} plot to: {save_path_pdf}")

    plt.show()

    print(f"\nGenerated CP band plot for α = {alpha}")


def print_cp_band_info(cp_bands_by_task: dict):
    """
    Print information about the CP bands.

    Args:
        cp_bands_by_task: Dictionary mapping task_name to CP band data
    """
    print("\n" + "="*79)
    print("CP Band Information".center(79))
    print("="*79)

    task_keys = sorted(cp_bands_by_task.keys())

    if task_keys == ['pooled']:
        # Pooled format
        print("\nFormat: POOLED")
        cp_band = cp_bands_by_task['pooled']
        print(f"Length: {len(cp_band)}, Min: {cp_band.min():.6f}, Max: {cp_band.max():.6f}, Mean: {cp_band.mean():.6f}")
        print("\nCP Band values:")
        print(cp_band)
    else:
        # Per-task format
        print(f"\nFormat: PER-TASK")
        print(f"Number of tasks: {len(task_keys)}\n")

        print(f"{'Task':<50} {'Length':<10} {'Min':<12} {'Max':<12} {'Mean':<12}")
        print("-" * 100)

        for task_key in task_keys:
            cp_band = cp_bands_by_task[task_key]
            print(f"{task_key:<50} {len(cp_band):<10} {cp_band.min():<12.6f} "
                  f"{cp_band.max():<12.6f} {cp_band.mean():<12.6f}")

        print("\n" + "="*100)

        # Print the actual CP band values for each task
        for task_key in task_keys:
            print(f"\nCP Band values for Task '{task_key}':")
            print(cp_bands_by_task[task_key])

    print("="*79)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CP bands from saved numpy files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--npy-path',
        type=str,
        default='/mnt/hdd/SAFE_ds/cover-pi-zero-simpler_0-3_indep_lstm/SAVED/cover_pi0_simpler_tasks0to3_4tasks_with_val_1200rollouts_PER_TASK_seeds0-7-42_20260304_031500/cp_bands_by_task.npy',
        help='Path to cp_band_by_alpha.npy file (default: logs/openvla-simpler_0-3-lstm-lstm/SAVED/openvla_simpler_tasks0to3_175854/cp_band_by_alpha.npy)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same directory as input file)'
    )
    parser.add_argument(
        '--exec-horizon',
        type=int,
        default=4,                  
        help='Execution horizon for x-axis scaling (default: 5)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Alpha value to use as primary (default: 0.2)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively'
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.npy_path):
        print(f"Error: File not found: {args.npy_path}", file=sys.stderr)
        sys.exit(1)

    # Set save directory to input file's directory if not specified
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.npy_path), 'cp_band_plots')

    # Load the CP bands
    cp_bands_by_task = load_cp_bands(args.npy_path, alpha=args.alpha)

    # Print information
    print_cp_band_info(cp_bands_by_task)

    # Create visualizations
    print(f"\nGenerating plots...")
    print(f"Save directory: {args.save_dir}")

    if args.no_show:
        plt.ioff()  # Turn off interactive mode

    plot_cp_bands(cp_bands_by_task, alpha=args.alpha, save_dir=args.save_dir, exec_horizon=args.exec_horizon)

    print(f"\n✓ Visualization complete!")
    print(f"  Plots saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
