"""
Evaluation and comparison script for OA-TSC experiments.
Independent script to fetch WandB results and generate comparisons.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def fetch_wandb_run_data(entity: str, project: str, run_id: str) -> dict:
    """
    Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get config
        config = dict(run.config)
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get history (if needed for plots)
        history = run.history()
        
        return {
            "run_id": run_id,
            "config": config,
            "summary": summary,
            "history": history,
            "name": run.name,
            "url": run.url,
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB data for {run_id}: {e}")
        return {
            "run_id": run_id,
            "config": {},
            "summary": {},
            "history": None,
            "error": str(e),
        }


def export_run_metrics(run_data: dict, results_dir: str):
    """
    Export per-run metrics to JSON.
    
    Args:
        run_data: Run data from WandB
        results_dir: Results directory path
    """
    run_id = run_data["run_id"]
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "summary": run_data["summary"],
            "config": run_data["config"],
        }, f, indent=2)
    
    print(f"Exported metrics: {metrics_path}")
    
    return metrics_path


def create_run_figures(run_data: dict, results_dir: str):
    """
    Create per-run figures.
    
    Args:
        run_data: Run data from WandB
        results_dir: Results directory path
        
    Returns:
        List of created figure paths
    """
    run_id = run_data["run_id"]
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = []
    
    # Only create figures if we have history data
    if run_data.get("history") is not None and not run_data["history"].empty:
        history = run_data["history"]
        
        # Figure 1: Accuracy over time (if available)
        if "correct" in history.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Compute running accuracy
            correct_vals = history["correct"].values
            running_acc = np.cumsum(correct_vals) / (np.arange(len(correct_vals)) + 1)
            
            ax.plot(running_acc, linewidth=2)
            ax.set_xlabel("Question Index")
            ax.set_ylabel("Running Accuracy")
            ax.set_title(f"Running Accuracy - {run_id}")
            ax.grid(True, alpha=0.3)
            
            fig_path = run_dir / "running_accuracy.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            figure_paths.append(fig_path)
            print(f"Created figure: {fig_path}")
        
        # Figure 2: Vote margin distribution (if available)
        if "vote_margin" in history.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            margins = history["vote_margin"].dropna().values
            ax.hist(margins, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel("Vote Margin")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Vote Margin Distribution - {run_id}")
            ax.grid(True, alpha=0.3)
            
            fig_path = run_dir / "vote_margin_distribution.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            figure_paths.append(fig_path)
            print(f"Created figure: {fig_path}")
        
        # Figure 3: OA-TSC specific - orthogonality vs correctness
        if "avg_orthogonality" in history.columns and "correct" in history.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            correct = history["correct"].values
            orth = history["avg_orthogonality"].values
            
            # Scatter plot
            colors = ['red' if c == 0 else 'green' for c in correct]
            ax.scatter(range(len(orth)), orth, c=colors, alpha=0.6, s=50)
            ax.set_xlabel("Question Index")
            ax.set_ylabel("Avg Orthogonality")
            ax.set_title(f"Orthogonality by Correctness - {run_id}")
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Correct'),
                Patch(facecolor='red', label='Incorrect')
            ]
            ax.legend(handles=legend_elements)
            
            fig_path = run_dir / "orthogonality_correctness.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            figure_paths.append(fig_path)
            print(f"Created figure: {fig_path}")
    
    return figure_paths


def compute_aggregated_metrics(all_run_data: list, primary_metric: str = "accuracy") -> dict:
    """
    Compute aggregated comparison metrics across runs.
    
    Args:
        all_run_data: List of run data dictionaries
        primary_metric: Primary metric name for comparison
        
    Returns:
        Aggregated metrics dictionary
    """
    # Extract metrics by run_id
    metrics_by_run = {}
    
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data.get("summary", {})
        
        metrics_by_run[run_id] = {
            "primary_metric": summary.get(primary_metric, None),
            "all_metrics": summary,
        }
    
    # Identify proposed vs baseline runs
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid]
    baseline_runs = [rid for rid in metrics_by_run.keys() if "comparative" in rid]
    
    # Find best proposed and baseline
    best_proposed = None
    best_proposed_score = -float('inf')
    
    for run_id in proposed_runs:
        score = metrics_by_run[run_id]["primary_metric"]
        if score is not None and score > best_proposed_score:
            best_proposed = run_id
            best_proposed_score = score
    
    best_baseline = None
    best_baseline_score = -float('inf')
    
    for run_id in baseline_runs:
        score = metrics_by_run[run_id]["primary_metric"]
        if score is not None and score > best_baseline_score:
            best_baseline = run_id
            best_baseline_score = score
    
    # Compute gap
    gap = None
    if best_proposed is not None and best_baseline is not None:
        gap = best_proposed_score - best_baseline_score
    
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": {
            "run_id": best_proposed,
            "score": best_proposed_score,
        },
        "best_baseline": {
            "run_id": best_baseline,
            "score": best_baseline_score,
        },
        "gap": gap,
    }
    
    return aggregated


def create_comparison_figures(all_run_data: list, aggregated_metrics: dict, 
                              results_dir: str, primary_metric: str = "accuracy"):
    """
    Create comparison figures across runs.
    
    Args:
        all_run_data: List of run data dictionaries
        aggregated_metrics: Aggregated metrics
        results_dir: Results directory path
        primary_metric: Primary metric name
        
    Returns:
        List of created figure paths
    """
    comp_dir = Path(results_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = []
    
    # Figure 1: Primary metric comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    run_ids = []
    scores = []
    colors = []
    
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        score = run_data["summary"].get(primary_metric, None)
        
        if score is not None:
            run_ids.append(run_id)
            scores.append(score)
            
            # Color code: proposed = blue, baseline = orange
            if "proposed" in run_id:
                colors.append('steelblue')
            else:
                colors.append('coral')
    
    if run_ids:
        bars = ax.bar(range(len(run_ids)), scores, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.set_ylabel(primary_metric.replace('_', ' ').title())
        ax.set_title(f"{primary_metric.replace('_', ' ').title()} Comparison")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='Proposed (OA-TSC)'),
            Patch(facecolor='coral', label='Baseline (DP-ASC)')
        ]
        ax.legend(handles=legend_elements)
        
        fig_path = comp_dir / f"{primary_metric}_comparison.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        figure_paths.append(fig_path)
        print(f"Created comparison figure: {fig_path}")
    
    # Figure 2: Multiple metrics comparison (if available)
    common_metrics = ["accuracy", "mean_weighted_vote_margin", "tokens_per_question"]
    
    # Check which metrics are available
    available_metrics = []
    for metric in common_metrics:
        if any(metric in run_data["summary"] for run_data in all_run_data):
            available_metrics.append(metric)
    
    if len(available_metrics) >= 2:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            run_ids = []
            scores = []
            colors = []
            
            for run_data in all_run_data:
                run_id = run_data["run_id"]
                score = run_data["summary"].get(metric, None)
                
                if score is not None:
                    run_ids.append(run_id)
                    scores.append(score)
                    
                    if "proposed" in run_id:
                        colors.append('steelblue')
                    else:
                        colors.append('coral')
            
            if run_ids:
                bars = ax.bar(range(len(run_ids)), scores, color=colors, alpha=0.8, edgecolor='black')
                ax.set_xticks(range(len(run_ids)))
                ax.set_xticklabels([rid.split('-')[0] for rid in run_ids], rotation=45, ha='right')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        fig_path = comp_dir / "multi_metric_comparison.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        figure_paths.append(fig_path)
        print(f"Created comparison figure: {fig_path}")
    
    return figure_paths


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate OA-TSC experiment results")
    parser.add_argument("results_dir", type=str, help="Results directory path")
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    parser.add_argument("--entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--project", type=str, default="2026-02-10", help="WandB project")
    parser.add_argument("--primary_metric", type=str, default="accuracy", help="Primary metric")
    
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    print(f"Results directory: {args.results_dir}")
    
    # Fetch data from WandB
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        run_data = fetch_wandb_run_data(args.entity, args.project, run_id)
        all_run_data.append(run_data)
    
    # Export per-run metrics and figures
    for run_data in all_run_data:
        print(f"\nProcessing {run_data['run_id']}...")
        
        # Export metrics
        export_run_metrics(run_data, args.results_dir)
        
        # Create figures
        create_run_figures(run_data, args.results_dir)
    
    # Compute aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated_metrics = compute_aggregated_metrics(all_run_data, args.primary_metric)
    
    # Export aggregated metrics
    comp_dir = Path(args.results_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    agg_path = comp_dir / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    print(f"Exported aggregated metrics: {agg_path}")
    
    # Create comparison figures
    print("\nCreating comparison figures...")
    comp_figures = create_comparison_figures(
        all_run_data, 
        aggregated_metrics, 
        args.results_dir,
        args.primary_metric
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Primary metric: {args.primary_metric}")
    
    if aggregated_metrics["best_proposed"]["run_id"]:
        print(f"\nBest proposed: {aggregated_metrics['best_proposed']['run_id']}")
        print(f"  Score: {aggregated_metrics['best_proposed']['score']:.4f}")
    
    if aggregated_metrics["best_baseline"]["run_id"]:
        print(f"\nBest baseline: {aggregated_metrics['best_baseline']['run_id']}")
        print(f"  Score: {aggregated_metrics['best_baseline']['score']:.4f}")
    
    if aggregated_metrics["gap"] is not None:
        print(f"\nGap (proposed - baseline): {aggregated_metrics['gap']:.4f}")
    
    print("\nGenerated files:")
    print(f"  Aggregated metrics: {agg_path}")
    for fig_path in comp_figures:
        print(f"  Figure: {fig_path}")
    
    print("="*60)


if __name__ == "__main__":
    main()
