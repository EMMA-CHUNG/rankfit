"""
Visualization utilities for RankFit analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_ranking_analysis(results, auc_score=None, title="Ranking Quality Analysis"):
    """
    Create comprehensive visualization of ranking quality.
    
    Parameters
    ----------
    results : dict
        Results from RankFitAnalyzer.calculate_metrics()
    auc_score : float, optional
        AUC score to display
    title : str
        Main title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    bin_stats = results['bin_stats']
    violations = results['violations']
    rankfit_t = results['rankfit_t']
    rankfit_v = results['rankfit_v']
    
    # Determine trend description
    if rankfit_t > 0.95:
        trend_desc = "Perfectly Decreasing"
    elif rankfit_t > 0.7:
        trend_desc = "Strongly Decreasing"
    elif rankfit_t > 0.5:
        trend_desc = "Weakly Decreasing"
    else:
        trend_desc = "Non-Decreasing"
    
    if len(violations) > 0:
        trend_desc += " (non-monotonic)"
    
    # Build metrics title
    metrics_parts = []
    if auc_score is not None:
        metrics_parts.append(f"AUC: {auc_score:.3f}")
    metrics_parts.extend([
        f"Violations: {len(violations)}",
        f"RankFit-V: {rankfit_v:.3f}",
        f"RankFit-T: {rankfit_t:.3f}",
        f"Trend: {trend_desc}"
    ])
    metrics_title = " | ".join(metrics_parts[:2]) + "\n" + " | ".join(metrics_parts[2:])
    
    # Plot 1: Event rates
    ax1.bar(bin_stats['bin'], bin_stats['actual_mean'], 
            color='skyblue', edgecolor='black', alpha=0.8, label='Event Rate')
    ax1.plot(bin_stats['bin'], bin_stats['actual_mean'], 'o-', color='navy')
    
    # Highlight violations
    if violations:
        for i, v in enumerate(violations):
            ax1.plot([v[0], v[1]], [v[2], v[3]], 'r-o', linewidth=3, 
                    markersize=10, label='Violation' if i == 0 else "")
    
    ax1.set_xlabel('Score Decile (0 = Top 10%)', fontsize=12)
    ax1.set_ylabel('Actual Event Rate', fontsize=12)
    ax1.set_title(metrics_title, fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.set_xticks(bin_stats['bin'])
    
    # Plot 2: Average scores
    ax2.bar(bin_stats['bin'], bin_stats['score_mean'], 
            color='lightgreen', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Score Decile (0 = Top 10%)', fontsize=12)
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('Average Score Distribution by Decile', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(bin_stats['bin'])
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    return fig