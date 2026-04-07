"""
Visualization utilities for RankFit analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


# Bin-count -> label lookup (falls back to generic "Bin")
_BIN_LABELS = {
    4:   ("Quartile",   25),
    5:   ("Quintile",   20),
    10:  ("Decile",     10),
    20:  ("Vigintile",   5),
    100: ("Percentile",  1),
}


def plot_ranking_analysis(results, auc_score=None,
                          title="Ranking Quality Analysis", n_bins=None):
    """
    Create a two-panel visualisation of ranking quality.

    Left panel  — Actual event rates per bin with violations highlighted.
    Right panel — Average model score per bin (confirms score ordering).

    Parameters
    ----------
    results : dict
        Output of ``RankFitAnalyzer.calculate_metrics()``.
    auc_score : float, optional
        AUC score to show in the subtitle.
    title : str
        Main chart title.
    n_bins : int, optional
        Number of bins. Used to generate the correct axis label
        (e.g. "Decile", "Quintile"). Inferred from ``results`` when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bin_stats  = results["bin_stats"]
    violations = results["violations"]
    rankfit_v  = results["rankfit_v"]
    rankfit_t  = results["rankfit_t"]

    # Infer n_bins when not supplied
    if n_bins is None:
        n_bins = int(bin_stats["bin"].max()) + 1

    bin_label, top_pct = _BIN_LABELS.get(n_bins, ("Bin", round(100 / n_bins, 1)))

    # Trend quality label
    if rankfit_t >= 0.95:
        trend_desc = "Perfectly decreasing"
    elif rankfit_t >= 0.80:
        trend_desc = "Strongly decreasing"
    elif rankfit_t >= 0.60:
        trend_desc = "Weakly decreasing"
    else:
        trend_desc = "Non-monotonic"

    if violations:
        trend_desc += " (violations present)"

    # Build subtitle
    parts = []
    if auc_score is not None:
        parts.append(f"AUC: {auc_score:.3f}")
    parts += [
        f"Violations: {len(violations)}",
        f"RankFit-V: {rankfit_v:.3f}",
        f"RankFit-T: {rankfit_t:.3f}",
        f"Trend: {trend_desc}",
    ]
    subtitle = "  |  ".join(parts[:2]) + "\n" + "  |  ".join(parts[2:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    bins  = bin_stats["bin"].values
    rates = bin_stats["actual_mean"].values

    # ── Left panel: event rates ──────────────────────────────────────────
    ax1.bar(bins, rates, color="steelblue", edgecolor="white",
            alpha=0.85, label="Event rate")
    ax1.plot(bins, rates, "o-", color="navy", linewidth=1.5, markersize=6)

    for idx, (b_i, b_j, r_i, r_j, _) in enumerate(violations):
        ax1.plot([b_i, b_j], [r_i, r_j], "r-o",
                 linewidth=3, markersize=10,
                 label="Ranking violation" if idx == 0 else "")
        ax1.annotate(
            f"\u26a0 Violation\n+{(r_j - r_i):.3f}",
            xy=(b_j, r_j),
            xytext=(b_j + 0.3, r_j + 0.005),
            fontsize=8,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )

    ax1.set_xlabel(f"Score {bin_label} (0 = Top {top_pct}%)", fontsize=12)
    ax1.set_ylabel("Actual Event Rate", fontsize=12)
    ax1.set_title(subtitle, fontsize=12)
    ax1.set_xticks(bins)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # ── Right panel: average scores ──────────────────────────────────────
    ax2.bar(bins, bin_stats["score_mean"].values,
            color="mediumseagreen", edgecolor="white", alpha=0.85)
    ax2.set_xlabel(f"Score {bin_label} (0 = Top {top_pct}%)", fontsize=12)
    ax2.set_ylabel("Average Predicted Score", fontsize=12)
    ax2.set_title("Average Score Distribution by Bin", fontsize=12)
    ax2.set_xticks(bins)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return fig


def plot_granularity_comparison(comparison, title="RankFit Across Granularities"):
    """
    Visualise RankFit metrics across multiple bin granularities.

    Produces a three-panel figure:
      - Top row: event rate bar charts for each bin size (one panel each)
      - Bottom row: RankFit-V and RankFit-T trend lines across granularities,
        plus a violations count bar chart

    Parameters
    ----------
    comparison : list of dict
        Output of ``RankFitAnalyzer.compare_granularities()``.
    title : str
        Main chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    n_gran = len(comparison)

    # ── Layout: top row = one event-rate chart per granularity
    #            bottom row = 3 summary panels
    fig = plt.figure(figsize=(max(18, 5 * n_gran), 12))
    gs  = gridspec.GridSpec(2, max(n_gran, 3), figure=fig,
                            hspace=0.45, wspace=0.35)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(n_gran)]
    ax_v     = fig.add_subplot(gs[1, 0])
    ax_t     = fig.add_subplot(gs[1, 1])
    ax_viols = fig.add_subplot(gs[1, 2])

    palette  = plt.cm.Blues(np.linspace(0.45, 0.85, n_gran))
    red      = "#E53935"
    label_map = {5: "Quintiles", 10: "Deciles",
                 20: "Vigintiles", 100: "Percentiles"}

    # ── Summary metric collectors ────────────────────────────────────────
    gran_labels  = []
    v_scores     = []
    t_scores     = []
    viol_counts  = []

    for idx, entry in enumerate(comparison):
        n_bins     = entry["n_bins"]
        bin_stats  = entry["bin_stats"]
        violations = entry["violations"]
        rv         = entry["rankfit_v"]
        rt         = entry["rankfit_t"]

        gran_label = label_map.get(n_bins, f"{n_bins} bins")
        gran_labels.append(gran_label)
        v_scores.append(rv)
        t_scores.append(rt)
        viol_counts.append(len(violations))

        bin_label, top_pct = _BIN_LABELS.get(n_bins, ("Bin", round(100 / n_bins, 1)))

        ax = top_axes[idx]
        bins_x = bin_stats["bin"].values
        rates  = bin_stats["actual_mean"].values

        # Bar colours: red for violation positions, blue otherwise
        viol_bins = {v[1] for v in violations}
        colors = [red if b in viol_bins else palette[idx] for b in bins_x]

        ax.bar(bins_x, rates, color=colors, edgecolor="white",
               alpha=0.9, width=0.8)
        ax.plot(bins_x, rates, "o-", color="navy",
                linewidth=1.2, markersize=4, zorder=3)

        # Annotate violations
        for b_i, b_j, r_i, r_j, mag in violations:
            ax.annotate("", xy=(b_j, r_j), xytext=(b_i, r_i),
                        arrowprops=dict(arrowstyle="->",
                                        color=red, lw=1.8))

        ax.set_title(
            f"{gran_label}  ({n_bins} bins)\n"
            f"V={rv:.3f}  T={rt:.3f}  violations={len(violations)}",
            fontsize=10, pad=6
        )
        ax.set_xlabel(f"Bin  (0 = top {top_pct}%)", fontsize=9)
        ax.set_ylabel("Event rate" if idx == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Show x-ticks only when few enough to be readable
        if n_bins <= 20:
            ax.set_xticks(bins_x)
        else:
            step = max(1, n_bins // 10)
            ax.set_xticks(bins_x[::step])

        # Violation legend patch
        if violations:
            patch = mpatches.Patch(color=red, alpha=0.85, label="Violation")
            ax.legend(handles=[patch], fontsize=8, loc="upper right")

    # ── Bottom panel 1: RankFit-V across granularities ──────────────────
    x = np.arange(n_gran)
    bar_colors_v = [
        "#43A047" if v >= 0.95 else
        "#66BB6A" if v >= 0.80 else
        "#FFA726" if v >= 0.60 else
        "#EF5350"
        for v in v_scores
    ]
    ax_v.bar(x, v_scores, color=bar_colors_v, edgecolor="white",
             alpha=0.9, width=0.55)
    ax_v.plot(x, v_scores, "o--", color="navy",
              linewidth=1.4, markersize=7, zorder=3)
    for xi, val in zip(x, v_scores):
        ax_v.text(xi, val + 0.01, f"{val:.3f}", ha="center",
                  fontsize=9, fontweight="bold")
    ax_v.axhline(0.80, color="orange", linewidth=1, linestyle=":",
                 label="Threshold (0.80)")
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(gran_labels, fontsize=9)
    ax_v.set_ylim(0, 1.12)
    ax_v.set_ylabel("RankFit-V", fontsize=10)
    ax_v.set_title("Violation score by granularity", fontsize=11)
    ax_v.legend(fontsize=8)
    ax_v.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Bottom panel 2: RankFit-T across granularities ──────────────────
    bar_colors_t = [
        "#43A047" if t >= 0.95 else
        "#66BB6A" if t >= 0.80 else
        "#FFA726" if t >= 0.60 else
        "#EF5350"
        for t in t_scores
    ]
    ax_t.bar(x, t_scores, color=bar_colors_t, edgecolor="white",
             alpha=0.9, width=0.55)
    ax_t.plot(x, t_scores, "o--", color="navy",
              linewidth=1.4, markersize=7, zorder=3)
    for xi, val in zip(x, t_scores):
        ax_t.text(xi, val + 0.01, f"{val:.3f}", ha="center",
                  fontsize=9, fontweight="bold")
    ax_t.axhline(0.80, color="orange", linewidth=1, linestyle=":",
                 label="Threshold (0.80)")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(gran_labels, fontsize=9)
    ax_t.set_ylim(0, 1.12)
    ax_t.set_ylabel("RankFit-T", fontsize=10)
    ax_t.set_title("Trend score by granularity", fontsize=11)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Bottom panel 3: violation count across granularities ────────────
    viol_colors = [red if v > 0 else "#66BB6A" for v in viol_counts]
    ax_viols.bar(x, viol_counts, color=viol_colors, edgecolor="white",
                 alpha=0.9, width=0.55)
    for xi, val in zip(x, viol_counts):
        ax_viols.text(xi, val + 0.05, str(val), ha="center",
                      fontsize=10, fontweight="bold",
                      color=red if val > 0 else "#2E7D32")
    ax_viols.set_xticks(x)
    ax_viols.set_xticklabels(gran_labels, fontsize=9)
    ax_viols.set_ylabel("Number of violations", fontsize=10)
    ax_viols.set_title("Violation count by granularity", fontsize=11)
    ax_viols.yaxis.get_major_locator().set_params(integer=True)
    ax_viols.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Key insight annotation ───────────────────────────────────────────
    max_v_idx = int(np.argmax(viol_counts))
    min_v_idx = int(np.argmin(viol_counts))
    if max_v_idx != min_v_idx:
        insight = (
            f"Finer bins reveal more violations: "
            f"{gran_labels[min_v_idx]} shows {viol_counts[min_v_idx]}, "
            f"{gran_labels[max_v_idx]} shows {viol_counts[max_v_idx]}."
        )
        fig.text(0.5, 0.01, insight, ha="center", fontsize=10,
                 color="#555555", style="italic")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig
