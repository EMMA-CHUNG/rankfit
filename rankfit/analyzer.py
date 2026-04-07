"""
Main RankFitAnalyzer class.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import calculate_rankfit_v, calculate_rankfit_t, find_violations
from .visualization import plot_ranking_analysis, plot_granularity_comparison


class RankFitAnalyzer:
    """
    Analyse the ranking quality of a binary classification model.

    Predictions are binned into equal-frequency (quantile) segments.
    Within each bin the actual event rate is computed and the two RankFit
    metrics assess whether those rates decrease monotonically from the
    highest-scored bin to the lowest-scored bin.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins. Common choices:

        * ``5``   – Quintiles (small datasets or quick checks)
        * ``10``  – Deciles (recommended default)
        * ``20``  – Vigintiles (detailed analysis)
        * ``100`` – Percentiles (high-stakes decisions; always use for
                    healthcare, fraud, or credit where fine-grained
                    violations matter)

    Examples
    --------
    >>> import numpy as np
    >>> from rankfit import RankFitAnalyzer
    >>> rng = np.random.default_rng(42)
    >>> y_scores = rng.random(1000)
    >>> y_true   = (rng.random(1000) < 0.1).astype(int)
    >>> analyzer = RankFitAnalyzer(n_bins=10)
    >>> results  = analyzer.calculate_metrics(y_scores, y_true)
    >>> print(f"RankFit-V: {results['rankfit_v']:.3f}")
    >>> print(f"RankFit-T: {results['rankfit_t']:.3f}")
    """

    def __init__(self, n_bins: int = 10):
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("`n_bins` must be an integer >= 2.")
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_metrics(self, y_scores, y_true):
        """
        Calculate RankFit-V and RankFit-T for a set of predictions.

        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            Model prediction scores or probabilities. Higher values should
            correspond to higher predicted probability of the positive class.
        y_true : array-like of shape (n_samples,)
            Ground-truth binary labels (0 or 1).

        Returns
        -------
        dict with keys:
            ``rankfit_v`` : float
                Violation score (0–1, higher is better).
            ``rankfit_t`` : float
                Trend score (0–1, higher is better).
            ``violations`` : list of tuple
                Each entry is ``(bin_i, bin_j, rate_i, rate_j, magnitude)``.
            ``bin_stats`` : pandas.DataFrame
                Per-bin statistics with columns
                ``bin, actual_mean, actual_count, score_mean``.
            ``auc`` : float or None
                ROC-AUC score, or ``None`` if only one class is present.

        Raises
        ------
        ValueError
            If ``y_scores`` and ``y_true`` have different lengths, or if
            ``y_true`` contains values other than 0 and 1.
        """
        y_scores = np.asarray(y_scores, dtype=float)
        y_true   = np.asarray(y_true,   dtype=float)

        self._validate_inputs(y_scores, y_true)

        df = pd.DataFrame({"score": y_scores, "actual": y_true})
        df["bin"] = self._assign_bins(df["score"])

        bin_stats = (
            df.groupby("bin")
            .agg(actual_mean=("actual", "mean"),
                 actual_count=("actual", "count"),
                 score_mean=("score", "mean"))
            .reset_index()
        )

        event_rates = bin_stats["actual_mean"].values
        violations  = find_violations(event_rates)

        auc = None
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_scores))

        return {
            "rankfit_v":  calculate_rankfit_v(event_rates, violations),
            "rankfit_t":  calculate_rankfit_t(event_rates),
            "violations": violations,
            "bin_stats":  bin_stats,
            "auc":        auc,
        }

    def plot_analysis(self, results, title="Ranking Quality Analysis"):
        """
        Visualise ranking quality from ``calculate_metrics()`` output.

        Parameters
        ----------
        results : dict
            Output of ``calculate_metrics()``.
        title : str, optional
            Main chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        return plot_ranking_analysis(results, auc_score=results.get("auc"),
                                     title=title, n_bins=self.n_bins)

    def summary(self, results):
        """
        Return a human-readable summary string.

        Parameters
        ----------
        results : dict
            Output of ``calculate_metrics()``.

        Returns
        -------
        str
        """
        v   = results["rankfit_v"]
        t   = results["rankfit_t"]
        auc = results["auc"]
        n_v = len(results["violations"])

        lines = [
            "=" * 45,
            "  RankFit Summary",
            "=" * 45,
        ]
        if auc is not None:
            lines.append(f"  AUC         : {auc:.4f}")
        lines += [
            f"  RankFit-V   : {v:.4f}  ({self._grade(v)})",
            f"  RankFit-T   : {t:.4f}  ({self._grade(t)})",
            f"  Violations  : {n_v}",
            f"  Bins        : {self.n_bins}",
            "=" * 45,
        ]
        if n_v:
            lines.append("  Violation details:")
            for b_i, b_j, r_i, r_j, mag in results["violations"]:
                lines.append(
                    f"    Bin {b_i} -> Bin {b_j} : "
                    f"{r_i:.4f} -> {r_j:.4f}  (magnitude {mag:.4f})"
                )
            lines.append("=" * 45)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_bins(self, scores: pd.Series) -> pd.Series:
        """Assign quantile bins; fall back to equal-width on heavy ties."""
        try:
            bins = pd.qcut(scores, q=self.n_bins, labels=False,
                           duplicates="drop")
        except ValueError:
            bins = pd.cut(scores, bins=self.n_bins, labels=False,
                          include_lowest=True)

        # Reverse so that bin 0 = highest scores (most predictive segment)
        return bins.max() - bins

    @staticmethod
    def _validate_inputs(y_scores, y_true):
        if len(y_scores) != len(y_true):
            raise ValueError(
                f"`y_scores` and `y_true` must have the same length "
                f"(got {len(y_scores)} and {len(y_true)})."
            )
        unique = np.unique(y_true)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                "`y_true` must contain only binary labels (0 and 1)."
            )

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.95:
            return "Excellent"
        if score >= 0.80:
            return "Good"
        if score >= 0.60:
            return "Moderate"
        return "Poor — review violations"

    def compare_granularities(self, y_scores, y_true,
                              bins=(5, 10, 20, 100),
                              title="RankFit Across Granularities"):
        """
        Run RankFit at multiple bin sizes and compare the results.

        This is the recommended diagnostic workflow for high-stakes models.
        Coarser bins (e.g. quintiles) can smooth over violations that only
        become visible at finer granularity (e.g. percentiles). Running
        across multiple bin sizes reveals whether violations are localised
        or structural.

        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            Model prediction scores or probabilities.
        y_true : array-like of shape (n_samples,)
            Ground-truth binary labels (0 or 1).
        bins : sequence of int, default=(5, 10, 20, 100)
            Bin counts to evaluate. Each must be >= 2.
        title : str, optional
            Main chart title.

        Returns
        -------
        comparison : list of dict
            One entry per bin size, each containing:
            ``n_bins, rankfit_v, rankfit_t, violations, bin_stats, auc``.
        fig : matplotlib.figure.Figure
            Side-by-side visualisation across all granularities.

        Examples
        --------
        >>> analyzer = RankFitAnalyzer()
        >>> comparison, fig = analyzer.compare_granularities(
        ...     y_scores, y_true, bins=[10, 100]
        ... )
        >>> for row in comparison:
        ...     print(f"{row['n_bins']:>4} bins  "
        ...           f"V={row['rankfit_v']:.3f}  "
        ...           f"T={row['rankfit_t']:.3f}  "
        ...           f"violations={len(row['violations'])}")
        """
        bins = list(bins)
        if len(bins) < 2:
            raise ValueError("`bins` must contain at least 2 values to compare.")
        if any(b < 2 for b in bins):
            raise ValueError("All values in `bins` must be >= 2.")

        y_scores = np.asarray(y_scores, dtype=float)
        y_true   = np.asarray(y_true,   dtype=float)
        self._validate_inputs(y_scores, y_true)

        comparison = []
        for n in sorted(bins):
            tmp     = RankFitAnalyzer(n_bins=n)
            results = tmp.calculate_metrics(y_scores, y_true)
            results["n_bins"] = n
            comparison.append(results)

        fig = plot_granularity_comparison(comparison, title=title)
        return comparison, fig
