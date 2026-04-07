"""
RankFit: Segment-level ranking quality metrics for machine learning models.

RankFit evaluates whether a model produces logically ordered, monotonic rankings
when predictions are grouped into operational segments such as deciles or quintiles.
It complements AUC by detecting ranking holes that AUC cannot see.

Metrics
-------
RankFit-V : Violation score (0–1). Penalises segments where a higher-scored
            bin has a lower actual event rate than a lower-scored bin.
            Higher is better. 1.0 = perfect monotonic ordering.

RankFit-T : Trend score (0–1). Measures global monotonic consistency using
            Kendall's tau. Higher is better. 1.0 = perfectly decreasing trend.

Quick start
-----------
>>> from rankfit import RankFitAnalyzer
>>> analyzer = RankFitAnalyzer(n_bins=10)
>>> results = analyzer.calculate_metrics(y_scores, y_true)
>>> print(f"RankFit-V: {results['rankfit_v']:.3f}")
>>> print(f"RankFit-T: {results['rankfit_t']:.3f}")
>>> analyzer.plot_analysis(results, title="My Model")
"""

from .analyzer import RankFitAnalyzer
from .metrics import calculate_rankfit_v, calculate_rankfit_t
from .visualization import plot_ranking_analysis, plot_granularity_comparison

__version__ = "0.2.0"
__author__ = "Emma Chung"
__email__ = "hsiaoyin.chung@gmail.com"
__license__ = "MIT"

__all__ = [
    "RankFitAnalyzer",
    "calculate_rankfit_v",
    "calculate_rankfit_t",
    "plot_ranking_analysis",
    "plot_granularity_comparison",
]
