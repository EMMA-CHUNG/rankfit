"""
RankFit: Segment-level ranking quality metrics for machine learning models.
"""

from .analyzer import RankFitAnalyzer
from .metrics import calculate_rankfit_v, calculate_rankfit_t
from .visualization import plot_ranking_analysis

__version__ = "0.1.0"
__all__ = [
    "RankFitAnalyzer",
    "calculate_rankfit_v",
    "calculate_rankfit_t",
    "plot_ranking_analysis",
]