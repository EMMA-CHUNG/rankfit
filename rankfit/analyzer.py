"""
Main RankFitAnalyzer class for calculating ranking quality metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from .metrics import calculate_rankfit_v, calculate_rankfit_t
from .visualization import plot_ranking_analysis


class RankFitAnalyzer:
    """
    Analyzes the ranking quality of a model by binning scores and evaluating
    the monotonicity of event rates across these bins.
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to create. Common choices:
        - 5: Quintiles (for small datasets)
        - 10: Deciles (recommended default)
        - 20: Vigintiles (for detailed analysis)
        - 100: Percentiles (for large datasets)
    
    Attributes
    ----------
    n_bins : int
        Number of bins used for analysis
    
    Examples
    --------
    >>> from rankfit import RankFitAnalyzer
    >>> analyzer = RankFitAnalyzer(n_bins=10)
    >>> results = analyzer.calculate_metrics(y_scores, y_true)
    >>> print(f"RankFit-V: {results['rankfit_v']:.3f}")
    """
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    
    def calculate_metrics(self, y_scores, y_true):
        """
        Calculate RankFit metrics for given scores and labels.
        
        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            Model prediction scores or probabilities
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'rankfit_v': Violation-based score (0-1, higher is better)
            - 'rankfit_t': Trend-based score (0-1, higher is better)
            - 'violations': List of violation details
            - 'bin_stats': DataFrame with bin-level statistics
        """
        # Create DataFrame
        df = pd.DataFrame({'score': y_scores, 'actual': y_true})
        
        # Create bins
        try:
            df['bin'] = pd.qcut(df['score'], q=self.n_bins, labels=False, 
                               duplicates='drop')
        except ValueError:
            # Handle case where there are too many ties
            df['bin'] = pd.cut(df['score'], bins=self.n_bins, labels=False, 
                              include_lowest=True)
        
        # Reverse bins so bin 0 = highest scores
        df['bin'] = df['bin'].max() - df['bin']
        
        # Calculate statistics per bin
        bin_stats = df.groupby('bin').agg(
            actual_mean=('actual', 'mean'),
            actual_count=('actual', 'count'),
            score_mean=('score', 'mean')
        ).reset_index()
        
        event_rates = bin_stats['actual_mean'].values
        
        # Find violations
        violations = []
        for i in range(len(event_rates) - 1):
            if event_rates[i + 1] > event_rates[i]:
                severity = event_rates[i + 1] - event_rates[i]
                violations.append((i, i + 1, event_rates[i], 
                                 event_rates[i + 1], severity))
        
        # Calculate metrics
        rankfit_v = calculate_rankfit_v(event_rates, violations)
        rankfit_t = calculate_rankfit_t(event_rates)
        
        return {
            'bin_stats': bin_stats,
            'rankfit_v': rankfit_v,
            'rankfit_t': rankfit_t,
            'violations': violations
        }
    
    def plot_analysis(self, results, auc_score=None, title="Ranking Quality Analysis"):
        """
        Create visualization of ranking quality analysis.
        
        Parameters
        ----------
        results : dict
            Results from calculate_metrics()
        auc_score : float, optional
            AUC score to display in title
        title : str
            Main title for the plot
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the analysis plots
        """
        return plot_ranking_analysis(results, auc_score, title)