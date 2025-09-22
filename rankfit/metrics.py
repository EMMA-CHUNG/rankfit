"""
Core metric calculations for RankFit.
"""

import numpy as np
from scipy import stats


def calculate_rankfit_v(event_rates, violations):
    """
    Calculate RankFit-V (Violation-based metric).
    
    Parameters
    ----------
    event_rates : array-like
        Event rates for each bin
    violations : list
        List of violation tuples
    
    Returns
    -------
    float
        RankFit-V score between 0 and 1
    """
    if len(event_rates) < 2:
        return 1.0
    
    # Sum violation magnitudes
    violation_magnitudes = [v[4] for v in violations]
    
    # Calculate range for normalization
    total_range = np.max(event_rates) - np.min(event_rates)
    
    if total_range == 0:
        return 1.0
    
    # Calculate score
    rankfit_v = max(0, 1 - (np.sum(violation_magnitudes) / total_range))
    
    return rankfit_v


def calculate_rankfit_t(event_rates):
    """
    Calculate RankFit-T (Trend-based metric).
    
    Parameters
    ----------
    event_rates : array-like
        Event rates for each bin
    
    Returns
    -------
    float
        RankFit-T score between 0 and 1
    """
    if len(event_rates) < 2:
        return 1.0
    
    # Calculate Kendall's tau
    # We expect negative correlation (higher bin number = lower rate)
    bin_indices = -np.arange(len(event_rates))
    tau, _ = stats.kendalltau(bin_indices, event_rates)
    
    # Transform to 0-1 scale
    # tau = 1: perfect positive correlation (what we want) -> 1.0
    # tau = 0: no correlation -> 0.5
    # tau = -1: perfect negative correlation -> 0.0
    rankfit_t = (tau + 1) / 2
    
    return rankfit_t if tau is not None else 0.5