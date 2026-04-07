"""
Core metric calculations for RankFit.

Both metrics operate on a sequence of event rates, one per ordered bin,
where bin 0 represents the highest-scored segment.
"""

import numpy as np
from scipy import stats


def calculate_rankfit_v(event_rates, violations=None):
    """
    Calculate RankFit-V (Violation Score).

    Detects and quantifies segments where a higher-scored bin has a lower
    actual event rate than a lower-scored bin. Violations are penalised
    proportionally to their magnitude relative to the total event rate range.

    Parameters
    ----------
    event_rates : array-like of shape (n_bins,)
        Actual event rate for each bin, ordered from highest-scored (index 0)
        to lowest-scored (index n-1).
    violations : list of tuples, optional
        Pre-computed violations as returned by ``find_violations()``.
        If None, violations are computed internally.

    Returns
    -------
    float
        RankFit-V score in [0, 1]. Higher is better.
        1.0 = perfectly monotonic ordering, no violations.
        0.0 = maximum possible violation magnitude.

    Examples
    --------
    >>> calculate_rankfit_v([0.10, 0.09, 0.08, 0.07, 0.06])
    1.0
    >>> calculate_rankfit_v([0.10, 0.08, 0.09, 0.07, 0.06])
    0.75
    """
    event_rates = np.asarray(event_rates, dtype=float)

    if len(event_rates) < 2:
        return 1.0

    total_range = float(np.max(event_rates) - np.min(event_rates))

    if total_range == 0.0:
        return 1.0

    if violations is None:
        violations = find_violations(event_rates)

    total_violation = sum(v[4] for v in violations)
    return float(max(0.0, 1.0 - (total_violation / total_range)))


def calculate_rankfit_t(event_rates):
    """
    Calculate RankFit-T (Trend Score).

    Measures global monotonic consistency between bin order and event rates
    using Kendall's rank correlation coefficient (tau).

    The bin index vector is negated before computing tau so that a perfectly
    decreasing event rate sequence (bin 0 highest, bin n-1 lowest) yields
    tau = +1, which maps to RankFit-T = 1.0.

    Parameters
    ----------
    event_rates : array-like of shape (n_bins,)
        Actual event rate for each bin, ordered from highest-scored (index 0)
        to lowest-scored (index n-1).

    Returns
    -------
    float
        RankFit-T score in [0, 1]. Higher is better.
        1.0 = perfectly decreasing trend.
        0.5 = no discernible trend.
        0.0 = perfectly increasing trend (worst possible).

    Examples
    --------
    >>> calculate_rankfit_t([0.10, 0.09, 0.08, 0.07, 0.06])
    1.0
    >>> round(calculate_rankfit_t([0.80, 0.90, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10]), 3)
    0.978
    """
    event_rates = np.asarray(event_rates, dtype=float)

    if len(event_rates) < 2:
        return 1.0

    # Negate bin indices so that descending event rates produce tau = +1.
    # Without negation a perfect model gives tau = -1 -> RankFit-T = 0.
    bin_indices = -np.arange(len(event_rates), dtype=float)
    tau, _ = stats.kendalltau(bin_indices, event_rates)

    if np.isnan(tau):
        return 0.5

    return float((tau + 1.0) / 2.0)


def find_violations(event_rates):
    """
    Identify all ranking violations in an event rate sequence.

    A violation occurs at position k when ``event_rates[k+1] > event_rates[k]``,
    i.e. a lower-scored bin has a higher event rate than a higher-scored bin.

    Parameters
    ----------
    event_rates : array-like of shape (n_bins,)
        Event rates ordered from highest-scored bin (index 0) to lowest-scored.

    Returns
    -------
    list of tuple
        Each tuple contains:
        ``(bin_i, bin_j, rate_i, rate_j, magnitude)``
        where ``bin_i < bin_j`` and ``rate_j > rate_i``.

    Examples
    --------
    >>> find_violations([0.10, 0.08, 0.09, 0.07, 0.06])
    [(1, 2, 0.08, 0.09, 0.01)]
    """
    event_rates = np.asarray(event_rates, dtype=float)
    violations = []
    for i in range(len(event_rates) - 1):
        diff = float(event_rates[i + 1] - event_rates[i])
        if diff > 0:
            violations.append((i, i + 1, float(event_rates[i]),
                               float(event_rates[i + 1]), diff))
    return violations
