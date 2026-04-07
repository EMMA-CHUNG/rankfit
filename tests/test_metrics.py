"""
Unit tests for rankfit.metrics.
"""

import numpy as np
import pytest

from rankfit.metrics import calculate_rankfit_v, calculate_rankfit_t, find_violations


# ── find_violations ──────────────────────────────────────────────────────────

class TestFindViolations:
    def test_perfect_ordering_no_violations(self):
        rates = [0.10, 0.09, 0.08, 0.07, 0.06]
        assert find_violations(rates) == []

    def test_single_violation_detected(self):
        rates = [0.10, 0.08, 0.09, 0.07, 0.06]
        viols = find_violations(rates)
        assert len(viols) == 1
        b_i, b_j, r_i, r_j, mag = viols[0]
        assert b_i == 1
        assert b_j == 2
        assert abs(mag - 0.01) < 1e-9

    def test_catastrophic_violation(self):
        rates = [0.10, 0.03, 0.09, 0.07, 0.06]
        viols = find_violations(rates)
        assert len(viols) == 1
        assert abs(viols[0][4] - 0.06) < 1e-9

    def test_multiple_violations(self):
        rates = [0.10, 0.12, 0.08, 0.11, 0.06]
        viols = find_violations(rates)
        assert len(viols) == 2

    def test_flat_rates_no_violations(self):
        rates = [0.05, 0.05, 0.05]
        assert find_violations(rates) == []


# ── calculate_rankfit_v ───────────────────────────────────────────────────────

class TestRankFitV:
    def test_perfect_ordering_returns_one(self):
        assert calculate_rankfit_v([0.10, 0.09, 0.08, 0.07, 0.06]) == 1.0

    def test_minor_violation(self):
        # rates [10, 8, 9, 7, 6]%: violation = 0.01, range = 0.04 -> 0.75
        result = calculate_rankfit_v([0.10, 0.08, 0.09, 0.07, 0.06])
        assert abs(result - 0.75) < 1e-9

    def test_catastrophic_violation(self):
        # rates [10, 3, 9, 7, 6]%: violation = 0.06, range = 0.07 -> 0.1428...
        result = calculate_rankfit_v([0.10, 0.03, 0.09, 0.07, 0.06])
        expected = 1.0 - 0.06 / 0.07
        assert abs(result - expected) < 1e-9

    def test_zero_range_returns_one(self):
        assert calculate_rankfit_v([0.05, 0.05, 0.05]) == 1.0

    def test_single_bin_returns_one(self):
        assert calculate_rankfit_v([0.10]) == 1.0

    def test_score_never_below_zero(self):
        # Pathological case with huge violations
        result = calculate_rankfit_v([0.01, 0.99, 0.01, 0.99])
        assert result >= 0.0

    def test_decile_example_from_paper(self):
        # From Section 4 of the paper:
        # D2 (0.90) > D1 (0.80), violation = 0.10, range = 0.80 -> 0.875
        rates = [0.80, 0.90, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
        result = calculate_rankfit_v(rates)
        assert abs(result - 0.875) < 1e-9


# ── calculate_rankfit_t ───────────────────────────────────────────────────────

class TestRankFitT:
    def test_perfect_decreasing_returns_one(self):
        result = calculate_rankfit_t([0.10, 0.09, 0.08, 0.07, 0.06])
        assert abs(result - 1.0) < 1e-9

    def test_perfect_increasing_returns_zero(self):
        result = calculate_rankfit_t([0.06, 0.07, 0.08, 0.09, 0.10])
        assert abs(result - 0.0) < 1e-9

    def test_minor_violation_five_bins(self):
        # [10, 8, 9, 7, 6]: C=9, D=1, tau=0.8 -> 0.9
        result = calculate_rankfit_t([0.10, 0.08, 0.09, 0.07, 0.06])
        assert abs(result - 0.9) < 1e-9

    def test_decile_example_from_paper(self):
        # C=44, D=1, tau=43/45=0.9556 -> RankFit-T=0.9778
        rates = [0.80, 0.90, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
        result = calculate_rankfit_t(rates)
        expected = (43 / 45 + 1.0) / 2.0
        assert abs(result - expected) < 1e-6

    def test_single_bin_returns_one(self):
        assert calculate_rankfit_t([0.10]) == 1.0

    def test_output_always_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            rates = rng.random(10).tolist()
            result = calculate_rankfit_t(rates)
            assert 0.0 <= result <= 1.0
