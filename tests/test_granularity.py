"""
Tests for compare_granularities() method and plot_granularity_comparison().
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI

import numpy as np
import pytest
import matplotlib.figure

from rankfit import RankFitAnalyzer, plot_granularity_comparison


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(7)
    n = 5000
    scores = rng.random(n)
    y_true = (rng.random(n) < (0.03 + 0.18 * scores)).astype(int)
    return scores, y_true


class TestCompareGranularities:

    def test_returns_comparison_and_figure(self, sample_data):
        scores, y_true = sample_data
        comparison, fig = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10]
        )
        assert isinstance(comparison, list)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_comparison_length_matches_bins(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10, 20]
        )
        assert len(comparison) == 3

    def test_bins_sorted_in_output(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[20, 5, 10]   # unsorted input
        )
        n_bins_out = [r["n_bins"] for r in comparison]
        assert n_bins_out == sorted(n_bins_out)

    def test_each_entry_has_required_keys(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10]
        )
        required = {"n_bins", "rankfit_v", "rankfit_t",
                    "violations", "bin_stats", "auc"}
        for entry in comparison:
            assert required.issubset(entry.keys())

    def test_scores_in_unit_interval(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10, 100]
        )
        for entry in comparison:
            assert 0.0 <= entry["rankfit_v"] <= 1.0
            assert 0.0 <= entry["rankfit_t"] <= 1.0

    def test_finer_bins_can_reveal_more_violations(self, sample_data):
        """
        Percentile-level analysis should find >= violations vs decile level
        for a realistic model (not guaranteed but almost always true).
        """
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[10, 100]
        )
        viols_10  = len(comparison[0]["violations"])
        viols_100 = len(comparison[1]["violations"])
        assert viols_100 >= viols_10

    def test_invalid_bins_too_few_raises(self, sample_data):
        scores, y_true = sample_data
        with pytest.raises(ValueError, match="at least 2 values"):
            RankFitAnalyzer().compare_granularities(
                scores, y_true, bins=[10]
            )

    def test_invalid_bin_value_raises(self, sample_data):
        scores, y_true = sample_data
        with pytest.raises(ValueError, match=">= 2"):
            RankFitAnalyzer().compare_granularities(
                scores, y_true, bins=[1, 10]
            )

    def test_default_bins_work(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true   # uses default bins=(5,10,20,100)
        )
        assert len(comparison) == 4
        assert [r["n_bins"] for r in comparison] == [5, 10, 20, 100]


class TestPlotGranularityComparison:

    def test_returns_figure(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10]
        )
        fig = plot_granularity_comparison(comparison)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self, sample_data):
        scores, y_true = sample_data
        comparison, _ = RankFitAnalyzer().compare_granularities(
            scores, y_true, bins=[5, 10]
        )
        fig = plot_granularity_comparison(comparison, title="My Title")
        titles = [t.get_text() for t in fig.texts]
        assert "My Title" in titles
