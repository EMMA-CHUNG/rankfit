"""
Integration tests for RankFitAnalyzer.
"""

import numpy as np
import pytest

from rankfit import RankFitAnalyzer


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 2000
    scores = rng.random(n)
    # Positives slightly more likely at high scores
    y_true = (rng.random(n) < (0.05 + 0.2 * scores)).astype(int)
    return scores, y_true


class TestRankFitAnalyzerInit:
    def test_default_n_bins(self):
        a = RankFitAnalyzer()
        assert a.n_bins == 10

    def test_custom_n_bins(self):
        a = RankFitAnalyzer(n_bins=5)
        assert a.n_bins == 5

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError):
            RankFitAnalyzer(n_bins=1)

    def test_non_int_n_bins_raises(self):
        with pytest.raises(ValueError):
            RankFitAnalyzer(n_bins=10.5)


class TestCalculateMetrics:
    def test_returns_expected_keys(self, sample_data):
        scores, y_true = sample_data
        results = RankFitAnalyzer().calculate_metrics(scores, y_true)
        assert set(results.keys()) == {"rankfit_v", "rankfit_t",
                                        "violations", "bin_stats", "auc"}

    def test_scores_in_unit_interval(self, sample_data):
        scores, y_true = sample_data
        results = RankFitAnalyzer().calculate_metrics(scores, y_true)
        assert 0.0 <= results["rankfit_v"] <= 1.0
        assert 0.0 <= results["rankfit_t"] <= 1.0

    def test_auc_computed(self, sample_data):
        scores, y_true = sample_data
        results = RankFitAnalyzer().calculate_metrics(scores, y_true)
        assert results["auc"] is not None
        assert 0.0 <= results["auc"] <= 1.0

    def test_auc_none_when_single_class(self):
        scores  = np.linspace(0, 1, 100)
        y_true  = np.zeros(100, dtype=int)
        results = RankFitAnalyzer().calculate_metrics(scores, y_true)
        assert results["auc"] is None

    def test_bin_stats_shape(self, sample_data):
        scores, y_true = sample_data
        results = RankFitAnalyzer(n_bins=10).calculate_metrics(scores, y_true)
        assert len(results["bin_stats"]) <= 10

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            RankFitAnalyzer().calculate_metrics([0.1, 0.2], [1])

    def test_invalid_labels_raises(self):
        with pytest.raises(ValueError):
            RankFitAnalyzer().calculate_metrics([0.1, 0.5, 0.9], [0, 2, 1])

    def test_perfect_model_high_scores(self):
        # Perfectly ranked model: event rate decreases strictly across all bins.
        # Assign positives so every decile has a strictly lower rate than the one above.
        n_bins = 10
        bin_size = 100
        n = n_bins * bin_size
        scores = np.linspace(1, 0, n)
        y_true = np.zeros(n, dtype=int)
        # Decile rates: 90%, 80%, 70%, ..., 10% — strictly decreasing
        for i in range(n_bins):
            n_pos = (n_bins - i) * bin_size // n_bins
            y_true[i * bin_size: i * bin_size + n_pos] = 1
        results = RankFitAnalyzer(n_bins=n_bins).calculate_metrics(scores, y_true)
        assert results["rankfit_v"] == 1.0
        assert results["rankfit_t"] == 1.0
        assert results["violations"] == []


class TestSummary:
    def test_summary_is_string(self, sample_data):
        scores, y_true = sample_data
        analyzer = RankFitAnalyzer()
        results  = analyzer.calculate_metrics(scores, y_true)
        summary  = analyzer.summary(results)
        assert isinstance(summary, str)
        assert "RankFit-V" in summary
        assert "RankFit-T" in summary
