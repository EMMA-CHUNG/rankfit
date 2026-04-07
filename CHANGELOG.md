# Changelog

## [0.2.0] - 2026-04-05
### Added
- `RankFitAnalyzer.compare_granularities()` — runs RankFit-V and RankFit-T
  at multiple bin sizes in a single call (e.g. quintiles, deciles, vigintiles,
  percentiles) and returns both a comparison list and a multi-panel figure.
- `plot_granularity_comparison()` — new public visualization function with:
  - Top row: one event-rate bar chart per granularity, violations highlighted in red
  - Bottom row: RankFit-V trend, RankFit-T trend, and violation count panels
  - Colour-coded bars (green = good, amber = moderate, red = poor)
  - Automatic insight annotation when finer bins reveal more violations
- 11 new tests in `tests/test_granularity.py` covering the new API.
- `plot_granularity_comparison` exported from top-level `rankfit` package.

### Changed
- Version bumped from 0.1.2 → 0.2.0.

## [0.1.2] - 2025-04-05
### Fixed
- Corrected `calculate_rankfit_t`: bin indices are now negated before computing
  Kendall's τ so that a perfectly decreasing event rate sequence correctly
  yields RankFit-T = 1.0 (previously would have returned 0.0).
- `calculate_rankfit_t` now checks `np.isnan(tau)` instead of `tau is not None`
  (scipy never returns None for tau).
- Removed unused `roc_auc_score` import from `analyzer.py`; AUC is now
  computed inside `calculate_metrics` and returned in the results dict.
- Visualization x-axis label is now dynamic based on `n_bins`
  (e.g. "Quintile", "Decile", "Percentile") instead of always showing "Decile".

### Added
- `find_violations()` exported as a standalone public function.
- `RankFitAnalyzer.summary()` method for human-readable metric reporting.
- Input validation in `RankFitAnalyzer.calculate_metrics()`.
- `_grade()` helper for qualitative score interpretation.
- Full unit-test suite covering metrics and analyzer.
- GitHub Actions CI/CD pipeline with multi-version matrix and PyPI publish.

## [0.1.1] - 2025-01-15
### Added
- Initial public release.
- `RankFitAnalyzer` with decile-level analysis.
- `calculate_rankfit_v` and `calculate_rankfit_t` metrics.
- Basic matplotlib visualization.
