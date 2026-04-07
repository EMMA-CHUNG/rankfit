# RankFit

[![PyPI version](https://badge.fury.io/py/rankfit.svg)](https://badge.fury.io/py/rankfit)
[![Python](https://img.shields.io/pypi/pyversions/rankfit.svg)](https://pypi.org/project/rankfit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/EMMA-CHUNG/rankfit/actions/workflows/ci.yml/badge.svg)](https://github.com/EMMA-CHUNG/rankfit/actions)

**Segment-level ranking quality metrics for machine learning models — beyond AUC.**

AUC tells you how your model discriminates globally. RankFit tells you whether it ranks your actual business segments correctly.

A model can achieve AUC 0.887 while hiding a catastrophic ranking violation in Decile 2 — causing you to target low-value customers while missing high-value ones. In a real-world case study, a RankFit-selected model with AUC 0.612 outperformed an AUC-selected model with 0.887, capturing 89% of high-value targets versus 67% and generating $82,500 more revenue per campaign.

---

## Installation

```bash
pip install rankfit
```

---

## Quick Start

```python
from rankfit import RankFitAnalyzer

analyzer = RankFitAnalyzer(n_bins=10)   # deciles (default)
results  = analyzer.calculate_metrics(y_scores, y_true)

print(analyzer.summary(results))
# =============================================
#   RankFit Summary
# =============================================
#   AUC         : 0.8871
#   RankFit-V   : 0.8750  (Good)
#   RankFit-T   : 0.9778  (Excellent)
#   Violations  : 1
# =============================================

analyzer.plot_analysis(results, title="My Model — Decile Analysis")
```

---

## Metrics

### RankFit-V (Violation Score)

Detects **ranking holes** — segments where a higher-scored bin has a *lower* actual event rate than a lower-scored bin. Violations are penalised proportionally to their magnitude.

```
RankFit-V = 1 − (Σ violation magnitudes) / (max rate − min rate)
```

| Score | Interpretation |
|-------|---------------|
| 1.000 | Perfect — no violations |
| ≥ 0.80 | Good — minor holes only |
| ≥ 0.60 | Moderate — review before deploying |
| < 0.60 | Poor — significant ranking failures |

### RankFit-T (Trend Score)

Measures **global monotonic consistency** using Kendall's τ, scaled to [0, 1].

```
RankFit-T = (τ + 1) / 2
```

| Score | Interpretation |
|-------|---------------|
| 1.000 | Perfectly decreasing trend |
| ≥ 0.80 | Strongly decreasing |
| ≥ 0.60 | Weakly decreasing |
| < 0.60 | Non-monotonic — poor ranking |

---

## Granularity Guide

| Use case | Recommended `n_bins` |
|----------|----------------------|
| Quick check / small dataset | `5` (quintiles) |
| Standard analysis | `10` (deciles) — default |
| Detailed analysis | `20` (vigintiles) |
| Healthcare, fraud, credit | `100` (percentiles) |

> **Tip:** For high-stakes decisions, always use percentiles. Coarser bins smooth over fine-grained violations that could cost lives or money.

---

## Recommended Workflow

```
1. Screen  → Use AUC to remove non-predictive models
2. Filter  → Disqualify models with RankFit-V < 0.80
3. Select  → Choose the model with the highest RankFit-T
```

---

## API Reference

### `RankFitAnalyzer(n_bins=10)`

| Method | Description |
|--------|-------------|
| `calculate_metrics(y_scores, y_true)` | Returns dict with `rankfit_v`, `rankfit_t`, `violations`, `bin_stats`, `auc` |
| `plot_analysis(results, title=...)` | Returns a `matplotlib.Figure` |
| `summary(results)` | Returns a formatted summary string |

### Standalone functions

```python
from rankfit.metrics import calculate_rankfit_v, calculate_rankfit_t, find_violations
```

---

## Development

```bash
git clone https://github.com/EMMA-CHUNG/rankfit.git
cd rankfit
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use RankFit in research, please cite:

```bibtex
@software{rankfit2025,
  author  = {Emma Chung},
  title   = {RankFit: Segment-Level Ranking Quality Metrics},
  year    = {2025},
  url     = {https://github.com/EMMA-CHUNG/rankfit},
}
```

---

## License

MIT © Emma Chung
