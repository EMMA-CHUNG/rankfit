# RankFit

Segment-level ranking quality metrics for machine learning models. RankFit helps you detect when your model ranks high-risk cases below low-risk ones, even when AUC looks good.

## Installation

```bash
pip install rankfit
```

## Quick Start

```python
from rankfit import RankFitAnalyzer
from sklearn.metrics import roc_auc_score

# Initialize analyzer
analyzer = RankFitAnalyzer(n_bins=10)

# Calculate metrics
results = analyzer.calculate_metrics(y_scores, y_true)

# Show results
print(f"AUC: {roc_auc_score(y_true, y_scores):.3f}")
print(f"RankFit-V: {results['rankfit_v']:.3f}")  # Violation score
print(f"RankFit-T: {results['rankfit_t']:.3f}")  # Trend score

# Visualize
auc = roc_auc_score(y_true, y_scores)
fig = analyzer.plot_analysis(results, auc, "Model Analysis")
fig.savefig("ranking_analysis.png")
```

## What RankFit Does

Traditional metrics like AUC can hide critical ranking failures. A model with 0.85 AUC might rank high-risk customers *below* low-risk ones in specific segments. RankFit catches these failures.

- **RankFit-V**: Measures ranking violations (1.0 = perfect, <0.7 = problems)
- **RankFit-T**: Measures trend strength (1.0 = perfect monotonic decrease)

## Example

```python
# Model with high AUC but poor ranking
auc_a = 0.85  # Looks good!
rankfit_v_a = 0.58  # Actually has ranking problems

# Model with lower AUC but perfect ranking  
auc_b = 0.65  # Looks worse
rankfit_v_b = 1.00  # But ranking is perfect
```

## Citation

If you use RankFit in your research, please cite:

```bibtex
@software{rankfit2025,  # ← @software is more accurate for a Python package
  title={RankFit: A Framework for Evaluating Segment-Level Ranking Quality},
  author={Emma Chung},
  year={2025},
  url={https://github.com/EMMA-CHUNG/rankfit},  # ← Add your GitHub URL
  version={0.1.1}
}
```

## License

MIT License - see LICENSE file for details.
