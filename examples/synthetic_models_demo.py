"""
Synthetic models demonstrating RankFit's ability to detect hidden failures.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from rankfit import RankFitAnalyzer


def create_high_auc_poor_ranking_model(n_samples=50000):
    """
    Creates a model with high AUC but severe ranking violation.
    """
    np.random.seed(42)
    
    # Decile 1 (Top 10%): High scores, very high event rate
    scores1 = np.random.uniform(0.9, 1.0, int(n_samples * 0.1))
    labels1 = np.random.choice([0, 1], p=[0.1, 0.9], size=len(scores1))
    
    # Decile 2 (10-20%): THE VIOLATION - High scores but very LOW event rate
    scores2 = np.random.uniform(0.8, 0.9, int(n_samples * 0.1))
    labels2 = np.random.choice([0, 1], p=[0.95, 0.05], size=len(scores2))
    
    # Other deciles: Properly ordered
    scores_other = np.random.uniform(0.0, 0.8, int(n_samples * 0.8))
    labels_other = (np.random.rand(len(scores_other)) < scores_other * 0.4).astype(int)
    
    scores = np.concatenate([scores1, scores2, scores_other])
    labels = np.concatenate([labels1, labels2, labels_other])
    
    return scores, labels


def create_low_auc_perfect_ranking_model(n_samples=50000):
    """
    Creates a model with low AUC but perfect ranking.
    """
    np.random.seed(0)
    
    scores = np.random.rand(n_samples)
    # Weak but perfectly monotonic relationship
    event_probability = 0.1 + (scores * 0.2)
    labels = (np.random.rand(n_samples) < event_probability).astype(int)
    
    return scores, labels


def main():
    print("Creating synthetic models...")
    
    # Generate models
    scores_a, labels_a = create_high_auc_poor_ranking_model()
    scores_b, labels_b = create_low_auc_perfect_ranking_model()
    
    # Calculate AUCs
    auc_a = roc_auc_score(labels_a, scores_a)
    auc_b = roc_auc_score(labels_b, scores_b)
    
    # Analyze with RankFit
    analyzer = RankFitAnalyzer(n_bins=10)
    results_a = analyzer.calculate_metrics(scores_a, labels_a)
    results_b = analyzer.calculate_metrics(scores_b, labels_b)
    
    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison: High AUC vs Perfect Ranking")
    print("="*60)
    
    print(f"\nModel A (High AUC, Poor Ranking):")
    print(f"  AUC: {auc_a:.3f}")
    print(f"  RankFit-V: {results_a['rankfit_v']:.3f}")
    print(f"  RankFit-T: {results_a['rankfit_t']:.3f}")
    print(f"  Violations: {len(results_a['violations'])}")
    
    print(f"\nModel B (Low AUC, Perfect Ranking):")
    print(f"  AUC: {auc_b:.3f}")
    print(f"  RankFit-V: {results_b['rankfit_v']:.3f}")
    print(f"  RankFit-T: {results_b['rankfit_t']:.3f}")
    print(f"  Violations: {len(results_b['violations'])}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    fig_a = analyzer.plot_analysis(results_a, auc_a, "Model A: High AUC, Poor Ranking")
    fig_a.savefig("model_a_analysis.png", dpi=300, bbox_inches='tight')
    
    fig_b = analyzer.plot_analysis(results_b, auc_b, "Model B: Low AUC, Perfect Ranking")
    fig_b.savefig("model_b_analysis.png", dpi=300, bbox_inches='tight')
    
    print("Analysis complete! Check the generated PNG files.")


if __name__ == "__main__":
    main()