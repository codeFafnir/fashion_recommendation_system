"""
Stage 7: Evaluation & Metrics
Final evaluation and comparison of all models

This stage performs:
1. Load all model predictions
2. Evaluate with multiple metrics (MAP@K, Precision@K, Recall@K, NDCG@K)
3. Create final ensemble
4. Generate submission file
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from tqdm.auto import tqdm
import gc

from config import EvaluationConfig, Config
from utils import print_section, force_garbage_collection, normalize_scores
from metrics import (
    calculate_map_at_k, calculate_precision_at_k,
    calculate_recall_at_k, calculate_ndcg_at_k,
    evaluate_all_metrics
)


def load_predictions():
    """Load all model predictions"""
    print_section("LOADING MODEL PREDICTIONS")
    
    val_data = pd.read_parquet(EvaluationConfig.MODEL_PATH / 'val_data.parquet')
    print(f"Loaded {len(val_data):,} validation samples")
    
    all_predictions = {}
    
    # Load LightGBM ensemble predictions
    try:
        ensemble_path = EvaluationConfig.MODEL_PATH / 'ensemble_predictions_val.parquet'
        if ensemble_path.exists():
            ensemble_preds = pd.read_parquet(ensemble_path)
            if 'ensemble_pred' in ensemble_preds.columns:
                all_predictions['lgb_ensemble'] = ensemble_preds['ensemble_pred'].values
                print("Loaded LightGBM ensemble predictions")
    except Exception as e:
        print(f"Could not load LightGBM ensemble: {e}")
    
    # Load Neural Tower predictions
    try:
        neural_path = EvaluationConfig.MODEL_PATH / 'neural_tower_predictions_val.parquet'
        if neural_path.exists():
            neural_preds = pd.read_parquet(neural_path)
            if 'pred_score' in neural_preds.columns:
                all_predictions['neural_tower'] = neural_preds['pred_score'].values
                print("Loaded Neural Tower predictions")
    except Exception as e:
        print(f"Could not load Neural Tower: {e}")
    
    print(f"\nTotal models loaded: {len(all_predictions)}")
    print(f"Models: {list(all_predictions.keys())}")
    
    return val_data, all_predictions


def evaluate_all_models(val_data, all_predictions):
    """Evaluate all models with multiple metrics"""
    print_section("EVALUATING ALL MODELS")
    
    evaluation_results = {}
    
    for model_name, predictions in tqdm(all_predictions.items(), desc="Evaluating models"):
        print(f"\nEvaluating {model_name}...")
        
        if len(predictions) != len(val_data):
            min_len = min(len(predictions), len(val_data))
            predictions = predictions[:min_len]
            val_data_eval = val_data.iloc[:min_len].copy()
        else:
            val_data_eval = val_data.copy()
        
        metrics = evaluate_all_metrics(
            val_data_eval,
            predictions.copy(),
            k_values=EvaluationConfig.K_VALUES
        )
        evaluation_results[model_name] = metrics
        
        print(f"  MAP@12: {metrics['MAP@12']:.6f}")
        print(f"  Precision@12: {metrics['Precision@12']:.6f}")
        print(f"  Recall@12: {metrics['Recall@12']:.6f}")
        print(f"  NDCG@12: {metrics['NDCG@12']:.6f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(evaluation_results).T
    comparison_df = comparison_df.sort_values('MAP@12', ascending=False)
    
    print_section("MODEL COMPARISON SUMMARY")
    print(comparison_df.to_string())
    
    # Save comparison
    comparison_df.to_csv(EvaluationConfig.MODEL_PATH / 'model_comparison.csv')
    
    return evaluation_results, comparison_df


def create_final_ensemble(val_data, all_predictions, evaluation_results):
    """Create final ensemble from all models"""
    print_section("CREATING FINAL ENSEMBLE")
    
    if len(all_predictions) == 0:
        print("No models available for ensemble!")
        return None, None
    
    # Normalize predictions
    normalized_predictions = {}
    for model_name, predictions in all_predictions.items():
        normalized_predictions[model_name] = normalize_scores(predictions)
        print(f"  {model_name}: normalized to [0, 1]")
    
    # Calculate performance-based weights
    if len(normalized_predictions) > 1:
        map_scores = {m: evaluation_results[m]['MAP@12'] for m in normalized_predictions.keys()}
        total_map = sum(map_scores.values())
        weights = {m: score / total_map for m, score in map_scores.items()}
    else:
        weights = {list(normalized_predictions.keys())[0]: 1.0}
    
    print(f"\nEnsemble weights:")
    for model_name, weight in weights.items():
        print(f"  {model_name}: {weight:.4f}")
    
    # Create weighted ensemble
    ensemble_pred = np.zeros(len(val_data))
    for model_name, weight in weights.items():
        if len(normalized_predictions[model_name]) == len(val_data):
            ensemble_pred += weight * normalized_predictions[model_name]
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_all_metrics(
        val_data,
        ensemble_pred.copy(),
        k_values=EvaluationConfig.K_VALUES
    )
    
    print(f"\nFinal Ensemble Results:")
    print(f"  MAP@12: {ensemble_metrics['MAP@12']:.6f}")
    print(f"  Precision@12: {ensemble_metrics['Precision@12']:.6f}")
    print(f"  Recall@12: {ensemble_metrics['Recall@12']:.6f}")
    print(f"  NDCG@12: {ensemble_metrics['NDCG@12']:.6f}")
    
    # Save ensemble predictions
    ensemble_df = val_data[['customer_id', 'article_id', 'label']].copy()
    ensemble_df['pred_score'] = ensemble_pred
    ensemble_df.to_parquet(
        EvaluationConfig.MODEL_PATH / 'final_ensemble_predictions_val.parquet',
        index=False
    )
    
    return ensemble_pred, ensemble_metrics


def generate_submission(val_data, best_predictions):
    """Generate submission file with top-12 predictions per user"""
    print_section("GENERATING SUBMISSION FILE")
    
    pred_df = val_data[['customer_id', 'article_id']].copy()
    pred_df['pred_score'] = best_predictions[:len(pred_df)]
    
    rankings = []
    for customer_id, group in tqdm(pred_df.groupby('customer_id'), desc="Ranking users"):
        group_sorted = group.sort_values('pred_score', ascending=False)
        top_articles = group_sorted.head(12)['article_id'].values
        predictions_str = ' '.join([str(art) for art in top_articles])
        
        rankings.append({
            'customer_id': customer_id,
            'prediction': predictions_str
        })
    
    submission_df = pd.DataFrame(rankings)
    submission_df = submission_df.sort_values('customer_id')
    
    print(f"\nGenerated rankings for {len(submission_df):,} users")
    print(f"Average articles per user: {submission_df['prediction'].str.split().str.len().mean():.2f}")
    
    # Save submission
    submission_path = EvaluationConfig.MODEL_PATH / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    # Display sample
    print("\nSample submission (first 5 rows):")
    print(submission_df.head().to_string(index=False))
    
    return submission_df


def print_final_summary(evaluation_results, ensemble_metrics, comparison_df):
    """Print final evaluation summary"""
    print_section("FINAL EVALUATION SUMMARY")
    
    # Best model
    best_model = comparison_df.index[0]
    best_map12 = comparison_df.loc[best_model, 'MAP@12']
    
    print(f"\nBest Single Model: {best_model}")
    print(f"  MAP@12: {best_map12:.6f}")
    
    # Model rankings
    print(f"\nModel Rankings (by MAP@12):")
    print("-" * 60)
    for idx, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
        marker = "1st" if idx == 1 else "2nd" if idx == 2 else "3rd" if idx == 3 else f"{idx}th"
        print(f"  {marker}: {model_name:30s} MAP@12: {row['MAP@12']:.6f}")
    
    # Ensemble results
    if ensemble_metrics:
        print(f"\nFinal Ensemble:")
        print(f"  MAP@12: {ensemble_metrics['MAP@12']:.6f}")
        print(f"  Precision@12: {ensemble_metrics['Precision@12']:.6f}")
        print(f"  Recall@12: {ensemble_metrics['Recall@12']:.6f}")
        print(f"  NDCG@12: {ensemble_metrics['NDCG@12']:.6f}")
    
    # Files saved
    print(f"\nGenerated Files:")
    print("-" * 60)
    print(f"  Model Comparison: {EvaluationConfig.MODEL_PATH / 'model_comparison.csv'}")
    print(f"  Ensemble Predictions: {EvaluationConfig.MODEL_PATH / 'final_ensemble_predictions_val.parquet'}")
    print(f"  Submission File: {EvaluationConfig.MODEL_PATH / 'submission.csv'}")


def run_stage7():
    """Run the complete Stage 7 pipeline"""
    print_section("STAGE 7: EVALUATION & METRICS")
    
    # Load predictions
    val_data, all_predictions = load_predictions()
    
    if len(all_predictions) == 0:
        print("No model predictions found!")
        return None
    
    # Evaluate all models
    evaluation_results, comparison_df = evaluate_all_models(val_data, all_predictions)
    
    # Create final ensemble
    ensemble_pred, ensemble_metrics = create_final_ensemble(
        val_data, all_predictions, evaluation_results
    )
    
    # Add ensemble to results if created
    if ensemble_metrics:
        evaluation_results['final_ensemble'] = ensemble_metrics
        comparison_df = pd.DataFrame(evaluation_results).T
        comparison_df = comparison_df.sort_values('MAP@12', ascending=False)
        comparison_df.to_csv(EvaluationConfig.MODEL_PATH / 'model_comparison.csv')
    
    # Generate submission
    best_predictions = ensemble_pred if ensemble_pred is not None else list(all_predictions.values())[0]
    submission_df = generate_submission(val_data, best_predictions)
    
    # Print final summary
    print_final_summary(evaluation_results, ensemble_metrics, comparison_df)
    
    print_section("STAGE 7 COMPLETE")
    print("Ready for submission!")
    
    force_garbage_collection()
    
    return {
        'evaluation_results': evaluation_results,
        'comparison_df': comparison_df,
        'submission_df': submission_df
    }


if __name__ == "__main__":
    run_stage7()

