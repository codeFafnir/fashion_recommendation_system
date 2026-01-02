"""
Stage 4A: LightGBM Training & Reranking
Train LightGBM models for ranking candidates

This stage performs:
1. Load and prepare training data
2. Train multiple LightGBM models (classifier, ranker variants)
3. Evaluate models with MAP@12
4. Feature importance analysis
5. Create ensemble predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
from datetime import datetime
import gc

from config import LightGBMConfig, Config
from utils import print_section, force_garbage_collection, print_memory
from metrics import evaluate_map_at_12


def load_training_data():
    """Load and prepare training data"""
    print_section("LOADING TRAINING DATA")
    
    train_data = pd.read_parquet(LightGBMConfig.MODEL_PATH / 'train_data.parquet')
    print(f"Loaded {len(train_data):,} training samples")
    print(f"Positives: {train_data['label'].sum():,} ({100*train_data['label'].mean():.2f}%)")
    
    val_data = pd.read_parquet(LightGBMConfig.MODEL_PATH / 'val_data.parquet')
    print(f"Loaded {len(val_data):,} validation samples")
    print(f"Positives: {val_data['label'].sum():,} ({100*val_data['label'].mean():.2f}%)")
    
    # Identify feature columns
    exclude_cols = ['customer_id', 'article_id', 'label', 'user_type', 'train_label', 'val_label']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"\nFeature columns: {len(feature_cols)}")
    
    # Handle missing values
    train_missing = train_data[feature_cols].isnull().sum()
    if train_missing.sum() > 0:
        for col in train_missing[train_missing > 0].index:
            if train_data[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                fill_value = train_data[col].median()
            else:
                fill_value = train_data[col].mode()[0] if len(train_data[col].mode()) > 0 else 0
            train_data[col] = train_data[col].fillna(fill_value)
            val_data[col] = val_data[col].fillna(fill_value)
    
    # Prepare feature matrices
    X_train = train_data[feature_cols].copy()
    y_train = train_data['label'].copy()
    X_val = val_data[feature_cols].copy()
    y_val = val_data['label'].copy()
    
    # Store IDs for evaluation
    train_customer_ids = train_data['customer_id'].copy()
    train_article_ids = train_data['article_id'].copy()
    val_customer_ids = val_data['customer_id'].copy()
    val_article_ids = val_data['article_id'].copy()
    
    # Identify categorical features
    categorical_features = [col for col in feature_cols
                           if X_train[col].dtype == 'category' or
                              X_train[col].dtype == 'object' or
                              col.endswith('_no') or col.endswith('_id') or col.endswith('_code')]
    
    print(f"Categorical features: {len(categorical_features)}")
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_cols': feature_cols,
        'categorical_features': categorical_features,
        'train_customer_ids': train_customer_ids,
        'val_customer_ids': val_customer_ids,
    }


def train_lightgbm_models(data):
    """Train multiple LightGBM models"""
    print_section("TRAINING LIGHTGBM MODELS")
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    val_data = data['val_data']
    categorical_features = data['categorical_features']
    train_customer_ids = data['train_customer_ids']
    val_customer_ids = data['val_customer_ids']
    
    # Convert categorical features to numeric codes
    for col in categorical_features:
        if col in X_train.columns:
            all_values = pd.concat([X_train[col], X_val[col]]).unique()
            train_cat = pd.Categorical(X_train[col], categories=sorted(all_values))
            val_cat = pd.Categorical(X_val[col], categories=sorted(all_values))
            X_train[col] = train_cat.codes.astype('int32')
            X_val[col] = val_cat.codes.astype('int32')
            
            if (X_train[col] == -1).any() or (X_val[col] == -1).any():
                max_code = max(X_train[col].max(), X_val[col].max())
                X_train[col] = X_train[col].replace(-1, max_code + 1)
                X_val[col] = X_val[col].replace(-1, max_code + 1)
    
    # Create LightGBM datasets
    train_dataset = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    
    val_dataset = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=categorical_features,
        reference=train_dataset,
        free_raw_data=False
    )
    
    # Prepare ranking datasets
    train_groups = train_customer_ids.value_counts().sort_index().values
    val_groups = val_customer_ids.value_counts().sort_index().values
    
    train_sort_idx = train_customer_ids.argsort()
    val_sort_idx = val_customer_ids.argsort()
    
    X_train_sorted = X_train.iloc[train_sort_idx].reset_index(drop=True)
    y_train_sorted = y_train.iloc[train_sort_idx].reset_index(drop=True)
    X_val_sorted = X_val.iloc[val_sort_idx].reset_index(drop=True)
    y_val_sorted = y_val.iloc[val_sort_idx].reset_index(drop=True)
    
    train_ranking_dataset = lgb.Dataset(
        X_train_sorted, label=y_train_sorted,
        categorical_feature=categorical_features,
        group=train_groups,
        free_raw_data=False
    )
    
    val_ranking_dataset = lgb.Dataset(
        X_val_sorted, label=y_val_sorted,
        categorical_feature=categorical_features,
        group=val_groups,
        reference=train_ranking_dataset,
        free_raw_data=False
    )
    
    # Model configurations
    models_config = {
        'lgb_classifier': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': LightGBMConfig.RANDOM_STATE,
            'force_col_wise': True,
        },
        'lgb_ranker_lambdarank': {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': LightGBMConfig.RANDOM_STATE,
            'force_col_wise': True,
            'label_gain': [0, 1],
        },
        'lgb_classifier_deep': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 15,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': LightGBMConfig.RANDOM_STATE,
            'force_col_wise': True,
        },
    }
    
    # Train models
    trained_models = {}
    model_predictions = {}
    model_scores = {}
    
    for model_name, params in models_config.items():
        print(f"\nTraining: {model_name}")
        
        if 'ranker' in model_name:
            train_ds = train_ranking_dataset
            val_ds = val_ranking_dataset
            use_sorted = True
        else:
            train_ds = train_dataset
            val_ds = val_dataset
            use_sorted = False
        
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=LightGBMConfig.N_ESTIMATORS,
            valid_sets=[train_ds, val_ds],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=LightGBMConfig.EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=LightGBMConfig.VERBOSE_EVAL)
            ]
        )
        
        # Make predictions
        if use_sorted:
            predictions = model.predict(X_val_sorted, num_iteration=model.best_iteration)
            val_revert_idx = val_sort_idx.argsort()
            predictions = predictions[val_revert_idx]
        else:
            predictions = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calculate MAP@12
        map_score = evaluate_map_at_12(val_data, predictions)
        
        trained_models[model_name] = model
        model_predictions[model_name] = predictions
        model_scores[model_name] = {
            'map_at_12': map_score,
            'best_iteration': model.best_iteration
        }
        
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  MAP@12: {map_score:.6f}")
        
        # Save model
        model_path = LightGBMConfig.MODEL_PATH / f'{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return trained_models, model_predictions, model_scores


def analyze_feature_importance(trained_models, feature_cols):
    """Analyze feature importance"""
    print_section("FEATURE IMPORTANCE ANALYSIS")
    
    importance_dfs = []
    
    for model_name, model in trained_models.items():
        importance = pd.DataFrame({
            'feature': feature_cols,
            f'{model_name}_gain': model.feature_importance(importance_type='gain'),
            f'{model_name}_split': model.feature_importance(importance_type='split')
        })
        importance_dfs.append(importance.set_index('feature'))
    
    feature_importance = pd.concat(importance_dfs, axis=1)
    feature_importance['avg_gain'] = feature_importance[[c for c in feature_importance.columns if 'gain' in c]].mean(axis=1)
    feature_importance = feature_importance.sort_values('avg_gain', ascending=False)
    
    print("\nTop 20 features by average gain:")
    print(feature_importance.head(20)['avg_gain'].to_string())
    
    # Save
    feature_importance.to_csv(LightGBMConfig.MODEL_PATH / 'feature_importance.csv')
    
    return feature_importance


def create_ensemble(model_predictions, model_scores, val_data):
    """Create ensemble predictions"""
    print_section("CREATING ENSEMBLE PREDICTIONS")
    
    # Normalize predictions
    normalized_predictions = {}
    for model_name, predictions in model_predictions.items():
        min_pred = predictions.min()
        max_pred = predictions.max()
        if max_pred > min_pred:
            normalized = (predictions - min_pred) / (max_pred - min_pred)
        else:
            normalized = predictions
        normalized_predictions[model_name] = normalized
    
    # Ensemble strategies
    best_model_name = max(model_scores.items(), key=lambda x: x[1]['map_at_12'])[0]
    
    ensemble_strategies = {
        'equal_weight': {m: 1.0 / len(model_predictions) for m in model_predictions.keys()},
        'best_only': {best_model_name: 1.0}
    }
    
    # Performance-based weights
    total_map = sum(scores['map_at_12'] for scores in model_scores.values())
    ensemble_strategies['performance_weight'] = {
        m: model_scores[m]['map_at_12'] / total_map
        for m in normalized_predictions.keys()
    }
    
    # Evaluate ensembles
    ensemble_predictions = {}
    ensemble_scores = {}
    
    for strategy_name, weights in ensemble_strategies.items():
        ensemble_pred = np.zeros(len(val_data))
        for model_name, weight in weights.items():
            if model_name in normalized_predictions:
                ensemble_pred += weight * normalized_predictions[model_name]
        
        ensemble_predictions[strategy_name] = ensemble_pred
        map_score = evaluate_map_at_12(val_data, ensemble_pred)
        ensemble_scores[strategy_name] = map_score
        
        print(f"  {strategy_name:25s}: MAP@12 = {map_score:.6f}")
    
    # Find best ensemble
    best_ensemble_name = max(ensemble_scores.items(), key=lambda x: x[1])[0]
    print(f"\nBest Ensemble: {best_ensemble_name} (MAP@12: {ensemble_scores[best_ensemble_name]:.6f})")
    
    # Save ensemble predictions
    val_data_with_preds = val_data.copy()
    val_data_with_preds['ensemble_pred'] = ensemble_predictions[best_ensemble_name]
    val_data_with_preds.to_parquet(
        LightGBMConfig.MODEL_PATH / 'ensemble_predictions_val.parquet',
        index=False
    )
    
    # Save metadata
    ensemble_metadata = {
        'strategies': {k: {m: float(v) for m, v in weights.items()} for k, weights in ensemble_strategies.items()},
        'scores': {k: float(v) for k, v in ensemble_scores.items()},
        'best_ensemble': best_ensemble_name,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(LightGBMConfig.MODEL_PATH / 'ensemble_metadata.json', 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    return ensemble_predictions, ensemble_scores


def run_stage4a():
    """Run the complete Stage 4A pipeline"""
    print_section("STAGE 4A: LIGHTGBM TRAINING")
    
    # Load data
    data = load_training_data()
    
    # Train models
    trained_models, model_predictions, model_scores = train_lightgbm_models(data)
    
    # Feature importance
    feature_importance = analyze_feature_importance(trained_models, data['feature_cols'])
    
    # Create ensemble
    ensemble_predictions, ensemble_scores = create_ensemble(
        model_predictions, model_scores, data['val_data']
    )
    
    # Save model metadata
    metadata = {
        'models': {name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in scores.items()} 
                   for name, scores in model_scores.items()},
        'best_model': max(model_scores.items(), key=lambda x: x[1]['map_at_12'])[0],
        'feature_columns': data['feature_cols'],
        'categorical_features': data['categorical_features'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(LightGBMConfig.MODEL_PATH / 'lgb_models_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print_section("TRAINING SUMMARY")
    print("\nModel Performance (MAP@12):")
    for model_name, scores in sorted(model_scores.items(), key=lambda x: x[1]['map_at_12'], reverse=True):
        print(f"  {model_name:30s}: {scores['map_at_12']:.6f}")
    
    best_model_name = metadata['best_model']
    print(f"\nBest Model: {best_model_name} (MAP@12: {model_scores[best_model_name]['map_at_12']:.6f})")
    
    force_garbage_collection()
    
    return trained_models, model_scores


if __name__ == "__main__":
    run_stage4a()

