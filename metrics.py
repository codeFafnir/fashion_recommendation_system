"""
Evaluation metrics for Fashion Recommendation System
Contains MAP@K, Precision@K, Recall@K, NDCG@K implementations
"""

import numpy as np
import pandas as pd


def calculate_map_at_k(y_true, y_pred, k=12):
    """
    Calculate Mean Average Precision at K (MAP@K)
    
    Args:
        y_true: list of sets containing true positive items for each user
        y_pred: list of lists containing predicted items (ranked) for each user
        k: number of top predictions to consider
    
    Returns:
        MAP@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    y_pred = [pred[:k] for pred in y_pred]
    
    aps = []
    for true_items, pred_items in zip(y_true, y_pred):
        if len(true_items) == 0:
            continue
        
        hits = 0
        precision_sum = 0.0
        
        for i, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                hits += 1
                precision_sum += hits / (i + 1)
        
        if hits > 0:
            denom = min(len(true_items), k)
            ap = precision_sum / denom
            aps.append(ap)
    
    return np.mean(aps) if len(aps) > 0 else 0.0


def calculate_precision_at_k(y_true, y_pred, k=12):
    """
    Calculate Precision at K
    
    Args:
        y_true: list of sets containing true positive items for each user
        y_pred: list of lists containing predicted items (ranked) for each user
        k: number of top predictions to consider
    
    Returns:
        Precision@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    y_pred = [pred[:k] for pred in y_pred]
    
    precisions = []
    for true_items, pred_items in zip(y_true, y_pred):
        if len(true_items) == 0:
            continue
        
        hits = len(set(pred_items) & true_items)
        precision = hits / len(pred_items) if len(pred_items) > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions) if len(precisions) > 0 else 0.0


def calculate_recall_at_k(y_true, y_pred, k=12):
    """
    Calculate Recall at K
    
    Args:
        y_true: list of sets containing true positive items for each user
        y_pred: list of lists containing predicted items (ranked) for each user
        k: number of top predictions to consider
    
    Returns:
        Recall@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    y_pred = [pred[:k] for pred in y_pred]
    
    recalls = []
    for true_items, pred_items in zip(y_true, y_pred):
        if len(true_items) == 0:
            continue
        
        hits = len(set(pred_items) & true_items)
        recall = hits / len(true_items) if len(true_items) > 0 else 0
        recalls.append(recall)
    
    return np.mean(recalls) if len(recalls) > 0 else 0.0


def calculate_ndcg_at_k(y_true, y_pred, k=12):
    """
    Calculate Normalized Discounted Cumulative Gain at K
    
    Args:
        y_true: list of sets containing true positive items for each user
        y_pred: list of lists containing predicted items (ranked) for each user
        k: number of top predictions to consider
    
    Returns:
        NDCG@K score
    """
    if len(y_true) == 0:
        return 0.0
    
    y_pred = [pred[:k] for pred in y_pred]
    
    ndcgs = []
    for true_items, pred_items in zip(y_true, y_pred):
        if len(true_items) == 0:
            continue
        
        dcg = 0.0
        for i, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        num_relevant = min(len(true_items), len(pred_items))
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg > 0:
            ndcg = dcg / idcg
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if len(ndcgs) > 0 else 0.0


def evaluate_map_at_12(df, predictions, customer_col='customer_id',
                       article_col='article_id', label_col='label', k=12):
    """
    Evaluate MAP@12 for a dataframe with predictions
    
    Args:
        df: DataFrame with customer_id, article_id, label columns
        predictions: Array of prediction scores (same length as df)
        customer_col: Name of customer ID column
        article_col: Name of article ID column
        label_col: Name of label column
        k: Number of top predictions to consider
    
    Returns:
        MAP@12 score
    """
    df_eval = df[[customer_col, article_col, label_col]].copy()
    df_eval['pred_score'] = predictions
    
    true_positives = (
        df_eval[df_eval[label_col] == 1]
        .groupby(customer_col)[article_col]
        .apply(list)
        .to_dict()
    )
    
    top_predictions = (
        df_eval.groupby(customer_col)
        .apply(lambda x: x.nlargest(k, 'pred_score')[article_col].tolist())
        .to_dict()
    )
    
    y_true = []
    y_pred = []
    
    for customer_id in true_positives.keys():
        if customer_id in top_predictions:
            y_true.append(set(true_positives[customer_id]))
            y_pred.append(top_predictions[customer_id])
    
    return calculate_map_at_k(y_true, y_pred, k=k)


def evaluate_all_metrics(df, predictions, k_values=None):
    """
    Evaluate all metrics for different K values
    
    Args:
        df: DataFrame with columns ['customer_id', 'article_id', 'label']
        predictions: Array of prediction scores
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 12]
    
    grouped = df.groupby('customer_id')
    
    y_true = []
    y_pred = []
    
    pred_copy = predictions.copy()
    
    for customer_id, group in grouped:
        true_items = set(group[group['label'] == 1]['article_id'].values)
        y_true.append(true_items)
        
        customer_df = group.copy()
        customer_df['pred_score'] = pred_copy[:len(customer_df)]
        customer_df = customer_df.sort_values('pred_score', ascending=False)
        pred_items = customer_df['article_id'].values.tolist()
        y_pred.append(pred_items)
        
        pred_copy = pred_copy[len(customer_df):]
    
    results = {}
    for k in k_values:
        results[f'MAP@{k}'] = calculate_map_at_k(y_true, y_pred, k)
        results[f'Precision@{k}'] = calculate_precision_at_k(y_true, y_pred, k)
        results[f'Recall@{k}'] = calculate_recall_at_k(y_true, y_pred, k)
        results[f'NDCG@{k}'] = calculate_ndcg_at_k(y_true, y_pred, k)
    
    return results

