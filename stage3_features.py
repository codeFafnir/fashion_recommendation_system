"""
Stage 3: Extracting Features
Feature engineering for user, item, and interaction features

This stage performs:
1. User statistics features
2. Item statistics features
3. User-item interaction features
4. Text-based semantic features (optional)
5. Label assignment for training data
6. Train/validation split
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
from tqdm.auto import tqdm
import json
import gc

from config import Config, config
from utils import (
    print_section, force_garbage_collection, print_memory,
    time_decay_score, chunk_dataframe
)


def compute_user_features(train_transactions, customers, max_date):
    """Compute user-level statistics features"""
    print_section("PART 1: USER FEATURES")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'user_features.parquet').exists():
        print("Found existing user_features.parquet, loading...")
        user_stats = pd.read_parquet(config.OUTPUT_PATH / 'user_features.parquet')
        print(f"Loaded {len(user_stats.columns)-1} user features")
        return user_stats
    
    print("Computing user statistics features...")
    
    # Basic purchase statistics
    user_stats = train_transactions.groupby('customer_id').agg({
        'article_id': ['count', 'nunique'],
        'price': ['mean', 'std', 'min', 'max'],
        't_dat': ['min', 'max']
    }).reset_index()
    
    user_stats.columns = [
        'customer_id', 'n_purchases', 'n_unique_items',
        'avg_price', 'std_price', 'min_price', 'max_price',
        'first_purchase', 'last_purchase'
    ]
    
    # Days since first/last purchase
    user_stats['days_since_first_purchase'] = (
        (max_date - user_stats['first_purchase']).dt.days
    ).astype(np.int16)
    user_stats['days_since_last_purchase'] = (
        (max_date - user_stats['last_purchase']).dt.days
    ).astype(np.int16)
    user_stats['purchase_span_days'] = (
        (user_stats['last_purchase'] - user_stats['first_purchase']).dt.days
    ).astype(np.int16)
    
    # Drop datetime columns
    user_stats = user_stats.drop(['first_purchase', 'last_purchase'], axis=1)
    
    # Purchase frequency
    user_stats['purchase_frequency'] = (
        user_stats['n_purchases'] / (user_stats['purchase_span_days'] + 1)
    ).astype(np.float32)
    
    # Item diversity (unique items / total purchases)
    user_stats['exploration_ratio'] = (
        user_stats['n_unique_items'] / user_stats['n_purchases']
    ).astype(np.float32)
    
    # Merge customer demographics
    if customers is not None:
        demo_cols = ['customer_id', 'age', 'FN', 'Active']
        available_cols = [col for col in demo_cols if col in customers.columns]
        if len(available_cols) > 1:
            user_stats = user_stats.merge(
                customers[available_cols],
                on='customer_id',
                how='left'
            )
    
    # Fill NaN values
    user_stats = user_stats.fillna(0)
    
    # Convert to optimal dtypes
    for col in user_stats.columns:
        if col != 'customer_id':
            if user_stats[col].dtype == 'float64':
                user_stats[col] = user_stats[col].astype(np.float32)
            elif user_stats[col].dtype == 'int64':
                user_stats[col] = user_stats[col].astype(np.int32)
    
    print(f"Created {len(user_stats.columns)-1} user features")
    print_memory()
    
    # Save
    user_stats.to_parquet(config.OUTPUT_PATH / 'user_features.parquet', index=False)
    
    return user_stats


def compute_item_features(train_transactions, articles, item_popularity, max_date):
    """Compute item-level statistics features"""
    print_section("PART 2: ITEM FEATURES")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'item_features.parquet').exists():
        print("Found existing item_features.parquet, loading...")
        item_stats = pd.read_parquet(config.OUTPUT_PATH / 'item_features.parquet')
        print(f"Loaded {len(item_stats.columns)-1} item features")
        return item_stats
    
    print("Computing item statistics features...")
    
    # Basic item statistics
    item_stats = train_transactions.groupby('article_id').agg({
        'customer_id': ['count', 'nunique'],
        'price': ['mean', 'std'],
        't_dat': ['min', 'max']
    }).reset_index()
    
    item_stats.columns = [
        'article_id', 'sales_count', 'unique_buyers',
        'avg_price', 'std_price', 'first_sale', 'last_sale'
    ]
    
    # Days since first/last sale
    item_stats['days_since_first_sale'] = (
        (max_date - item_stats['first_sale']).dt.days
    ).astype(np.int16)
    item_stats['days_since_last_sale'] = (
        (max_date - item_stats['last_sale']).dt.days
    ).astype(np.int16)
    
    item_stats = item_stats.drop(['first_sale', 'last_sale'], axis=1)
    
    # Recent popularity
    recent_date = max_date - timedelta(days=config.RECENT_DAYS)
    recent_trans = train_transactions[train_transactions['t_dat'] >= recent_date]
    recent_counts = recent_trans.groupby('article_id').size().reset_index(name='sales_recent')
    item_stats = item_stats.merge(recent_counts, on='article_id', how='left')
    item_stats['sales_recent'] = item_stats['sales_recent'].fillna(0).astype(np.int32)
    
    # Sales trend
    mid_date = max_date - timedelta(days=config.RECENT_DAYS)
    old_cutoff = max_date - timedelta(days=config.MEDIUM_DAYS)
    
    recent_period = train_transactions[train_transactions['t_dat'] >= mid_date]
    old_period = train_transactions[
        (train_transactions['t_dat'] >= old_cutoff) & (train_transactions['t_dat'] < mid_date)
    ]
    
    item_recent_count = recent_period.groupby('article_id').size().reset_index(name='sales_recent_period')
    item_old_count = old_period.groupby('article_id').size().reset_index(name='sales_old_period')
    
    item_trend = item_recent_count.merge(item_old_count, on='article_id', how='outer').fillna(0)
    item_trend['sales_trend'] = (
        (item_trend['sales_recent_period'] - item_trend['sales_old_period']) /
        (item_trend['sales_old_period'] + 1)
    ).astype(np.float32)
    
    item_stats = item_stats.merge(
        item_trend[['article_id', 'sales_trend']],
        on='article_id',
        how='left'
    )
    item_stats['sales_trend'] = item_stats['sales_trend'].fillna(0).astype(np.float32)
    
    # Merge article metadata
    article_features = articles[[
        'article_id', 'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id',
        'department_no', 'index_group_no', 'section_no', 'garment_group_no'
    ]].copy()
    
    item_stats = item_stats.merge(article_features, on='article_id', how='left')
    
    # Add popularity scores
    if item_popularity is not None:
        item_stats = item_stats.merge(
            item_popularity[['article_id', 'popularity_score']],
            on='article_id',
            how='left'
        )
        item_stats['popularity_score'] = item_stats['popularity_score'].fillna(0).astype(np.float32)
    
    # Convert to optimal dtypes
    for col in item_stats.columns:
        if col != 'article_id':
            if item_stats[col].dtype == 'float64':
                item_stats[col] = item_stats[col].astype(np.float32)
            elif item_stats[col].dtype == 'int64':
                item_stats[col] = item_stats[col].astype(np.int32)
    
    print(f"Created {len(item_stats.columns)-1} item features")
    print_memory()
    
    # Save
    item_stats.to_parquet(config.OUTPUT_PATH / 'item_features.parquet', index=False)
    
    return item_stats


def compute_interaction_features(candidates, train_transactions, articles,
                                 user_stats, item_stats):
    """Compute user-item interaction features"""
    print_section("PART 3: USER-ITEM INTERACTION FEATURES")
    
    print("Computing interaction features...")
    
    # Build user purchase history
    user_purchases = (
        train_transactions
        .groupby('customer_id')['article_id']
        .apply(set)
        .to_dict()
    )
    
    user_purchase_list = (
        train_transactions
        .sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(list)
        .to_dict()
    )
    
    # User category preferences
    user_categories = (
        train_transactions
        .merge(articles[['article_id', 'product_type_no']], on='article_id')
        .groupby(['customer_id', 'product_type_no'])
        .size()
        .reset_index(name='count')
    )
    
    user_top_category = (
        user_categories
        .sort_values(['customer_id', 'count'], ascending=[True, False])
        .groupby('customer_id')
        .first()
        .reset_index()
        [['customer_id', 'product_type_no']]
        .rename(columns={'product_type_no': 'top_category'})
    )
    
    # User price preferences
    user_price_stats = train_transactions.groupby('customer_id')['price'].agg(['mean', 'std']).reset_index()
    user_price_stats.columns = ['customer_id', 'user_avg_price', 'user_std_price']
    
    # Process candidates in chunks
    n_chunks = max(1, len(candidates) // config.CHUNK_SIZE)
    candidate_chunks = np.array_split(candidates, n_chunks)
    
    feature_chunks = []
    
    for chunk_idx, chunk in enumerate(tqdm(candidate_chunks, desc="Feature chunks")):
        chunk_features = chunk.copy()
        
        # Has user purchased this item before?
        chunk_features['has_purchased_item'] = chunk_features.apply(
            lambda row: 1 if row['article_id'] in user_purchases.get(row['customer_id'], set()) else 0,
            axis=1
        ).astype(np.int8)
        
        # Days since last purchase of item
        def days_since_purchase(row):
            user_items = user_purchase_list.get(row['customer_id'], [])
            if row['article_id'] in user_items:
                try:
                    idx = user_items.index(row['article_id'])
                    return min(idx, 365)
                except:
                    return 365
            return 365
        
        chunk_features['days_since_item_purchase'] = chunk_features.apply(
            days_since_purchase, axis=1
        ).astype(np.int16)
        
        # Merge item metadata
        chunk_features = chunk_features.merge(
            item_stats[['article_id', 'product_type_no', 'avg_price', 'popularity_score']],
            on='article_id',
            how='left'
        )
        
        # Category match
        chunk_features = chunk_features.merge(user_top_category, on='customer_id', how='left')
        chunk_features['category_match'] = (
            chunk_features['product_type_no'] == chunk_features['top_category']
        ).astype(np.int8)
        chunk_features = chunk_features.drop(['product_type_no', 'top_category'], axis=1)
        
        # Price match features
        chunk_features = chunk_features.merge(user_price_stats, on='customer_id', how='left')
        chunk_features['price_vs_user_avg'] = (
            (chunk_features['avg_price'] - chunk_features['user_avg_price']) /
            (chunk_features['user_std_price'] + 0.01)
        ).astype(np.float32)
        
        chunk_features['is_cheaper_than_usual'] = (
            chunk_features['avg_price'] < chunk_features['user_avg_price']
        ).astype(np.int8)
        
        chunk_features = chunk_features.drop(['user_avg_price', 'user_std_price', 'avg_price'], axis=1)
        
        # Rank features
        for score_col in ['repurchase_score', 'popularity_score', 'copurchase_score',
                          'userknn_score', 'category_score']:
            if score_col in chunk_features.columns:
                chunk_features[f'{score_col}_rank'] = (
                    chunk_features.groupby('customer_id')[score_col]
                    .rank(method='dense', ascending=False)
                    .astype(np.int16)
                )
        
        # Overall candidate rank
        if 'n_strategies' in chunk_features.columns:
            chunk_features['overall_rank'] = (
                chunk_features.groupby('customer_id')['n_strategies']
                .rank(method='dense', ascending=False)
                .astype(np.int16)
            )
        
        chunk_features = chunk_features.fillna(0)
        feature_chunks.append(chunk_features)
        
        if chunk_idx % 10 == 0:
            force_garbage_collection()
    
    # Combine chunks
    all_features = pd.concat(feature_chunks, ignore_index=True)
    del feature_chunks
    force_garbage_collection()
    
    print(f"Created interaction features for {len(all_features):,} candidates")
    
    return all_features


def merge_all_features(all_features, user_stats, item_stats):
    """Merge user, item, and interaction features"""
    print_section("MERGING ALL FEATURES")
    
    # Merge user features
    all_features = all_features.merge(user_stats, on='customer_id', how='left')
    print("Merged user features")
    
    # Merge remaining item features
    remaining_item_cols = [col for col in item_stats.columns if col not in all_features.columns]
    remaining_item_cols.append('article_id')
    all_features = all_features.merge(item_stats[remaining_item_cols], on='article_id', how='left')
    print("Merged item features")
    
    # Fill missing values
    numerical_cols = all_features.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        all_features[numerical_cols] = all_features[numerical_cols].fillna(0)
    
    print(f"\nTotal features: {len(all_features.columns) - 2} (excluding customer_id, article_id)")
    print(f"Total candidate-feature pairs: {len(all_features):,}")
    print_memory()
    
    return all_features


def assign_labels(all_features, val_transactions):
    """Assign labels for training and validation"""
    print_section("ASSIGNING LABELS")
    
    # Create ground truth set
    val_purchases = set(
        zip(val_transactions['customer_id'], val_transactions['article_id'])
    )
    
    # Assign labels
    all_features['label'] = all_features.apply(
        lambda row: 1 if (row['customer_id'], row['article_id']) in val_purchases else 0,
        axis=1
    ).astype(np.int8)
    
    n_positive = all_features['label'].sum()
    n_negative = len(all_features) - n_positive
    
    print(f"Positive samples: {n_positive:,} ({100*n_positive/len(all_features):.2f}%)")
    print(f"Negative samples: {n_negative:,} ({100*n_negative/len(all_features):.2f}%)")
    
    return all_features


def create_train_val_split(all_features, val_users):
    """Create train/validation split based on users"""
    print_section("CREATING TRAIN/VAL SPLIT")
    
    # Mark user types
    all_features['user_type'] = all_features['customer_id'].apply(
        lambda x: 'validation' if x in val_users else 'train'
    )
    
    # Split data
    train_data = all_features[all_features['user_type'] == 'train'].copy()
    val_data = all_features[all_features['user_type'] == 'validation'].copy()
    
    print(f"Training data: {len(train_data):,} samples")
    print(f"Validation data: {len(val_data):,} samples")
    print(f"Training users: {train_data['customer_id'].nunique():,}")
    print(f"Validation users: {val_data['customer_id'].nunique():,}")
    
    # Balance training data
    train_pos = train_data[train_data['label'] == 1]
    train_neg = train_data[train_data['label'] == 0]
    
    # Sample negatives (e.g., 1.5:1 ratio)
    n_neg_sample = int(len(train_pos) * 1.5)
    if len(train_neg) > n_neg_sample:
        train_neg_sampled = train_neg.sample(n=n_neg_sample, random_state=config.RANDOM_STATE)
        train_data = pd.concat([train_pos, train_neg_sampled], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"\nBalanced training data: {len(train_data):,} samples")
        print(f"  Positive: {len(train_pos):,}")
        print(f"  Negative: {len(train_neg_sampled):,}")
    
    # Save
    train_data.to_parquet(config.MODEL_PATH / 'train_data.parquet', index=False)
    val_data.to_parquet(config.MODEL_PATH / 'val_data.parquet', index=False)
    
    print(f"\nSaved train_data.parquet and val_data.parquet")
    
    return train_data, val_data


def save_feature_metadata(all_features):
    """Save feature metadata"""
    feature_names = [col for col in all_features.columns
                     if col not in ['customer_id', 'article_id', 'label', 'user_type']]
    
    # Categorize features
    user_features = [f for f in feature_names if any(x in f.lower() for x in [
        'user', 'customer', 'purchase', 'age', 'active', 'fn', 'trend'
    ])]
    
    item_features = [f for f in feature_names if any(x in f.lower() for x in [
        'item', 'article', 'sales', 'product', 'colour', 'color',
        'department', 'section', 'garment', 'frequency', 'count'
    ]) and f not in user_features]
    
    interaction_features = [f for f in feature_names if any(x in f.lower() for x in [
        'score', 'rank', 'strategies', 'match', 'purchased', 'category_match',
        'price_vs', 'cheaper'
    ]) and f not in user_features and f not in item_features]
    
    feature_metadata = {
        'total_features': len(feature_names),
        'feature_list': feature_names,
        'user_features': user_features,
        'item_features': item_features,
        'interaction_features': interaction_features,
    }
    
    with open(config.OUTPUT_PATH / 'feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    with open(config.OUTPUT_PATH / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"\nFeature breakdown:")
    print(f"  User features: {len(user_features)}")
    print(f"  Item features: {len(item_features)}")
    print(f"  Interaction features: {len(interaction_features)}")


def run_stage3(data=None):
    """Run the complete Stage 3 pipeline"""
    print_section("STAGE 3: EXTRACTING FEATURES")
    
    if data is None:
        # Load from saved files
        train_transactions = pd.read_parquet(config.OUTPUT_PATH / 'train_transactions.parquet')
        val_transactions = pd.read_parquet(config.OUTPUT_PATH / 'val_transactions.parquet')
        customers = pd.read_parquet(config.OUTPUT_PATH / 'customers.parquet')
        articles = pd.read_parquet(config.OUTPUT_PATH / 'articles.parquet')
        candidates = pd.read_parquet(config.OUTPUT_PATH / 'candidates.parquet')
        val_ground_truth = pd.read_parquet(config.OUTPUT_PATH / 'val_ground_truth.parquet')
        
        if (config.OUTPUT_PATH / 'item_popularity.parquet').exists():
            item_popularity = pd.read_parquet(config.OUTPUT_PATH / 'item_popularity.parquet')
        else:
            item_popularity = None
        
        max_date = train_transactions['t_dat'].max()
        val_users = set(val_ground_truth['customer_id'].unique())
    else:
        train_transactions = data.get('train_transactions')
        val_transactions = data.get('val_transactions')
        customers = data.get('customers')
        articles = data.get('articles')
        candidates = data.get('candidates')
        item_popularity = data.get('item_popularity')
        max_date = data.get('max_date')
        val_users = data.get('val_users')
    
    # Compute features
    user_stats = compute_user_features(train_transactions, customers, max_date)
    force_garbage_collection()
    
    item_stats = compute_item_features(train_transactions, articles, item_popularity, max_date)
    force_garbage_collection()
    
    all_features = compute_interaction_features(
        candidates, train_transactions, articles, user_stats, item_stats
    )
    force_garbage_collection()
    
    all_features = merge_all_features(all_features, user_stats, item_stats)
    force_garbage_collection()
    
    # Assign labels
    all_features = assign_labels(all_features, val_transactions)
    
    # Save feature metadata
    save_feature_metadata(all_features)
    
    # Create train/val split
    train_data, val_data = create_train_val_split(all_features, val_users)
    
    # Save training features
    all_features.to_parquet(config.OUTPUT_PATH / 'training_features.parquet', index=False)
    
    print_section("STAGE 3 COMPLETE")
    print(f"Total features: {len(all_features.columns) - 4}")
    print(f"Training samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    
    force_garbage_collection()
    
    return train_data, val_data


if __name__ == "__main__":
    run_stage3()

