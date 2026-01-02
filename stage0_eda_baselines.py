"""
Stage 0: EDA and Baselines
Exploratory Data Analysis and Baseline Models

This stage performs:
1. Raw H&M dataset analysis
2. Training data analysis
3. Validation data analysis
4. Baseline model evaluation (Popularity, User-CF, Item-CF, SVD)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import gc

from config import EDAConfig, Config
from utils import print_section, force_garbage_collection, reduce_mem_usage
from metrics import calculate_map_at_k, evaluate_map_at_12

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def run_raw_dataset_eda():
    """Run EDA on raw H&M dataset"""
    print_section("EDA: RAW H&M DATASET")
    
    config = EDAConfig()
    
    try:
        transactions = pd.read_csv(
            config.DATA_PATH / 'transactions_train.csv',
            dtype={'article_id': str, 'customer_id': str}
        )
        articles = pd.read_csv(config.DATA_PATH / 'articles.csv')
        customers = pd.read_csv(config.DATA_PATH / 'customers.csv')
        
        transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
        
        # Basic statistics
        print(f"\nTransactions: {len(transactions):,} rows")
        print(f"Articles: {len(articles):,} items")
        print(f"Customers: {len(customers):,} users")
        print(f"Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")
        
        # Create temporal analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        daily_trans = transactions.groupby(transactions['t_dat'].dt.date).size()
        axes[0, 0].plot(daily_trans.index, daily_trans.values, linewidth=2)
        axes[0, 0].set_title('Daily Transaction Volume', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Transactions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        user_activity = transactions.groupby('customer_id').size()
        axes[1, 0].hist(user_activity.values, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('User Purchase Activity Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Purchases per User')
        axes[1, 0].set_ylabel('Number of Users')
        axes[1, 0].set_yscale('log')
        
        item_popularity = transactions.groupby('article_id').size()
        axes[1, 1].hist(item_popularity.values, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title('Item Popularity Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Purchases per Item')
        axes[1, 1].set_ylabel('Number of Items')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / '1_raw_dataset_temporal.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return transactions, articles, customers
        
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return None, None, None


def run_training_data_eda():
    """Run EDA on training data"""
    print_section("EDA: TRAINING DATA")
    
    config = EDAConfig()
    
    try:
        train_data = pd.read_parquet(config.MODEL_PATH / 'train_data.parquet')
        
        print(f"\nTotal samples: {len(train_data):,}")
        print(f"Unique users: {train_data['customer_id'].nunique():,}")
        print(f"Unique items: {train_data['article_id'].nunique():,}")
        print(f"Positive samples: {train_data['label'].sum():,} ({100*train_data['label'].mean():.2f}%)")
        
        # Label distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        label_counts = train_data['label'].value_counts()
        axes[0].pie(label_counts.values, labels=['Negative (0)', 'Positive (1)'],
                    autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        axes[0].set_title('Label Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        
        axes[1].bar(['Negative (0)', 'Positive (1)'], label_counts.values,
                    color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
        axes[1].set_title('Label Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / '4_train_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return train_data
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def run_baseline_popularity(train_transactions, val_data):
    """Run popularity-based baseline"""
    print_section("BASELINE 1: POPULARITY-BASED RECOMMENDATION")
    
    config = Config()
    
    item_popularity = train_transactions.groupby('article_id').size().reset_index(name='popularity')
    item_popularity = item_popularity.sort_values('popularity', ascending=False)
    
    val_data_pop = val_data.copy()
    val_data_pop = val_data_pop.merge(
        item_popularity[['article_id', 'popularity']],
        on='article_id',
        how='left'
    )
    val_data_pop['pred_score'] = val_data_pop['popularity'].fillna(0)
    
    map12_popularity = evaluate_map_at_12(val_data_pop, val_data_pop['pred_score'].values)
    print(f"\nPopularity-Based Baseline MAP@12: {map12_popularity:.6f}")
    
    return map12_popularity, val_data_pop


def run_baseline_user_cf(train_transactions, val_data, all_users, all_items):
    """Run user-based collaborative filtering baseline"""
    print_section("BASELINE 2: USER-BASED COLLABORATIVE FILTERING")
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    
    train_transactions = train_transactions.copy()
    train_transactions['user_idx'] = train_transactions['customer_id'].map(user_to_idx)
    train_transactions['item_idx'] = train_transactions['article_id'].map(item_to_idx)
    
    rows = train_transactions['user_idx'].values
    cols = train_transactions['item_idx'].values
    data = np.ones(len(train_transactions), dtype=np.float32)
    
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))
    
    user_similarity = cosine_similarity(user_item_matrix, dense_output=False)
    
    val_users = val_data['customer_id'].unique()
    val_predictions = []
    
    for user in tqdm(val_users, desc="Predicting for users"):
        if user not in user_to_idx:
            user_predictions = np.zeros(len(val_data[val_data['customer_id'] == user]))
        else:
            user_idx = user_to_idx[user]
            user_sim = user_similarity[user_idx].toarray().flatten()
            similar_users_idx = np.argsort(user_sim)[::-1][1:51]
            similar_users_items = user_item_matrix[similar_users_idx].sum(axis=0).A1
            
            user_candidates = val_data[val_data['customer_id'] == user]['article_id'].values
            user_candidate_idx = [item_to_idx.get(item, -1) for item in user_candidates]
            
            user_scores = []
            for item_idx in user_candidate_idx:
                if item_idx >= 0:
                    score = similar_users_items[item_idx]
                else:
                    score = 0
                user_scores.append(score)
            
            user_predictions = np.array(user_scores)
        
        val_predictions.extend(user_predictions)
    
    val_data_ubcf = val_data.copy()
    val_data_ubcf['pred_score'] = val_predictions
    
    map12_ubcf = evaluate_map_at_12(val_data_ubcf, val_data_ubcf['pred_score'].values)
    print(f"\nUser-Based CF MAP@12: {map12_ubcf:.6f}")
    
    force_garbage_collection()
    return map12_ubcf, val_data_ubcf


def run_baseline_item_cf(train_transactions, val_data, all_users, all_items, user_item_matrix=None):
    """Run item-based collaborative filtering baseline"""
    print_section("BASELINE 3: ITEM-BASED COLLABORATIVE FILTERING")
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    
    if user_item_matrix is None:
        train_transactions = train_transactions.copy()
        train_transactions['user_idx'] = train_transactions['customer_id'].map(user_to_idx)
        train_transactions['item_idx'] = train_transactions['article_id'].map(item_to_idx)
        
        rows = train_transactions['user_idx'].values
        cols = train_transactions['item_idx'].values
        data = np.ones(len(train_transactions), dtype=np.float32)
        
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))
    
    item_item_matrix = user_item_matrix.T
    item_similarity = cosine_similarity(item_item_matrix, dense_output=False)
    
    val_users = val_data['customer_id'].unique()
    val_predictions = []
    
    for user in tqdm(val_users, desc="Predicting for users"):
        if user not in user_to_idx:
            user_predictions = np.zeros(len(val_data[val_data['customer_id'] == user]))
        else:
            user_idx = user_to_idx[user]
            user_items = user_item_matrix[user_idx].nonzero()[1]
            
            if len(user_items) == 0:
                user_predictions = np.zeros(len(val_data[val_data['customer_id'] == user]))
            else:
                user_candidates = val_data[val_data['customer_id'] == user]['article_id'].values
                user_candidate_idx = [item_to_idx.get(item, -1) for item in user_candidates]
                
                user_scores = []
                for candidate_idx in user_candidate_idx:
                    if candidate_idx >= 0:
                        item_sims = item_similarity[candidate_idx].toarray().flatten()
                        score = np.mean([item_sims[i] for i in user_items])
                    else:
                        score = 0
                    user_scores.append(score)
                
                user_predictions = np.array(user_scores)
        
        val_predictions.extend(user_predictions)
    
    val_data_ibcf = val_data.copy()
    val_data_ibcf['pred_score'] = val_predictions
    
    map12_ibcf = evaluate_map_at_12(val_data_ibcf, val_data_ibcf['pred_score'].values)
    print(f"\nItem-Based CF MAP@12: {map12_ibcf:.6f}")
    
    force_garbage_collection()
    return map12_ibcf, val_data_ibcf


def run_baseline_svd(train_transactions, val_data, all_users, all_items, n_factors=50):
    """Run SVD-based baseline"""
    print_section("BASELINE 4: SVD-BASED RECOMMENDATION")
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    train_transactions = train_transactions.copy()
    train_transactions['user_idx'] = train_transactions['customer_id'].map(user_to_idx)
    train_transactions['item_idx'] = train_transactions['article_id'].map(item_to_idx)
    
    rows = train_transactions['user_idx'].values
    cols = train_transactions['item_idx'].values
    data = np.ones(len(train_transactions), dtype=np.float32)
    
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))
    
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T
    
    val_users = val_data['customer_id'].unique()
    val_predictions = []
    
    for user in tqdm(val_users, desc="Predicting for users"):
        if user not in user_to_idx:
            user_predictions = np.zeros(len(val_data[val_data['customer_id'] == user]))
        else:
            user_idx = user_to_idx[user]
            user_embedding = user_factors[user_idx]
            
            user_candidates = val_data[val_data['customer_id'] == user]['article_id'].values
            user_candidate_idx = [item_to_idx.get(item, -1) for item in user_candidates]
            
            user_scores = []
            for item_idx in user_candidate_idx:
                if item_idx >= 0:
                    score = np.dot(user_embedding, item_factors[item_idx])
                else:
                    score = 0
                user_scores.append(score)
            
            user_predictions = np.array(user_scores)
        
        val_predictions.extend(user_predictions)
    
    val_data_svd = val_data.copy()
    val_data_svd['pred_score'] = val_predictions
    
    map12_svd = evaluate_map_at_12(val_data_svd, val_data_svd['pred_score'].values)
    print(f"\nSVD-Based Baseline MAP@12: {map12_svd:.6f}")
    
    force_garbage_collection()
    return map12_svd, val_data_svd


def run_all_baselines():
    """Run all baseline models and compare"""
    print_section("RUNNING ALL BASELINE MODELS")
    
    config = Config()
    
    # Load data
    train_transactions = pd.read_parquet(config.OUTPUT_PATH / 'train_transactions.parquet')
    val_data = pd.read_parquet(config.MODEL_PATH / 'val_data.parquet')
    
    all_users = sorted(train_transactions['customer_id'].unique().tolist())
    all_items = sorted(train_transactions['article_id'].unique().tolist())
    
    results = {}
    
    # Run baselines
    map12_pop, _ = run_baseline_popularity(train_transactions, val_data)
    results['Popularity'] = map12_pop
    
    map12_ubcf, _ = run_baseline_user_cf(train_transactions, val_data, all_users, all_items)
    results['User-CF'] = map12_ubcf
    
    map12_ibcf, _ = run_baseline_item_cf(train_transactions, val_data, all_users, all_items)
    results['Item-CF'] = map12_ibcf
    
    map12_svd, _ = run_baseline_svd(train_transactions, val_data, all_users, all_items)
    results['SVD'] = map12_svd
    
    # Summary
    print_section("BASELINE COMPARISON SUMMARY")
    print("\nModel Performance (MAP@12):")
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:20s}: {score:.6f}")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nBest Baseline: {best_model[0]} (MAP@12: {best_model[1]:.6f})")
    
    return results


if __name__ == "__main__":
    run_all_baselines()

