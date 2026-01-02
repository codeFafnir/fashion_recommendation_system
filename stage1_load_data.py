"""
Stage 1: Loading Dataset
Data loading, temporal splitting, and user/item sampling

This stage performs:
1. Load raw H&M data (transactions, articles, customers)
2. Select temporal window
3. User-based stratified sampling
4. Item filtering
5. Memory optimization
6. Save processed datasets
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import gc

from config import Config, config
from utils import (
    print_section, force_garbage_collection, reduce_mem_usage,
    print_memory, optimize_categorical_columns
)


def load_raw_data():
    """Load raw H&M dataset"""
    print_section("STEP 1: LOADING DATA")
    
    transactions = pd.read_csv(
        config.DATA_PATH / 'transactions_train.csv',
        dtype={
            'article_id': 'int32',
            'price': 'float32',
            'sales_channel_id': 'int8'
        },
        parse_dates=['t_dat']
    )
    print(f"Transactions: {len(transactions):,} rows")
    print(f"Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")
    
    customers = pd.read_csv(
        config.DATA_PATH / 'customers.csv',
        dtype={
            'FN': 'float32',
            'Active': 'float32',
            'age': 'float32'
        }
    )
    customers = reduce_mem_usage(customers, verbose=False)
    print(f"Customers: {len(customers):,} rows")
    
    articles = pd.read_csv(
        config.DATA_PATH / 'articles.csv',
        dtype={'article_id': 'int32'}
    )
    articles = reduce_mem_usage(articles, verbose=False)
    print(f"Articles: {len(articles):,} rows")
    
    return transactions, customers, articles


def select_temporal_window(transactions):
    """Select temporal window for training/validation"""
    print_section("STEP 2: SELECTING TEMPORAL WINDOW")
    
    max_date = transactions['t_dat'].max()
    print(f"Last transaction date: {max_date}")
    
    window_start = max_date - timedelta(weeks=config.TOTAL_WEEKS)
    print(f"Using {config.TOTAL_WEEKS} weeks of data")
    print(f"Window: {window_start.date()} to {max_date.date()}")
    
    transactions = transactions[transactions['t_dat'] >= window_start].copy()
    print(f"Retained {len(transactions):,} transactions")
    
    transactions['week'] = ((transactions['t_dat'] - window_start).dt.days // 7).astype(np.int8)
    
    return transactions, max_date


def stratified_user_sampling(transactions):
    """Perform user-based stratified sampling"""
    print_section("STEP 3: USER-BASED STRATIFIED SAMPLING")
    
    user_activity = transactions.groupby('customer_id').agg({
        'article_id': 'count',
        'week': ['min', 'max', 'nunique']
    }).reset_index()
    
    user_activity.columns = ['customer_id', 'total_purchases', 'first_week', 'last_week', 'active_weeks']
    user_activity['week_span'] = user_activity['last_week'] - user_activity['first_week'] + 1
    
    print(f"Total users in window: {len(user_activity):,}")
    print(f"Avg purchases per user: {user_activity['total_purchases'].mean():.2f}")
    
    # Separate cold start and regular users
    if config.INCLUDE_COLD_START:
        cold_start_users = user_activity[
            user_activity['total_purchases'] <= config.COLD_START_MAX_PURCHASES
        ].copy()
        regular_users = user_activity[
            user_activity['total_purchases'] >= config.MIN_USER_PURCHASES
        ].copy()
        
        n_cold_start_target = int(config.TARGET_USERS * config.COLD_START_RATIO)
        n_regular_target = config.TARGET_USERS - n_cold_start_target
        
        print(f"\nCold start users: {len(cold_start_users):,}")
        print(f"Regular users: {len(regular_users):,}")
    else:
        regular_users = user_activity[
            user_activity['total_purchases'] >= config.MIN_USER_PURCHASES
        ].copy()
        cold_start_users = pd.DataFrame()
        n_cold_start_target = 0
        n_regular_target = config.TARGET_USERS
    
    # Sample cold start users
    sampled_cold_start = []
    if config.INCLUDE_COLD_START and len(cold_start_users) > 0:
        n_cold_sample = min(n_cold_start_target, len(cold_start_users))
        sampled_cold_start = cold_start_users['customer_id'].sample(
            n=n_cold_sample,
            random_state=config.RANDOM_STATE
        ).tolist()
        print(f"Sampled {len(sampled_cold_start):,} cold start users")
    
    # Stratified sampling for regular users
    if config.STRATIFY_BY_ACTIVITY and len(regular_users) > 0:
        regular_users['activity_level'] = pd.cut(
            regular_users['total_purchases'],
            bins=config.ACTIVITY_BINS,
            labels=config.ACTIVITY_LABELS
        )
        
        activity_dist = regular_users['activity_level'].value_counts().sort_index()
        samples_per_stratum = (activity_dist / activity_dist.sum() * n_regular_target).round().astype(int)
        
        diff = n_regular_target - samples_per_stratum.sum()
        if diff != 0:
            largest_stratum = samples_per_stratum.idxmax()
            samples_per_stratum[largest_stratum] += diff
        
        sampled_regular = []
        for level in config.ACTIVITY_LABELS:
            stratum_users = regular_users[regular_users['activity_level'] == level]['customer_id']
            n_sample = min(samples_per_stratum[level], len(stratum_users))
            if n_sample > 0:
                sampled = stratum_users.sample(n=n_sample, random_state=config.RANDOM_STATE)
                sampled_regular.extend(sampled.tolist())
    else:
        n_sample = min(n_regular_target, len(regular_users))
        sampled_regular = regular_users['customer_id'].sample(
            n=n_sample,
            random_state=config.RANDOM_STATE
        ).tolist()
    
    selected_users = set(sampled_cold_start + sampled_regular)
    print(f"\nTotal selected users: {len(selected_users):,}")
    
    return selected_users, user_activity


def filter_transactions(transactions, selected_users, max_date):
    """Filter transactions to sampled users and create train/val split"""
    print_section("STEP 4: FILTERING TRANSACTIONS")
    
    transactions = transactions[transactions['customer_id'].isin(selected_users)].copy()
    print(f"Retained {len(transactions):,} transactions")
    
    # Create train/val split
    val_end_date = max_date
    val_start_date = val_end_date - timedelta(weeks=config.N_VAL_WEEKS)
    train_end_date = val_start_date - timedelta(days=1)
    
    print(f"\nTrain: up to {train_end_date.date()} ({config.N_TRAIN_WEEKS} weeks)")
    print(f"Val: {val_start_date.date()} to {val_end_date.date()} ({config.N_VAL_WEEKS} week)")
    
    train_transactions = transactions[transactions['t_dat'] <= train_end_date].copy()
    val_transactions = transactions[transactions['t_dat'] > train_end_date].copy()
    
    print(f"\nTraining transactions: {len(train_transactions):,}")
    print(f"Validation transactions: {len(val_transactions):,}")
    
    val_users = set(val_transactions['customer_id'].unique())
    print(f"Users in validation: {len(val_users):,}")
    
    return train_transactions, val_transactions, val_users


def filter_items(train_transactions, val_transactions, articles):
    """Filter items based on minimum purchases"""
    print_section("STEP 5: ITEM FILTERING")
    
    item_counts = train_transactions['article_id'].value_counts()
    print(f"Unique items in training: {len(item_counts):,}")
    
    valid_items = set(item_counts[item_counts >= config.MIN_ITEM_PURCHASES].index)
    print(f"Items with >= {config.MIN_ITEM_PURCHASES} purchases: {len(valid_items):,}")
    
    val_items = set(val_transactions['article_id'].unique())
    print(f"Items in validation: {len(val_items):,}")
    
    selected_items = valid_items.union(val_items)
    print(f"Total selected items: {len(selected_items):,}")
    
    train_transactions = train_transactions[train_transactions['article_id'].isin(selected_items)].copy()
    val_transactions = val_transactions[val_transactions['article_id'].isin(selected_items)].copy()
    articles = articles[articles['article_id'].isin(selected_items)].copy()
    
    print(f"\nAfter filtering:")
    print(f"Training transactions: {len(train_transactions):,}")
    print(f"Validation transactions: {len(val_transactions):,}")
    print(f"Articles retained: {len(articles):,}")
    
    return train_transactions, val_transactions, articles, selected_items


def optimize_memory(train_transactions, val_transactions, customers, articles):
    """Optimize memory usage"""
    print_section("STEP 6: MEMORY OPTIMIZATION")
    
    train_transactions = reduce_mem_usage(train_transactions)
    val_transactions = reduce_mem_usage(val_transactions)
    
    # Optimize categorical columns
    article_cat_cols = [
        'product_code', 'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
        'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'
    ]
    articles = optimize_categorical_columns(articles, article_cat_cols)
    
    customer_cat_cols = ['club_member_status', 'fashion_news_frequency', 'postal_code']
    customers = optimize_categorical_columns(customers, customer_cat_cols)
    
    print_memory()
    
    return train_transactions, val_transactions, customers, articles


def save_processed_data(train_transactions, val_transactions, val_users, 
                        customers, articles, selected_users):
    """Save processed datasets"""
    print_section("STEP 7: SAVING PROCESSED DATA")
    
    # Save transactions
    train_transactions.to_parquet(config.OUTPUT_PATH / 'train_transactions.parquet', index=False)
    print(f"Saved train_transactions.parquet")
    
    val_transactions.to_parquet(config.OUTPUT_PATH / 'val_transactions.parquet', index=False)
    print(f"Saved val_transactions.parquet")
    
    # Save filtered customers and articles
    customers_filtered = customers[customers['customer_id'].isin(selected_users)]
    customers_filtered.to_parquet(config.OUTPUT_PATH / 'customers.parquet', index=False)
    print(f"Saved customers.parquet")
    
    articles.to_parquet(config.OUTPUT_PATH / 'articles.parquet', index=False)
    print(f"Saved articles.parquet")
    
    # Save ground truth for validation
    val_ground_truth = val_transactions.groupby('customer_id')['article_id'].apply(list).reset_index()
    val_ground_truth.columns = ['customer_id', 'purchased_articles']
    val_ground_truth.to_parquet(config.OUTPUT_PATH / 'val_ground_truth.parquet', index=False)
    print(f"Saved val_ground_truth.parquet")
    
    # Get all unique users and items
    all_users = sorted(train_transactions['customer_id'].unique().tolist())
    all_items = sorted(train_transactions['article_id'].unique().tolist())
    
    print(f"\nFinal dataset statistics:")
    print(f"Training users: {len(all_users):,}")
    print(f"Training items: {len(all_items):,}")
    print(f"Validation users: {len(val_users):,}")
    
    return all_users, all_items


def run_stage1():
    """Run the complete Stage 1 pipeline"""
    print_section("STAGE 1: LOADING AND PREPARING DATA")
    
    # Step 1: Load raw data
    transactions, customers, articles = load_raw_data()
    
    # Step 2: Select temporal window
    transactions, max_date = select_temporal_window(transactions)
    
    # Step 3: User sampling
    selected_users, user_activity = stratified_user_sampling(transactions)
    
    # Step 4: Filter transactions
    train_transactions, val_transactions, val_users = filter_transactions(
        transactions, selected_users, max_date
    )
    del transactions
    force_garbage_collection()
    
    # Step 5: Filter items
    train_transactions, val_transactions, articles, selected_items = filter_items(
        train_transactions, val_transactions, articles
    )
    
    # Filter customers to selected users
    customers = customers[customers['customer_id'].isin(selected_users)].copy()
    
    # Step 6: Optimize memory
    train_transactions, val_transactions, customers, articles = optimize_memory(
        train_transactions, val_transactions, customers, articles
    )
    
    # Step 7: Save processed data
    all_users, all_items = save_processed_data(
        train_transactions, val_transactions, val_users,
        customers, articles, selected_users
    )
    
    # Save user activity stats
    user_activity_filtered = user_activity[user_activity['customer_id'].isin(selected_users)]
    user_activity_filtered.to_parquet(config.OUTPUT_PATH / 'user_activity_stats.parquet', index=False)
    
    print_section("STAGE 1 COMPLETE")
    print(f"Training transactions: {len(train_transactions):,}")
    print(f"Validation transactions: {len(val_transactions):,}")
    print(f"Unique users: {len(all_users):,}")
    print(f"Unique items: {len(all_items):,}")
    
    force_garbage_collection()
    
    return {
        'train_transactions': train_transactions,
        'val_transactions': val_transactions,
        'customers': customers,
        'articles': articles,
        'all_users': all_users,
        'all_items': all_items,
        'val_users': val_users,
        'max_date': max_date
    }


if __name__ == "__main__":
    data = run_stage1()

