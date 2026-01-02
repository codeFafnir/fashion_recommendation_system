"""
Stage 2: Generating Candidates
Candidate generation using multiple recall strategies

This stage performs:
1. Repurchase candidates (items user bought before)
2. Popularity-based candidates
3. Co-purchase candidates (item-to-item CF)
4. User-KNN collaborative filtering
5. Category-based recommendations
6. Merge all candidates
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import pickle
import gc

from config import Config, config
from utils import (
    print_section, force_garbage_collection, print_memory,
    time_decay_score
)


def generate_repurchase_candidates(train_transactions, all_users, max_date):
    """Generate candidates based on user repurchase history"""
    print_section("STRATEGY 1: REPURCHASE")
    
    # Calculate user-item purchase frequency with recency weighting
    train_transactions = train_transactions.copy()
    train_transactions['days_ago'] = (max_date - train_transactions['t_dat']).dt.days
    train_transactions['weight'] = time_decay_score(
        train_transactions['days_ago'].values, 0.05
    ).astype(np.float32)
    
    # Aggregate scores per user-item pair
    user_item_scores = (
        train_transactions
        .groupby(['customer_id', 'article_id'])
        .agg({'weight': 'sum'})
        .reset_index()
    )
    user_item_scores.columns = ['customer_id', 'article_id', 'repurchase_score']
    
    # Normalize scores per user
    user_item_scores['repurchase_score'] = (
        user_item_scores.groupby('customer_id')['repurchase_score']
        .transform(lambda x: x / x.max())
        .astype(np.float32)
    )
    
    # Get top candidates per user
    repurchase_candidates = (
        user_item_scores
        .sort_values(['customer_id', 'repurchase_score'], ascending=[True, False])
        .groupby('customer_id', as_index=False)
        .head(config.N_REPURCHASE_CANDIDATES)
        [['customer_id', 'article_id', 'repurchase_score']]
    )
    
    print(f"Generated {len(repurchase_candidates):,} repurchase candidates")
    print_memory()
    
    # Save intermediate result
    repurchase_candidates.to_parquet(config.OUTPUT_PATH / 'temp_repurchase.parquet', index=False)
    
    return repurchase_candidates


def generate_popularity_candidates(train_transactions, all_users, max_date):
    """Generate candidates based on item popularity"""
    print_section("STRATEGY 2: POPULARITY")
    
    cutoff_date = max_date - timedelta(weeks=config.POPULARITY_WINDOW_WEEKS)
    recent_trans = train_transactions[train_transactions['t_dat'] >= cutoff_date].copy()
    
    print(f"Using {len(recent_trans):,} recent transactions")
    
    recent_trans['days_ago'] = (max_date - recent_trans['t_dat']).dt.days
    recent_trans = recent_trans.dropna(subset=['days_ago'])
    recent_trans['days_ago'] = recent_trans['days_ago'].astype(np.int16)
    recent_trans['weight'] = time_decay_score(recent_trans['days_ago'].values, 0.1).astype(np.float32)
    
    # Calculate popularity scores
    item_popularity = (
        recent_trans
        .groupby('article_id', as_index=False)
        .agg({'weight': 'sum', 'customer_id': 'nunique'})
        .rename(columns={'weight': 'weighted_purchases', 'customer_id': 'unique_buyers'})
    )
    
    item_popularity['popularity_score'] = (
        0.7 * item_popularity['weighted_purchases'] +
        0.3 * item_popularity['unique_buyers']
    )
    item_popularity['popularity_score'] = (
        item_popularity['popularity_score'] / item_popularity['popularity_score'].max()
    ).astype(np.float32)
    
    # Get top items
    top_items = item_popularity.nlargest(config.N_POPULARITY_CANDIDATES, 'popularity_score')
    print(f"Top {len(top_items)} popular items")
    
    # Create candidates for all users
    pop_chunks = []
    for user_chunk in tqdm(np.array_split(all_users, 20), desc="Popularity chunks"):
        chunk_df = pd.DataFrame({
            'customer_id': np.repeat(user_chunk, len(top_items)),
            'article_id': np.tile(top_items['article_id'].values, len(user_chunk))
        })
        
        rank_penalty = np.tile(1 - np.arange(len(top_items)) * 0.01, len(user_chunk))
        scores = np.tile(top_items['popularity_score'].values, len(user_chunk))
        chunk_df['popularity_score'] = (scores * rank_penalty).astype(np.float32)
        
        pop_chunks.append(chunk_df)
    
    popularity_candidates = pd.concat(pop_chunks, ignore_index=True)
    del pop_chunks
    force_garbage_collection()
    
    print(f"Generated {len(popularity_candidates):,} popularity candidates")
    print_memory()
    
    # Save
    popularity_candidates.to_parquet(config.OUTPUT_PATH / 'temp_popularity.parquet', index=False)
    item_popularity.to_parquet(config.OUTPUT_PATH / 'item_popularity.parquet', index=False)
    
    return popularity_candidates, item_popularity


def generate_copurchase_candidates(train_transactions, all_users):
    """Generate candidates based on co-purchase patterns (item-to-item CF)"""
    print_section("STRATEGY 3: CO-PURCHASE (Item-to-Item CF)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_copurchase.parquet').exists():
        print("Found existing co-purchase candidates, loading...")
        copurchase_candidates = pd.read_parquet(config.OUTPUT_PATH / 'temp_copurchase.parquet')
        print(f"Loaded {len(copurchase_candidates):,} co-purchase candidates")
        return copurchase_candidates
    
    print("Building co-purchase matrix...")
    
    # Create basket ID (same user, same day)
    train_transactions = train_transactions.copy()
    train_transactions['basket_id'] = (
        train_transactions['customer_id'].astype(str) + '_' +
        train_transactions['t_dat'].astype(str)
    )
    
    # Get baskets with multiple items
    basket_items = (
        train_transactions
        .groupby('basket_id')['article_id']
        .apply(list)
        .reset_index()
    )
    basket_items = basket_items[basket_items['article_id'].apply(len) >= 2]
    print(f"Baskets with 2+ items: {len(basket_items):,}")
    
    # Build co-purchase counts
    copurchase_counts = defaultdict(lambda: defaultdict(int))
    
    for items in tqdm(basket_items['article_id'], desc="Processing baskets"):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1, item2 = items[i], items[j]
                copurchase_counts[item1][item2] += 1
                copurchase_counts[item2][item1] += 1
    
    print(f"Built co-purchase matrix for {len(copurchase_counts):,} items")
    
    # Compute item-to-item similarity scores
    item_to_items = {}
    
    for item1 in tqdm(copurchase_counts.keys(), desc="Computing similarities"):
        copurchased = copurchase_counts[item1]
        copurchased = {
            item2: count
            for item2, count in copurchased.items()
            if count >= config.MIN_ITEM_SUPPORT
        }
        
        if copurchased:
            top_items = sorted(
                copurchased.items(),
                key=lambda x: x[1],
                reverse=True
            )[:config.MAX_ITEM_NEIGHBORS]
            
            max_count = top_items[0][1]
            item_to_items[item1] = [
                (item2, count / max_count)
                for item2, count in top_items
            ]
    
    print(f"Computed similarities for {len(item_to_items):,} items")
    
    # Save item-to-item similarity matrix
    with open(config.OUTPUT_PATH / 'item_to_items.pkl', 'wb') as f:
        pickle.dump(item_to_items, f)
    
    # Generate co-purchase candidates for each user
    user_recent_items = (
        train_transactions
        .sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(lambda x: list(x.unique()[:10]))
        .to_dict()
    )
    
    copurchase_candidates = []
    
    for user in tqdm(all_users, desc="User co-purchase recommendations"):
        if user not in user_recent_items:
            continue
        
        user_items = user_recent_items[user]
        candidate_scores = defaultdict(float)
        
        for user_item in user_items:
            if user_item in item_to_items:
                for similar_item, score in item_to_items[user_item]:
                    if similar_item not in user_items:
                        candidate_scores[similar_item] += score
        
        if candidate_scores:
            top_candidates = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:config.N_COPURCHASE_CANDIDATES]
            
            for item, score in top_candidates:
                copurchase_candidates.append({
                    'customer_id': user,
                    'article_id': item,
                    'copurchase_score': score
                })
    
    copurchase_candidates = pd.DataFrame(copurchase_candidates)
    print(f"Generated {len(copurchase_candidates):,} co-purchase candidates")
    
    copurchase_candidates.to_parquet(config.OUTPUT_PATH / 'temp_copurchase.parquet', index=False)
    
    del basket_items, copurchase_counts, item_to_items
    force_garbage_collection()
    
    return copurchase_candidates


def generate_userknn_candidates(train_transactions, all_users, all_items, val_users, max_date):
    """Generate candidates based on user-KNN collaborative filtering"""
    print_section("STRATEGY 4: USER-KNN COLLABORATIVE FILTERING")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_userknn.parquet').exists():
        print("Found existing user-KNN candidates, loading...")
        userknn_candidates = pd.read_parquet(config.OUTPUT_PATH / 'temp_userknn.parquet')
        print(f"Loaded {len(userknn_candidates):,} user-KNN candidates")
        return userknn_candidates
    
    print("Building user-item matrix...")
    
    user_to_idx = {user: idx for idx, user in enumerate(all_users)}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    
    n_users = len(all_users)
    n_items = len(all_items)
    
    # Use recent transactions for user similarity
    recent_date = max_date - timedelta(weeks=4)
    recent_user_items = train_transactions[train_transactions['t_dat'] >= recent_date].copy()
    
    user_item_matrix = lil_matrix((n_users, n_items), dtype=np.int8)
    
    for _, row in tqdm(recent_user_items.iterrows(), total=len(recent_user_items), desc="Building matrix"):
        user_idx = user_to_idx[row['customer_id']]
        item_idx = item_to_idx[row['article_id']]
        user_item_matrix[user_idx, item_idx] = 1
    
    user_item_matrix = user_item_matrix.tocsr()
    
    # Save user-item matrix
    from scipy.sparse import save_npz
    save_npz(config.OUTPUT_PATH / 'user_item_matrix.npz', user_item_matrix)
    
    with open(config.OUTPUT_PATH / 'user_to_idx.pkl', 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open(config.OUTPUT_PATH / 'item_to_idx.pkl', 'wb') as f:
        pickle.dump(item_to_idx, f)
    
    # Compute similarities for validation users
    print(f"Computing similarities for {len(val_users):,} validation users...")
    
    val_user_indices = [user_to_idx[user] for user in val_users if user in user_to_idx]
    
    val_user_matrix_norm = normalize(user_item_matrix[val_user_indices], norm='l2', axis=1)
    user_item_matrix_norm = normalize(user_item_matrix, norm='l2', axis=1)
    
    # Batch processing
    batch_size = 1000
    userknn_candidates = []
    
    for i in tqdm(range(0, len(val_user_indices), batch_size), desc="Similarity batches"):
        batch_indices = val_user_indices[i:i+batch_size]
        batch_matrix = val_user_matrix_norm[i:i+batch_size]
        
        similarities = cosine_similarity(batch_matrix, user_item_matrix_norm)
        
        for j, user_idx in enumerate(batch_indices):
            user = all_users[user_idx]
            user_sims = similarities[j]
            
            similar_user_indices = np.argsort(user_sims)[::-1][1:config.N_SIMILAR_USERS+1]
            
            candidate_scores = defaultdict(float)
            user_purchased = set(
                train_transactions[train_transactions['customer_id'] == user]['article_id']
            )
            
            for sim_user_idx in similar_user_indices:
                sim_score = user_sims[sim_user_idx]
                if sim_score < 0.01:
                    continue
                
                sim_user = all_users[sim_user_idx]
                sim_user_items = train_transactions[
                    train_transactions['customer_id'] == sim_user
                ]['article_id'].unique()
                
                for item in sim_user_items:
                    if item not in user_purchased:
                        candidate_scores[item] += sim_score
            
            if candidate_scores:
                top_candidates = sorted(
                    candidate_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:config.N_USERKNN_CANDIDATES]
                
                for item, score in top_candidates:
                    userknn_candidates.append({
                        'customer_id': user,
                        'article_id': item,
                        'userknn_score': score
                    })
    
    userknn_candidates = pd.DataFrame(userknn_candidates)
    print(f"Generated {len(userknn_candidates):,} user-KNN candidates")
    
    userknn_candidates.to_parquet(config.OUTPUT_PATH / 'temp_userknn.parquet', index=False)
    
    del user_item_matrix, val_user_matrix_norm, user_item_matrix_norm
    force_garbage_collection()
    
    return userknn_candidates


def generate_category_candidates(train_transactions, articles, all_users):
    """Generate candidates based on category preferences"""
    print_section("STRATEGY 5: CATEGORY-BASED RECOMMENDATIONS")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_category.parquet').exists():
        print("Found existing category candidates, loading...")
        category_candidates = pd.read_parquet(config.OUTPUT_PATH / 'temp_category.parquet')
        print(f"Loaded {len(category_candidates):,} category candidates")
        return category_candidates
    
    print("Computing user category preferences...")
    
    user_categories = (
        train_transactions
        .merge(articles[['article_id', 'product_type_no', 'product_group_name']], on='article_id')
        .groupby(['customer_id', 'product_type_no'])
        .size()
        .reset_index(name='count')
    )
    
    user_top_categories = (
        user_categories
        .sort_values(['customer_id', 'count'], ascending=[True, False])
        .groupby('customer_id')
        .head(3)
    )
    
    user_top_categories.to_parquet(config.OUTPUT_PATH / 'user_category_preferences.parquet', index=False)
    
    # Get popular items per category
    category_popular_items = (
        train_transactions
        .merge(articles[['article_id', 'product_type_no']], on='article_id')
        .groupby(['product_type_no', 'article_id'])
        .size()
        .reset_index(name='count')
        .sort_values(['product_type_no', 'count'], ascending=[True, False])
        .groupby('product_type_no')
        .head(config.N_TOP_CATEGORY_ITEMS)
    )
    
    # Generate candidates
    category_candidates = []
    
    user_cats_dict = user_top_categories.groupby('customer_id')['product_type_no'].apply(list).to_dict()
    cat_items_dict = category_popular_items.groupby('product_type_no')['article_id'].apply(list).to_dict()
    
    for user in tqdm(all_users, desc="Category recommendations"):
        if user not in user_cats_dict:
            continue
        
        user_cats = user_cats_dict[user]
        user_purchased = set(
            train_transactions[train_transactions['customer_id'] == user]['article_id']
        )
        
        for cat in user_cats:
            if cat not in cat_items_dict:
                continue
            
            cat_items = cat_items_dict[cat]
            n_added = 0
            
            for item in cat_items:
                if item not in user_purchased and n_added < config.N_CATEGORY_CANDIDATES:
                    category_candidates.append({
                        'customer_id': user,
                        'article_id': item,
                        'category_score': 1.0 - n_added * 0.05
                    })
                    n_added += 1
    
    category_candidates = pd.DataFrame(category_candidates)
    print(f"Generated {len(category_candidates):,} category candidates")
    
    category_candidates.to_parquet(config.OUTPUT_PATH / 'temp_category.parquet', index=False)
    
    return category_candidates


def merge_all_candidates(repurchase_candidates, popularity_candidates,
                         copurchase_candidates, userknn_candidates,
                         category_candidates):
    """Merge all candidate sources"""
    print_section("MERGING ALL CANDIDATES")
    
    # Start with repurchase
    candidates = repurchase_candidates[['customer_id', 'article_id', 'repurchase_score']].copy()
    
    # Merge popularity
    candidates = candidates.merge(
        popularity_candidates[['customer_id', 'article_id', 'popularity_score']],
        on=['customer_id', 'article_id'],
        how='outer'
    )
    
    # Merge co-purchase
    candidates = candidates.merge(
        copurchase_candidates[['customer_id', 'article_id', 'copurchase_score']],
        on=['customer_id', 'article_id'],
        how='outer'
    )
    
    # Merge user-KNN
    candidates = candidates.merge(
        userknn_candidates[['customer_id', 'article_id', 'userknn_score']],
        on=['customer_id', 'article_id'],
        how='outer'
    )
    
    # Merge category
    candidates = candidates.merge(
        category_candidates[['customer_id', 'article_id', 'category_score']],
        on=['customer_id', 'article_id'],
        how='outer'
    )
    
    # Fill NaN and count strategies
    score_cols = ['repurchase_score', 'popularity_score', 'copurchase_score',
                  'userknn_score', 'category_score']
    candidates[score_cols] = candidates[score_cols].fillna(0)
    
    candidates['n_strategies'] = (candidates[score_cols] > 0).sum(axis=1).astype(np.int8)
    
    print(f"\nTotal unique candidates: {len(candidates):,}")
    print(f"Unique users: {candidates['customer_id'].nunique():,}")
    print(f"Unique items: {candidates['article_id'].nunique():,}")
    
    # Strategy coverage
    print("\nStrategy coverage:")
    for col in score_cols:
        coverage = (candidates[col] > 0).mean() * 100
        print(f"  {col}: {coverage:.1f}%")
    
    print(f"\nCandidates per user: {candidates.groupby('customer_id').size().mean():.1f}")
    
    # Save
    candidates.to_parquet(config.OUTPUT_PATH / 'candidates.parquet', index=False)
    print(f"\nSaved candidates.parquet")
    
    return candidates


def run_stage2(data=None):
    """Run the complete Stage 2 pipeline"""
    print_section("STAGE 2: GENERATING CANDIDATES")
    
    if data is None:
        # Load from saved files
        train_transactions = pd.read_parquet(config.OUTPUT_PATH / 'train_transactions.parquet')
        articles = pd.read_parquet(config.OUTPUT_PATH / 'articles.parquet')
        val_ground_truth = pd.read_parquet(config.OUTPUT_PATH / 'val_ground_truth.parquet')
        
        all_users = sorted(train_transactions['customer_id'].unique().tolist())
        all_items = sorted(train_transactions['article_id'].unique().tolist())
        val_users = set(val_ground_truth['customer_id'].unique())
        max_date = train_transactions['t_dat'].max()
    else:
        train_transactions = data['train_transactions']
        articles = data['articles']
        all_users = data['all_users']
        all_items = data['all_items']
        val_users = data['val_users']
        max_date = data['max_date']
    
    # Generate candidates from each strategy
    repurchase_candidates = generate_repurchase_candidates(train_transactions, all_users, max_date)
    force_garbage_collection()
    
    popularity_candidates, item_popularity = generate_popularity_candidates(
        train_transactions, all_users, max_date
    )
    force_garbage_collection()
    
    copurchase_candidates = generate_copurchase_candidates(train_transactions, all_users)
    force_garbage_collection()
    
    userknn_candidates = generate_userknn_candidates(
        train_transactions, all_users, all_items, val_users, max_date
    )
    force_garbage_collection()
    
    category_candidates = generate_category_candidates(train_transactions, articles, all_users)
    force_garbage_collection()
    
    # Merge all candidates
    candidates = merge_all_candidates(
        repurchase_candidates, popularity_candidates,
        copurchase_candidates, userknn_candidates,
        category_candidates
    )
    
    print_section("STAGE 2 COMPLETE")
    print(f"Total candidates: {len(candidates):,}")
    
    return candidates


if __name__ == "__main__":
    run_stage2()

