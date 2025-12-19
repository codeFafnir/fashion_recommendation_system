"""
Enhanced Feature Engineering with Text-Based Article Features
Memory-Optimized Implementation

This module adds:
1. TF-IDF text features from product names/descriptions
2. Category semantic features
3. User preference embeddings based on text
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
import pickle
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class TextFeatureConfig:
    # Paths
    DATA_PATH = Path('/Users/raghu/Desktop/Quarter_1/CSE_258R/assignment2/fashion_recommender_2')
    OUTPUT_PATH = Path('/Users/raghu/Desktop/Quarter_1/CSE_258R/assignment2/fashion_recommender_features_2')
    
    # Text feature parameters
    MAX_TFIDF_FEATURES = 100  # Keep small for memory
    SVD_COMPONENTS = 20  # Dimension reduction
    MIN_DF = 5  # Minimum document frequency
    MAX_DF = 0.5  # Maximum document frequency
    
    # Processing
    CHUNK_SIZE = 50000
    
config = TextFeatureConfig()

# ============================================================================
# STAGE 2 ENHANCEMENT: CO-PURCHASE WITH TEXT SIMILARITY
# ============================================================================

def create_text_corpus(articles):
    """
    Create text corpus from article metadata
    Memory-efficient approach
    """
    print("Creating text corpus from articles...")
    
    # Text columns to use
    text_cols = [
        'prod_name', 'product_type_name', 'product_group_name',
        'graphical_appearance_name', 'colour_group_name',
        'perceived_colour_value_name', 'perceived_colour_master_name',
        'department_name', 'index_name', 'index_group_name',
        'section_name', 'garment_group_name'
    ]
    
    # Check which columns exist
    available_cols = [col for col in text_cols if col in articles.columns]
    print(f"  Available text columns: {len(available_cols)}/{len(text_cols)}")
    
    if not available_cols:
        print("  ⚠️  No text columns found, skipping text features")
        return None, available_cols
    
    # Combine text columns efficiently
    def safe_join(row):
        texts = []
        for col in available_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                texts.append(str(val).lower().strip())
        return ' '.join(texts) if texts else ''
    
    # Process in chunks to save memory
    corpus = []
    chunk_size = 10000
    
    for i in range(0, len(articles), chunk_size):
        chunk = articles.iloc[i:i+chunk_size]
        chunk_corpus = chunk[available_cols].apply(safe_join, axis=1)
        corpus.extend(chunk_corpus.tolist())
        
        if i % 50000 == 0:
            print(f"    Processed {i:,} articles...")
            gc.collect()
    
    print(f"  ✓ Created corpus for {len(corpus):,} articles")
    return corpus, available_cols


def compute_text_embeddings(articles, corpus):
    """
    Compute TF-IDF and reduce dimensionality
    Returns article_id -> embedding mapping
    """
    if corpus is None:
        return None, None
    
    print("\nComputing text embeddings...")
    
    # Remove empty documents
    valid_indices = [i for i, doc in enumerate(corpus) if doc.strip()]
    valid_corpus = [corpus[i] for i in valid_indices]
    valid_articles = articles.iloc[valid_indices]['article_id'].values
    
    print(f"  Valid documents: {len(valid_corpus):,}")
    
    # TF-IDF with strict parameters for memory
    print("  Computing TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_TFIDF_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        ngram_range=(1, 2),
        stop_words='english',
        dtype=np.float32
    )
    
    tfidf_matrix = vectorizer.fit_transform(valid_corpus)
    print(f"  ✓ TF-IDF shape: {tfidf_matrix.shape}")
    
    # Dimensionality reduction
    print(f"  Reducing to {config.SVD_COMPONENTS} dimensions...")
    svd = TruncatedSVD(n_components=config.SVD_COMPONENTS, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    embeddings = normalize(embeddings, norm='l2', axis=1)
    
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Create mapping
    article_embeddings = dict(zip(valid_articles, embeddings))
    
    # Clean up
    del tfidf_matrix, embeddings, valid_corpus
    gc.collect()
    
    return article_embeddings, vectorizer


def compute_user_text_preferences(train_transactions, article_embeddings, articles):
    """
    Compute user preferences in embedding space
    Based on their purchase history
    """
    if article_embeddings is None:
        return None
    
    print("\nComputing user text preferences...")
    
    # Get user purchase history with embeddings
    user_prefs = defaultdict(list)
    
    print("  Building user preference vectors...")
    for _, row in train_transactions.iterrows():
        user = row['customer_id']
        item = row['article_id']
        
        if item in article_embeddings:
            user_prefs[user].append(article_embeddings[item])
    
    # Average embeddings for each user
    user_embeddings = {}
    for user, item_embeds in user_prefs.items():
        if item_embeds:
            user_embeddings[user] = np.mean(item_embeds, axis=0)
    
    print(f"  ✓ Computed preferences for {len(user_embeddings):,} users")
    
    del user_prefs
    gc.collect()
    
    return user_embeddings


# ============================================================================
# STAGE 2 ADDITION: TEXT-BASED SIMILARITY CANDIDATES
# ============================================================================

def generate_text_similarity_candidates(
    all_users, 
    train_transactions,
    article_embeddings,
    user_embeddings,
    n_candidates=15
):
    """
    Generate candidates based on text similarity
    For each user, recommend items similar to their preferences
    """
    if article_embeddings is None or user_embeddings is None:
        print("⚠️  Text embeddings not available, skipping text-based candidates")
        return pd.DataFrame(columns=['customer_id', 'article_id', 'text_similarity_score'])
    
    print("\nGenerating text similarity candidates...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get all available articles and their embeddings
    available_items = list(article_embeddings.keys())
    item_embed_matrix = np.array([article_embeddings[item] for item in available_items])
    
    candidates = []
    
    # Get user purchase history
    user_purchased = (
        train_transactions
        .groupby('customer_id')['article_id']
        .apply(set)
        .to_dict()
    )
    
    print(f"  Processing {len(all_users):,} users...")
    
    for i, user in enumerate(all_users):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i:,} users...")
            gc.collect()
        
        # Skip if user has no embedding
        if user not in user_embeddings:
            continue
        
        user_embed = user_embeddings[user].reshape(1, -1)
        
        # Compute similarity with all items
        similarities = cosine_similarity(user_embed, item_embed_matrix)[0]
        
        # Get top N items not yet purchased
        purchased_items = user_purchased.get(user, set())
        
        # Sort by similarity
        top_indices = np.argsort(similarities)[::-1]
        
        count = 0
        for idx in top_indices:
            item = available_items[idx]
            if item not in purchased_items and count < n_candidates:
                candidates.append({
                    'customer_id': user,
                    'article_id': item,
                    'text_similarity_score': similarities[idx]
                })
                count += 1
            
            if count >= n_candidates:
                break
    
    candidates_df = pd.DataFrame(candidates)
    print(f"  ✓ Generated {len(candidates_df):,} text similarity candidates")
    
    del item_embed_matrix, user_purchased
    gc.collect()
    
    return candidates_df


# ============================================================================
# STAGE 3 ENHANCEMENT: ADDITIONAL FEATURES
# ============================================================================

def create_category_encoding_features(articles):
    """
    Create frequency-based encodings for categorical text features
    Memory efficient alternative to one-hot encoding
    """
    print("\nCreating category encoding features...")
    
    category_cols = [
        'product_type_name', 'product_group_name',
        'colour_group_name', 'department_name',
        'index_group_name', 'section_name', 'garment_group_name'
    ]
    
    available_cats = [col for col in category_cols if col in articles.columns]
    
    if not available_cats:
        print("  ⚠️  No categorical columns found")
        return articles
    
    print(f"  Processing {len(available_cats)} categorical columns...")
    
    article_features = articles[['article_id']].copy()
    
    for col in available_cats:
        # Frequency encoding
        freq_map = articles[col].value_counts(normalize=True).to_dict()
        article_features[f'{col}_frequency'] = articles[col].map(freq_map).astype(np.float32)
        
        # Count encoding
        count_map = articles[col].value_counts().to_dict()
        article_features[f'{col}_count'] = articles[col].map(count_map).astype(np.int32)
    
    print(f"  ✓ Created {len(available_cats)*2} category encoding features")
    
    return article_features


def create_text_match_features(candidates_chunk, user_embeddings, article_embeddings):
    """
    Create text-based matching features for candidate pairs
    """
    if user_embeddings is None or article_embeddings is None:
        return candidates_chunk
    
    # Text similarity score
    similarities = []
    for _, row in candidates_chunk.iterrows():
        user = row['customer_id']
        item = row['article_id']
        
        if user in user_embeddings and item in article_embeddings:
            user_vec = user_embeddings[user]
            item_vec = article_embeddings[item]
            sim = np.dot(user_vec, item_vec)
            similarities.append(sim)
        else:
            similarities.append(0.0)
    
    candidates_chunk['user_item_text_similarity'] = similarities
    
    return candidates_chunk


def create_semantic_diversity_features(train_transactions, articles, article_embeddings):
    """
    Compute user's semantic diversity in their purchase history
    """
    if article_embeddings is None:
        return None
    
    print("\nComputing semantic diversity features...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get user purchase history
    user_items = (
        train_transactions
        .groupby('customer_id')['article_id']
        .apply(list)
        .to_dict()
    )
    
    diversity_features = []
    
    for user, items in user_items.items():
        # Get embeddings for user's items
        item_vecs = [article_embeddings[item] for item in items if item in article_embeddings]
        
        if len(item_vecs) < 2:
            diversity_features.append({
                'customer_id': user,
                'semantic_diversity': 0.0,
                'semantic_range': 0.0
            })
            continue
        
        # Compute pairwise similarities
        item_matrix = np.array(item_vecs)
        sims = cosine_similarity(item_matrix)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = sims[np.triu_indices_from(sims, k=1)]
        
        # Diversity = 1 - average similarity
        avg_sim = upper_tri.mean()
        diversity = 1 - avg_sim
        
        # Range = max - min similarity
        sem_range = upper_tri.max() - upper_tri.min()
        
        diversity_features.append({
            'customer_id': user,
            'semantic_diversity': diversity,
            'semantic_range': sem_range
        })
    
    diversity_df = pd.DataFrame(diversity_features)
    print(f"  ✓ Computed diversity for {len(diversity_df):,} users")
    
    return diversity_df


# ============================================================================
# STAGE 3 ENHANCEMENT: TEXT FEATURE PIPELINE
# ============================================================================

def enhance_features_with_text(
    all_features,
    articles,
    train_transactions,
    article_embeddings,
    user_embeddings,
    chunk_size=50000
):
    """
    Add text-based features to the feature matrix
    Process in chunks for memory efficiency
    """
    print("\nEnhancing features with text semantics...")
    
    # 1. Category encoding features
    category_features = create_category_encoding_features(articles)
    
    # 2. Semantic diversity features
    diversity_features = create_semantic_diversity_features(
        train_transactions, 
        articles, 
        article_embeddings
    )
    
    # 3. Merge category features (item-level)
    print("\n  Merging category features...")
    all_features = all_features.merge(
        category_features,
        on='article_id',
        how='left'
    )
    
    # 4. Merge diversity features (user-level)
    if diversity_features is not None:
        print("  Merging semantic diversity features...")
        all_features = all_features.merge(
            diversity_features,
            on='customer_id',
            how='left'
        )
        all_features['semantic_diversity'] = all_features['semantic_diversity'].fillna(0).astype(np.float32)
        all_features['semantic_range'] = all_features['semantic_range'].fillna(0).astype(np.float32)
    
    # 5. Add text similarity features (in chunks)
    if user_embeddings is not None and article_embeddings is not None:
        print("  Computing user-item text similarities in chunks...")
        
        n_chunks = max(1, len(all_features) // chunk_size)
        chunks = np.array_split(all_features, n_chunks)
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"    Processing chunk {i+1}/{len(chunks)}...")
            
            enhanced_chunk = create_text_match_features(
                chunk,
                user_embeddings,
                article_embeddings
            )
            enhanced_chunks.append(enhanced_chunk)
            
            if i % 10 == 0:
                gc.collect()
        
        all_features = pd.concat(enhanced_chunks, ignore_index=True)
        del enhanced_chunks
        gc.collect()
    
    print("  ✓ Text enhancement complete")
    
    return all_features


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def integrate_text_features_stage2(
    all_users,
    train_transactions,
    articles,
    output_path
):
    """
    Stage 2: Add text-based candidates to recall
    """
    print("\n" + "="*80)
    print("  STAGE 2 ENHANCEMENT: TEXT-BASED CANDIDATES")
    print("="*80 + "\n")
    
    # 1. Create text corpus
    corpus, text_cols = create_text_corpus(articles)
    
    if corpus is None:
        print("⚠️  Skipping text features - no text columns found")
        return None, None, None, None
    
    # 2. Compute embeddings
    article_embeddings, vectorizer = compute_text_embeddings(articles, corpus)
    
    # 3. Compute user preferences
    user_embeddings = compute_user_text_preferences(
        train_transactions,
        article_embeddings,
        articles
    )
    
    # 4. Generate candidates
    text_candidates = generate_text_similarity_candidates(
        all_users,
        train_transactions,
        article_embeddings,
        user_embeddings,
        n_candidates=15
    )
    
    # 5. Save candidates
    if len(text_candidates) > 0:
        output_file = output_path / 'temp_text_similarity.parquet'
        text_candidates.to_parquet(output_file, index=False)
        print(f"\n✓ Saved text similarity candidates to {output_file}")
    
    # Save embeddings for Stage 3
    print("\nSaving embeddings for Stage 3...")
    
    with open(output_path / 'article_embeddings.pkl', 'wb') as f:
        pickle.dump(article_embeddings, f)
    
    with open(output_path / 'user_embeddings.pkl', 'wb') as f:
        pickle.dump(user_embeddings, f)
    
    print("✓ Saved embeddings")
    
    return text_candidates, article_embeddings, user_embeddings, text_cols


def integrate_text_features_stage3(
    all_features,
    articles,
    train_transactions,
    embeddings_path
):
    """
    Stage 3: Add text-based features to feature engineering
    """
    print("\n" + "="*80)
    print("  STAGE 3 ENHANCEMENT: TEXT-BASED FEATURES")
    print("="*80 + "\n")
    
    # Load embeddings
    print("Loading saved embeddings...")
    
    try:
        with open(embeddings_path / 'article_embeddings.pkl', 'rb') as f:
            article_embeddings = pickle.load(f)
        
        with open(embeddings_path / 'user_embeddings.pkl', 'rb') as f:
            user_embeddings = pickle.load(f)
        
        print(f"  ✓ Loaded {len(article_embeddings):,} article embeddings")
        print(f"  ✓ Loaded {len(user_embeddings):,} user embeddings")
    except FileNotFoundError:
        print("  ⚠️  Embeddings not found, skipping text features")
        return all_features
    
    # Enhance features
    all_features = enhance_features_with_text(
        all_features,
        articles,
        train_transactions,
        article_embeddings,
        user_embeddings,
        chunk_size=config.CHUNK_SIZE
    )
    
    return all_features
