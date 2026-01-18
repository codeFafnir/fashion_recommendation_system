"""
Stage 2: Generating Candidates
Candidate generation using multiple recall strategies with PySpark

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
from tqdm.auto import tqdm
import pickle
import gc

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, ArrayType, IntegerType, StringType, StructType, StructField

from config import Config, config
from utils import (
    print_section, force_garbage_collection, print_memory,
    time_decay_score
)
from spark_utils import (
    get_spark_session, spark_to_pandas, pandas_to_spark,
    cache_and_count, top_n_per_group, repartition_by_key
)


def generate_repurchase_candidates_spark(spark, train_transactions_spark, all_users, max_date):
    """Generate candidates based on user repurchase history using PySpark"""
    print_section("STRATEGY 1: REPURCHASE (SPARK)")
    
    # Calculate days_ago and weight
    train_with_weight = train_transactions_spark.withColumn(
        "days_ago",
        F.datediff(F.lit(max_date), F.col("t_dat"))
    ).withColumn(
        "weight",
        F.exp(-0.05 * F.col("days_ago")).cast(FloatType())
    )
    
    # Aggregate scores per user-item pair
    user_item_scores = train_with_weight.groupBy("customer_id", "article_id").agg(
        F.sum("weight").alias("repurchase_score")
    )
    
    # Normalize scores per user (divide by max in group)
    window = Window.partitionBy("customer_id")
    user_item_scores = user_item_scores.withColumn(
        "max_score",
        F.max("repurchase_score").over(window)
    ).withColumn(
        "repurchase_score",
        (F.col("repurchase_score") / F.col("max_score")).cast(FloatType())
    ).drop("max_score")
    
    # Get top N candidates per user
    repurchase_candidates = top_n_per_group(
        user_item_scores,
        group_by="customer_id",
        order_by="repurchase_score",
        n=config.N_REPURCHASE_CANDIDATES
    )
    
    repurchase_candidates = cache_and_count(repurchase_candidates, "Repurchase candidates")
    
    # Save intermediate result
    repurchase_candidates_pd = spark_to_pandas(repurchase_candidates.coalesce(10))
    repurchase_candidates_pd.to_parquet(config.OUTPUT_PATH / 'temp_repurchase.parquet', index=False)
    
    print_memory()
    
    return repurchase_candidates


def generate_popularity_candidates_spark(spark, train_transactions_spark, all_users_spark, max_date):
    """Generate candidates based on item popularity using PySpark"""
    print_section("STRATEGY 2: POPULARITY (SPARK)")
    
    cutoff_date = max_date - timedelta(weeks=config.POPULARITY_WINDOW_WEEKS)
    
    # Filter to recent transactions
    recent_trans = train_transactions_spark.filter(F.col("t_dat") >= cutoff_date)
    
    # Calculate weights
    recent_trans = recent_trans.withColumn(
        "days_ago",
        F.datediff(F.lit(max_date), F.col("t_dat"))
    ).withColumn(
        "weight",
        F.exp(-0.1 * F.col("days_ago")).cast(FloatType())
    )
    
    recent_count = recent_trans.count()
    print(f"Using {recent_count:,} recent transactions")
    
    # Calculate popularity scores
    item_popularity = recent_trans.groupBy("article_id").agg(
        F.sum("weight").alias("weighted_purchases"),
        F.countDistinct("customer_id").alias("unique_buyers")
    )
    
    # Compute combined popularity score
    max_stats = item_popularity.agg(
        F.max("weighted_purchases").alias("max_weighted"),
        F.max("unique_buyers").alias("max_buyers")
    ).collect()[0]
    
    item_popularity = item_popularity.withColumn(
        "popularity_score",
        (
            0.7 * F.col("weighted_purchases") / max_stats["max_weighted"] +
            0.3 * F.col("unique_buyers") / max_stats["max_buyers"]
        ).cast(FloatType())
    )
    
    # Get top popular items
    top_items = item_popularity.orderBy(F.col("popularity_score").desc()).limit(
        config.N_POPULARITY_CANDIDATES
    )
    top_items = top_items.cache()
    
    print(f"Top {top_items.count()} popular items")
    
    # Cross join with all users to create candidates
    # Add rank penalty
    top_items_with_rank = top_items.withColumn(
        "rank",
        F.row_number().over(Window.orderBy(F.col("popularity_score").desc()))
    ).withColumn(
        "rank_penalty",
        (1 - (F.col("rank") - 1) * 0.01).cast(FloatType())
    )
    
    # Cross join with users
    popularity_candidates = all_users_spark.crossJoin(
        top_items_with_rank.select("article_id", "popularity_score", "rank_penalty")
    ).withColumn(
        "popularity_score",
        (F.col("popularity_score") * F.col("rank_penalty")).cast(FloatType())
    ).drop("rank_penalty")
    
    popularity_candidates = cache_and_count(popularity_candidates, "Popularity candidates")
    
    # Save results
    popularity_candidates_pd = spark_to_pandas(popularity_candidates.coalesce(10))
    popularity_candidates_pd.to_parquet(config.OUTPUT_PATH / 'temp_popularity.parquet', index=False)
    
    item_popularity_pd = spark_to_pandas(item_popularity)
    item_popularity_pd.to_parquet(config.OUTPUT_PATH / 'item_popularity.parquet', index=False)
    
    top_items.unpersist()
    print_memory()
    
    return popularity_candidates, item_popularity


def generate_copurchase_candidates_spark(spark, train_transactions_spark, all_users_spark):
    """Generate candidates based on co-purchase patterns using PySpark"""
    print_section("STRATEGY 3: CO-PURCHASE (SPARK)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_copurchase.parquet').exists():
        print("Found existing co-purchase candidates, loading...")
        copurchase_candidates_pd = pd.read_parquet(config.OUTPUT_PATH / 'temp_copurchase.parquet')
        print(f"Loaded {len(copurchase_candidates_pd):,} co-purchase candidates")
        copurchase_candidates = pandas_to_spark(spark, copurchase_candidates_pd)
        return copurchase_candidates
    
    print("Building co-purchase matrix with Spark...")
    
    # Create basket ID (same user, same day)
    baskets = train_transactions_spark.withColumn(
        "basket_id",
        F.concat(F.col("customer_id"), F.lit("_"), F.col("t_dat").cast("string"))
    )
    
    # Get items per basket
    basket_items = baskets.groupBy("basket_id").agg(
        F.collect_set("article_id").alias("items"),
        F.first("customer_id").alias("customer_id")
    ).filter(F.size("items") >= 2)
    
    basket_count = basket_items.count()
    print(f"Baskets with 2+ items: {basket_count:,}")
    
    # Explode to get pairs - self join on basket_id
    basket_exploded = basket_items.select(
        "basket_id",
        F.explode("items").alias("item1")
    )
    
    # Self join to get pairs
    pairs = basket_exploded.alias("a").join(
        basket_exploded.alias("b"),
        (F.col("a.basket_id") == F.col("b.basket_id")) & 
        (F.col("a.item1") < F.col("b.item1"))
    ).select(
        F.col("a.item1").alias("item1"),
        F.col("b.item1").alias("item2")
    )
    
    # Count co-purchases
    copurchase_counts = pairs.groupBy("item1", "item2").count()
    
    # Also add reverse pairs
    copurchase_counts_reversed = copurchase_counts.select(
        F.col("item2").alias("item1"),
        F.col("item1").alias("item2"),
        "count"
    )
    
    all_copurchases = copurchase_counts.union(copurchase_counts_reversed)
    
    # Filter by minimum support
    all_copurchases = all_copurchases.filter(F.col("count") >= config.MIN_ITEM_SUPPORT)
    
    # Normalize scores per source item and get top neighbors
    window = Window.partitionBy("item1")
    item_similarities = all_copurchases.withColumn(
        "max_count",
        F.max("count").over(window)
    ).withColumn(
        "similarity",
        (F.col("count") / F.col("max_count")).cast(FloatType())
    ).drop("max_count", "count")
    
    # Get top N similar items per item
    item_similarities = top_n_per_group(
        item_similarities,
        group_by="item1",
        order_by="similarity",
        n=config.MAX_ITEM_NEIGHBORS
    )
    
    item_similarities = item_similarities.cache()
    print(f"Computed similarities for {item_similarities.select('item1').distinct().count():,} items")
    
    # Save item-to-item similarities
    item_sim_pd = spark_to_pandas(item_similarities)
    item_to_items = {}
    for item1, group in item_sim_pd.groupby('item1'):
        item_to_items[item1] = list(zip(group['item2'].tolist(), group['similarity'].tolist()))
    
    with open(config.OUTPUT_PATH / 'item_to_items.pkl', 'wb') as f:
        pickle.dump(item_to_items, f)
    
    # Get user recent items
    user_recent_items = train_transactions_spark.withColumn(
        "rank",
        F.row_number().over(Window.partitionBy("customer_id").orderBy(F.col("t_dat").desc()))
    ).filter(F.col("rank") <= 10).select(
        "customer_id",
        F.col("article_id").alias("user_item")
    )
    
    # Join with item similarities to get recommendations
    user_recommendations = user_recent_items.join(
        item_similarities,
        user_recent_items["user_item"] == item_similarities["item1"],
        "inner"
    ).select(
        "customer_id",
        F.col("item2").alias("article_id"),
        "similarity"
    )
    
    # Filter out items user already purchased
    user_purchased = train_transactions_spark.select(
        "customer_id",
        F.col("article_id").alias("purchased_item")
    ).distinct()
    
    user_recommendations = user_recommendations.join(
        user_purchased,
        (user_recommendations["customer_id"] == user_purchased["customer_id"]) &
        (user_recommendations["article_id"] == user_purchased["purchased_item"]),
        "left_anti"
    )
    
    # Aggregate scores per user-item
    copurchase_candidates = user_recommendations.groupBy(
        "customer_id", "article_id"
    ).agg(
        F.sum("similarity").alias("copurchase_score")
    )
    
    # Get top N per user
    copurchase_candidates = top_n_per_group(
        copurchase_candidates,
        group_by="customer_id",
        order_by="copurchase_score",
        n=config.N_COPURCHASE_CANDIDATES
    )
    
    copurchase_candidates = cache_and_count(copurchase_candidates, "Co-purchase candidates")
    
    # Save
    copurchase_candidates_pd = spark_to_pandas(copurchase_candidates.coalesce(10))
    copurchase_candidates_pd.to_parquet(config.OUTPUT_PATH / 'temp_copurchase.parquet', index=False)
    
    item_similarities.unpersist()
    force_garbage_collection()
    
    return copurchase_candidates


def generate_userknn_candidates_spark(spark, train_transactions_spark, all_users_spark, 
                                      all_items, val_users_set, max_date):
    """Generate candidates based on user-KNN collaborative filtering using PySpark"""
    print_section("STRATEGY 4: USER-KNN (SPARK)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_userknn.parquet').exists():
        print("Found existing user-KNN candidates, loading...")
        userknn_candidates_pd = pd.read_parquet(config.OUTPUT_PATH / 'temp_userknn.parquet')
        print(f"Loaded {len(userknn_candidates_pd):,} user-KNN candidates")
        userknn_candidates = pandas_to_spark(spark, userknn_candidates_pd)
        return userknn_candidates
    
    print("Building user-item matrix with Spark...")
    
    # Use recent transactions for user similarity
    recent_date = max_date - timedelta(weeks=4)
    recent_trans = train_transactions_spark.filter(F.col("t_dat") >= recent_date)
    
    # Create user-item interactions (binary)
    user_items = recent_trans.select("customer_id", "article_id").distinct()
    user_items = user_items.cache()
    
    # For user similarity, we need to compare users based on their item purchases
    # Using approximate similarity via item overlap
    
    # Get validation users
    val_users_df = spark.createDataFrame(
        [(u,) for u in val_users_set], 
        ["customer_id"]
    )
    
    # Get items for validation users
    val_user_items = user_items.join(val_users_df, "customer_id", "inner")
    
    # Find similar users based on shared items
    # Join val_user_items with all user_items on article_id
    user_similarity = val_user_items.alias("v").join(
        user_items.alias("a"),
        F.col("v.article_id") == F.col("a.article_id"),
        "inner"
    ).filter(
        F.col("v.customer_id") != F.col("a.customer_id")
    ).groupBy(
        F.col("v.customer_id").alias("val_user"),
        F.col("a.customer_id").alias("similar_user")
    ).agg(
        F.count("*").alias("shared_items")
    )
    
    # Get top N similar users per validation user
    user_similarity = top_n_per_group(
        user_similarity,
        group_by="val_user",
        order_by="shared_items",
        n=config.N_SIMILAR_USERS
    )
    
    user_similarity = user_similarity.withColumn(
        "sim_score",
        F.col("shared_items").cast(FloatType())
    )
    
    # Get items from similar users
    similar_user_items = user_similarity.join(
        user_items,
        user_similarity["similar_user"] == user_items["customer_id"],
        "inner"
    ).select(
        F.col("val_user").alias("customer_id"),
        "article_id",
        "sim_score"
    )
    
    # Filter out items the validation user already purchased
    val_user_purchased = val_user_items.select(
        "customer_id",
        F.col("article_id").alias("purchased_item")
    ).distinct()
    
    similar_user_items = similar_user_items.join(
        val_user_purchased,
        (similar_user_items["customer_id"] == val_user_purchased["customer_id"]) &
        (similar_user_items["article_id"] == val_user_purchased["purchased_item"]),
        "left_anti"
    )
    
    # Aggregate scores
    userknn_candidates = similar_user_items.groupBy(
        "customer_id", "article_id"
    ).agg(
        F.sum("sim_score").alias("userknn_score")
    )
    
    # Get top N per user
    userknn_candidates = top_n_per_group(
        userknn_candidates,
        group_by="customer_id",
        order_by="userknn_score",
        n=config.N_USERKNN_CANDIDATES
    )
    
    userknn_candidates = cache_and_count(userknn_candidates, "User-KNN candidates")
    
    # Save user-item matrix info
    user_to_idx = {user: idx for idx, user in enumerate(
        user_items.select("customer_id").distinct().toPandas()["customer_id"].tolist()
    )}
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    
    with open(config.OUTPUT_PATH / 'user_to_idx.pkl', 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open(config.OUTPUT_PATH / 'item_to_idx.pkl', 'wb') as f:
        pickle.dump(item_to_idx, f)
    
    # Save candidates
    userknn_candidates_pd = spark_to_pandas(userknn_candidates.coalesce(10))
    userknn_candidates_pd.to_parquet(config.OUTPUT_PATH / 'temp_userknn.parquet', index=False)
    
    user_items.unpersist()
    force_garbage_collection()
    
    return userknn_candidates


def generate_category_candidates_spark(spark, train_transactions_spark, articles_spark, all_users_spark):
    """Generate candidates based on category preferences using PySpark"""
    print_section("STRATEGY 5: CATEGORY-BASED (SPARK)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'temp_category.parquet').exists():
        print("Found existing category candidates, loading...")
        category_candidates_pd = pd.read_parquet(config.OUTPUT_PATH / 'temp_category.parquet')
        print(f"Loaded {len(category_candidates_pd):,} category candidates")
        category_candidates = pandas_to_spark(spark, category_candidates_pd)
        return category_candidates
    
    print("Computing user category preferences with Spark...")
    
    # Join transactions with articles to get product_type_no
    trans_with_category = train_transactions_spark.join(
        articles_spark.select("article_id", "product_type_no"),
        "article_id",
        "inner"
    )
    
    # Compute user category preferences
    user_categories = trans_with_category.groupBy(
        "customer_id", "product_type_no"
    ).count()
    
    # Get top 3 categories per user
    user_top_categories = top_n_per_group(
        user_categories,
        group_by="customer_id",
        order_by="count",
        n=3
    )
    
    # Save user category preferences
    user_top_categories_pd = spark_to_pandas(user_top_categories.coalesce(5))
    user_top_categories_pd.to_parquet(config.OUTPUT_PATH / 'user_category_preferences.parquet', index=False)
    
    # Get popular items per category
    category_items = trans_with_category.groupBy(
        "product_type_no", "article_id"
    ).count()
    
    category_popular_items = top_n_per_group(
        category_items,
        group_by="product_type_no",
        order_by="count",
        n=config.N_TOP_CATEGORY_ITEMS
    )
    
    # Add rank within category
    window = Window.partitionBy("product_type_no").orderBy(F.col("count").desc())
    category_popular_items = category_popular_items.withColumn(
        "cat_rank",
        F.row_number().over(window)
    ).withColumn(
        "category_score",
        (1.0 - (F.col("cat_rank") - 1) * 0.05).cast(FloatType())
    )
    
    # Join user preferences with category popular items
    category_candidates = user_top_categories.join(
        category_popular_items.select("product_type_no", "article_id", "category_score"),
        "product_type_no",
        "inner"
    ).select("customer_id", "article_id", "category_score")
    
    # Filter out items user already purchased
    user_purchased = train_transactions_spark.select(
        "customer_id",
        F.col("article_id").alias("purchased_item")
    ).distinct()
    
    category_candidates = category_candidates.join(
        user_purchased,
        (category_candidates["customer_id"] == user_purchased["customer_id"]) &
        (category_candidates["article_id"] == user_purchased["purchased_item"]),
        "left_anti"
    )
    
    # Get top N per user
    category_candidates = top_n_per_group(
        category_candidates,
        group_by="customer_id",
        order_by="category_score",
        n=config.N_CATEGORY_CANDIDATES
    )
    
    category_candidates = cache_and_count(category_candidates, "Category candidates")
    
    # Save
    category_candidates_pd = spark_to_pandas(category_candidates.coalesce(10))
    category_candidates_pd.to_parquet(config.OUTPUT_PATH / 'temp_category.parquet', index=False)
    
    return category_candidates


def merge_all_candidates_spark(spark, repurchase_candidates, popularity_candidates,
                         copurchase_candidates, userknn_candidates,
                         category_candidates):
    """Merge all candidate sources using PySpark"""
    print_section("MERGING ALL CANDIDATES (SPARK)")
    
    # Start with repurchase
    candidates = repurchase_candidates.select(
        "customer_id", "article_id", "repurchase_score"
    )
    
    # Full outer join with popularity
    candidates = candidates.join(
        popularity_candidates.select("customer_id", "article_id", "popularity_score"),
        ["customer_id", "article_id"],
        "outer"
    )
    
    # Full outer join with co-purchase
    candidates = candidates.join(
        copurchase_candidates.select("customer_id", "article_id", "copurchase_score"),
        ["customer_id", "article_id"],
        "outer"
    )
    
    # Full outer join with user-KNN
    candidates = candidates.join(
        userknn_candidates.select("customer_id", "article_id", "userknn_score"),
        ["customer_id", "article_id"],
        "outer"
    )
    
    # Full outer join with category
    candidates = candidates.join(
        category_candidates.select("customer_id", "article_id", "category_score"),
        ["customer_id", "article_id"],
        "outer"
    )
    
    # Fill NaN with 0
    score_cols = ['repurchase_score', 'popularity_score', 'copurchase_score',
                  'userknn_score', 'category_score']
    
    for col in score_cols:
        candidates = candidates.withColumn(
            col,
            F.coalesce(F.col(col), F.lit(0.0)).cast(FloatType())
        )
    
    # Count number of strategies that contributed
    candidates = candidates.withColumn(
        "n_strategies",
        (
            F.when(F.col("repurchase_score") > 0, 1).otherwise(0) +
            F.when(F.col("popularity_score") > 0, 1).otherwise(0) +
            F.when(F.col("copurchase_score") > 0, 1).otherwise(0) +
            F.when(F.col("userknn_score") > 0, 1).otherwise(0) +
            F.when(F.col("category_score") > 0, 1).otherwise(0)
        ).cast("int")
    )
    
    candidates = candidates.cache()
    
    # Print statistics
    total_candidates = candidates.count()
    unique_users = candidates.select("customer_id").distinct().count()
    unique_items = candidates.select("article_id").distinct().count()
    
    print(f"\nTotal unique candidates: {total_candidates:,}")
    print(f"Unique users: {unique_users:,}")
    print(f"Unique items: {unique_items:,}")
    
    # Strategy coverage
    print("\nStrategy coverage:")
    for col in score_cols:
        coverage = candidates.filter(F.col(col) > 0).count() / total_candidates * 100
        print(f"  {col}: {coverage:.1f}%")
    
    avg_per_user = total_candidates / unique_users
    print(f"\nCandidates per user: {avg_per_user:.1f}")
    
    # Convert to pandas and save
    candidates_pd = spark_to_pandas(candidates.coalesce(20))
    candidates_pd.to_parquet(config.OUTPUT_PATH / 'candidates.parquet', index=False)
    print(f"\nSaved candidates.parquet")
    
    return candidates


def run_stage2(data=None):
    """Run the complete Stage 2 pipeline with PySpark"""
    print_section("STAGE 2: GENERATING CANDIDATES WITH PYSPARK")
    
    # Initialize Spark
    spark = get_spark_session(
        app_name="FashionRecommendation_Stage2",
        memory=config.SPARK_MEMORY
    )
    print(f"Spark session initialized")
    
    try:
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
    
        # Convert to Spark DataFrames
        train_transactions_spark = pandas_to_spark(spark, train_transactions)
        train_transactions_spark = train_transactions_spark.cache()
        
        articles_spark = pandas_to_spark(spark, articles)
        articles_spark = articles_spark.cache()
        
        # Create users DataFrame
        all_users_spark = spark.createDataFrame(
            [(u,) for u in all_users],
            ["customer_id"]
        )
        
    # Generate candidates from each strategy
        repurchase_candidates = generate_repurchase_candidates_spark(
            spark, train_transactions_spark, all_users, max_date
        )
    force_garbage_collection()
    
        popularity_candidates, item_popularity = generate_popularity_candidates_spark(
            spark, train_transactions_spark, all_users_spark, max_date
    )
    force_garbage_collection()
    
        copurchase_candidates = generate_copurchase_candidates_spark(
            spark, train_transactions_spark, all_users_spark
        )
    force_garbage_collection()
    
        userknn_candidates = generate_userknn_candidates_spark(
            spark, train_transactions_spark, all_users_spark, all_items, val_users, max_date
    )
    force_garbage_collection()
    
        category_candidates = generate_category_candidates_spark(
            spark, train_transactions_spark, articles_spark, all_users_spark
        )
    force_garbage_collection()
    
    # Merge all candidates
        candidates = merge_all_candidates_spark(
            spark, repurchase_candidates, popularity_candidates,
        copurchase_candidates, userknn_candidates,
        category_candidates
    )
        
        # Convert final result to pandas
        candidates_pd = pd.read_parquet(config.OUTPUT_PATH / 'candidates.parquet')
    
    print_section("STAGE 2 COMPLETE")
        print(f"Total candidates: {len(candidates_pd):,}")
        
        return candidates_pd
        
    finally:
        # Stop Spark session
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    run_stage2()
