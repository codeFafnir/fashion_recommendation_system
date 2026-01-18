"""
Stage 3: Extracting Features
Feature engineering for user, item, and interaction features using PySpark

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

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, IntegerType

from config import Config, config
from utils import (
    print_section, force_garbage_collection, print_memory,
    time_decay_score, chunk_dataframe
)
from spark_utils import (
    get_spark_session, spark_to_pandas, pandas_to_spark,
    cache_and_count, top_n_per_group
)


def compute_user_features_spark(spark, train_transactions_spark, customers_spark, max_date):
    """Compute user-level statistics features using PySpark"""
    print_section("PART 1: USER FEATURES (SPARK)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'user_features.parquet').exists():
        print("Found existing user_features.parquet, loading...")
        user_stats = pd.read_parquet(config.OUTPUT_PATH / 'user_features.parquet')
        print(f"Loaded {len(user_stats.columns)-1} user features")
        return user_stats
    
    print("Computing user statistics features with Spark...")
    
    # Basic purchase statistics
    user_stats = train_transactions_spark.groupBy("customer_id").agg(
        F.count("article_id").alias("n_purchases"),
        F.countDistinct("article_id").alias("n_unique_items"),
        F.avg("price").alias("avg_price"),
        F.stddev("price").alias("std_price"),
        F.min("price").alias("min_price"),
        F.max("price").alias("max_price"),
        F.min("t_dat").alias("first_purchase"),
        F.max("t_dat").alias("last_purchase")
    )
    
    # Days since first/last purchase
    user_stats = user_stats.withColumn(
        "days_since_first_purchase",
        F.datediff(F.lit(max_date), F.col("first_purchase")).cast(IntegerType())
    ).withColumn(
        "days_since_last_purchase",
        F.datediff(F.lit(max_date), F.col("last_purchase")).cast(IntegerType())
    ).withColumn(
        "purchase_span_days",
        F.datediff(F.col("last_purchase"), F.col("first_purchase")).cast(IntegerType())
    )
    
    # Drop datetime columns
    user_stats = user_stats.drop("first_purchase", "last_purchase")
    
    # Purchase frequency
    user_stats = user_stats.withColumn(
        "purchase_frequency",
        (F.col("n_purchases") / (F.col("purchase_span_days") + 1)).cast(FloatType())
    )
    
    # Item diversity (unique items / total purchases)
    user_stats = user_stats.withColumn(
        "exploration_ratio",
        (F.col("n_unique_items") / F.col("n_purchases")).cast(FloatType())
    )
    
    # Merge customer demographics if available
    if customers_spark is not None:
        demo_cols = ["customer_id", "age", "FN", "Active"]
        available_cols = [c for c in demo_cols if c in customers_spark.columns]
        if len(available_cols) > 1:
            user_stats = user_stats.join(
                customers_spark.select(available_cols),
                "customer_id",
                "left"
            )
    
    # Fill NaN values
    for col in user_stats.columns:
        if col != "customer_id":
            user_stats = user_stats.withColumn(
                col,
                F.coalesce(F.col(col), F.lit(0))
            )
    
    # Cast to appropriate types
    float_cols = ["avg_price", "std_price", "min_price", "max_price", 
                  "purchase_frequency", "exploration_ratio", "age", "FN", "Active"]
    int_cols = ["n_purchases", "n_unique_items", "days_since_first_purchase",
                "days_since_last_purchase", "purchase_span_days"]
    
    for col in float_cols:
        if col in user_stats.columns:
            user_stats = user_stats.withColumn(col, F.col(col).cast(FloatType()))
    
    for col in int_cols:
        if col in user_stats.columns:
            user_stats = user_stats.withColumn(col, F.col(col).cast(IntegerType()))
    
    user_stats = user_stats.cache()
    feature_count = len(user_stats.columns) - 1
    print(f"Created {feature_count} user features")
    print_memory()
    
    # Convert to pandas and save
    user_stats_pd = spark_to_pandas(user_stats.coalesce(10))
    user_stats_pd.to_parquet(config.OUTPUT_PATH / 'user_features.parquet', index=False)
    
    return user_stats_pd


def compute_item_features_spark(spark, train_transactions_spark, articles_spark, 
                                item_popularity_spark, max_date):
    """Compute item-level statistics features using PySpark"""
    print_section("PART 2: ITEM FEATURES (SPARK)")
    
    # Check if already computed
    if (config.OUTPUT_PATH / 'item_features.parquet').exists():
        print("Found existing item_features.parquet, loading...")
        item_stats = pd.read_parquet(config.OUTPUT_PATH / 'item_features.parquet')
        print(f"Loaded {len(item_stats.columns)-1} item features")
        return item_stats
    
    print("Computing item statistics features with Spark...")
    
    # Basic item statistics
    item_stats = train_transactions_spark.groupBy("article_id").agg(
        F.count("*").alias("sales_count"),
        F.countDistinct("customer_id").alias("unique_buyers"),
        F.avg("price").alias("avg_price"),
        F.stddev("price").alias("std_price"),
        F.min("t_dat").alias("first_sale"),
        F.max("t_dat").alias("last_sale")
    )
    
    # Days since first/last sale
    item_stats = item_stats.withColumn(
        "days_since_first_sale",
        F.datediff(F.lit(max_date), F.col("first_sale")).cast(IntegerType())
    ).withColumn(
        "days_since_last_sale",
        F.datediff(F.lit(max_date), F.col("last_sale")).cast(IntegerType())
    )
    
    item_stats = item_stats.drop("first_sale", "last_sale")
    
    # Recent sales
    recent_date = max_date - timedelta(days=config.RECENT_DAYS)
    recent_trans = train_transactions_spark.filter(F.col("t_dat") >= recent_date)
    recent_counts = recent_trans.groupBy("article_id").agg(
        F.count("*").alias("sales_recent")
    )
    
    item_stats = item_stats.join(recent_counts, "article_id", "left")
    item_stats = item_stats.withColumn(
        "sales_recent",
        F.coalesce(F.col("sales_recent"), F.lit(0)).cast(IntegerType())
    )
    
    # Sales trend
    mid_date = max_date - timedelta(days=config.RECENT_DAYS)
    old_cutoff = max_date - timedelta(days=config.MEDIUM_DAYS)
    
    recent_period = train_transactions_spark.filter(F.col("t_dat") >= mid_date)
    old_period = train_transactions_spark.filter(
        (F.col("t_dat") >= old_cutoff) & (F.col("t_dat") < mid_date)
    )
    
    item_recent = recent_period.groupBy("article_id").agg(
        F.count("*").alias("sales_recent_period")
    )
    item_old = old_period.groupBy("article_id").agg(
        F.count("*").alias("sales_old_period")
    )
    
    item_trend = item_recent.join(item_old, "article_id", "outer")
    item_trend = item_trend.withColumn(
        "sales_recent_period",
        F.coalesce(F.col("sales_recent_period"), F.lit(0))
    ).withColumn(
        "sales_old_period",
        F.coalesce(F.col("sales_old_period"), F.lit(0))
    ).withColumn(
        "sales_trend",
        ((F.col("sales_recent_period") - F.col("sales_old_period")) /
         (F.col("sales_old_period") + 1)).cast(FloatType())
    )
    
    item_stats = item_stats.join(
        item_trend.select("article_id", "sales_trend"),
        "article_id",
        "left"
    )
    item_stats = item_stats.withColumn(
        "sales_trend",
        F.coalesce(F.col("sales_trend"), F.lit(0.0)).cast(FloatType())
    )
    
    # Merge article metadata
    article_features = articles_spark.select(
        "article_id", "product_type_no", "graphical_appearance_no",
        "colour_group_code", "perceived_colour_value_id",
        "department_no", "index_group_no", "section_no", "garment_group_no"
    )
    
    item_stats = item_stats.join(article_features, "article_id", "left")
    
    # Add popularity scores if available
    if item_popularity_spark is not None:
        item_stats = item_stats.join(
            item_popularity_spark.select("article_id", "popularity_score"),
            "article_id",
            "left"
        )
        item_stats = item_stats.withColumn(
            "popularity_score",
            F.coalesce(F.col("popularity_score"), F.lit(0.0)).cast(FloatType())
        )
    
    # Fill NaN and cast types
    for col in item_stats.columns:
        if col != "article_id":
            item_stats = item_stats.withColumn(
                col,
                F.coalesce(F.col(col), F.lit(0))
            )
    
    item_stats = item_stats.cache()
    feature_count = len(item_stats.columns) - 1
    print(f"Created {feature_count} item features")
    print_memory()
    
    # Convert to pandas and save
    item_stats_pd = spark_to_pandas(item_stats.coalesce(10))
    item_stats_pd.to_parquet(config.OUTPUT_PATH / 'item_features.parquet', index=False)
    
    return item_stats_pd


def compute_interaction_features_spark(spark, candidates_spark, train_transactions_spark, 
                                       articles_spark, user_stats_spark, item_stats_spark):
    """Compute user-item interaction features using PySpark"""
    print_section("PART 3: USER-ITEM INTERACTION FEATURES (SPARK)")
    
    print("Computing interaction features with Spark...")
    
    # Build user purchase history
    user_purchases = train_transactions_spark.select(
        "customer_id", "article_id"
    ).distinct()
    
    # User category preferences
    trans_with_cat = train_transactions_spark.join(
        articles_spark.select("article_id", "product_type_no"),
        "article_id",
        "inner"
    )
    
    user_cat_counts = trans_with_cat.groupBy("customer_id", "product_type_no").count()
    
    # Get top category per user
    window = Window.partitionBy("customer_id").orderBy(F.col("count").desc())
    user_top_cat = user_cat_counts.withColumn(
        "rank",
        F.row_number().over(window)
    ).filter(F.col("rank") == 1).select(
        "customer_id",
        F.col("product_type_no").alias("top_category")
    )
    
    # User price stats
    user_price_stats = train_transactions_spark.groupBy("customer_id").agg(
        F.avg("price").alias("user_avg_price"),
        F.stddev("price").alias("user_std_price")
    )
    user_price_stats = user_price_stats.withColumn(
        "user_std_price",
        F.coalesce(F.col("user_std_price"), F.lit(0.01))
    )
    
    # Add interaction features to candidates
    all_features = candidates_spark
    
    # Has user purchased this item before?
    all_features = all_features.join(
        user_purchases.withColumn("has_purchased_item", F.lit(1)),
        ["customer_id", "article_id"],
        "left"
    ).withColumn(
        "has_purchased_item",
        F.coalesce(F.col("has_purchased_item"), F.lit(0)).cast(IntegerType())
    )
    
    # Get item metadata for category matching and price comparison
    item_meta = item_stats_spark.select("article_id", "product_type_no", "avg_price")
    all_features = all_features.join(item_meta, "article_id", "left")
    
    # Category match
    all_features = all_features.join(user_top_cat, "customer_id", "left")
    all_features = all_features.withColumn(
        "category_match",
        F.when(F.col("product_type_no") == F.col("top_category"), 1).otherwise(0).cast(IntegerType())
    ).drop("product_type_no", "top_category")
    
    # Price match features
    all_features = all_features.join(user_price_stats, "customer_id", "left")
    all_features = all_features.withColumn(
        "price_vs_user_avg",
        ((F.col("avg_price") - F.col("user_avg_price")) /
         (F.col("user_std_price") + 0.01)).cast(FloatType())
    ).withColumn(
        "is_cheaper_than_usual",
        F.when(F.col("avg_price") < F.col("user_avg_price"), 1).otherwise(0).cast(IntegerType())
    ).drop("user_avg_price", "user_std_price", "avg_price")
    
    # Add rank features for each score column
    score_cols = ['repurchase_score', 'popularity_score', 'copurchase_score',
                  'userknn_score', 'category_score']
    
    for score_col in score_cols:
        if score_col in all_features.columns:
            window = Window.partitionBy("customer_id").orderBy(F.col(score_col).desc())
            all_features = all_features.withColumn(
                f"{score_col}_rank",
                F.dense_rank().over(window).cast(IntegerType())
            )
    
    # Overall candidate rank based on n_strategies
    if "n_strategies" in all_features.columns:
        window = Window.partitionBy("customer_id").orderBy(F.col("n_strategies").desc())
        all_features = all_features.withColumn(
            "overall_rank",
            F.dense_rank().over(window).cast(IntegerType())
        )
    
    # Fill NaN values
    for col in all_features.columns:
        if col not in ["customer_id", "article_id"]:
            all_features = all_features.withColumn(
                col,
                F.coalesce(F.col(col), F.lit(0))
            )
    
    all_features = all_features.cache()
    print(f"Created interaction features for {all_features.count():,} candidates")
    
    return all_features


def merge_all_features_spark(spark, all_features_spark, user_stats_pd, item_stats_pd):
    """Merge user, item, and interaction features using PySpark"""
    print_section("MERGING ALL FEATURES (SPARK)")
    
    # Convert stats back to Spark
    user_stats_spark = pandas_to_spark(spark, user_stats_pd)
    item_stats_spark = pandas_to_spark(spark, item_stats_pd)
    
    # Merge user features
    all_features = all_features_spark.join(
        user_stats_spark,
        "customer_id",
        "left"
    )
    print("Merged user features")
    
    # Merge remaining item features (exclude those already in all_features)
    existing_cols = set(all_features.columns)
    item_cols_to_add = [c for c in item_stats_spark.columns if c not in existing_cols]
    item_cols_to_add.append("article_id")
    
    all_features = all_features.join(
        item_stats_spark.select(item_cols_to_add),
        "article_id",
        "left"
    )
    print("Merged item features")
    
    # Fill missing values
    for col in all_features.columns:
        if col not in ["customer_id", "article_id"]:
            all_features = all_features.withColumn(
                col,
                F.coalesce(F.col(col), F.lit(0))
            )
    
    all_features = all_features.cache()
    feature_count = len(all_features.columns) - 2
    row_count = all_features.count()
    
    print(f"\nTotal features: {feature_count} (excluding customer_id, article_id)")
    print(f"Total candidate-feature pairs: {row_count:,}")
    print_memory()
    
    return all_features


def assign_labels_spark(spark, all_features_spark, val_transactions_pd):
    """Assign labels for training and validation using PySpark"""
    print_section("ASSIGNING LABELS (SPARK)")
    
    # Create ground truth DataFrame
    val_purchases = val_transactions_pd[['customer_id', 'article_id']].drop_duplicates()
    val_purchases['label'] = 1
    val_purchases_spark = pandas_to_spark(spark, val_purchases)
    
    # Assign labels via left join
    all_features = all_features_spark.join(
        val_purchases_spark,
        ["customer_id", "article_id"],
        "left"
    ).withColumn(
        "label",
        F.coalesce(F.col("label"), F.lit(0)).cast(IntegerType())
    )
    
    all_features = all_features.cache()
    
    # Print statistics
    total = all_features.count()
    n_positive = all_features.filter(F.col("label") == 1).count()
    n_negative = total - n_positive
    
    print(f"Positive samples: {n_positive:,} ({100*n_positive/total:.2f}%)")
    print(f"Negative samples: {n_negative:,} ({100*n_negative/total:.2f}%)")
    
    return all_features


def create_train_val_split_spark(spark, all_features_spark, val_users_set):
    """Create train/validation split based on users using PySpark"""
    print_section("CREATING TRAIN/VAL SPLIT (SPARK)")
    
    # Create validation users DataFrame
    val_users_df = spark.createDataFrame(
        [(u,) for u in val_users_set],
        ["val_customer_id"]
    )
    
    # Mark user types
    all_features = all_features_spark.join(
        val_users_df,
        all_features_spark["customer_id"] == val_users_df["val_customer_id"],
        "left"
    ).withColumn(
        "user_type",
        F.when(F.col("val_customer_id").isNotNull(), "validation").otherwise("train")
    ).drop("val_customer_id")
    
    # Split data
    train_data = all_features.filter(F.col("user_type") == "train")
    val_data = all_features.filter(F.col("user_type") == "validation")
    
    train_count = train_data.count()
    val_count = val_data.count()
    train_users = train_data.select("customer_id").distinct().count()
    val_users = val_data.select("customer_id").distinct().count()
    
    print(f"Training data: {train_count:,} samples")
    print(f"Validation data: {val_count:,} samples")
    print(f"Training users: {train_users:,}")
    print(f"Validation users: {val_users:,}")
    
    # Balance training data
    train_pos = train_data.filter(F.col("label") == 1)
    train_neg = train_data.filter(F.col("label") == 0)
    
    n_pos = train_pos.count()
    n_neg = train_neg.count()
    n_neg_sample = int(n_pos * 1.5)
    
    if n_neg > n_neg_sample:
        sample_fraction = n_neg_sample / n_neg
        train_neg_sampled = train_neg.sample(
            fraction=sample_fraction,
            seed=config.RANDOM_STATE
        )
        train_data = train_pos.union(train_neg_sampled)
        
        # Shuffle
        train_data = train_data.orderBy(F.rand(seed=config.RANDOM_STATE))
        
        print(f"\nBalanced training data: {train_data.count():,} samples")
        print(f"  Positive: {n_pos:,}")
        print(f"  Negative (sampled): {train_neg_sampled.count():,}")
    
    # Convert to pandas and save
    train_data_pd = spark_to_pandas(train_data.coalesce(20))
    val_data_pd = spark_to_pandas(val_data.coalesce(10))
    
    train_data_pd.to_parquet(config.MODEL_PATH / 'train_data.parquet', index=False)
    val_data_pd.to_parquet(config.MODEL_PATH / 'val_data.parquet', index=False)
    
    print(f"\nSaved train_data.parquet and val_data.parquet")
    
    return train_data_pd, val_data_pd


def save_feature_metadata(all_features_columns):
    """Save feature metadata"""
    feature_names = [col for col in all_features_columns
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
    """Run the complete Stage 3 pipeline with PySpark"""
    print_section("STAGE 3: EXTRACTING FEATURES WITH PYSPARK")
    
    # Initialize Spark
    spark = get_spark_session(
        app_name="FashionRecommendation_Stage3",
        memory=config.SPARK_MEMORY
    )
    print(f"Spark session initialized")
    
    try:
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
        
        # Convert to Spark DataFrames
        train_transactions_spark = pandas_to_spark(spark, train_transactions)
        train_transactions_spark = train_transactions_spark.cache()
        
        customers_spark = pandas_to_spark(spark, customers)
        articles_spark = pandas_to_spark(spark, articles)
        candidates_spark = pandas_to_spark(spark, candidates)
        candidates_spark = candidates_spark.cache()
        
        item_popularity_spark = None
        if item_popularity is not None:
            item_popularity_spark = pandas_to_spark(spark, item_popularity)
        
        # Compute features
        user_stats_pd = compute_user_features_spark(
            spark, train_transactions_spark, customers_spark, max_date
        )
        force_garbage_collection()
        
        # Convert user_stats back to spark for item features
        user_stats_spark = pandas_to_spark(spark, user_stats_pd)
        
        item_stats_pd = compute_item_features_spark(
            spark, train_transactions_spark, articles_spark, item_popularity_spark, max_date
        )
        force_garbage_collection()
        
        # Convert item_stats back to spark for interaction features
        item_stats_spark = pandas_to_spark(spark, item_stats_pd)
        
        all_features_spark = compute_interaction_features_spark(
            spark, candidates_spark, train_transactions_spark, articles_spark,
            user_stats_spark, item_stats_spark
        )
        force_garbage_collection()
        
        all_features_spark = merge_all_features_spark(
            spark, all_features_spark, user_stats_pd, item_stats_pd
        )
        force_garbage_collection()
        
        # Assign labels
        all_features_spark = assign_labels_spark(spark, all_features_spark, val_transactions)
        
        # Save feature metadata
        save_feature_metadata(all_features_spark.columns)
        
        # Create train/val split
        train_data_pd, val_data_pd = create_train_val_split_spark(
            spark, all_features_spark, val_users
        )
        
        # Save all training features
        all_features_pd = spark_to_pandas(all_features_spark.coalesce(20))
        all_features_pd.to_parquet(config.OUTPUT_PATH / 'training_features.parquet', index=False)
        
        print_section("STAGE 3 COMPLETE")
        print(f"Total features: {len(all_features_pd.columns) - 4}")
        print(f"Training samples: {len(train_data_pd):,}")
        print(f"Validation samples: {len(val_data_pd):,}")
        
        force_garbage_collection()
        
        return train_data_pd, val_data_pd
        
    finally:
        # Stop Spark session
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    run_stage3()
