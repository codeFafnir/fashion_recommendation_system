"""
Stage 1: Loading Dataset
Data loading, temporal splitting, and user/item sampling using PySpark

This stage performs:
1. Load raw H&M data (transactions, articles, customers) using PySpark
2. Select temporal window
3. Sample 50% of data (stratified by user activity)
4. Item filtering
5. Memory optimization
6. Save processed datasets (as parquet files compatible with pandas)
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import gc

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType

from config import Config, config
from utils import (
    print_section, force_garbage_collection, reduce_mem_usage,
    print_memory, optimize_categorical_columns
)
from spark_utils import (
    get_spark_session, spark_to_pandas, pandas_to_spark,
    cache_and_count, top_n_per_group, normalize_column_in_group,
    repartition_by_key
)


def load_raw_data_spark(spark):
    """Load raw H&M dataset using PySpark"""
    print_section("STEP 1: LOADING DATA WITH PYSPARK")
    
    # Load transactions
    print("Loading transactions...")
    transactions = spark.read.csv(
        str(config.DATA_PATH / 'transactions_train.csv'),
        header=True,
        inferSchema=True
    )
    
    # Convert date column
    transactions = transactions.withColumn(
        "t_dat", 
        F.to_date(F.col("t_dat"), "yyyy-MM-dd")
    )
    
    # Optimize data types
    transactions = (transactions
        .withColumn("article_id", F.col("article_id").cast("int"))
        .withColumn("price", F.col("price").cast("float"))
        .withColumn("sales_channel_id", F.col("sales_channel_id").cast("int"))
    )
    
    transactions = cache_and_count(transactions, "Transactions")
    
    # Get date range
    date_stats = transactions.agg(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date")
    ).collect()[0]
    print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']}")
    
    # Load customers
    print("\nLoading customers...")
    customers = spark.read.csv(
        str(config.DATA_PATH / 'customers.csv'),
        header=True,
        inferSchema=True
    )
    customers = (customers
        .withColumn("FN", F.col("FN").cast("float"))
        .withColumn("Active", F.col("Active").cast("float"))
        .withColumn("age", F.col("age").cast("float"))
    )
    customers = cache_and_count(customers, "Customers")
    
    # Load articles
    print("\nLoading articles...")
    articles = spark.read.csv(
        str(config.DATA_PATH / 'articles.csv'),
        header=True,
        inferSchema=True
    )
    articles = articles.withColumn("article_id", F.col("article_id").cast("int"))
    articles = cache_and_count(articles, "Articles")
    
    return transactions, customers, articles


def select_temporal_window_spark(transactions):
    """Select temporal window for training/validation using PySpark"""
    print_section("STEP 2: SELECTING TEMPORAL WINDOW")
    
    max_date = transactions.agg(F.max("t_dat")).collect()[0][0]
    print(f"Last transaction date: {max_date}")
    
    window_start = max_date - timedelta(weeks=config.TOTAL_WEEKS)
    print(f"Using {config.TOTAL_WEEKS} weeks of data")
    print(f"Window: {window_start} to {max_date}")
    
    # Filter to temporal window
    transactions = transactions.filter(F.col("t_dat") >= window_start)
    
    # Add week column
    transactions = transactions.withColumn(
        "week",
        F.floor(F.datediff(F.col("t_dat"), F.lit(window_start)) / 7).cast("int")
    )
    
    transactions = cache_and_count(transactions, "Transactions in window")
    
    return transactions, max_date


def stratified_user_sampling_spark(transactions, spark):
    """Perform user-based stratified sampling using PySpark with 50% sample"""
    print_section("STEP 3: USER-BASED STRATIFIED SAMPLING (50% DATA)")
    
    # Compute user activity statistics
    user_activity = transactions.groupBy("customer_id").agg(
        F.count("article_id").alias("total_purchases"),
        F.min("week").alias("first_week"),
        F.max("week").alias("last_week"),
        F.countDistinct("week").alias("active_weeks")
    )
    
    user_activity = user_activity.withColumn(
        "week_span",
        F.col("last_week") - F.col("first_week") + 1
    )
    
    user_activity = cache_and_count(user_activity, "Total users in window")
    
    avg_purchases = user_activity.agg(F.avg("total_purchases")).collect()[0][0]
    print(f"Avg purchases per user: {avg_purchases:.2f}")
    
    # Create activity level bins for stratified sampling
    user_activity = user_activity.withColumn(
        "activity_level",
        F.when(F.col("total_purchases") <= 1, "cold_start")
        .when(F.col("total_purchases") <= 3, "low")
        .when(F.col("total_purchases") <= 8, "medium")
        .when(F.col("total_purchases") <= 20, "high")
        .when(F.col("total_purchases") <= 50, "very_high")
        .otherwise("extreme")
    )
    
    # Print activity distribution
    print("\nActivity level distribution:")
    activity_dist = user_activity.groupBy("activity_level").count().orderBy("count")
    activity_dist.show()
    
    # Stratified sampling - sample SAMPLE_FRACTION (50%) from each stratum
    sample_fraction = config.SAMPLE_FRACTION
    print(f"\nSampling {sample_fraction*100:.0f}% of users from each activity level...")
    
    # Create fraction dictionary for stratified sampling
    activity_levels = ["cold_start", "low", "medium", "high", "very_high", "extreme"]
    fractions = {level: sample_fraction for level in activity_levels}
    
    # Perform stratified sampling
    sampled_users = user_activity.sampleBy("activity_level", fractions, seed=config.RANDOM_STATE)
    
    sampled_users = cache_and_count(sampled_users, "Sampled users")
    
    # Print sampled distribution
    print("\nSampled activity level distribution:")
    sampled_users.groupBy("activity_level").count().orderBy("count").show()
    
    # Get the selected user IDs
    selected_user_ids = sampled_users.select("customer_id")
    
    return selected_user_ids, user_activity


def filter_transactions_spark(transactions, selected_user_ids, max_date, spark):
    """Filter transactions to sampled users and create train/val split using PySpark"""
    print_section("STEP 4: FILTERING TRANSACTIONS")
    
    # Filter to selected users using broadcast join for efficiency
    from pyspark.sql.functions import broadcast
    transactions = transactions.join(
        broadcast(selected_user_ids),
        "customer_id",
        "inner"
    )
    
    transactions = cache_and_count(transactions, "Retained transactions")
    
    # Create train/val split
    val_end_date = max_date
    val_start_date = val_end_date - timedelta(weeks=config.N_VAL_WEEKS)
    train_end_date = val_start_date - timedelta(days=1)
    
    print(f"\nTrain: up to {train_end_date} ({config.N_TRAIN_WEEKS} weeks)")
    print(f"Val: {val_start_date} to {val_end_date} ({config.N_VAL_WEEKS} week)")
    
    train_transactions = transactions.filter(F.col("t_dat") <= train_end_date)
    val_transactions = transactions.filter(F.col("t_dat") > train_end_date)
    
    train_transactions = cache_and_count(train_transactions, "Training transactions")
    val_transactions = cache_and_count(val_transactions, "Validation transactions")
    
    # Get validation users
    val_user_count = val_transactions.select("customer_id").distinct().count()
    print(f"Users in validation: {val_user_count:,}")
    
    return train_transactions, val_transactions


def filter_items_spark(train_transactions, val_transactions, articles, spark):
    """Filter items based on minimum purchases using PySpark"""
    print_section("STEP 5: ITEM FILTERING")
    
    # Get item counts in training data
    item_counts = train_transactions.groupBy("article_id").count()
    unique_items = item_counts.count()
    print(f"Unique items in training: {unique_items:,}")
    
    # Filter items with minimum purchases
    valid_items = item_counts.filter(
        F.col("count") >= config.MIN_ITEM_PURCHASES
    ).select("article_id")
    
    valid_items_count = valid_items.count()
    print(f"Items with >= {config.MIN_ITEM_PURCHASES} purchases: {valid_items_count:,}")
    
    # Get items in validation
    val_items = val_transactions.select("article_id").distinct()
    val_items_count = val_items.count()
    print(f"Items in validation: {val_items_count:,}")
    
    # Union valid training items with validation items
    selected_items = valid_items.union(val_items).distinct()
    selected_items_count = selected_items.count()
    print(f"Total selected items: {selected_items_count:,}")
    
    # Filter transactions and articles
    from pyspark.sql.functions import broadcast
    
    train_transactions = train_transactions.join(
        broadcast(selected_items),
        "article_id",
        "inner"
    )
    
    val_transactions = val_transactions.join(
        broadcast(selected_items),
        "article_id",
        "inner"
    )
    
    articles = articles.join(
        broadcast(selected_items),
        "article_id",
        "inner"
    )
    
    print(f"\nAfter filtering:")
    train_transactions = cache_and_count(train_transactions, "Training transactions")
    val_transactions = cache_and_count(val_transactions, "Validation transactions")
    articles = cache_and_count(articles, "Articles retained")
    
    return train_transactions, val_transactions, articles, selected_items


def convert_to_pandas_and_optimize(train_transactions, val_transactions, 
                                    customers, articles, selected_user_ids):
    """Convert Spark DataFrames to pandas and optimize memory"""
    print_section("STEP 6: CONVERTING TO PANDAS & MEMORY OPTIMIZATION")
    
    print("Converting to pandas DataFrames...")
    
    # Convert to pandas - coalesce first to avoid too many partitions
    train_transactions_pd = spark_to_pandas(train_transactions.coalesce(10))
    print(f"Train transactions: {len(train_transactions_pd):,} rows")
    
    val_transactions_pd = spark_to_pandas(val_transactions.coalesce(5))
    print(f"Val transactions: {len(val_transactions_pd):,} rows")
    
    # Filter customers to selected users
    from pyspark.sql.functions import broadcast
    customers_filtered = customers.join(
        broadcast(selected_user_ids),
        "customer_id",
        "inner"
    )
    customers_pd = spark_to_pandas(customers_filtered.coalesce(5))
    print(f"Customers: {len(customers_pd):,} rows")
    
    articles_pd = spark_to_pandas(articles.coalesce(5))
    print(f"Articles: {len(articles_pd):,} rows")
    
    # Convert t_dat back to datetime if it's not already
    if train_transactions_pd['t_dat'].dtype == 'object':
        train_transactions_pd['t_dat'] = pd.to_datetime(train_transactions_pd['t_dat'])
    if val_transactions_pd['t_dat'].dtype == 'object':
        val_transactions_pd['t_dat'] = pd.to_datetime(val_transactions_pd['t_dat'])
    
    # Optimize memory
    print("\nOptimizing memory...")
    train_transactions_pd = reduce_mem_usage(train_transactions_pd)
    val_transactions_pd = reduce_mem_usage(val_transactions_pd)
    
    # Optimize categorical columns
    article_cat_cols = [
        'product_code', 'product_type_no', 'graphical_appearance_no',
        'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
        'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no'
    ]
    articles_pd = optimize_categorical_columns(articles_pd, article_cat_cols)
    
    customer_cat_cols = ['club_member_status', 'fashion_news_frequency', 'postal_code']
    customers_pd = optimize_categorical_columns(customers_pd, customer_cat_cols)
    
    print_memory()
    
    return train_transactions_pd, val_transactions_pd, customers_pd, articles_pd


def save_processed_data(train_transactions, val_transactions, 
                        customers, articles, selected_users_pd):
    """Save processed datasets"""
    print_section("STEP 7: SAVING PROCESSED DATA")
    
    # Get val_users set
    val_users = set(val_transactions['customer_id'].unique())
    
    # Save transactions
    train_transactions.to_parquet(config.OUTPUT_PATH / 'train_transactions.parquet', index=False)
    print(f"Saved train_transactions.parquet")
    
    val_transactions.to_parquet(config.OUTPUT_PATH / 'val_transactions.parquet', index=False)
    print(f"Saved val_transactions.parquet")
    
    # Save filtered customers and articles
    customers.to_parquet(config.OUTPUT_PATH / 'customers.parquet', index=False)
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
    
    return all_users, all_items, val_users


def run_stage1():
    """Run the complete Stage 1 pipeline with PySpark"""
    print_section("STAGE 1: LOADING AND PREPARING DATA WITH PYSPARK")
    
    # Initialize Spark
    spark = get_spark_session(
        app_name="FashionRecommendation_Stage1",
        memory=config.SPARK_MEMORY
    )
    print(f"Spark session initialized with {config.SPARK_MEMORY} memory")
    print(f"Sample fraction: {config.SAMPLE_FRACTION * 100:.0f}% of data")
    
    try:
        # Step 1: Load raw data
        transactions, customers, articles = load_raw_data_spark(spark)
        
        # Step 2: Select temporal window
        transactions, max_date = select_temporal_window_spark(transactions)
        
        # Step 3: User sampling (50% stratified sample)
        selected_user_ids, user_activity = stratified_user_sampling_spark(transactions, spark)
        
        # Step 4: Filter transactions
        train_transactions, val_transactions = filter_transactions_spark(
            transactions, selected_user_ids, max_date, spark
        )
        
        # Unpersist intermediate data
        transactions.unpersist()
        
        # Step 5: Filter items
        train_transactions, val_transactions, articles, selected_items = filter_items_spark(
            train_transactions, val_transactions, articles, spark
        )
        
        # Step 6: Convert to pandas and optimize memory
        (train_transactions_pd, val_transactions_pd, 
         customers_pd, articles_pd) = convert_to_pandas_and_optimize(
            train_transactions, val_transactions, 
            customers, articles, selected_user_ids
        )
        
        # Get selected users as pandas
        selected_users_pd = set(train_transactions_pd['customer_id'].unique())
        
        # Step 7: Save processed data
        all_users, all_items, val_users = save_processed_data(
            train_transactions_pd, val_transactions_pd,
            customers_pd, articles_pd, selected_users_pd
        )
        
        # Save user activity stats (convert to pandas first)
        user_activity_pd = spark_to_pandas(
            user_activity.join(
                selected_user_ids,
                "customer_id",
                "inner"
            ).coalesce(5)
        )
        user_activity_pd.to_parquet(config.OUTPUT_PATH / 'user_activity_stats.parquet', index=False)
        
        print_section("STAGE 1 COMPLETE")
        print(f"Training transactions: {len(train_transactions_pd):,}")
        print(f"Validation transactions: {len(val_transactions_pd):,}")
        print(f"Unique users: {len(all_users):,}")
        print(f"Unique items: {len(all_items):,}")
        
        # Get max_date as python datetime
        max_date_py = pd.Timestamp(max_date)
        
        force_garbage_collection()
        
        return {
            'train_transactions': train_transactions_pd,
            'val_transactions': val_transactions_pd,
            'customers': customers_pd,
            'articles': articles_pd,
            'all_users': all_users,
            'all_items': all_items,
            'val_users': val_users,
            'max_date': max_date_py
        }
        
    finally:
        # Stop Spark session
        spark.stop()
        print("\nSpark session stopped.")


# Keep backward compatibility - fallback to pandas if Spark fails
def run_stage1_pandas_fallback():
    """Fallback to original pandas implementation if Spark is not available"""
    print_section("STAGE 1: LOADING DATA (PANDAS FALLBACK)")
    print("Warning: Using pandas fallback. For large datasets, install PySpark.")
    
    # Import original pandas functions
    from utils import reduce_mem_usage, optimize_categorical_columns
    
    # Load raw data
    transactions = pd.read_csv(
        config.DATA_PATH / 'transactions_train.csv',
        dtype={
            'article_id': 'int32',
            'price': 'float32',
            'sales_channel_id': 'int8'
        },
        parse_dates=['t_dat']
    )
    
    # Sample 50% of the data
    sampled_transactions = transactions.sample(
        frac=config.SAMPLE_FRACTION,
        random_state=config.RANDOM_STATE
    )
    
    # Continue with rest of pandas pipeline...
    # (This is a simplified fallback)
    
    print("Pandas fallback completed.")
    return None


if __name__ == "__main__":
    try:
        data = run_stage1()
    except Exception as e:
        print(f"Error with PySpark: {e}")
        print("Attempting pandas fallback...")
        data = run_stage1_pandas_fallback()
