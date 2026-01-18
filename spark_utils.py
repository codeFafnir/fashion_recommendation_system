"""
Spark Utilities for Fashion Recommendation System
PySpark initialization and utility functions for large-scale data processing
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, DateType, LongType, DoubleType
)
from pyspark.sql.window import Window
import pandas as pd
import numpy as np


def get_spark_session(app_name="FashionRecommendation", memory="8g"):
    """
    Initialize or get existing SparkSession with optimized settings
    
    Args:
        app_name: Name of the Spark application
        memory: Driver and executor memory allocation
    
    Returns:
        SparkSession instance
    """
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "100")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def stop_spark_session():
    """Stop the current SparkSession"""
    spark = SparkSession.builder.getOrCreate()
    spark.stop()


def spark_to_pandas(spark_df, coalesce_partitions=None):
    """
    Convert Spark DataFrame to pandas DataFrame efficiently
    
    Args:
        spark_df: PySpark DataFrame
        coalesce_partitions: Optional number of partitions to coalesce before conversion
    
    Returns:
        pandas DataFrame
    """
    if coalesce_partitions:
        spark_df = spark_df.coalesce(coalesce_partitions)
    return spark_df.toPandas()


def pandas_to_spark(spark, pandas_df, schema=None):
    """
    Convert pandas DataFrame to Spark DataFrame
    
    Args:
        spark: SparkSession instance
        pandas_df: pandas DataFrame
        schema: Optional schema for the Spark DataFrame
    
    Returns:
        PySpark DataFrame
    """
    if schema:
        return spark.createDataFrame(pandas_df, schema=schema)
    return spark.createDataFrame(pandas_df)


def get_transactions_schema():
    """Get schema for transactions data"""
    return StructType([
        StructField("t_dat", DateType(), True),
        StructField("customer_id", StringType(), True),
        StructField("article_id", IntegerType(), True),
        StructField("price", FloatType(), True),
        StructField("sales_channel_id", IntegerType(), True)
    ])


def get_customers_schema():
    """Get schema for customers data"""
    return StructType([
        StructField("customer_id", StringType(), True),
        StructField("FN", FloatType(), True),
        StructField("Active", FloatType(), True),
        StructField("club_member_status", StringType(), True),
        StructField("fashion_news_frequency", StringType(), True),
        StructField("age", FloatType(), True),
        StructField("postal_code", StringType(), True)
    ])


def get_articles_schema():
    """Get schema for articles data"""
    return StructType([
        StructField("article_id", IntegerType(), True),
        StructField("product_code", IntegerType(), True),
        StructField("prod_name", StringType(), True),
        StructField("product_type_no", IntegerType(), True),
        StructField("product_type_name", StringType(), True),
        StructField("product_group_name", StringType(), True),
        StructField("graphical_appearance_no", IntegerType(), True),
        StructField("graphical_appearance_name", StringType(), True),
        StructField("colour_group_code", IntegerType(), True),
        StructField("colour_group_name", StringType(), True),
        StructField("perceived_colour_value_id", IntegerType(), True),
        StructField("perceived_colour_value_name", StringType(), True),
        StructField("perceived_colour_master_id", IntegerType(), True),
        StructField("perceived_colour_master_name", StringType(), True),
        StructField("department_no", IntegerType(), True),
        StructField("department_name", StringType(), True),
        StructField("index_code", StringType(), True),
        StructField("index_name", StringType(), True),
        StructField("index_group_no", IntegerType(), True),
        StructField("index_group_name", StringType(), True),
        StructField("section_no", IntegerType(), True),
        StructField("section_name", StringType(), True),
        StructField("garment_group_no", IntegerType(), True),
        StructField("garment_group_name", StringType(), True),
        StructField("detail_desc", StringType(), True)
    ])


def time_decay_udf(decay_rate=0.1):
    """
    Create a UDF for time decay score calculation
    
    Args:
        decay_rate: Exponential decay rate
    
    Returns:
        Spark Column expression for decay score
    """
    def calc_decay(days_ago):
        return F.exp(-decay_rate * days_ago)
    return calc_decay


def repartition_by_key(df, key_column, num_partitions=None):
    """
    Repartition DataFrame by a key column for optimized joins
    
    Args:
        df: PySpark DataFrame
        key_column: Column name to partition by
        num_partitions: Number of partitions (optional)
    
    Returns:
        Repartitioned DataFrame
    """
    if num_partitions:
        return df.repartition(num_partitions, key_column)
    return df.repartition(key_column)


def cache_and_count(df, name="DataFrame"):
    """
    Cache a DataFrame and print its count
    
    Args:
        df: PySpark DataFrame
        name: Name for logging
    
    Returns:
        Cached DataFrame
    """
    df = df.cache()
    count = df.count()
    print(f"{name}: {count:,} rows")
    return df


def stratified_sample(df, strata_col, fractions, seed=42):
    """
    Perform stratified sampling on a DataFrame
    
    Args:
        df: PySpark DataFrame
        strata_col: Column to stratify by
        fractions: Dictionary of strata values to sample fractions
        seed: Random seed
    
    Returns:
        Sampled DataFrame
    """
    return df.sampleBy(strata_col, fractions, seed)


def broadcast_join(df_large, df_small, on_column, how="inner"):
    """
    Perform broadcast join for efficient joining with small tables
    
    Args:
        df_large: Large DataFrame
        df_small: Small DataFrame (will be broadcast)
        on_column: Column(s) to join on
        how: Join type
    
    Returns:
        Joined DataFrame
    """
    from pyspark.sql.functions import broadcast
    return df_large.join(broadcast(df_small), on_column, how)


def add_row_number(df, partition_by=None, order_by=None, col_name="row_num"):
    """
    Add row number column to DataFrame
    
    Args:
        df: PySpark DataFrame
        partition_by: Column(s) to partition by
        order_by: Column(s) to order by
        col_name: Name for the row number column
    
    Returns:
        DataFrame with row number column
    """
    if partition_by is None:
        window = Window.orderBy(order_by)
    else:
        window = Window.partitionBy(partition_by).orderBy(order_by)
    
    return df.withColumn(col_name, F.row_number().over(window))


def top_n_per_group(df, group_by, order_by, n, ascending=False):
    """
    Get top N rows per group
    
    Args:
        df: PySpark DataFrame
        group_by: Column(s) to group by
        order_by: Column(s) to order by
        n: Number of rows to keep per group
        ascending: Sort order
    
    Returns:
        DataFrame with top N rows per group
    """
    if ascending:
        window = Window.partitionBy(group_by).orderBy(F.col(order_by).asc())
    else:
        window = Window.partitionBy(group_by).orderBy(F.col(order_by).desc())
    
    return (df
        .withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") <= n)
        .drop("_rank")
    )


def normalize_column_in_group(df, group_by, col_name, new_col_name=None):
    """
    Normalize a column within groups (min-max scaling per group)
    
    Args:
        df: PySpark DataFrame
        group_by: Column to group by
        col_name: Column to normalize
        new_col_name: Name for normalized column (defaults to original)
    
    Returns:
        DataFrame with normalized column
    """
    if new_col_name is None:
        new_col_name = col_name
    
    window = Window.partitionBy(group_by)
    
    return (df
        .withColumn("_max", F.max(col_name).over(window))
        .withColumn(new_col_name, F.col(col_name) / F.col("_max"))
        .drop("_max")
    )


def save_to_parquet(df, path, mode="overwrite", partition_by=None):
    """
    Save DataFrame to parquet with optimal settings
    
    Args:
        df: PySpark DataFrame
        path: Output path
        mode: Save mode ('overwrite', 'append', etc.)
        partition_by: Column(s) to partition by (optional)
    """
    writer = df.write.mode(mode)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.parquet(str(path))


def load_parquet(spark, path):
    """
    Load parquet file(s) into DataFrame
    
    Args:
        spark: SparkSession
        path: Path to parquet file or directory
    
    Returns:
        PySpark DataFrame
    """
    return spark.read.parquet(str(path))


def print_spark_memory_stats():
    """Print Spark memory statistics"""
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    
    # Get memory info from executors
    status = sc.statusTracker()
    executor_ids = status.getExecutorInfos()
    
    print("\nSpark Memory Stats:")
    print(f"  Active executors: {len(executor_ids)}")
    print(f"  Default parallelism: {sc.defaultParallelism}")

