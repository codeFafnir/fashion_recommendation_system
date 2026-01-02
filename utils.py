"""
Utility functions for Fashion Recommendation System
Shared utilities used across all stages
"""

import pandas as pd
import numpy as np
import gc
import psutil
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage of a DataFrame by downcasting numeric types
    
    Args:
        df: pandas DataFrame
        verbose: whether to print memory reduction info
    
    Returns:
        DataFrame with optimized memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)')
    
    return df


def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()


def print_memory():
    """Print current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"  Memory: {mem_info.rss / 1024**3:.2f} GB")


def print_section(title):
    """Pretty print section headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def time_decay_score(days_ago, decay_rate=0.1):
    """
    Calculate time decay score
    
    Args:
        days_ago: array of days since event
        decay_rate: exponential decay rate
    
    Returns:
        Array of decay scores
    """
    return np.exp(-decay_rate * days_ago)


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(a, b, default=0):
    """
    Safe division that handles divide by zero
    
    Args:
        a: numerator
        b: denominator
        default: value to return when b is 0
    
    Returns:
        a/b or default if b is 0
    """
    if isinstance(b, (int, float)):
        return a / b if b != 0 else default
    else:
        result = np.where(b != 0, a / b, default)
        return result


def chunk_dataframe(df, chunk_size):
    """
    Split DataFrame into chunks
    
    Args:
        df: pandas DataFrame
        chunk_size: number of rows per chunk
    
    Yields:
        DataFrame chunks
    """
    n_chunks = max(1, len(df) // chunk_size)
    for chunk in np.array_split(df, n_chunks):
        yield chunk


def optimize_categorical_columns(df, columns):
    """
    Convert specified columns to categorical type
    
    Args:
        df: pandas DataFrame
        columns: list of column names to convert
    
    Returns:
        DataFrame with categorical columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def fill_missing_values(df, strategy='zero'):
    """
    Fill missing values in DataFrame
    
    Args:
        df: pandas DataFrame
        strategy: 'zero', 'mean', or 'median'
    
    Returns:
        DataFrame with filled values
    """
    if strategy == 'zero':
        return df.fillna(0)
    elif strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    else:
        return df.fillna(0)


def normalize_scores(scores):
    """
    Normalize scores to [0, 1] range
    
    Args:
        scores: array of scores
    
    Returns:
        Normalized scores
    """
    min_score = scores.min()
    max_score = scores.max()
    if max_score > min_score:
        return (scores - min_score) / (max_score - min_score)
    return scores

