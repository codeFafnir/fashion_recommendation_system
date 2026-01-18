# H&M Personalized Fashion Recommendation System

A multi-stage recommendation system for fashion products using LightGBM and Neural Networks, with **PySpark support for large-scale data processing**.

## Project Structure

```
fashion_recommendation_system/
├── config.py               # Shared configuration for all stages
├── utils.py                # Utility functions (memory, GC, printing)
├── spark_utils.py          # PySpark utilities for distributed processing
├── metrics.py              # Evaluation metrics (MAP@K, NDCG, Precision, Recall)
├── main.py                 # Main entry point to run all stages
├── stage0_eda_baselines.py # Stage 0: EDA and Baseline models
├── stage1_load_data.py     # Stage 1: Data loading with PySpark (50% sample)
├── stage2_candidates.py    # Stage 2: Candidate generation with PySpark
├── stage3_features.py      # Stage 3: Feature extraction with PySpark
├── stage4a_lightgbm.py     # Stage 4A: LightGBM training
├── stage4b_neural.py       # Stage 4B: Neural tower training
├── stage7_evaluation.py    # Stage 7: Evaluation and metrics
├── text_features.py        # Text-based feature extraction utilities
└── notebook3.ipynb         # Original notebook (reference)
```

## Requirements

```bash
# Core dependencies
pip install pandas numpy pyspark scikit-learn lightgbm torch tqdm

# PySpark requires Java 8 or 11
# Verify with: java -version
```

## Usage

```bash
# Run all stages
python main.py

# Run specific stage(s)
python main.py --stage 1        # Load data only
python main.py --stage 1 2 3    # Run stages 1, 2, and 3
python main.py --stage 7        # Run evaluation only
```

## PySpark Configuration

The system now uses **PySpark** for data-intensive stages (1, 2, 3) to handle large datasets efficiently. Key configuration options in `config.py`:

```python
# PySpark settings
USE_SPARK = True          # Enable PySpark for large-scale processing
SAMPLE_FRACTION = 0.5     # Use 50% of data (adjustable)
SPARK_MEMORY = "8g"       # Driver/executor memory
SPARK_PARTITIONS = 200    # Number of shuffle partitions
```

**Benefits of PySpark:**
- Process larger datasets that don't fit in memory
- Distributed computing for faster candidate generation
- Scalable to cluster deployment
- Automatic memory management and optimization

## Stage Overview

| Stage | Name | Description |
|-------|------|-------------|
| 0 | EDA & Baselines | Exploratory data analysis and baseline models |
| 1 | Load Data | Data loading, temporal splits, user sampling |
| 2 | Candidates | Multi-strategy candidate generation |
| 3 | Features | Feature engineering (user, item, interaction) |
| 4A | LightGBM | Gradient boosting model training |
| 4B | Neural | Two-tower neural network training |
| 7 | Evaluation | Final metrics and submission generation |

---

# Stage-by-Stage Descriptions

---

## Stage 1: Data Loading & Preprocessing (PySpark)

Stage 1 established the foundation of our recommendation system by transforming raw H&M transaction data into a clean, structured, and memory-efficient dataset optimized for machine learning. **This stage now uses PySpark for distributed processing**, enabling efficient handling of the full dataset.

### Data Sources
The process loads three primary data sources using PySpark's distributed CSV reader:
- Transaction records containing 31.8 million purchase events spanning from September 2018 to September 2020
- A product catalog with 105,542 unique articles containing rich metadata (product types, colors, departments, sections, garment groups)
- Customer demographic data for 1.37 million users including age, gender indicators, and activity status

### Sampling Strategy (50% Data)
Instead of fixed user count sampling, **the system now uses stratified sampling with 50% of the data** (`SAMPLE_FRACTION = 0.5` in config). This approach:
- Samples 50% of users from each activity stratum (cold_start, low, medium, high, very_high, extreme)
- Maintains the natural distribution of user activity patterns
- Allows processing of larger datasets for better model generalization
- Is easily adjustable via the `SAMPLE_FRACTION` configuration parameter

### PySpark Processing Pipeline
1. **Load raw data** using Spark's distributed CSV reader with schema inference
2. **Temporal windowing**: Filter to 24-week window, split into 11 training + 1 validation weeks
3. **Stratified sampling**: Sample 50% of users from each activity level using `sampleBy()`
4. **Transaction filtering**: Use broadcast joins for efficient filtering
5. **Item filtering**: Remove products with insufficient sales (< 5 purchases)
6. **Conversion to pandas**: Convert final results to pandas DataFrames for compatibility with downstream stages
7. **Memory optimization**: Apply dtype optimization and save as parquet

The preprocessing pipeline implemented sophisticated temporal windowing, splitting the 24-week dataset into 11 training weeks and 1 validation week to ensure realistic time-based evaluation. Item filtering removed products with insufficient sales data (minimum 5 purchases), ensuring only items with meaningful interaction history were included. 

All outputs are saved in parquet format, maintaining compatibility with subsequent stages. The Spark session is automatically stopped after processing to free resources.

---

## Stage 2: Candidate Generation with PySpark

Stage 2 implemented a comprehensive multi-strategy candidate generation system using **PySpark for distributed computation**, producing a diverse pool of potential recommendations for each user. This serves as the recall layer that narrows down from millions of items to hundreds of relevant candidates.

### PySpark-Accelerated Strategies

**1. Repurchase Strategy (Spark)**
- Computes user-item purchase history with time decay weighting using Spark window functions
- Normalizes scores per user and selects top N candidates per user efficiently

**2. Popularity-Based Strategy (Spark)**
- Aggregates sales volume and unique buyer counts using distributed groupBy
- Cross-joins with users using Spark's efficient broadcast mechanism for small DataFrames

**3. Co-Purchase Strategy (Spark)**
- Builds co-purchase matrix from basket analysis using self-joins
- Computes item-to-item similarity scores with distributed processing
- Uses left_anti joins to filter already-purchased items efficiently

**4. User-KNN Collaborative Filtering (Spark)**
- Finds similar users based on item overlap using distributed joins
- Aggregates recommendations from similar users' purchases
- Scales to millions of users with Spark's distributed computation

**5. Category-Based Strategy (Spark)**
- Computes user category preferences using groupBy and window functions
- Joins with category-level popular items efficiently

### Candidate Merging
All strategies are merged using Spark's distributed outer joins, tracking which strategies recommended each candidate. The merging process computes strategy scores (repurchase_score, popularity_score, copurchase_score, userknn_score, category_score) and counts how many strategies recommended each item. The merged candidate pool is converted to pandas and saved as `candidates.parquet` for use in subsequent feature engineering stages.

---

## Stage 3: Feature Engineering with PySpark

Stage 3 constructed a comprehensive feature set using **PySpark for distributed computation**, capturing user characteristics, item properties, user-item interactions, visual attributes, and semantic relationships.

### PySpark Feature Computation

**User-Level Features (Spark)**
Computed using Spark groupBy and aggregation:
- Activity metrics: purchase count, unique items, exploration ratio
- Temporal features: days since first/last purchase, purchase frequency
- Price preferences: avg/std/min/max price
- Demographics from customer data merge

**Item-Level Features (Spark)**
Computed with distributed aggregations:
- Sales metrics: total sales, unique buyers
- Temporal: days since first/last sale
- Recent performance: sales in last 7/21 days, sales trend
- Category metadata from articles

**User-Item Interaction Features (Spark)**
Computed with distributed joins and window functions:
- Purchase history indicators via left joins
- Category match using user preference joins
- Price compatibility features
- Rank features using window functions (dense_rank)

### Label Assignment & Train/Val Split
- Labels assigned via left join with validation purchases
- Train/val split based on user membership in validation set
- Balanced sampling of negatives (1.5:1 ratio) using Spark's sample()
- Final data converted to pandas and saved as parquet

**Image Features (512 dimensions)** are extracted separately using FashionCLIP (handled in text_features.py, not requiring Spark).

**Text Features** complement visual features by extracting semantic information from article descriptions.

All features are processed with careful attention to data quality: missing values filled with 0, types optimized for memory. The final feature matrix is saved as `training_features.parquet` for model training. Spark session is stopped after processing to free resources.

---

## Stage 4: Model Training

Stage 4 implemented a dual-model training approach combining gradient boosting (LightGBM) and deep learning (Neural Towers) to create a robust ensemble recommendation system optimized for the MAP@12 metric.

### Part A: LightGBM Reranking Models

We trained four distinct LightGBM model variants, each optimized for different aspects of the recommendation task. The **LightGBM Classifier** used binary classification with `binary_logloss` objective, treating recommendation as a purchase prediction problem with fast training and inference. The **LightGBM Ranker (LambdaRank)** employed learning-to-rank with the `lambdarank` objective, directly optimizing for ranking quality using group-based training where each group represented a user's candidate items, making it ideal for MAP@12 optimization. The **LightGBM Ranker (XENDCG)** used an alternative ranking objective (`rank_xendcg`) providing model diversity through different optimization approaches. The **LightGBM Classifier (Deep)** featured increased model capacity with deeper trees (num_leaves=127, max_depth=15) to capture complex feature interactions.

All models were trained with early stopping (50 rounds patience), time-based cross-validation, and feature importance tracking. Categorical features were properly encoded as integer codes, and group information was provided for ranking models to ensure proper per-user ranking. Training was optimized for M4 MacBook Air with appropriate batch sizes and memory management. After training, we created ensemble models: a weighted ensemble with configurable weights for each model, and an average ensemble that simply averaged predictions. Model checkpoints, predictions, and metadata were saved for evaluation and submission generation.

### Part B: Two-Tower Neural Network Training

We implemented a two-tower neural network architecture that processes user and item features through specialized towers before fusion, enabling efficient retrieval and clear separation between user and item representations.

**User Tower Architecture:**
The User Tower takes user-level features (demographics, purchase history, behavioral patterns) and processes them through a three-layer MLP (user_dim → 256 → 128 → 128) with batch normalization, ReLU activation, and dropout (0.3) to produce a 128-dimensional user embedding capturing user preferences and behavior patterns.

**Item Tower Architecture:**
The Item Tower processes all item-related features, including product attributes, popularity metrics, and image embeddings (FashionCLIP 512-dimensional vectors), through a similar three-layer MLP (item_dim → 256 → 128 → 128) producing a 128-dimensional item embedding. By combining item metadata with visual features in a single tower, the model learns unified item representations that capture both semantic and visual characteristics.

**Fusion Layer:**
The fusion layer concatenates user and item embeddings (128 + 128 = 256 dimensions) and processes them through a deep MLP (256 → 256 → 128 → 64 → 1) with batch normalization, ReLU activations, dropout regularization, and a final sigmoid activation for binary classification.

**Training Configuration:**
The model was trained using weighted Binary Cross-Entropy loss (to handle class imbalance), AdamW optimizer with weight decay (2e-4), learning rate of 3e-4 with ReduceLROnPlateau scheduling, batch size of 4096 optimized for M4 MacBook Air, and early stopping with 5 epochs patience based on validation MAP@12. Training utilized MPS (Metal Performance Shaders) acceleration for Apple Silicon, significantly speeding up training compared to CPU-only execution.

**Advantages of Two-Tower Architecture:**
- **Efficient Retrieval**: User and item embeddings can be precomputed separately, enabling fast approximate nearest neighbor search at inference time
- **Scalability**: New items can be added by computing only their item embeddings without retraining
- **Interpretability**: Clear separation between user preferences and item characteristics

### Label Creation & Dataset Preparation

Before training, we implemented sophisticated label creation logic that correctly handles two distinct user types. For **validation users** (users who appear in the validation ground truth), labels were created from their future purchases during the validation period, enabling proper evaluation. For **training-only users** (users not in validation set), labels were created from their past purchases during the training period, providing additional training data. This vectorized approach used `np.where` to efficiently assign labels based on user type, ensuring correct temporal evaluation.

The labeled dataset was split using stratified sampling (85% train, 15% validation) preserving label distribution, then downsampled for local training efficiency. The downsampling strategy kept all positive samples while sampling negatives to achieve a 40% positive, 60% negative ratio, resulting in 685,494 training samples and 120,970 validation samples—balanced for training while manageable for M4 MacBook Air memory constraints.

### Evaluation & Ensemble

Both LightGBM and Neural Tower models were evaluated using comprehensive metrics including MAP@12 (competition metric), Precision@K, Recall@K, and NDCG@K for K values 1, 3, 5, 10, and 12. Model predictions were ranked by MAP@12 performance, with detailed comparison reports saved to CSV. The final ensemble combined the best-performing models (typically Neural Tower + LightGBM Ranker variants) using weighted averaging, with configurable weights optimized for validation performance. The ensemble typically outperformed individual models, and final predictions were formatted as top-12 recommendations per user in Kaggle submission format.

---

## Summary

This multi-stage pipeline transformed raw H&M transaction data into a production-ready recommendation system through:

1. **PySpark-accelerated data preprocessing** - Efficiently process 50% of the full dataset with distributed computing
2. **Multi-strategy candidate generation** - Five complementary recall strategies implemented with Spark
3. **Comprehensive feature engineering** - User, item, and interaction features computed distributedly
4. **Ensemble model training** - LightGBM + Neural Network models optimized for MAP@12

### Key Features
- **Scalable**: Uses PySpark for Stages 1-3, enabling processing of larger datasets
- **Configurable**: Adjust `SAMPLE_FRACTION` in config.py to control data size
- **Compatible**: Outputs remain as pandas DataFrames/parquet, compatible with downstream stages
- **Memory-efficient**: Aggressive dtype optimization and Spark memory management
- **Apple Silicon optimized**: MPS acceleration for neural network training

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PySpark Processing Layer                     │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Stage 1       │    Stage 2      │    Stage 3                  │
│   Load Data     │   Candidates    │   Features                  │
│   (50% sample)  │   (5 strategies)│   (User/Item/Interaction)   │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Parquet Files (Pandas compatible)          │
└─────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌─────────────────────┐                 ┌─────────────────────────┐
│   Stage 4A          │                 │   Stage 4B              │
│   LightGBM          │                 │   Neural Towers         │
│   (pandas/sklearn)  │                 │   (PyTorch/MPS)         │
└─────────┬───────────┘                 └────────────┬────────────┘
          │                                          │
          └─────────────────┬────────────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Stage 7           │
                 │   Evaluation        │
                 │   (MAP@12)          │
                 └─────────────────────┘
```

