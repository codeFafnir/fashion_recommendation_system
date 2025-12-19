# Stage-by-Stage Project Descriptions
## H&M Personalized Fashion Recommendation System

---

## Stage 1: Data Loading & Preprocessing

Stage 1 established the foundation of our recommendation system by transforming raw H&M transaction data into a clean, structured, and memory-efficient dataset optimized for machine learning. The process began by loading three primary data sources: transaction records containing 31.8 million purchase events spanning from September 2018 to September 2020, a product catalog with 105,542 unique articles containing rich metadata (product types, colors, departments, sections, garment groups), and customer demographic data for 1.37 million users including age, gender indicators, and activity status. 

The preprocessing pipeline implemented sophisticated temporal windowing, splitting the 24-week dataset into 11 training weeks and 1 validation week to ensure realistic time-based evaluation. To manage computational constraints while maintaining data quality, we employed stratified user sampling targeting 50,000 users, with activity-based stratification ensuring representation across different purchase frequency levels (low, medium, high, very high, and extreme activity bins). A critical aspect of our approach was cold-start handling, where 15% of sampled users were intentionally selected as cold-start users (those with ≤1 purchase) to simulate real-world scenarios where new or infrequent customers require recommendations. 

Item filtering removed products with insufficient sales data (minimum 5 purchases), ensuring only items with meaningful interaction history were included. Throughout this process, aggressive memory optimization was applied using efficient data types: integer columns were downcast to int8, int16, or int32 based on value ranges, float columns to float32 where precision allowed, and categorical columns converted to pandas category dtype. This optimization reduced memory usage from several gigabytes to approximately 0.35 GB for the transaction dataset, making it feasible to process on a MacBook Air M4 with unified memory architecture. The final preprocessed dataset contained 47,543 training users, 15,932 unique articles, and 4,943 validation users, with all data saved in optimized parquet format for efficient subsequent loading and processing.

---

## Stage 2: Candidate Generation (Recall Strategies)

Stage 2 implemented a comprehensive multi-strategy candidate generation system that produced a diverse pool of potential recommendations for each user, serving as the recall layer that narrows down from millions of items to hundreds of relevant candidates. We developed and integrated six complementary recall strategies, each capturing different aspects of user-item relationships and recommendation signals.

The **Repurchase Strategy** identified items users had previously purchased, scoring them based on recency (more recent purchases weighted higher) and frequency (items bought multiple times scored higher), effectively capturing user loyalty patterns and repeat purchase behavior. The **Popularity-Based Strategy** recommended globally trending items by aggregating sales volume, unique buyer counts, and recent purchase trends, ensuring coverage of items currently in demand across the entire user base. The **Co-Purchase Strategy** (Item-to-Item Collaborative Filtering) discovered complementary item relationships by computing Jaccard similarity between item purchase patterns—items frequently bought together (e.g., a shirt and matching pants) received higher co-purchase scores, enabling cross-selling recommendations.

The **User-KNN Collaborative Filtering** strategy found similar users based on purchase history overlap, then recommended items purchased by these similar users, effectively leveraging collaborative filtering principles to handle diverse user preference patterns. The **Category-Based Strategy** combined user category preferences (derived from their purchase history) with item popularity within those categories, ensuring recommendations aligned with user interests while maintaining category diversity. Finally, the **Text Similarity Strategy** utilized semantic embeddings generated from article descriptions (product type names, color names, department names, etc.), computing cosine similarity between user preference embeddings and item embeddings to recommend semantically similar items (e.g., "summer dress" → "beachwear").

All six strategies were merged using outer joins, preserving candidates from all strategies while tracking which strategies recommended each candidate through binary indicator columns. The merging process also computed strategy scores (repurchase_score, popularity_score, copurchase_score, userknn_score, category_score, text_similarity_score) and created a composite metric counting how many strategies recommended each item. This multi-strategy approach generated approximately 100-200 candidate items per user, providing comprehensive coverage while maintaining computational efficiency. The merged candidate pool was saved as `all_candidates_merged.parquet` for use in subsequent feature engineering stages.

---

## Stage 3: Feature Engineering

Stage 3 constructed a comprehensive feature set totaling 584 features that captured user characteristics, item properties, user-item interactions, visual attributes, and semantic relationships. This extensive feature engineering process transformed raw data and candidate pairs into rich numerical representations suitable for machine learning models.

**User-Level Features (43 features)** were extracted from customer purchase history and demographics, including activity metrics (total purchase count, unique items purchased, exploration rate measuring diversity of purchases), temporal features (days since first purchase, days since last purchase, average days between purchases, purchase frequency), demographic attributes (age, gender indicator FN, Active status), behavioral patterns (average purchase value, price range preferences, standard deviation of purchase values), and category preferences (top 3 preferred categories, category diversity index, category purchase counts). 

**Item-Level Features (29 features)** captured product characteristics and performance metrics, including categorical attributes (product type, color group, department, index group, section, garment group encoded as frequency and count features), popularity metrics (total sales count, unique buyer count, sales frequency, recent sales trends), temporal item features (days since first sale, days since last sale, sales velocity), and recent performance indicators (sales in last week, recent unique buyers, sales trend direction).

**User-Item Interaction Features** encoded the relationship between specific users and items, including purchase history indicators (has user bought this item before, purchase count for this item, recency of last purchase), category alignment (user's preference score for the item's category, category purchase frequency), price compatibility (user's average price vs item price, price difference, price compatibility score), and temporal interaction features (days since user last interacted with this item category).

**Image Features (512 dimensions)** were extracted using FashionCLIP, a pre-trained vision-language model optimized for fashion. Each product image was processed through the FashionCLIP encoder to generate 512-dimensional embeddings capturing visual style, color patterns, design elements, and aesthetic characteristics. These embeddings enable visual similarity matching, allowing the model to recommend items that look similar even if they differ in metadata.

**Text Features** complemented visual features by extracting semantic information from article descriptions. Using TF-IDF vectorization and dimensionality reduction, we created article embeddings and user preference embeddings based on their purchase history, enabling semantic similarity matching (e.g., "summer dress" → "beachwear"). Additional text-based features included category encoding (frequency and count encodings for categorical text attributes) and semantic diversity metrics (measuring the variety of semantic styles in a user's purchase history).

All features were processed with careful attention to data quality: missing values were imputed (0 for numerical features, mode for categorical), categorical features were encoded as numeric codes for LightGBM compatibility, and features were standardized using StandardScaler for neural network training. The final feature matrix was saved as `training_features.parquet`, containing all 584 features merged with candidate user-item pairs, ready for model training.

---

## Stage 4: Model Training

Stage 4 implemented a dual-model training approach combining gradient boosting (LightGBM) and deep learning (Neural Towers) to create a robust ensemble recommendation system optimized for the MAP@12 metric.

### Part A: LightGBM Reranking Models

We trained four distinct LightGBM model variants, each optimized for different aspects of the recommendation task. The **LightGBM Classifier** used binary classification with `binary_logloss` objective, treating recommendation as a purchase prediction problem with fast training and inference. The **LightGBM Ranker (LambdaRank)** employed learning-to-rank with the `lambdarank` objective, directly optimizing for ranking quality using group-based training where each group represented a user's candidate items, making it ideal for MAP@12 optimization. The **LightGBM Ranker (XENDCG)** used an alternative ranking objective (`rank_xendcg`) providing model diversity through different optimization approaches. The **LightGBM Classifier (Deep)** featured increased model capacity with deeper trees (num_leaves=127, max_depth=15) to capture complex feature interactions.

All models were trained with early stopping (50 rounds patience), time-based cross-validation, and feature importance tracking. Categorical features were properly encoded as integer codes, and group information was provided for ranking models to ensure proper per-user ranking. Training was optimized for M4 MacBook Air with appropriate batch sizes and memory management. After training, we created ensemble models: a weighted ensemble with configurable weights for each model, and an average ensemble that simply averaged predictions. Model checkpoints, predictions, and metadata were saved for evaluation and submission generation.

### Part B: Neural Towers Training

We implemented a sophisticated three-tower neural network architecture that processes different feature types through specialized towers before fusion. The **User Tower** takes 43 user features and processes them through a two-layer MLP (43 → 256 → 128) with batch normalization, ReLU activation, and dropout (0.3) to produce a 128-dimensional user embedding capturing user preferences and behavior patterns. The **Item Tower** processes 29 item features through a similar architecture (29 → 128 → 64) producing a 64-dimensional item embedding representing product characteristics. The **Image Tower** handles 512-dimensional FashionCLIP embeddings through a two-layer MLP (512 → 256 → 128) producing a 128-dimensional visual embedding capturing aesthetic and style information.

The **Fusion Layer** concatenates all three embeddings (128 + 64 + 128 = 320 dimensions) and processes them through a deep MLP (320 → 256 → 128 → 64 → 1) with batch normalization, ReLU activations, dropout regularization, and a final sigmoid activation for binary classification. The complete model contains 346,689 trainable parameters and was trained using Binary Cross-Entropy loss, AdamW optimizer with weight decay (1e-5), learning rate of 1e-3 with ReduceLROnPlateau scheduling, batch size of 2048 optimized for M4 MacBook Air, and early stopping with 5 epochs patience based on validation MAP@12. Training utilized MPS (Metal Performance Shaders) acceleration for Apple Silicon, significantly speeding up training compared to CPU-only execution.

### Label Creation & Dataset Preparation

Before training, we implemented sophisticated label creation logic that correctly handles two distinct user types. For **validation users** (users who appear in the validation ground truth), labels were created from their future purchases during the validation period, enabling proper evaluation. For **training-only users** (users not in validation set), labels were created from their past purchases during the training period, providing additional training data. This vectorized approach used `np.where` to efficiently assign labels based on user type, ensuring correct temporal evaluation.

The labeled dataset was split using stratified sampling (85% train, 15% validation) preserving label distribution, then downsampled for local training efficiency. The downsampling strategy kept all positive samples while sampling negatives to achieve a 40% positive, 60% negative ratio, resulting in 685,494 training samples and 120,970 validation samples—balanced for training while manageable for M4 MacBook Air memory constraints.

### Evaluation & Ensemble

Both LightGBM and Neural Tower models were evaluated using comprehensive metrics including MAP@12 (competition metric), Precision@K, Recall@K, and NDCG@K for K values 1, 3, 5, 10, and 12. Model predictions were ranked by MAP@12 performance, with detailed comparison reports saved to CSV. The final ensemble combined the best-performing models (typically Neural Tower + LightGBM Ranker variants) using weighted averaging, with configurable weights optimized for validation performance. The ensemble typically outperformed individual models, and final predictions were formatted as top-12 recommendations per user in Kaggle submission format.

---

## Summary

This four-stage pipeline transformed raw H&M transaction data into a production-ready recommendation system through systematic data preprocessing, multi-strategy candidate generation, comprehensive feature engineering, and ensemble model training. Each stage built upon the previous, with careful attention to memory efficiency, computational optimization for Apple Silicon, and evaluation rigor to ensure competitive performance on the MAP@12 metric.

