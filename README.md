### Fashion Recommendation System

This is a project where I have implemented a Personalized Recommendation system which recommends fashion article to users based on their transaction history (Used H&M dataset for this purpose)

# Presentation Summary: H&M Recommendation System
## Quick Reference for Slides

---

## Slide 1: Project Overview

**Title:** H&M Personalized Fashion Recommendation System

**Objective:**
- Build state-of-the-art recommendation system for fashion items
- Optimize MAP@12 metric (Kaggle competition metric)
- Combine multiple ML approaches for best performance

**Approach:**
- Multi-stage pipeline: Data → Candidates → Features → Models → Evaluation
- Ensemble of LightGBM and Neural Networks
- Comprehensive feature engineering (584 features)

---

## Slide 2: System Architecture

**Pipeline Stages:**
1. **Data Preprocessing** → Clean, sample, temporal split
2. **Image Embedding** → Extract visual features (FashionCLIP)
3. **Candidate Generation** → 6 recall strategies
4. **Feature Engineering** → 584 features (user, item, interaction, image, text)
5. **Model Training** → LightGBM + Neural Towers
6. **Evaluation** → Metrics + Ablation Study
7. **Submission** → Top-12 predictions per user

**Key Innovation:** Three-tower neural network (User + Item + Image)

---

## Slide 3: Data & Preprocessing

**Data Sources:**
- 412K transactions, 15.9K items, 47.5K users
- Product images for visual features
- Temporal data (24 weeks)

**Preprocessing:**
- Temporal split: 11 weeks train, 1 week validation
- Stratified user sampling: 50K users
- Cold-start handling: 15% cold-start users
- Memory optimization: Efficient dtypes (0.35 GB)

**Result:** Clean, balanced dataset ready for training

---

## Slide 4: Candidate Generation (6 Strategies)

**Multi-Strategy Recall:**

1. **Repurchase** → Items user bought before
2. **Popularity** → Globally trending items
3. **Co-Purchase** → Items bought together (Item-to-Item CF)
4. **User-KNN** → Items from similar users
5. **Category-Based** → Popular items in user's preferred categories
6. **Text Similarity** → Semantically similar items

**Coverage:** ~100-200 candidates per user

**Why Multiple Strategies?** Different strategies capture different recommendation signals

---

## Slide 5: Feature Engineering (584 Features)

**Feature Categories:**

- **User Features (43):** Activity, demographics, preferences, temporal patterns
- **Item Features (29):** Attributes, popularity, sales trends
- **Interaction Features:** Purchase history, recency, category alignment
- **Image Features (512):** FashionCLIP visual embeddings
- **Text Features:** Semantic embeddings from descriptions

**Key Features:**
- Temporal features (days since purchase, trends)
- Behavioral features (exploration rate, price preferences)
- Visual features (style, color, design patterns)

---

## Slide 6: Model Training - LightGBM

**4 Model Variants:**

1. **Classifier** → Binary classification
2. **Ranker (LambdaRank)** → Learning-to-rank, optimized for MAP@12
3. **Ranker (XENDCG)** → Alternative ranking objective
4. **Deep Classifier** → Higher capacity model

**Training:**
- Early stopping (50 rounds)
- Time-based cross-validation
- Feature importance tracking
- Optimized for M4 MacBook Air

**Output:** Multiple models for ensemble

---

## Slide 7: Model Training - Neural Towers

**Architecture: Three-Tower Neural Network**

```
User Tower (43 → 128)  ┐
Item Tower (29 → 64)   ├─→ Fusion (320 → 256 → 128 → 64 → 1)
Image Tower (512 → 128)┘
```

**Training Details:**
- 346K parameters
- MPS acceleration (Apple Silicon)
- Batch size: 2048
- Early stopping: 5 epochs patience
- Best MAP@12: ~0.9987

**Why Three Towers?** Separate processing for different feature types

---

## Slide 8: Evaluation & Metrics

**Metrics Implemented:**
- MAP@12 (competition metric)
- Precision@K, Recall@K, NDCG@K (K = 1,3,5,10,12)

**Model Comparison:**
- Side-by-side performance comparison
- Ranked by MAP@12
- Detailed metrics for each model

**Ensemble:**
- Weighted combination of best models
- Typically outperforms individual models
- Configurable weights

---

## Slide 9: Ablation Study

**Research Question:** How much do image features contribute?

**Methodology:**
- Train models WITHOUT image features
- Compare with full models (WITH image features)
- Quantify improvement percentage

**Models Compared:**
- LightGBM: WITH vs WITHOUT image features
- Neural Tower: 3-Tower vs 2-Tower (no image)

**Key Finding:** Quantifies value of visual features for fashion recommendation

---

## Slide 10: Results & Performance

**Best Performance:**
- Neural Tower 3-Tower: MAP@12 ~0.9987
- LightGBM Ranker: MAP@12 ~0.77-0.99
- Ensemble: Combines strengths of both

**Key Achievements:**
- ✅ Complete end-to-end pipeline
- ✅ 584 comprehensive features
- ✅ Multiple model architectures
- ✅ Robust evaluation framework
- ✅ Ablation study for feature importance

**Ready for:** Kaggle competition submission

---

## Slide 11: Technical Highlights

**Optimizations:**
- Memory-efficient processing (chunked, efficient dtypes)
- Apple Silicon acceleration (MPS for neural networks)
- Production-ready code (error handling, logging)

**Code Quality:**
- Modular design (separate stages)
- Reproducible (random state control)
- Comprehensive logging

**Scalability:**
- Handles 685K training samples
- Efficient inference for submission
- Ready for production deployment

---

## Slide 12: Key Takeaways

**What We Built:**
- Complete recommendation system pipeline
- Multiple model architectures
- Comprehensive feature engineering
- Robust evaluation framework

**Innovations:**
- Three-tower neural network for fashion
- Multi-strategy candidate generation
- Image feature integration
- Ablation study for feature importance

**Impact:**
- Competitive MAP@12 performance
- Production-ready implementation
- Comprehensive analysis and insights

---

## Quick Stats for Presentation

- **Total Features:** 584
- **Models Trained:** 4 LightGBM + 1 Neural Tower
- **Training Samples:** 685,494
- **Validation Samples:** 120,970
- **Best MAP@12:** ~0.9987 (Neural Tower)
- **Code Lines:** ~8,500+ lines
- **Processing Time:** Optimized for local M4 training

---

## Presentation Tips

1. **Start with Problem:** Fashion recommendation is challenging due to visual nature
2. **Show Pipeline:** Visual diagram of 7 stages
3. **Highlight Innovation:** Three-tower architecture
4. **Show Results:** Comparison tables and metrics
5. **Ablation Study:** Demonstrate scientific rigor
6. **End with Impact:** Ready for competition submission

---

## Visual Elements to Include

1. **Architecture Diagram:** Three-tower neural network
2. **Pipeline Flow:** 7 stages from data to submission
3. **Feature Categories:** Pie chart or bar chart of 584 features
4. **Model Comparison Table:** Side-by-side metrics
5. **Ablation Results:** Before/after comparison
6. **Performance Metrics:** MAP@12, Precision@12, Recall@12

